#!/usr/bin/env python3
"""
R2E-Gym Evaluation Script

Scores OpenHands-generated patches against the R2E-Gym environments. A faithful,
self-contained reimplementation of R2E-Gym's ``_calculate_reward_r2e`` (no
dependency on the ``r2egym`` pip package): for each instance it starts a fresh
container from the dataset's ``docker_image``, reproduces R2E-Gym's ``setup_env``,
applies the model patch in ``/testbed``, runs the baked ``run_tests.sh``, and
awards reward 1.0 iff the parsed pytest results match ``expected_output_json``
exactly.

Disk safety: each base image is removed after its instance (peak disk is roughly
``--workers`` x image size); pass ``--keep-images`` to disable.

Usage:
    uv run r2egym-eval path/to/output.jsonl
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.r2egym import constants
from benchmarks.r2egym.config import EVAL_DEFAULTS
from benchmarks.r2egym.dataset import get_dataset
from benchmarks.utils.patch_utils import remove_files_from_patch
from openhands.sdk import get_logger


logger = get_logger(__name__)

# PATH R2E-Gym exports into every ``docker exec`` (see docker.py DOCKER_PATH).
DOCKER_PATH = (
    "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
REPO_PATH = constants.REPO_PATH_IN_IMAGE  # /testbed
ALT_PATH = "/root"
# Files R2E-Gym hides from the agent by relocating them to /root (SKIP_FILES_NEW).
SKIP_FILES_NEW = ["run_tests.sh", "r2e_tests"]

# Strip ANSI escape codes and carriage returns, exactly like R2E-Gym's run().
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\r")


# --------------------------------------------------------------------------- #
# Faithful copies of R2E-Gym's parser + reward logic
# (r2egym/repo_analysis/execution_log_parser.py, docker._calculate_reward_r2e)
# --------------------------------------------------------------------------- #
def parse_log_pytest(log: str | None) -> dict[str, str]:
    """Parse a pytest ``short test summary info`` block into {test: status}."""
    if log is None:
        return {}
    test_status_map: dict[str, str] = {}
    if "short test summary info" not in log:
        return test_status_map
    log = log.split("short test summary info")[1]
    log = log.strip()
    for line in log.split("\n"):
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            test_status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            test_status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            try:
                test_name = ".".join(line.split("::")[1:])
            except IndexError:
                test_name = line
            test_name = test_name.split(" - ")[0]
            test_status_map[test_name] = "ERROR"
    return test_status_map


def decolor_dict_keys(d: dict[str, str]) -> dict[str, str]:
    def decolor(k: str) -> str:
        return re.sub(r"\x1b\[\d+m", "", k)

    return {decolor(k): v for k, v in d.items()}


def compute_reward(test_output: str, expected_output_json: str) -> float:
    """R2E-Gym exact-match reward: 1.0 iff parsed == expected, else 0.0."""
    parse = decolor_dict_keys(parse_log_pytest(test_output))
    expected = decolor_dict_keys(json.loads(expected_output_json))

    parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
    expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

    if len(parse) != len(expected):
        return 0.0
    for k in parse.keys():
        if not k:
            continue
        if k not in expected or parse[k] != expected[k]:
            return 0.0
    return 1.0


# --------------------------------------------------------------------------- #
# Raw-docker execution helpers (mirrors r2egym DockerRuntime.run / setup_env)
# --------------------------------------------------------------------------- #
def _dexec(container: str, cmd: str, timeout: int) -> tuple[str, int]:
    """Run ``cmd`` inside the container (cwd=/testbed, R2E PATH) with a timeout.

    Combines stdout+stderr and strips ANSI/\\r, matching R2E-Gym's run().
    """
    # Run the (possibly compound) command inside an inner shell so the container
    # `timeout` wraps a single program; `timeout {t} <compound>` would otherwise be
    # a shell syntax error and silently no-op via the outer `|| true`.
    wrapped = f"timeout {timeout} /bin/sh -c {shlex.quote(cmd)}"
    try:
        p = subprocess.run(
            [
                "docker",
                "exec",
                "-w",
                REPO_PATH,
                "-e",
                f"PATH={DOCKER_PATH}",
                container,
                "/bin/sh",
                "-c",
                wrapped,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )
    except subprocess.TimeoutExpired:
        return f"The command took too long to execute (>{timeout}s)", -1
    out = (p.stdout or "") + (p.stderr or "")
    out = _ANSI_RE.sub("", out)
    return out, p.returncode


def _setup_env(container: str) -> None:
    """Reproduce R2E-Gym's standard-r2e setup_env. Each step is best-effort."""
    steps = [
        f"ln -s {REPO_PATH}/.venv {ALT_PATH}/.venv",
        f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python",
        f"ln -s {REPO_PATH}/.venv/bin/python {ALT_PATH}/.local/bin/python3",
        f"find {REPO_PATH}/.venv/bin -type f -executable "
        f"-exec ln -sf {{}} {ALT_PATH}/.local/bin/ \\;",
        "uv pip install chardet",
        "find . -name '*.pyc' -delete",
        "find . -name '__pycache__' -exec rm -rf {} +",
        "find /r2e_tests -name '*.pyc' -delete",
        "find /r2e_tests -name '__pycache__' -exec rm -rf {} +",
    ]
    for skip_file in SKIP_FILES_NEW:
        steps.append(f"mv {REPO_PATH}/{skip_file} {ALT_PATH}/{skip_file}")
    steps.append(f"mv /r2e_tests {ALT_PATH}/r2e_tests")
    steps.append(f"ln -s {ALT_PATH}/r2e_tests {REPO_PATH}/r2e_tests")

    for step in steps:
        # Best-effort: an already-done step or a missing path must not abort setup.
        _dexec(container, f"{step} >/dev/null 2>&1 || true", timeout=120)


def _resolve_run_tests_path(container: str) -> str:
    """Locate the baked test script (normally /root/run_tests.sh)."""
    for candidate in (
        f"{ALT_PATH}/run_tests.sh",
        f"{REPO_PATH}/run_tests.sh",
        "/run_tests.sh",
    ):
        out, rc = _dexec(container, f"test -f {candidate} && echo OK", timeout=30)
        if rc == 0 and "OK" in out:
            return candidate
    return f"{ALT_PATH}/run_tests.sh"


def _container_name(instance_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)[:80]
    return f"r2e_eval_{safe}"


def evaluate_one(
    instance_id: str,
    docker_image: str,
    patch: str,
    expected_output_json: str,
    timeout: int,
    keep_images: bool,
) -> dict:
    """Score a single instance in a fresh container; always clean up."""
    result: dict = {
        "instance_id": instance_id,
        "docker_image": docker_image,
        "resolved": False,
        "reward": 0.0,
        "empty_patch": not bool(patch.strip()),
        "patch_applied": None,
        "error": None,
    }
    container = _container_name(instance_id)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True, text=True)

    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container,
            "--entrypoint",
            "tail",
            docker_image,
            "-f",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        result["error"] = f"docker run failed: {run.stderr.strip()[:400]}"
        return result

    try:
        _setup_env(container)

        if patch.strip():
            with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
                f.write(patch)
                patch_host = f.name
            try:
                cp = subprocess.run(
                    ["docker", "cp", patch_host, f"{container}:/patch.diff"],
                    capture_output=True,
                    text=True,
                )
            finally:
                os.unlink(patch_host)
            if cp.returncode != 0:
                result["error"] = f"docker cp failed: {cp.stderr.strip()[:400]}"
                return result
            apply_out, apply_rc = _dexec(
                container, "git apply --whitespace=fix /patch.diff", timeout=300
            )
            result["patch_applied"] = apply_rc == 0
            if apply_rc != 0:
                result["apply_output"] = apply_out[:2000]
        else:
            result["patch_applied"] = False

        run_tests_path = _resolve_run_tests_path(container)
        test_out, _ = _dexec(container, f"bash {run_tests_path}", timeout=timeout)
        reward = compute_reward(test_out, expected_output_json)
        result["reward"] = reward
        result["resolved"] = reward == 1.0
        return result
    except Exception as e:  # noqa: BLE001 - never let one instance kill the run
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], capture_output=True, text=True
        )
        if not keep_images:
            rmi = subprocess.run(
                ["docker", "rmi", "-f", docker_image],
                capture_output=True,
                text=True,
            )
            if rmi.returncode == 0:
                logger.info("[cleanup] removed image %s", docker_image)


def load_predictions(input_file: str) -> dict[str, str]:
    """Read OpenHands output.jsonl -> {instance_id: git_patch}."""
    preds: dict[str, str] = {}
    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Line %d: invalid JSON - %s", line_num, e)
                continue
            instance_id = data.get("instance_id")
            if not instance_id:
                logger.warning("Line %d: missing instance_id", line_num)
                continue
            patch = (data.get("test_result") or {}).get("git_patch", "") or ""
            patch = remove_files_from_patch(patch, constants.SETUP_FILES_TO_REMOVE)
            preds[instance_id] = patch
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OpenHands patches against R2E-Gym environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run r2egym-eval output.jsonl
    uv run r2egym-eval output.jsonl --dataset R2E-Gym/R2E-Gym-Subset --split train
    uv run r2egym-eval output.jsonl --workers 2 --timeout 600 --keep-images
        """,
    )
    parser.add_argument("input_file", help="Path to the OpenHands output.jsonl file")
    parser.add_argument("--dataset", help="R2E-Gym dataset (for expected outputs)")
    parser.add_argument("--split", help="Dataset split")
    parser.add_argument(
        "--workers", type=int, help="Concurrent instances (peak disk scales with this)"
    )
    parser.add_argument(
        "--timeout", type=int, help="Per-instance test-run timeout (seconds)"
    )
    parser.add_argument(
        "--select", help="Path to a text file of instance IDs to evaluate"
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Do NOT remove each base image after use (uses more disk)",
    )
    parser.add_argument(
        "--output-file",
        help="Where to write the report JSON "
        "(default: input_file with .report.json extension)",
    )
    parser.set_defaults(**EVAL_DEFAULTS)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error("Input file does not exist: %s", input_file)
        sys.exit(1)

    predictions = load_predictions(str(input_file))
    logger.info("Loaded %d predictions from %s", len(predictions), input_file)

    # Load expected outputs + base images from the dataset.
    df = get_dataset(
        dataset_name=args.dataset,
        split=args.split,
        eval_limit=None,
        selected_instances_file=args.select,
    )
    meta: dict[str, dict] = {}
    for _, row in df.iterrows():
        meta[str(row["instance_id"])] = {
            "docker_image": str(row["docker_image"]),
            "expected_output_json": row["expected_output_json"],
        }

    # Intersect predictions with the (optionally selected) dataset rows.
    to_eval = [iid for iid in predictions if iid in meta]
    missing_meta = [iid for iid in predictions if iid not in meta]
    if missing_meta:
        logger.warning(
            "%d predicted instances not found in dataset (skipped): %s",
            len(missing_meta),
            ", ".join(missing_meta[:5]) + ("..." if len(missing_meta) > 5 else ""),
        )
    if not to_eval:
        logger.error("No predictions overlap the dataset; nothing to evaluate.")
        sys.exit(1)

    logger.info(
        "Evaluating %d instances with %d workers (timeout=%ds, keep_images=%s)",
        len(to_eval),
        args.workers,
        args.timeout,
        args.keep_images,
    )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                evaluate_one,
                iid,
                meta[iid]["docker_image"],
                predictions[iid],
                meta[iid]["expected_output_json"],
                args.timeout,
                args.keep_images,
            ): iid
            for iid in to_eval
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = (
                "RESOLVED"
                if res["resolved"]
                else ("EMPTY" if res["empty_patch"] else "unresolved")
            )
            if res["error"]:
                status = f"ERROR ({res['error'][:60]})"
            logger.info(
                "[%d/%d] %s -> %s",
                len(results),
                len(to_eval),
                res["instance_id"],
                status,
            )

    resolved = [r for r in results if r["resolved"]]
    empty = [r for r in results if r["empty_patch"] and not r["resolved"]]
    errored = [r for r in results if r["error"]]
    total = len(results)
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "total_predictions": len(predictions),
        "total_evaluated": total,
        "resolved": len(resolved),
        "unresolved": total - len(resolved),
        "empty_patch": len(empty),
        "errored": len(errored),
        "resolve_rate": (len(resolved) / total) if total else 0.0,
        "resolved_ids": sorted(r["instance_id"] for r in resolved),
        "skipped_no_metadata": missing_meta,
        "results": sorted(results, key=lambda r: r["instance_id"]),
    }

    out_path = (
        Path(args.output_file)
        if args.output_file
        else input_file.with_suffix(".report.json")
    )
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 80)
    print(
        f"R2E-Gym evaluation: {len(resolved)}/{total} resolved "
        f"({report['resolve_rate'] * 100:.1f}%)"
    )
    print(f"  empty patches: {len(empty)}   errored: {len(errored)}")
    print(f"Report written to: {out_path}")
    print(json.dumps({"report_json": str(out_path)}))


if __name__ == "__main__":
    main()
