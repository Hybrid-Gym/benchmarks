"""Build verify-and-repair candidates from graded rollouts of other models.

A source is one graded rollout: its evaluated patches (``eval_snapshot.jsonl`` or
``output.jsonl``; rows with ``instance_id`` and ``test_result.git_patch``, last row
wins) and the grader's report (``output.report.json``, whose ``results`` carry
per-instance ``resolved`` and ``patch_applied``).

Each instance gets at most one candidate. Every instance with a flawed patch
(applied, non-empty, unresolved) gets one of those. Correct (resolved) patches are
added for other instances so that they make up ``--correct-fraction`` of the result,
so the agent also sees attempts that only need verifying. All choices are a
deterministic function of the instance id, and ``--max-per-repo`` limits pool skew
(R2E-Gym-Lite is about half numpy).

Usage:
    uv run python -m benchmarks.hybridgym_verifyrepair.build_candidates \\
        --source gpt5mini=<run_dir> --source dv4f=<run_dir> \\
        --select pool.txt --out candidates.jsonl
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from benchmarks.utils.patch_utils import remove_noise_from_patch


def _unit(key: str) -> float:
    """Deterministic pseudo-random number in [0, 1) derived from ``key``."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 16**8


def _by_hash(ids: Iterable[str]) -> list[str]:
    """Sort ids in a fixed pseudo-random order."""
    return sorted(ids, key=lambda i: _unit("order:" + i))


def _patches(path: Path) -> dict[str, str]:
    patches: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            patch = (row.get("test_result") or {}).get("git_patch") or ""
            patches[row["instance_id"]] = remove_noise_from_patch(patch)
    return patches


def load_source(run_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return (flawed, resolved) patches keyed by instance id for one graded run."""
    snapshot = run_dir / "eval_snapshot.jsonl"
    patches = _patches(snapshot if snapshot.exists() else run_dir / "output.jsonl")
    report = json.loads((run_dir / "output.report.json").read_text())
    if "results" not in report:
        raise ValueError(
            f"{run_dir}: expected an R2E-Gym report with per-instance 'results'"
        )
    flawed: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for result in report["results"]:
        iid = result["instance_id"]
        patch = patches.get(iid, "")
        if not patch.strip():
            continue
        if result.get("resolved"):
            resolved[iid] = patch
        elif result.get("patch_applied"):
            flawed[iid] = patch
    return flawed, resolved


def build_candidates(
    sources: dict[str, tuple[dict[str, str], dict[str, str]]],
    correct_fraction: float,
    max_per_repo: int | None,
    select: set[str] | None = None,
) -> list[dict[str, object]]:
    """Pick one candidate per instance; see the module docstring."""
    flawed: dict[str, list[tuple[str, str]]] = {}
    correct: dict[str, list[tuple[str, str]]] = {}
    for name, (source_flawed, source_resolved) in sorted(sources.items()):
        for iid, patch in source_flawed.items():
            flawed.setdefault(iid, []).append((name, patch))
        for iid, patch in source_resolved.items():
            correct.setdefault(iid, []).append((name, patch))
    if select is not None:
        flawed = {k: v for k, v in flawed.items() if k in select}
        correct = {k: v for k, v in correct.items() if k in select}

    if not 0 <= correct_fraction < 1:
        raise ValueError(f"correct_fraction must be in [0, 1), got {correct_fraction}")
    per_repo: Counter[str] = Counter()

    def fits(iid: str) -> bool:
        repo = iid.split("__")[0]
        if max_per_repo is not None and per_repo[repo] >= max_per_repo:
            return False
        per_repo[repo] += 1
        return True

    # Flawed candidates take the per-repo slots first; correct ones then fill up to
    # the target fraction of the (capped) flawed count within the same caps.
    chosen = [(iid, False, flawed[iid]) for iid in _by_hash(flawed) if fits(iid)]
    n_correct = round(correct_fraction * len(chosen) / (1 - correct_fraction))
    for iid in _by_hash(set(correct) - set(flawed)):
        if n_correct == 0:
            break
        if fits(iid):
            chosen.append((iid, True, correct[iid]))
            n_correct -= 1

    rows: list[dict[str, object]] = []
    for iid, is_correct, pool in sorted(chosen, key=lambda c: _unit("order:" + c[0])):
        name, patch = pool[int(_unit("pick:" + iid) * len(pool))]
        rows.append(
            {
                "instance_id": iid,
                "candidate_patch": patch,
                "source": name,
                "candidate_resolved": is_correct,
                "repo": iid.split("__")[0],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build verify-and-repair candidates from graded rollouts."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="name=RUN_DIR of a graded rollout (repeatable)",
    )
    parser.add_argument("--select", help="Only use instance ids listed in this file")
    parser.add_argument("--correct-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-repo", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Keep the first N rows")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sources = {}
    for spec in args.source:
        name, run_dir = spec.split("=", 1)
        sources[name] = load_source(Path(run_dir))
    select = None
    if args.select:
        select = {x.strip() for x in open(args.select) if x.strip()}
    rows = build_candidates(sources, args.correct_fraction, args.max_per_repo, select)
    if args.limit is not None:
        rows = rows[: args.limit]

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    kinds = Counter("correct" if r["candidate_resolved"] else "flawed" for r in rows)
    repos = Counter(str(r["repo"]) for r in rows)
    print(f"{len(rows)} candidates -> {args.out}: {dict(kinds)}; repos {dict(repos)}")


if __name__ == "__main__":
    main()
