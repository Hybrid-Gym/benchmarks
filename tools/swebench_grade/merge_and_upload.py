#!/usr/bin/env python3
"""Merge a swebench harness report's resolved_ids into predictions.jsonl and
push predictions.jsonl + eval_report.json back to the source HF dataset path.

The harness (swebench.harness.run_evaluation) ignores --report_dir for the
report file itself: it always writes `<model_name_or_path>.<run_id>.json` to
the current working directory, where model_name_or_path comes from the
predictions.jsonl rows (NOT necessarily the --model/--run_id label this script
was invoked with -- those only match by convention when a run covers the full
500-instance set with run_id == the checkpoint name). This script derives the
report filename from the actual predictions.jsonl content instead of assuming
it equals args.model, and sizes resolve_rate off len(rows) rather than the
report's own total_instances (which is always 500, the full SWE-bench Verified
dataset size, even when predictions.jsonl only covers a subset).
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


GOLDEN_PATCHES_PATH = Path(__file__).parent / "golden_patches.json"


def patch2file_paths(patch: str) -> set:
    """Files touched by a unified diff. Ported from extra_eval.py (yiqing branch)."""
    file_paths = set()
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                file_paths.add(parts[2][2:])  # strip 'a/' prefix
    return file_paths


def compute_localized_rate(rows: list, total: int) -> tuple:
    """Returns (non_empty_rate, localized_rate) against golden_patches.json."""
    golden = json.loads(GOLDEN_PATCHES_PATH.read_text())
    golden_files = {iid: patch2file_paths(p) for iid, p in golden.items()}

    non_empty = 0
    localized = 0
    for row in rows:
        if not row.get("model_patch"):
            continue
        non_empty += 1
        gf = golden_files.get(row["instance_id"])
        if gf and (patch2file_paths(row["model_patch"]) & gf):
            localized += 1

    return (non_empty / total if total else 0.0, localized / total if total else 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dir", required=True, help="local dir with predictions.jsonl")
    p.add_argument("--repo", required=True, help="HF dataset repo, e.g. org/name")
    p.add_argument(
        "--flat",
        action="store_true",
        help="Push to predictions.jsonl / eval_report.json at repo root instead of "
        "under a <model>/ prefix (for one-model-per-repo datasets).",
    )
    args = p.parse_args()

    model_dir = Path(args.dir)
    predictions_path = model_dir / "predictions.jsonl"

    rows = [
        json.loads(line)
        for line in predictions_path.read_text().splitlines()
        if line.strip()
    ]
    model_name_or_path = rows[0]["model_name_or_path"]
    report_path = Path(f"{model_name_or_path}.{args.model}.json")

    if not report_path.exists():
        raise SystemExit(
            f"harness report not found at {report_path} (cwd={Path.cwd()}); "
            "did run_evaluation.py finish successfully?"
        )

    report = json.loads(report_path.read_text())
    resolved_ids = set(report["resolved_ids"])

    for row in rows:
        row["resolved"] = row["instance_id"] in resolved_ids

    predictions_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    total = len(rows)
    resolved_n = report["resolved_instances"]
    non_empty_rate, localized_rate = compute_localized_rate(rows, total)
    eval_report = {
        "model": args.model,
        "harness_run_id": args.model,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "total_instances": total,
        "resolved_instances": resolved_n,
        "resolve_rate": resolved_n / total if total else 0.0,
        "unresolved_instances": report["unresolved_instances"],
        "empty_patch_instances": report["empty_patch_instances"],
        "non_empty_rate": non_empty_rate,
        "localized_rate": localized_rate,
        "error_instances": report["error_instances"],
        "error_ids": report["error_ids"],
    }
    eval_report_path = model_dir / "eval_report.json"
    eval_report_path.write_text(json.dumps(eval_report, indent=2) + "\n")

    print(f"=== {args.model} ===")
    print(f"  resolved: {resolved_n}/{total} ({eval_report['resolve_rate']:.1%})")
    print(
        f"  empty_patch: {eval_report['empty_patch_instances']} (non_empty_rate {non_empty_rate:.1%})"
    )
    print(f"  localized_rate: {localized_rate:.1%}")
    print(f"  errors (failed to build/apply/run): {eval_report['error_instances']}")
    if eval_report["error_ids"]:
        print(f"  error_ids: {eval_report['error_ids']}")

    prefix = "" if args.flat else f"{args.model}/"
    api = HfApi()
    for local_path, repo_path in [
        (predictions_path, f"{prefix}predictions.jsonl"),
        (eval_report_path, f"{prefix}eval_report.json"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  uploaded -> {args.repo}:{repo_path}")

    report_path.unlink()


if __name__ == "__main__":
    main()
