#!/usr/bin/env python3
"""Merge a swebench harness report's resolved_ids into predictions.jsonl and
push predictions.jsonl + eval_report.json back to the source HF dataset path.

The harness (swebench.harness.run_evaluation) ignores --report_dir for the
report file itself: it always writes `<model>.<run_id>.json` to the current
working directory. This script expects that file to exist at repo root
(cwd == model == run_id, which is how run_swebench_eval.sh invokes it).
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dir", required=True, help="local dir with predictions.jsonl")
    p.add_argument("--repo", required=True, help="HF dataset repo, e.g. org/name")
    args = p.parse_args()

    model_dir = Path(args.dir)
    predictions_path = model_dir / "predictions.jsonl"
    report_path = Path(f"{args.model}.{args.model}.json")

    if not report_path.exists():
        raise SystemExit(
            f"harness report not found at {report_path} (cwd={Path.cwd()}); "
            "did run_evaluation.py finish successfully?"
        )

    report = json.loads(report_path.read_text())
    resolved_ids = set(report["resolved_ids"])

    rows = [
        json.loads(line)
        for line in predictions_path.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        row["resolved"] = row["instance_id"] in resolved_ids

    predictions_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    total = report["total_instances"]
    resolved_n = report["resolved_instances"]
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
        "error_instances": report["error_instances"],
        "incomplete_instances": len(report.get("incomplete_ids", [])),
        "error_ids": report["error_ids"],
    }
    eval_report_path = model_dir / "eval_report.json"
    eval_report_path.write_text(json.dumps(eval_report, indent=2) + "\n")

    print(f"=== {args.model} ===")
    print(f"  resolved: {resolved_n}/{total} ({eval_report['resolve_rate']:.1%})")
    print(f"  empty_patch: {eval_report['empty_patch_instances']}")
    print(f"  errors (failed to build/apply/run): {eval_report['error_instances']}")
    if eval_report["error_ids"]:
        print(f"  error_ids: {eval_report['error_ids']}")

    api = HfApi()
    for local_path, repo_path in [
        (predictions_path, f"{args.model}/predictions.jsonl"),
        (eval_report_path, f"{args.model}/eval_report.json"),
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
