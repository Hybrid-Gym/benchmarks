#!/usr/bin/env python3
"""Rebuild a full-dataset eval report by scanning per-instance report.json files
directly from logs/run_evaluation/, instead of trusting run_evaluation.py's own
make_run_report(). Needed after a partial retry (--instance_ids subset), since
that call filters full_dataset down to the subset and would otherwise produce a
report scoped to only the retried instances rather than the whole 500.
"""

import argparse
import json
from pathlib import Path


RUN_EVAL_LOG_DIR = Path("logs/run_evaluation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dir", required=True, help="local dir with predictions.jsonl")
    args = p.parse_args()

    model_dir = Path(args.dir)
    predictions_path = model_dir / "predictions.jsonl"
    rows = [
        json.loads(line)
        for line in predictions_path.read_text().splitlines()
        if line.strip()
    ]

    report_base = RUN_EVAL_LOG_DIR / args.model / args.model

    resolved_ids, unresolved_ids, error_ids, empty_patch_ids = [], [], [], []
    for row in rows:
        iid = row["instance_id"]
        if not row.get("model_patch"):
            empty_patch_ids.append(iid)
            continue
        report_file = report_base / iid / "report.json"
        if not report_file.exists():
            error_ids.append(iid)
            continue
        try:
            content = report_file.read_text().strip()
            report = json.loads(content)
            if report[iid]["resolved"]:
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)
        except (json.JSONDecodeError, KeyError):
            error_ids.append(iid)

    total = len(rows)
    out = {
        "model": args.model,
        "harness_run_id": args.model,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "total_instances": total,
        "resolved_instances": len(resolved_ids),
        "resolve_rate": len(resolved_ids) / total if total else 0.0,
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "resolved_ids": sorted(resolved_ids),
        "error_ids": sorted(error_ids),
    }
    report_path = Path(f"{args.model}.{args.model}.json")
    report_path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"=== {args.model} (rebuilt from full report.json scan) ===")
    print(
        f"  resolved: {out['resolved_instances']}/{total} ({out['resolve_rate']:.1%})"
    )
    print(f"  unresolved: {out['unresolved_instances']}")
    print(f"  empty_patch: {out['empty_patch_instances']}")
    print(f"  errors: {out['error_instances']}")
    if out["error_ids"]:
        print(f"  error_ids: {out['error_ids']}")
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
