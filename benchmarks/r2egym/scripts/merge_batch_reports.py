#!/usr/bin/env python3
"""Merge the per-batch reports written by batch_eval.sh into one output.report.json.

batch_eval.sh evaluates a run in batches of ~25 so it can wait for Docker Hub pull
budget and free disk between them, and writes one `<BATCH_DIR>/batch_NN.report.json`
per batch. Nothing then combined them, so a finished eval left no run-level report --
the file every downstream step (resolved labels, HF push) actually reads.

The output schema matches what r2egym-eval writes for a single-shot eval, so the two
paths are interchangeable downstream.

Usage:
    python3 benchmarks/r2egym/scripts/merge_batch_reports.py \
        --batch-dir eval_outputs/tmp/gpt5mini_batches \
        --out <run_dir>/output.report.json
"""

import argparse
import json
import sys
from pathlib import Path


def merge(batch_dir: Path) -> tuple[dict, int, int]:
    reports = sorted(batch_dir.glob("batch_*.report.json"))
    if not reports:
        raise SystemExit(f"no batch_*.report.json under {batch_dir}")

    # Last write wins per instance: a batch that was re-run (its report rewritten)
    # supersedes the earlier verdict, and re-running a batch is the documented way to
    # repair one. Dedupe rather than concatenating, so a re-run cannot double-count.
    by_id: dict[str, dict] = {}
    dupes = 0
    for path in reports:
        try:
            results = json.loads(path.read_text())["results"]
        except (json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"{path} is not a batch report: {e}")
        for row in results:
            if row["instance_id"] in by_id:
                dupes += 1
            by_id[row["instance_id"]] = row

    results = [by_id[k] for k in sorted(by_id)]
    resolved_ids = [r["instance_id"] for r in results if r.get("resolved")]
    total = len(results)
    return (
        {
            "total_instances": total,
            "resolved_instances": len(resolved_ids),
            "resolve_rate": round(100.0 * len(resolved_ids) / total, 2)
            if total
            else 0.0,
            "errored_instances": sum(1 for r in results if r.get("error")),
            "resolved_ids": resolved_ids,
            "results": results,
        },
        len(reports),
        dupes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing report (refused by default)",
    )
    args = ap.parse_args()

    # A run-level report may have come from a single-shot eval covering instances the
    # batches never had; silently replacing it would lose those rows.
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists; pass --force to overwrite")

    report, n_batches, dupes = merge(args.batch_dir)
    args.out.write_text(json.dumps(report, indent=2))

    print(
        f"merged {n_batches} batches -> {args.out}\n"
        f"  total={report['total_instances']} "
        f"resolved={report['resolved_instances']} "
        f"({report['resolve_rate']}%) "
        f"errored={report['errored_instances']}"
        + (f"\n  note: {dupes} duplicate rows, last-write-wins" if dupes else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
