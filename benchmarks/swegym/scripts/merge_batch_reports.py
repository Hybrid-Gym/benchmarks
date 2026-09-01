#!/usr/bin/env python3
"""Merge per-batch SWE-Gym eval reports into one report per model.

The batch runner writes `<batch>.<model>.json`, each covering ~20 instances. This
folds them into `report_<model>.json` with the same field names the single-shot
harness produces, so the r2egym push path (`resolved_ids` -> per-row `resolved`
labels) consumes it unchanged.

Deliberately reports COVERAGE rather than assuming it: a batch that failed leaves no
file, and silently merging the rest would understate the denominator and inflate the
resolve rate. Missing batches are listed.

Usage:
    python benchmarks/swegym/scripts/merge_batch_reports.py \
        --reports-dir eval_outputs/swegym_outputs/eval_reports \
        --batch-dir  eval_outputs/swegym_outputs/eval_batches \
        --models gpt5mini qwen80b kimi25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ID_FIELDS = [
    "resolved_ids",
    "unresolved_ids",
    "error_ids",
    "empty_patch_ids",
    "completed_ids",
    "submitted_ids",
    "unstopped_ids",
]


def merge_model(reports_dir: Path, batch_dir: Path, model: str) -> dict:
    batches = sorted(batch_dir.glob("batch_*.txt"))
    merged: dict[str, set[str]] = {k: set() for k in ID_FIELDS}
    missing: list[str] = []

    for b in batches:
        r = reports_dir / f"{b.stem}.{model}.json"
        if not r.exists():
            missing.append(b.stem)
            continue
        try:
            d = json.loads(r.read_text())
        except json.JSONDecodeError:
            missing.append(b.stem)
            continue
        for k in ID_FIELDS:
            merged[k].update(d.get(k) or [])

    selected = set()
    for b in batches:
        selected.update(
            line.strip() for line in b.read_text().splitlines() if line.strip()
        )

    out: dict = {"model": model}
    for k in ID_FIELDS:
        out[k] = sorted(merged[k])
        out[k.replace("_ids", "_instances")] = len(merged[k])

    out["selected_instances"] = len(selected)
    out["batches_total"] = len(batches)
    out["batches_missing"] = missing
    out["coverage"] = (
        round(len(merged["completed_ids"]) / len(selected), 4) if selected else 0.0
    )
    # Resolve rate over what was actually scored, and over the full selection. They
    # differ whenever coverage < 1, and quoting only the first would overstate.
    comp = len(merged["completed_ids"]) or 1
    out["resolve_rate_of_completed"] = round(len(merged["resolved_ids"]) / comp, 4)
    out["resolve_rate_of_selected"] = (
        round(len(merged["resolved_ids"]) / len(selected), 4) if selected else 0.0
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    reports_dir, batch_dir = Path(args.reports_dir), Path(args.batch_dir)
    for model in args.models:
        out = merge_model(reports_dir, batch_dir, model)
        dest = reports_dir / f"report_{model}.json"
        dest.write_text(json.dumps(out, indent=2))
        miss = out["batches_missing"]
        print(
            f"{model:10s} resolved={out['resolved_instances']:5d} "
            f"completed={out['completed_instances']:5d}/{out['selected_instances']} "
            f"({out['coverage'] * 100:.1f}% coverage)  "
            f"rate_of_completed={out['resolve_rate_of_completed'] * 100:.1f}%  "
            f"-> {dest.name}"
        )
        if miss:
            print(
                f"           MISSING {len(miss)} batch(es): {', '.join(miss[:8])}"
                f"{' ...' if len(miss) > 8 else ''}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
