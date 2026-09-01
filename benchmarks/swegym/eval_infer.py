#!/usr/bin/env python3
"""Prepare SWE-Gym rollouts for the vendored SWE-Bench-Fork evaluator.

The stock `swebench` (4.1.0) cannot score SWE-Gym at all -- its spec map covers 66
repos and none of SWE-Gym's 11. The working harness is `SWE-Gym/SWE-Bench-Fork`,
vendored at `vendor/swegym-swebench` and installed into its own `.venv-swegym-eval`
so it cannot disturb the 4.1.0 the SWE-bench benchmark uses.

This module does the two things that harness will not do for us:

  prepare   rollout output.jsonl  ->  predictions.jsonl the harness accepts
  batches   instance id list      ->  batch_NNN.txt files sized for our disk

Why batching is not optional: an eval image averages ~4GB and the harness does NOT
delete images when it is done (it reports `Unremoved images: N` and leaves them).
1500 instances resident at once is ~6TB against ~120GB of free disk on a shared box,
so batches must be small and the caller must delete each batch's images afterwards.

Usage:
    python -m benchmarks.swegym.eval_infer prepare \
        --run-dir eval_outputs/swegym_outputs/.../swegym-gpt5mini-1500 \
        --model-name swegym-gpt5mini-1500 --out preds.jsonl [--select ids.txt]

    python -m benchmarks.swegym.eval_infer batches \
        --select eval_outputs/swegym_select_1500.txt --size 20 --out-dir batches/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_last_rows(output_jsonl: Path) -> dict[str, dict[str, Any]]:
    """Map instance_id -> its LAST row.

    A retried instance appears more than once and only the final attempt reflects
    what the run actually produced; taking the first (or all) rows would score a
    superseded attempt. This is the same last-row-wins rule the r2egym push needed.
    """
    rows: dict[str, dict[str, Any]] = {}
    with open(output_jsonl, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("instance_id")
            if iid:
                rows[iid] = row
    return rows


def extract_patch(row: dict[str, Any]) -> str:
    """The model patch lives at test_result.git_patch in our rollout schema."""
    return ((row.get("test_result") or {}).get("git_patch") or "").strip()


def cmd_prepare(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_jsonl = run_dir / "output.jsonl"
    if not out_jsonl.exists():
        print(f"no output.jsonl in {run_dir}", file=sys.stderr)
        return 1

    rows = load_last_rows(out_jsonl)

    select: set[str] | None = None
    if args.select:
        select = {
            line.strip() for line in open(args.select, encoding="utf-8") if line.strip()
        }

    written = empty = skipped = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for iid, row in sorted(rows.items()):
            if select is not None and iid not in select:
                skipped += 1
                continue
            patch = extract_patch(row)
            if not patch:
                empty += 1
            # Empty patches are written anyway: the harness scores them as
            # unresolved, which is the truthful denominator. Dropping them would
            # silently inflate the resolve rate.
            fh.write(
                json.dumps(
                    {
                        "instance_id": iid,
                        "model_patch": patch,
                        "model_name_or_path": args.model_name,
                    }
                )
                + "\n"
            )
            written += 1

    print(
        f"wrote {written} predictions ({empty} with empty patch, {skipped} outside "
        f"selection) -> {args.out}"
    )
    return 0


def cmd_batches(args: argparse.Namespace) -> int:
    ids = [line.strip() for line in open(args.select, encoding="utf-8") if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable, deterministic split so a resumed run reuses the same batch files and
    # the "skip batches whose report exists" logic stays valid.
    ids.sort()
    n = 0
    for i in range(0, len(ids), args.size):
        chunk = ids[i : i + args.size]
        p = out_dir / f"batch_{i // args.size:04d}.txt"
        p.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        n += 1
    print(f"wrote {n} batches of <= {args.size} ids -> {out_dir}")
    return 0


def sanitized_image(instance_id: str) -> str:
    """Published (Docker Hub) image name: lowercase AND `__` -> `_s_`.

    Rule lives in build_images.py:32.
    """
    return (
        f"xingyaoww/sweb.eval.x86_64.{instance_id.lower().replace('__', '_s_')}:latest"
    )


def local_image(instance_id: str) -> str:
    """Local tag the harness looks for: lowercase, but `__` PRESERVED.

    These two transforms differ and the difference is easy to miss, because it only
    shows up on the one SWE-Gym repo with capitals. `make_test_spec` lowercases the
    instance id (logging "contains capital letters. Converting to lowercase.") and
    builds `sweb.eval.x86_64.project-monai__monai-1010:latest` -- keeping the double
    underscore. Retagging to the original-case id instead makes every image lookup
    miss for all 233 Project-MONAI instances, and the harness would then try to BUILD
    them, which is exactly what the retag exists to avoid.
    """
    return f"sweb.eval.x86_64.{instance_id.lower()}:latest"


def cmd_images(args: argparse.Namespace) -> int:
    """Print `<remote> <local>` pairs for a batch, for the shell to pull and retag."""
    for line in open(args.select, encoding="utf-8"):
        iid = line.strip()
        if iid:
            print(f"{sanitized_image(iid)} {local_image(iid)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="swegym-eval-prep")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="rollout output.jsonl -> predictions.jsonl")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--select")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("batches", help="split an id list into batch files")
    p.add_argument("--select", required=True)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_batches)

    p = sub.add_parser(
        "images", help="print '<remote> <local>' image pairs for a batch"
    )
    p.add_argument("--select", required=True)
    p.set_defaults(func=cmd_images)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
