"""Attach judge verdicts to a pushed HF trajectory dataset as extra label columns.

judge.py writes verdicts to a local JSONL keyed by instance_id; this joins them back
onto the dataset they were judged from so the labels travel with the data and can be
used for data selection (`ds.filter(lambda r: r["read_after_narrowing"])`).

Adds four columns and leaves the existing ones untouched:
  broad_then_narrow, multi_round_refinement, read_after_narrowing  (bool)
  judge_notes                                                      (str)

Usage:
  python tools/funclocalize_judge/push_labels.py \\
      --repo synthetic-code-training/func_localize_claude45_1457i \\
      --verdicts-dir eval_outputs/funclocalize_judge \\
      --dry-run

  # push a labelled copy instead of a new revision of the source repo
  python tools/funclocalize_judge/push_labels.py --repo ... --dest-suffix _labeled

A verdicts file may hold several rows per instance (a retry appends rather than
rewrites), so the LAST row per id wins and error rows are dropped — the same rule
judge.py uses when it resumes. Every dataset row must end up with a verdict or the
push aborts: a partially labelled dataset silently reads as "these ones are False".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from datasets import Dataset


LABELS = ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing")


def load_verdicts(path: Path) -> dict[str, dict]:
    """Last non-error verdict per instance id."""
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        out[row["instance_id"]] = row
    return out


def label_dataset(repo: str, split: str, verdicts: dict[str, dict]) -> Dataset:
    from datasets import load_dataset

    # load_dataset is typed as a union over Dataset/IterableDataset/*Dict; asking for
    # a concrete split always yields a Dataset, so narrow it before column work.
    ds = cast("Dataset", load_dataset(repo, split=split))
    ids = cast("list[str]", ds["instance_id"])
    missing = [i for i in ids if i not in verdicts]
    if missing:
        sys.exit(
            f"error: {len(missing)} of {len(ids)} rows in {repo} have no verdict "
            f"(e.g. {missing[:3]}); re-run judge.py before pushing"
        )

    def add_labels(row: dict) -> dict[str, object]:
        v = verdicts[row["instance_id"]]
        cols: dict[str, object] = {k: bool(v[k]) for k in LABELS}
        cols["judge_notes"] = v.get("notes", "")
        return cols

    return ds.map(add_labels)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo", action="append", default=[], required=True, help="Source dataset repo"
    )
    p.add_argument(
        "--verdicts-dir",
        default="eval_outputs/funclocalize_judge",
        help="Holds <dataset-name>.verdicts.jsonl for each --repo",
    )
    p.add_argument("--split", default="train")
    p.add_argument(
        "--dest-suffix",
        default="",
        help="Push to <repo><suffix> instead of a new revision of --repo",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for repo in args.repo:
        name = repo.split("/")[-1]
        vpath = Path(args.verdicts_dir) / f"{name}.verdicts.jsonl"
        if not vpath.exists():
            sys.exit(f"error: no verdicts for {repo} at {vpath}")

        verdicts = load_verdicts(vpath)
        ds = label_dataset(repo, args.split, verdicts)
        dest = f"{repo}{args.dest_suffix}"

        rates = {k: sum(ds[k]) for k in LABELS}
        print(f"\n{repo} -> {dest}")
        print(
            f"  rows {ds.num_rows}  verdicts {len(verdicts)}  columns {ds.column_names}"
        )
        for k, n in rates.items():
            print(f"  {k:<24} {n:>5} / {ds.num_rows}  ({100 * n / ds.num_rows:.1f}%)")

        if args.dry_run:
            print("  [dry-run] not pushed")
            continue
        ds.push_to_hub(dest, split=args.split)
        print(f"  pushed -> https://huggingface.co/datasets/{dest}")


if __name__ == "__main__":
    main()
