"""Re-grade candidates and drop the ones whose label cannot be trusted.

A candidate's label (``candidate_resolved``) comes from one grading of another
model's rollout. Two kinds of labels are unreliable:

- flaky: a second grading disagrees with the first (a few R2E-Gym tests are flaky);
- not a flaw: the patch is unresolved only because tests that are expected to
  fail (they fail at the base commit and with the reference fix, e.g. a C
  extension that is not built) pass or do not run with it. R2E-Gym's reward
  requires every test status to match, but such a patch fails no test that
  should pass.

A candidate is kept if the re-grade agrees with its label and, for a flawed one,
some test expected to pass does not pass; those tests are stored as
``failing_tests``. Grading uses the R2E-Gym grader:

    uv run python -m benchmarks.hybridgym_verifyrepair.relabel_candidates preds \\
        --candidates cand.jsonl --out preds.jsonl
    uv run r2egym-eval preds.jsonl --output-file regrade.report.json
    uv run python -m benchmarks.hybridgym_verifyrepair.relabel_candidates filter \\
        --candidates cand.jsonl --report regrade.report.json --out cand.relabeled.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def relabel(
    candidates: list[dict[str, Any]], regrades: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return the kept candidates and the number dropped per reason."""
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for row in candidates:
        result = regrades.get(row["instance_id"])
        if result is None:
            dropped["not re-graded"] += 1
            continue
        if result.get("error"):
            dropped["re-grade error"] += 1
            continue
        if bool(result["resolved"]) != bool(row["candidate_resolved"]):
            dropped["label changed on re-grade"] += 1
            continue
        if row["candidate_resolved"]:
            kept.append(row)
            continue
        # mismatched_tests maps test -> "got/expected".
        failing = sorted(
            test
            for test, status in (result.get("mismatched_tests") or {}).items()
            if status.endswith("/PASSED")
        )
        if not failing:
            dropped["flawed only on tests expected to fail"] += 1
            continue
        kept.append({**row, "failing_tests": failing})
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-grade candidates, drop bad labels."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preds = sub.add_parser("preds", help="Write candidates as r2egym-eval input")
    preds.add_argument("--candidates", type=Path, required=True)
    preds.add_argument("--out", type=Path, required=True)
    filt = sub.add_parser("filter", help="Keep candidates with a trustworthy label")
    filt.add_argument("--candidates", type=Path, required=True)
    filt.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="r2egym-eval report of these candidate patches (repeatable; later wins)",
    )
    filt.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidates = _read_jsonl(args.candidates)
    if args.command == "preds":
        with open(args.out, "w") as f:
            for row in candidates:
                pred = {
                    "instance_id": row["instance_id"],
                    "test_result": {"git_patch": row["candidate_patch"]},
                }
                f.write(json.dumps(pred) + "\n")
        print(f"{len(candidates)} predictions -> {args.out}")
        return

    regrades: dict[str, dict[str, Any]] = {}
    for report in args.report:
        for result in json.loads(report.read_text())["results"]:
            regrades[result["instance_id"]] = result
    kept, dropped = relabel(candidates, regrades)
    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    kinds = Counter("correct" if r["candidate_resolved"] else "flawed" for r in kept)
    print(f"kept {len(kept)} {dict(kinds)} -> {args.out}; dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
