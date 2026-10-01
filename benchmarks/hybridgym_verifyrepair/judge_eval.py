"""Score ``--task judge`` rollouts: does the agent's verdict match the candidate's grade?

The verdict is the last ``VERDICT: CORRECT`` / ``VERDICT: INCORRECT`` line of the
agent's final ``finish`` message; a missing verdict counts as wrong. The label is the
candidate's ``candidate_resolved`` field (its grade from the benchmark's own grader).

The report is written next to ``output.jsonl`` as ``output.report.json``.
``resolved_ids`` lists the instances with a correct verdict, which is what
``convert_and_push`` reads as ``resolved``. Two filter flags are recorded per
instance: ``ran_code`` (the agent ran Python or a test runner at least once) and
``patch_unchanged`` (the final diff still equals the candidate, i.e. the agent did
not edit the code it was asked to judge or leave files behind).

Usage:
    uv run hybridgym-verifyrepair-judge-eval <output.jsonl> --candidates cand.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.utils.patch_utils import remove_noise_from_patch


VERDICT_RE = re.compile(
    r"^[\s*_`#>-]*VERDICT[*_`]*\s*:[\s*_`]*(CORRECT|INCORRECT)\b",
    re.IGNORECASE | re.MULTILINE,
)
CODE_RUN_RE = re.compile(
    r"(^|[;&|(]\s*|\s)(python3?|pytest|py\.test|tox|\S*runtests\.py|bin/test)(\s|$)"
)


def parse_verdict(message: str) -> bool | None:
    """Return True for CORRECT, False for INCORRECT, None if there is no verdict."""
    matches = VERDICT_RE.findall(message or "")
    if not matches:
        return None
    return matches[-1].upper() == "CORRECT"


def _actions(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (e.get("tool_name") or "", e.get("action") or {})
        for e in row.get("history") or []
        if e.get("kind") == "ActionEvent"
    ]


def final_message(row: dict[str, Any]) -> str:
    """Message of the agent's last ``finish`` call ('' if it never finished)."""
    messages = [a.get("message") or "" for t, a in _actions(row) if t == "finish"]
    return messages[-1] if messages else ""


def ran_code(row: dict[str, Any]) -> bool:
    return any(
        tool == "terminal"
        and CODE_RUN_RE.search(action.get("command") or "") is not None
        and "pip install" not in (action.get("command") or "")
        for tool, action in _actions(row)
    )


def _changed_lines(patch: str) -> list[str]:
    return [
        line
        for line in remove_noise_from_patch(patch).splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


def score(
    rows: list[dict[str, Any]], candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build the report; for repeated instance ids the last row wins."""
    last = {row["instance_id"]: row for row in rows}
    results = []
    confusion: Counter[str] = Counter()
    for iid in sorted(last):
        row = last[iid]
        candidate = candidates[iid]
        label = bool(candidate["candidate_resolved"])
        verdict = parse_verdict(final_message(row))
        patch = (row.get("test_result") or {}).get("git_patch") or ""
        results.append(
            {
                "instance_id": iid,
                "candidate_resolved": label,
                "verdict": verdict,
                "resolved": verdict is label,
                "ran_code": ran_code(row),
                "patch_unchanged": _changed_lines(patch)
                == _changed_lines(candidate["candidate_patch"]),
                "error": bool(row.get("error")),
            }
        )
        said = {True: "said_correct", False: "said_incorrect", None: "no_verdict"}
        confusion[f"{'correct' if label else 'flawed'}/{said[verdict]}"] += 1
    resolved_ids = [r["instance_id"] for r in results if r["resolved"]]
    return {
        "task": "verifyrepair-judge",
        "total_instances": len(results),
        "resolved_instances": len(resolved_ids),
        "accuracy": len(resolved_ids) / len(results) if results else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "ran_code": sum(r["ran_code"] for r in results),
        "patch_unchanged": sum(r["patch_unchanged"] for r in results),
        "resolved_ids": resolved_ids,
        "results": results,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score --task judge rollouts.")
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Candidates file used for the run (needs candidate_resolved)",
    )
    args = parser.parse_args()

    candidates = {r["instance_id"]: r for r in _read_jsonl(args.candidates)}
    report = score(_read_jsonl(args.output_jsonl), candidates)
    out = args.output_jsonl.with_name("output.report.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("resolved_ids", "results")}
    print(json.dumps(summary, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
