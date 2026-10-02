"""Audit judge trajectories whose verdict matches the label (run after judge_eval).

A matching verdict can still be a guess: the agent may judge a patch correct after
a run that never exercised the issue, or call it wrong for a defect it does not
have. An LLM sees the issue, the candidate, the reference fix and the failing
hidden tests (none of which the agent saw beyond the first two) plus a digest of
the agent's runs, and answers:

- evidence: did one of the agent's runs show the behavior its verdict relies on?
- reason: is the agent's account (the defect it names, or why the patch works)
  right?

Each checked result in ``output.report.json`` gets ``evidence``, ``reason`` and
``check_note``. ``verdict_correct_ids`` keeps judge_eval's list and
``resolved_ids`` (what ``convert_and_push`` reads) is narrowed to the instances
with both answers "yes". Results that already have an answer are not re-checked.

Usage:
    uv run python -m benchmarks.hybridgym_verifyrepair.reason_check <output.jsonl> \\
        --candidates cand.relabeled.jsonl --llm-config checker.json
"""

import argparse
import ast
import difflib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

from benchmarks.r2egym.config import INFER_DEFAULTS
from benchmarks.r2egym.dataset import get_dataset


PROMPT = """You audit one trajectory of a "judge" task. The agent was given a GitHub issue and a candidate patch (left as uncommitted changes) and had to decide, by running code, whether the patch resolves the issue. Its verdict matches the ground truth; decide whether it reached it for the right reasons, so that the trajectory is good training data.

Ground truth: the patch {truth}. The reference fix and the failing hidden tests below were not shown to the agent.

Answer two questions:
1. evidence: Does at least one of the agent's runs in the digest exercise the behavior the issue describes on the patched code, with output that supports the verdict? For CORRECT, the issue's scenario must run and behave as the issue expects; for INCORRECT, a run must show the failure or wrong output the agent relies on (failing existing tests count). A run that crashed for an unrelated reason, a script that re-implements the logic or only tests the patch's own design instead of the issue's behavior, or a claim with no supporting run does not count.
2. reason: For INCORRECT, is the defect the agent names a real defect of this patch, consistent with the failing hidden tests or the reference fix? For CORRECT, is the agent's account of what the patch does and why it resolves the issue accurate?

Reply with JSON only: {{"evidence": "yes" or "no", "reason": "yes" or "no", "note": "<one sentence>"}}

=== ISSUE ===
{issue}

=== CANDIDATE PATCH ===
{candidate}

=== REFERENCE FIX (non-test files) ===
{gold}

=== FAILING HIDDEN TESTS ===
{tests}

=== AGENT DIGEST (actions, output tails, final answer; verdict {verdict}) ===
{digest}
"""
MAX_ATTEMPTS = 4


def _is_test_path(path: str) -> bool:
    padded = f"/{path}"
    return path.split("/")[-1].startswith("test") or any(
        part in padded for part in ("/tests/", "/test/")
    )


def gold_diff(parsed_commit: dict[str, Any]) -> str:
    """Unified diff of the reference commit's non-test files."""
    out: list[str] = []
    for fd in parsed_commit["file_diffs"]:
        match = re.search(r"'path': '([^']+)'", str(fd["header"]))
        path = match.group(1) if match else "?"
        if _is_test_path(path):
            continue
        old = (fd.get("old_file_content") or "").splitlines(keepends=True)
        new = (fd.get("new_file_content") or "").splitlines(keepends=True)
        out += difflib.unified_diff(old, new, f"a/{path}", f"b/{path}", n=3)
    return "".join(out)


def hidden_test_sources(test_files: list[str], names: list[str]) -> str:
    """Source of each named test (``Class.method`` or ``function``)."""
    found: list[str] = []
    for code in test_files:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        lines = code.splitlines()
        for node in ast.walk(tree):
            funcs: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
            if isinstance(node, ast.ClassDef):
                funcs = [
                    (f"{node.name}.{sub.name}", sub)
                    for sub in node.body
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs = [(node.name, node)]
            # Parametrized test ids ("test_x[pyloop]") match their function.
            wanted = {n.split("[")[0] for n in names}
            for name, fn in funcs:
                if name in wanted:
                    body = lines[fn.lineno - 1 : fn.end_lineno]
                    found.append(f"# {name}\n" + "\n".join(body))
    return "\n\n".join(found)


def digest(history: list[dict[str, Any]], limit: int = 30000) -> str:
    """Agent actions with output tails and its final answer."""
    outputs: dict[str, str] = {}
    for e in history:
        if e.get("kind") == "ObservationEvent":
            content = (e.get("observation") or {}).get("content") or []
            if content and isinstance(content[0], dict):
                outputs[e.get("tool_call_id") or ""] = content[0].get("text") or ""
    lines: list[str] = []
    for e in history:
        kind = e.get("kind")
        if kind == "ActionEvent":
            action = e.get("action") or {}
            tool = e.get("tool_name")
            if tool == "terminal":
                output = outputs.get(e.get("tool_call_id") or "", "")
                lines.append(
                    f"$ {str(action.get('command'))[:300]}\n  -> {output[-600:]}"
                )
            elif tool == "file_editor":
                body = action.get("file_text") or action.get("new_str") or ""
                lines.append(
                    f"[{action.get('command')} {action.get('path')}] {body[:800]}"
                )
            elif tool == "finish":
                lines.append(f"[FINISH] {action.get('message')}")
        elif kind == "MessageEvent" and e.get("source") == "agent":
            content = (e.get("llm_message") or {}).get("content") or []
            lines.append("[ANSWER] " + "\n".join(c.get("text") or "" for c in content))
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    return text[: limit * 2 // 5] + "\n...[cut]...\n" + text[-limit * 3 // 5 :]


def parse_reply(raw: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    return {
        "evidence": str(data.get("evidence", "?")).lower(),
        "reason": str(data.get("reason", "?")).lower(),
        "check_note": str(data.get("note", raw[-200:])),
    }


def apply_checks(report: dict[str, Any], checks: dict[str, dict[str, str]]) -> None:
    """Store the answers and narrow ``resolved_ids`` to the fully supported ones."""
    report.setdefault("verdict_correct_ids", list(report["resolved_ids"]))
    for result in report["results"]:
        result.update(checks.get(result["instance_id"], {}))
    report["resolved_ids"] = [
        r["instance_id"]
        for r in report["results"]
        if r["instance_id"] in report["verdict_correct_ids"]
        and r.get("evidence") == "yes"
        and r.get("reason") == "yes"
    ]
    report["resolved_instances"] = len(report["resolved_ids"])


def _ask(client: OpenAI, model: str, prompt: str) -> str:
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                # Thinking models spend most of a small budget before answering.
                max_tokens=16000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 - any gateway failure is retryable
            last = e
            time.sleep(8 * 2**attempt * (0.5 + random.random()))
    raise last if last else RuntimeError("no attempts made")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit matching judge verdicts.")
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--llm-config",
        type=Path,
        required=True,
        help="JSON with model/base_url/api_key",
    )
    parser.add_argument("--dataset", default=INFER_DEFAULTS["dataset"])
    parser.add_argument("--split", default=INFER_DEFAULTS["split"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    report_path = args.output_jsonl.with_name("output.report.json")
    report = json.loads(report_path.read_text())
    verdict_correct = set(report.get("verdict_correct_ids", report["resolved_ids"]))
    todo = [
        r["instance_id"]
        for r in report["results"]
        if r["instance_id"] in verdict_correct
        and r.get("evidence") not in ("yes", "no")
    ]
    with open(args.candidates) as f:
        candidates = {r["instance_id"]: r for r in map(json.loads, f)}
    with open(args.output_jsonl) as f:
        rows = {r["instance_id"]: r for r in map(json.loads, f)}
    by_id = {r["instance_id"]: r for r in report["results"]}
    df = get_dataset(args.dataset, args.split)
    fields = ("problem_statement", "parsed_commit_content", "execution_result_content")
    data = {
        str(r["instance_id"]): {k: str(r[k]) for k in fields}
        for _, r in df[df.instance_id.isin(todo)].iterrows()
    }

    config = json.loads(args.llm_config.read_text())
    model = config["model"].removeprefix("openai/")
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    def check(iid: str) -> tuple[str, dict[str, str]]:
        cand, ds = candidates[iid], data[iid]
        failing = cand.get("failing_tests") or []
        tests = json.loads(ds["execution_result_content"])["test_file_codes"]
        failing_src = hidden_test_sources(tests, failing) or ", ".join(failing)
        prompt = PROMPT.format(
            truth="passes the hidden tests (it is correct)"
            if cand["candidate_resolved"]
            else "fails the hidden tests (it is flawed)",
            issue=ds["problem_statement"],
            candidate=cand["candidate_patch"][:8000],
            gold=gold_diff(json.loads(ds["parsed_commit_content"]))[:8000],
            tests=(failing_src or "none")[:6000],
            verdict="CORRECT" if by_id[iid]["verdict"] else "INCORRECT",
            digest=digest(rows[iid].get("history") or []),
        )
        try:
            return iid, parse_reply(_ask(client, model, prompt))
        except Exception as e:  # noqa: BLE001 - leave it unchecked, retry next run
            return iid, {
                "evidence": "?",
                "reason": "?",
                "check_note": f"error: {e}"[:300],
            }

    with ThreadPoolExecutor(args.workers) as ex:
        checks = dict(ex.map(check, todo))
    apply_checks(report, checks)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    answered = [r for r in report["results"] if r.get("evidence") in ("yes", "no")]
    print(
        f"checked {len(checks)} (answered in total {len(answered)}); "
        f"verdict correct {len(report['verdict_correct_ids'])} -> "
        f"kept {len(report['resolved_ids'])}; report -> {report_path}"
    )


if __name__ == "__main__":
    main()
