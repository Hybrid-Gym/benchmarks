"""
Extract file localization steps and their outputs from the
synthetic-code-training/func_localize_claude45_1457i HuggingFace dataset.

File localization = any tool call that searches for / reads files to find
the target function, before the agent makes its first edit (str_replace).

Supported message formats:
  Format A (OpenHands / XML-style):
    assistant content contains  <function=TOOL>\n<parameter=k>v</parameter>\n</function>
    result arrives as next user message:  "EXECUTION RESULT of [TOOL]:\n..."
  Format B (structured JSON embedded in content string):
    assistant content is a JSON object (or contains a "function_calls" key)
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool: str
    params: dict
    result: Optional[str] = None  # filled in after parsing next user message


@dataclass
class LocalizationStep:
    step_index: int           # 0-based index among all tool calls in the trajectory
    strategy: str             # human-readable label
    tool: str
    command: Optional[str]    # bash command (terminal tool) or file_editor sub-command
    path: Optional[str]
    extra_params: dict = field(default_factory=dict)
    result: Optional[str] = None


@dataclass
class TrajectoryRecord:
    instance_id: str
    resolved: bool
    localization_steps: list[LocalizationStep]
    total_tool_calls: int


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# ---- Format A: XML-style tool calls ----
_XML_CALL_RE = re.compile(
    r"<function=(?P<tool>\w+)>\n?(?P<body>.*?)</function>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=(?P<key>\w+)>(?P<value>.*?)</parameter>",
    re.DOTALL,
)
_RESULT_RE = re.compile(
    r"^EXECUTION RESULT of \[(?P<tool>\w+)\]:\n?(?P<output>.*)",
    re.DOTALL | re.MULTILINE,
)


def parse_xml_calls(content: str) -> list[ToolCall]:
    calls = []
    for m in _XML_CALL_RE.finditer(content):
        tool = m.group("tool")
        body = m.group("body")
        params = {pm.group("key"): pm.group("value").strip()
                  for pm in _XML_PARAM_RE.finditer(body)}
        calls.append(ToolCall(tool=tool, params=params))
    return calls


def parse_xml_result(content: str) -> Optional[str]:
    m = _RESULT_RE.match(content.strip())
    return m.group("output").strip() if m else None


# ---- Format B: JSON embedded in content ----

def try_parse_json_calls(content: str) -> list[ToolCall]:
    """Try to parse content as JSON with a function_calls array."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    calls_raw = obj.get("function_calls") or obj.get("tool_calls") or []
    if not calls_raw:
        return []
    calls = []
    for c in calls_raw:
        tool = c.get("tool") or c.get("name") or c.get("function", {}).get("name", "")
        params = c.get("parameters") or c.get("arguments") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"raw": params}
        calls.append(ToolCall(tool=tool, params=params))
    return calls


# ---- Unified message parser ----

def extract_tool_calls_from_messages(messages: list[dict]) -> list[ToolCall]:
    """
    Walk the message list and pair each assistant tool-call with the
    following user result message.  Returns a flat list of ToolCall objects
    (with .result populated where possible).
    """
    calls: list[ToolCall] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            # Try Format B first (JSON)
            new_calls = try_parse_json_calls(content)
            # Fall back to Format A (XML)
            if not new_calls:
                new_calls = parse_xml_calls(content)

            if new_calls:
                # Attach result from the immediately following user message
                result_text: Optional[str] = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    result_text = parse_xml_result(messages[i + 1].get("content", "") or "")
                    if result_text is None:
                        # Format B: result might just be the raw user content
                        result_text = (messages[i + 1].get("content") or "").strip()
                for c in new_calls:
                    c.result = result_text
                calls.extend(new_calls)
        i += 1
    return calls


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------

def classify_localization_strategy(call: ToolCall) -> Optional[str]:
    """
    Return a strategy label if the call is a localization step, else None.
    Localization steps are tool calls that *find* the target file/function.
    Editing steps (str_replace, insert) are NOT localization.
    """
    tool = call.tool.lower()
    params = call.params

    if tool == "think":
        return "think: reasoning about file location"

    if tool == "task_tracker":
        cmd = params.get("command") or params.get("subcommand") or ""
        if cmd in ("plan", "view"):
            return "task_tracker: multi-step planning"
        return None  # update/complete not localization

    if tool == "file_editor":
        command = params.get("command") or ""
        if command == "view":
            path = params.get("path", "")
            # directory listing vs file read
            if path and not path.endswith(".py") and "." not in path.split("/")[-1]:
                return "file_editor: directory listing"
            return "file_editor: read candidate file"
        # str_replace / insert / create = editing, not localization
        return None

    if tool == "terminal":
        cmd = (params.get("command") or params.get("cmd") or
               params.get("input") or "").strip()
        cmd_lower = cmd.lower()
        if not cmd:
            return None
        # grep / ripgrep
        if re.search(r'\b(grep|rg)\b', cmd_lower):
            if re.search(r'def\s', cmd):
                return "terminal: grep for function definition"
            return "terminal: grep for keyword in files"
        # find pipeline
        if re.search(r'\bfind\b.*\bxargs\b', cmd_lower) or re.search(r'\bfind\b.*-name.*\.py', cmd_lower):
            return "terminal: find+xargs pipeline"
        # ls / tree
        if re.search(r'\b(ls|tree)\b', cmd_lower):
            return "terminal: directory listing (ls/tree)"
        # python ast / import checks
        if re.search(r'py_compile|ast\.parse|get_docstring', cmd_lower):
            return None  # verification, not localization
        # general exploration commands
        if re.search(r'\b(cat|head|tail|less|more)\b', cmd_lower):
            return "terminal: read file contents"
        return None  # unclassified terminal command; skip

    return None


# ---------------------------------------------------------------------------
# Per-trajectory processing
# ---------------------------------------------------------------------------

def process_trajectory(record: dict) -> TrajectoryRecord:
    instance_id = record.get("instance_id", "unknown")
    resolved = bool(record.get("resolved", False))
    messages = record.get("messages", [])

    all_calls = extract_tool_calls_from_messages(messages)

    localization_steps: list[LocalizationStep] = []
    editing_started = False

    for idx, call in enumerate(all_calls):
        tool = call.tool.lower()
        params = call.params

        # Detect first editing action → localization phase ends
        if tool == "file_editor":
            command = params.get("command") or ""
            if command in ("str_replace", "insert", "create"):
                editing_started = True
                continue
        if tool in ("finish",):
            editing_started = True
            continue

        if editing_started:
            continue

        strategy = classify_localization_strategy(call)
        if strategy is None:
            continue

        path = params.get("path") or None
        command = (params.get("command") or params.get("cmd") or
                   params.get("input") or params.get("subcommand") or None)
        extra = {k: v for k, v in params.items()
                 if k not in ("command", "cmd", "input", "path", "subcommand")}

        localization_steps.append(LocalizationStep(
            step_index=idx,
            strategy=strategy,
            tool=call.tool,
            command=command,
            path=path,
            extra_params=extra,
            result=call.result,
        ))

    return TrajectoryRecord(
        instance_id=instance_id,
        resolved=resolved,
        localization_steps=localization_steps,
        total_tool_calls=len(all_calls),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def record_to_dict(rec: TrajectoryRecord) -> dict:
    return {
        "instance_id": rec.instance_id,
        "resolved": rec.resolved,
        "total_tool_calls": rec.total_tool_calls,
        "num_localization_steps": len(rec.localization_steps),
        "localization_steps": [
            {
                "step_index": s.step_index,
                "strategy": s.strategy,
                "tool": s.tool,
                "command": s.command,
                "path": s.path,
                "extra_params": s.extra_params,
                "result_preview": (s.result[:300] + "...") if s.result and len(s.result) > 300 else s.result,
            }
            for s in rec.localization_steps
        ],
    }


def print_summary(records: list[TrajectoryRecord]) -> None:
    from collections import Counter

    total = len(records)
    resolved = sum(r.resolved for r in records)
    strategy_counter: Counter = Counter()
    step_counts = []

    for rec in records:
        step_counts.append(len(rec.localization_steps))
        for s in rec.localization_steps:
            strategy_counter[s.strategy] += 1

    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total trajectories : {total}")
    print(f"Resolved           : {resolved} ({100*resolved/total:.1f}%)")
    print(f"Avg localization   : {avg_steps:.1f} steps/trajectory")
    print(f"\nStrategy distribution (total localization steps):")
    for strategy, count in strategy_counter.most_common():
        print(f"  {count:5d}  {strategy}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract file localization steps from func_localize_claude45 dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default="localization_steps.jsonl",
        help="Output JSONL file path (default: localization_steps.jsonl)",
    )
    parser.add_argument(
        "--max-records", "-n",
        type=int,
        default=None,
        help="Max number of records to process (default: all)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print strategy summary without writing JSONL",
    )
    args = parser.parse_args()

    print(f"Loading dataset synthetic-code-training/func_localize_claude45_1457i ...")
    ds = load_dataset(
        "synthetic-code-training/func_localize_claude45_1457i",
        split=args.split,
        trust_remote_code=True,
    )

    n = min(len(ds), args.max_records) if args.max_records else len(ds)
    print(f"Processing {n} records ...")

    records: list[TrajectoryRecord] = []
    for i in range(n):
        row = ds[i]
        rec = process_trajectory(row)
        records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{n}")

    print_summary(records)

    if not args.summary_only:
        with open(args.output, "w") as f:
            for rec in records:
                f.write(json.dumps(record_to_dict(rec)) + "\n")
        print(f"Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
