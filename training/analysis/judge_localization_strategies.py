"""
Sample ~100 trajectories from synthetic-code-training/func_localize_claude45_1457i
and use an LLM judge to classify which file-localization strategies each trajectory uses.

Strategies focus on two behavioural dimensions:
  A) Keyword searching patterns
  B) File reading trigger patterns

Usage:
  export NVIDIA_API_KEY=...          # or OPENAI_API_KEY for OpenAI-compatible endpoint
  python judge_localization_strategies.py [--n 100] [--output results.jsonl]
"""

import ast
import re
import json
import random
import argparse
import os
from textwrap import dedent
from datasets import load_dataset
from openai import OpenAI


# ---------------------------------------------------------------------------
# Strategy definitions  (these are what the LLM judges)
# ---------------------------------------------------------------------------

STRATEGIES = {
    # ── Keyword search strategies ──────────────────────────────────────────
    "broad_then_narrow_search": {
        "label": "Broad-then-Narrow Keyword Search",
        "description": dedent("""\
            The agent first issues a broad search (e.g. grep for a general topic word
            like 'publish' or 'register') to find candidate files, then follows up with
            a narrower search (e.g. 'def publish', 'def.*register') to pinpoint the
            exact function definition.  At least two grep/find rounds are used, with
            the second being strictly more specific than the first."""),
    },
    "direct_specific_search": {
        "label": "Direct Specific Search (no broad phase)",
        "description": dedent("""\
            The agent skips any broad keyword pass and immediately searches for a
            specific pattern — typically 'def <function_name>', a unique error message
            string, or a very distinctive code token.  There is only one grep/find
            round before a file is opened."""),
    },
    "multi_round_refinement": {
        "label": "Multi-Round Search Refinement (3+ rounds)",
        "description": dedent("""\
            The agent runs three or more grep/find rounds, each more targeted than the
            last, before committing to a single candidate file.  The rounds may combine
            different tools (grep by keyword, then grep by def pattern, then grep inside
            a specific file)."""),
    },
    "intra_file_grep": {
        "label": "Intra-File Grep After Opening",
        "description": dedent("""\
            After opening a file (or after identifying a candidate file), the agent
            runs a grep or search *within* that specific file (not repo-wide) to locate
            the exact line range of the target function.  Signal: grep with an explicit
            file path argument rather than -r/--recursive."""),
    },
    "no_keyword_search": {
        "label": "No Keyword Search (pure file navigation)",
        "description": dedent("""\
            The agent never runs grep or find at all.  It navigates entirely through
            directory listings, import chain tracing, and reading files directly."""),
    },

    # ── File reading trigger strategies ───────────────────────────────────
    "read_after_few_candidates": {
        "label": "Read Files Only After Narrowing to Few Candidates",
        "description": dedent("""\
            The agent delays opening any file for full reading until its grep/search
            results show only a small number of candidate files (roughly ≤5).  It does
            not open files speculatively; it first reduces the candidate set through
            search, then reads."""),
    },
    "immediate_file_read": {
        "label": "Immediate File Read (no prior search)",
        "description": dedent("""\
            The agent opens a specific source file directly — without running any grep
            or find first — based on semantic understanding of the function description
            and the repo's directory structure.  The first substantive action after
            viewing the repo root is a file open."""),
    },
    "full_file_before_range": {
        "label": "Read Full File Before Zooming to Line Range",
        "description": dedent("""\
            When the agent opens a file it always reads the entire file first (no
            view_range / line range specified), building a mental map of the file
            structure, before later re-opening or re-reading a specific line range to
            study the target function in detail."""),
    },
    "candidate_comparison_read": {
        "label": "Candidate-Comparison Read",
        "description": dedent("""\
            The agent opens multiple different files (≥2) that all seem plausible, reads
            each of them, and then explicitly compares them (in a think step or in
            its reasoning) to decide which one contains the target function."""),
    },
    "import_chain_read": {
        "label": "Import-Chain–Driven File Read",
        "description": dedent("""\
            The agent opens a file because it encountered an import statement or module
            reference while reading a different file.  It is tracing the dependency
            chain to find where a function is actually defined (e.g. `from .submod
            import X` → opens submod.py)."""),
    },
}


# ---------------------------------------------------------------------------
# Dataset loading + trajectory serialisation
# ---------------------------------------------------------------------------

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

_TASK_LIST_PARAM_RE = re.compile(
    r'(<parameter=task_list>)(.*?)(</parameter>)',
    re.DOTALL
)


def fix_task_list_json(content: str) -> str:
    """Convert Python single-quote task_list values to JSON double-quote syntax."""
    def _replace(m: re.Match) -> str:
        raw = m.group(2)
        try:
            parsed = ast.literal_eval(raw)
            return m.group(1) + json.dumps(parsed) + m.group(3)
        except (ValueError, SyntaxError):
            return m.group(0)
    return _TASK_LIST_PARAM_RE.sub(_replace, content)


def fix_messages_task_list(messages: list[dict]) -> tuple[list[dict], bool]:
    """Fix task_list JSON syntax in all messages. Returns (fixed_messages, was_changed)."""
    changed = False
    result = []
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str) and "<parameter=task_list>" in content:
            new_content = fix_task_list_json(content)
            if new_content != content:
                changed = True
                msg = {**msg, "content": new_content}
        result.append(msg)
    return result, changed


def _parse_xml_calls(content: str):
    for m in _XML_CALL_RE.finditer(content):
        tool = m.group("tool")
        params = {pm.group("key"): pm.group("value").strip()
                  for pm in _XML_PARAM_RE.finditer(m.group("body"))}
        yield tool, params


def _try_json_calls(content: str):
    try:
        obj = json.loads(content)
    except Exception:
        return
    for c in (obj.get("function_calls") or obj.get("tool_calls") or []):
        tool = (c.get("tool") or c.get("name") or
                c.get("function", {}).get("name", ""))
        params = c.get("parameters") or c.get("arguments") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        yield tool, params


def serialise_trajectory(messages: list[dict], max_chars: int = 12_000) -> str:
    """
    Render the tool-call sequence of a trajectory as a compact, readable text.
    We include assistant calls + results but skip pure reasoning/system text to
    stay within the LLM context window.
    """
    lines = []
    total = 0

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            # collect tool calls
            calls = list(_try_json_calls(content)) or list(_parse_xml_calls(content))
            for tool, params in calls:
                # Format each call compactly
                cmd = (params.get("command") or params.get("cmd") or
                       params.get("input") or params.get("subcommand") or "")
                path = params.get("path") or ""
                vr = params.get("view_range") or ""

                if tool == "think":
                    body = params.get("thought") or params.get("content") or str(params)
                    line = f"[think] {body[:300]}"
                elif tool == "terminal":
                    line = f"[terminal] {cmd[:400]}"
                elif tool == "file_editor":
                    line = f"[file_editor command={cmd} path={path}" + (f" view_range={vr}" if vr else "") + "]"
                elif tool == "task_tracker":
                    line = f"[task_tracker command={cmd}]"
                elif tool == "finish":
                    line = "[finish]"
                else:
                    line = f"[{tool}] {str(params)[:200]}"

                lines.append(line)
                total += len(line)

        elif role == "user" and i > 1:
            # result of previous tool call
            result = content.strip()
            m = _RESULT_RE.match(result)
            if m:
                output = m.group("output").strip()
            else:
                output = result
            # Truncate long results
            if len(output) > 600:
                output = output[:600] + "\n... [truncated]"
            result_line = f"  → {output}"
            lines.append(result_line)
            total += len(result_line)

        if total > max_chars:
            lines.append("... [trajectory truncated for length]")
            break

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = dedent("""\
    You are an expert software-engineering researcher analysing agent trajectories.
    Your task is to judge which file-localisation strategies an agent used in a
    single trajectory.

    A trajectory is shown as a sequence of tool calls and their outputs.
    The task is: given only a *behavioural description* of a Python function
    (no name provided), find that function in a repository and write its docstring.

    You will be given:
    1. The list of strategies to judge (each with a name and precise definition).
    2. The serialised trajectory.

    For EACH strategy, output exactly one JSON key whose value is a boolean (true/false)
    and a one-sentence justification.

    Output format (strict JSON, no extra text):
    {
      "strategy_key_1": {"used": true, "reason": "..."},
      "strategy_key_2": {"used": false, "reason": "..."},
      ...
    }
""")


def build_user_prompt(trajectory_text: str) -> str:
    strategy_block = "\n\n".join(
        f'Strategy key: "{key}"\nLabel: {info["label"]}\nDefinition: {info["description"]}'
        for key, info in STRATEGIES.items()
    )
    return dedent(f"""\
        ## Strategies to Judge

        {strategy_block}

        ---

        ## Trajectory

        {trajectory_text}

        ---

        Judge each strategy key and return strict JSON as described.""")


def judge_trajectory(
    client: OpenAI,
    model: str,
    trajectory_text: str,
    temperature: float = 0.1,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(trajectory_text)},
        ],
        temperature=temperature,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"_parse_error": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM-judge file localization strategies in func_localize dataset"
    )
    parser.add_argument("--n", type=int, default=100,
                        help="Number of trajectories to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--split", default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--output", "-o", default="strategy_judgements.jsonl",
                        help="Output JSONL path (default: strategy_judgements.jsonl)")
    parser.add_argument("--model", default="nvidia/deepseek-ai/evals-deepseek-v4-pro",
                        help="LLM model name")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY",
                        help="Env var holding the API key (default: NVIDIA_API_KEY)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print one trajectory + prompt without calling the LLM")
    args = parser.parse_args()

    # ── API client ──────────────────────────────────────────────────────────
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit(f"Set {args.api_key_env} (or OPENAI_API_KEY) before running.")

    client = OpenAI(api_key=api_key or "dummy", base_url=args.base_url)

    # ── Load dataset ────────────────────────────────────────────────────────
    print("Loading dataset ...")
    ds = load_dataset(
        "synthetic-code-training/func_localize_claude45_1457i",
        split=args.split,
        trust_remote_code=True,
    )
    print(f"  {len(ds)} total records")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.n, len(ds)))
    print(f"  Sampling {len(indices)} records (seed={args.seed})")

    # ── Dry run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        row = ds[indices[0]]
        traj = serialise_trajectory(row["messages"])
        prompt = build_user_prompt(traj)
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT (first 3000 chars) ===")
        print(prompt[:3000])
        print("\n=== TRAJECTORY PREVIEW ===")
        print(traj[:2000])
        return

    # ── Judge ────────────────────────────────────────────────────────────────
    results = []
    errors = 0
    task_list_fixed_count = 0

    with open(args.output, "w") as out_f:
        for rank, idx in enumerate(indices):
            row = ds[idx]
            instance_id = row.get("instance_id", f"row_{idx}")
            resolved = bool(row.get("resolved", False))

            fixed_messages, tl_changed = fix_messages_task_list(row.get("messages", []))
            if tl_changed:
                task_list_fixed_count += 1
            traj_text = serialise_trajectory(fixed_messages)
            print(f"[{rank+1:3d}/{len(indices)}] {instance_id} ...", end=" ", flush=True)

            try:
                judgement = judge_trajectory(client, args.model, traj_text)
                status = "ok"
            except Exception as e:
                judgement = {"_error": str(e)}
                errors += 1
                status = f"ERROR: {e}"

            record = {
                "instance_id": instance_id,
                "resolved": resolved,
                "judgement": judgement,
            }
            out_f.write(json.dumps(record) + "\n")
            results.append(record)
            print(status)

    # ── Aggregate summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STRATEGY USAGE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<45} {'Used':>6}  {'%':>5}")
    print("-" * 60)

    for key, info in STRATEGIES.items():
        used_count = sum(
            1 for r in results
            if isinstance(r["judgement"].get(key), dict)
            and r["judgement"][key].get("used") is True
        )
        pct = 100 * used_count / len(results) if results else 0
        label = info["label"][:44]
        print(f"{label:<45} {used_count:>6}  {pct:>4.0f}%")

    print(f"\nTotal: {len(results)} records, {errors} errors")
    print(f"Task-list JSON fixed: {task_list_fixed_count} / {len(indices)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
