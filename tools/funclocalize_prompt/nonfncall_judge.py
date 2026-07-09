"""Judge trajectories from the HF dataset (non-fncall messages format).

Converts non-fncall messages (from HF dataset like func_localize_gpt5mini_1346i)
into the history-event format that tools/funclocalize_judge/judge.py expects,
then runs the LLM judge on each trajectory.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

# Load judge.py from tools/funclocalize_judge/judge.py
_JUDGE_PY = Path(__file__).parent.parent / "funclocalize_judge" / "judge.py"
_spec = importlib.util.spec_from_file_location("funclocalize_judge", _JUDGE_PY)
_judge_mod = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_judge_mod)  # type: ignore

# Re-export useful items from judge.py
judge_one = _judge_mod.judge_one
aggregate = _judge_mod.aggregate
print_aggregate = _judge_mod.print_aggregate
STRATEGY_KEYS = _judge_mod.STRATEGY_KEYS


# ── non-fncall message parser ──────────────────────────────────────────────

# Matches <function=TOOL_NAME>..body..</function>
_FUNC_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
# Matches <parameter=NAME>..value..</parameter>
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def _parse_function_call(tool_name: str, body: str) -> dict:
    """Parse a single <function=...> block into {name, arguments}."""
    params: dict[str, Any] = {}
    for m in _PARAM_RE.finditer(body):
        params[m.group(1)] = m.group(2).strip()
    return {"name": tool_name, "arguments": params}


def nonfncall_messages_to_history(messages: list[dict]) -> list[dict]:
    """Convert non-fncall HF-dataset messages to history events for judge.py.

    The non-fncall format embeds tool calls as::

        <function=terminal>
        <parameter=command>grep -r keyword .</parameter>
        </function>

    inside assistant messages, and tool results appear as user messages:
    ``EXECUTION RESULT of [function]: ...``
    """
    history: list[dict] = []
    first_user_seen = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            continue  # skip system prompt

        if role == "user":
            if not first_user_seen:
                # First user message = task description
                history.append(
                    {
                        "kind": "MessageEvent",
                        "llm_message": {"role": "user", "content": content},
                    }
                )
                first_user_seen = True
            else:
                # Tool result / execution output
                history.append(
                    {
                        "kind": "ObservationEvent",
                        "observation": {"content": content},
                    }
                )

        elif role == "assistant":
            # Thinking text = everything outside <function=...> blocks
            thinking = _FUNC_RE.sub("", content).strip()
            if thinking:
                history.append(
                    {
                        "kind": "MessageEvent",
                        "llm_message": {"role": "assistant", "content": thinking},
                    }
                )

            for m in _FUNC_RE.finditer(content):
                tc = _parse_function_call(m.group(1), m.group(2))
                history.append({"kind": "ActionEvent", "tool_call": tc})

    return history


def build_judge_rows(hf_rows: list[dict]) -> list[dict]:
    """Convert HF dataset rows to judge-compatible rows with history events."""
    out = []
    for row in hf_rows:
        history = nonfncall_messages_to_history(row.get("messages") or [])
        out.append(
            {
                "instance_id": row.get("instance_id", "unknown"),
                "history": history,
            }
        )
    return out


def _is_rate_limit_error(result: dict) -> bool:
    err = result.get("error", "") or ""
    return bool("429" in err or "rate" in err.lower() or "block" in err.lower())


def _judge_one_with_retry(
    client: OpenAI,
    model: str,
    row: dict,
    max_retries: int = 1,
    base_wait: float = 10.0,
    use_rule_fallback: bool = True,
) -> dict:
    """Call judge_one with exponential backoff on rate-limit (429) errors.

    Falls back to rule_based_judge_one if all retries are exhausted.
    Defaults to 2 retries (waits: 30s, 60s) for fast fail-over.
    """
    import time

    for attempt in range(max_retries + 1):
        result = judge_one(client, model, row)
        if _is_rate_limit_error(result):
            if attempt < max_retries:
                wait = base_wait * (2 ** attempt)
                print(
                    f"  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if use_rule_fallback:
                print(
                    f"  [fallback] rule-based judge for {row.get('instance_id', '?')}",
                    flush=True,
                )
                return rule_based_judge_one(row)
        return result
    return result  # type: ignore[return-value]


def judge_rows(
    rows: list[dict],
    client: OpenAI,
    model: str,
    workers: int = 1,
    request_delay: float = 2.0,
) -> list[dict]:
    """Judge a list of rows (already in judge-compatible history-event format).

    Uses sequential processing (workers=1) to avoid WAF rate limits.
    After the first rate-limit failure, automatically switches all remaining
    rows to the rule-based judge to avoid long waits.
    """
    import time

    verdicts: list[dict] = []
    use_rule_based = False  # flip to True once we detect persistent rate limiting

    if workers <= 1:
        for i, row in enumerate(rows, 1):
            if use_rule_based:
                verdicts.append(rule_based_judge_one(row))
            else:
                result = _judge_one_with_retry(client, model, row)
                verdicts.append(result)
                # If rate limit hit (even if rule-based fallback ran), switch all remaining to rule-based
                notes = result.get("notes", "") or ""
                if _is_rate_limit_error(result) or "rule-based" in notes:
                    print(
                        f"  [auto-fallback] Switching all remaining rows to rule-based judge.",
                        flush=True,
                    )
                    use_rule_based = True

            if i % 10 == 0:
                print(f"  judged {i}/{len(rows)}", flush=True)
            if i < len(rows) and not use_rule_based and request_delay > 0:
                time.sleep(request_delay)
            # No delay needed between rule-based calls
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_judge_one_with_retry, client, model, r) for r in rows]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                verdicts.append(fut.result())
                if i % 10 == 0:
                    print(f"  judged {i}/{len(rows)}", flush=True)

    return verdicts


# ── rule-based judge (fallback when LLM API is unavailable) ──────────────────

_BROAD_PATTERNS = re.compile(
    r"\b(grep\s+-[rR]|rg\s+|find\s+\.|ls\s+(-[a-zA-Z]+\s+)?/workspace)\b",
    re.IGNORECASE,
)
_NARROW_PATTERNS = re.compile(r"\b(grep|rg)\b.*\s+--(include|path|glob)\b|grep.*\s+-r\s+\S+/\S+", re.IGNORECASE)


def rule_based_judge_one(row: dict) -> dict:
    """Heuristic judge for the 3 rubrics — no LLM API required.

    Approximates the LLM judge using regex patterns on the trajectory summary.
    Less accurate but robust to API unavailability.
    """
    iid = row.get("instance_id", "unknown")
    events = _judge_mod.extract_localization_phase(row.get("history") or [])
    if not events:
        return {"instance_id": iid, "error": "no localization actions",
                "broad_then_narrow": None, "multi_round_refinement": None,
                "read_after_narrowing": None, "notes": ""}

    actions = [ev for ev in events if ev.get("kind") == "ActionEvent"]
    terminal_actions = [
        ev for ev in actions
        if (ev.get("tool_call") or {}).get("name") == "terminal"
    ]
    view_actions = [
        ev for ev in actions
        if (ev.get("tool_call") or {}).get("name") == "file_editor"
        and _judge_mod._parse_args((ev.get("tool_call") or {}).get("arguments")).get("command") == "view"
    ]

    def _cmd(ev: dict) -> str:
        return str(_judge_mod._parse_args((ev.get("tool_call") or {}).get("arguments")).get("command", ""))

    def _is_broad(cmd: str) -> bool:
        return bool(_BROAD_PATTERNS.search(cmd)) and "/workspace" not in cmd.split()[0]

    # Rule 1: BROAD-THEN-NARROW
    # First 1-2 terminal actions should be broad; later ones narrow
    broad_then_narrow = False
    if terminal_actions:
        first_cmds = [_cmd(ev) for ev in terminal_actions[:2]]
        any_broad = any(_is_broad(c) for c in first_cmds)
        # Check that the first action is not a targeted file read
        first_action_is_file_read = (
            actions and
            (actions[0].get("tool_call") or {}).get("name") == "file_editor" and
            _judge_mod._parse_args((actions[0].get("tool_call") or {}).get("arguments")).get("command") == "view"
        )
        broad_then_narrow = any_broad and not first_action_is_file_read

    # Rule 2: MULTI-ROUND REFINEMENT
    # ≥2 distinct search commands
    search_cmds = [_cmd(ev) for ev in terminal_actions]
    multi_round_refinement = len(search_cmds) >= 2

    # Rule 3: READ-AFTER-NARROWING
    # Full file reads (view with no range or large range) only after ≥2 searches
    def _is_full_read(ev: dict) -> bool:
        args = _judge_mod._parse_args((ev.get("tool_call") or {}).get("arguments"))
        if args.get("command") != "view":
            return False
        vr = args.get("view_range")
        if vr is None:
            return True
        # Parse range and check if > 100 lines
        try:
            r = eval(str(vr))  # noqa: S307 (safe: only numbers)
            if isinstance(r, (list, tuple)) and len(r) == 2:
                return (r[1] - r[0]) > 100
        except Exception:
            pass
        return False

    # Find the action index of first full read
    all_indexed = list(enumerate(actions))
    first_full_read_idx = next(
        (i for i, ev in all_indexed if ev.get("kind") == "ActionEvent" and
         (ev.get("tool_call") or {}).get("name") == "file_editor" and _is_full_read(ev)),
        len(actions),  # no full read = compliant
    )
    # Count search commands before the first full read
    searches_before_read = sum(
        1 for i, ev in all_indexed[:first_full_read_idx]
        if ev.get("kind") == "ActionEvent" and
        (ev.get("tool_call") or {}).get("name") == "terminal"
    )
    read_after_narrowing = (first_full_read_idx == len(actions)) or (searches_before_read >= 2)

    return {
        "instance_id": iid,
        "broad_then_narrow": broad_then_narrow,
        "multi_round_refinement": multi_round_refinement,
        "read_after_narrowing": read_after_narrowing,
        "notes": "rule-based (no LLM)",
    }


def compute_scores(verdicts: list[dict]) -> dict:
    """Compute per-rubric compliance scores (fraction, not percent)."""
    agg = aggregate(verdicts)
    total = max(agg["total"], 1)
    scores = {}
    for k in STRATEGY_KEYS:
        scores[k] = agg[k] / total
    scores["total"] = agg["total"]
    scores["errors"] = agg["errors"]
    return scores
