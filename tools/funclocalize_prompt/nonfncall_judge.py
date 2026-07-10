"""Judge trajectories from the HF dataset (non-fncall messages format).

Converts non-fncall messages (from HF dataset like func_localize_gpt5mini_1346i)
into the history-event format that tools/funclocalize_judge/judge.py expects,
then runs the LLM judge on each trajectory.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import re
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
    max_retries: int = 8,
    base_wait: float = 30.0,
    max_wait: float = 300.0,
) -> dict:
    """Call judge_one with exponential backoff on rate-limit (429) errors.

    Retries up to max_retries times (waits: 30s, 60s, 120s, 240s, 300s, ...).
    If all retries exhausted, returns the error result as-is.
    """
    import time

    for attempt in range(max_retries + 1):
        result = judge_one(client, model, row)
        if _is_rate_limit_error(result):
            if attempt < max_retries:
                wait = min(base_wait * (2 ** attempt), max_wait)
                print(
                    f"  [rate limit] waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...",
                    flush=True,
                )
                time.sleep(wait)
                continue
            print(
                f"  [error] rate limit persists after {max_retries} retries for {row.get('instance_id', '?')}",
                flush=True,
            )
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
    Retries on rate-limit errors with exponential backoff — no rule-based fallback.
    """
    import time

    verdicts: list[dict] = []

    if workers <= 1:
        for i, row in enumerate(rows, 1):
            verdicts.append(_judge_one_with_retry(client, model, row))
            if i % 10 == 0:
                print(f"  judged {i}/{len(rows)}", flush=True)
            if i < len(rows) and request_delay > 0:
                time.sleep(request_delay)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_judge_one_with_retry, client, model, r) for r in rows]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                verdicts.append(fut.result())
                if i % 10 == 0:
                    print(f"  judged {i}/{len(rows)}", flush=True)

    return verdicts


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
