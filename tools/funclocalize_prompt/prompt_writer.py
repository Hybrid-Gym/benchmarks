"""LLM-based prompt writer/reviser for the funclocalize pipeline.

Uses the NVIDIA inference API to write and iteratively refine Jinja2 prompt
templates that improve agent compliance with the localization rubrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
WRITER_MODEL = "openai/openai/gpt-5-mini"
MAX_TOKENS = 4096

# Load rubrics text
_RUBRICS_PATH = Path(__file__).parent / "rubrics.txt"
RUBRICS_TEXT = _RUBRICS_PATH.read_text()

# Load default.j2 as the starting template
_DEFAULT_J2 = (
    Path(__file__).parent.parent.parent
    / "benchmarks"
    / "hybridgym_funclocalize"
    / "prompts"
    / "default.j2"
)
DEFAULT_TEMPLATE = _DEFAULT_J2.read_text()


SYSTEM_PROMPT = """\
You are an expert at writing concise, effective task prompts for AI coding agents.
Your goal is to produce a Jinja2 prompt template that maximizes agent compliance
with specific localization-strategy rubrics.

The agent task: given a Python repository and a description of a function/class,
the agent must locate the target in the codebase and write its docstring.

Available Jinja2 variables in the template:
  {{ workspace_dir_name }}  — the repo directory name in /workspace/
  {{ module_type }}         — "function" or "class"
  {{ description }}         — textual description of the target
  {{ is_multi }}            — True if multiple targets (multi-target block uses target_list)
  {{ num_targets }}         — number of targets (for multi)
  {{ target_list }}         — formatted list of targets (for multi)

IMPORTANT: The template must include BOTH the {% if is_multi %} and {% else %} blocks
from the default template, preserving the multi-target handling.

Return ONLY the Jinja2 template text, no explanation, no markdown fences.
"""


def _make_client(api_key: str, base_url: str = NVIDIA_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm(client: OpenAI, model: str, messages: list[dict], max_retries: int = 3) -> str:
    """Call LLM with exponential backoff. Raises if all retries exhausted."""
    import time

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            # Empty content likely means all tokens were reasoning — retry or fall through
            print(f"  [empty response] attempt {attempt + 1}, retrying...", flush=True)
            if attempt < max_retries:
                time.sleep(5.0)
                continue
            raise RuntimeError("LLM returned empty content after all retries")
        except RuntimeError:
            raise
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str
            if is_rate_limit and attempt < max_retries:
                wait = 30.0 * (2 ** attempt)
                print(f"  [rate limit] waiting {wait:.0f}s before retry {attempt + 1}...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _format_scores(scores: dict, label: str = "") -> str:
    lines = [f"Scores ({label}):" if label else "Scores:"]
    total = scores.get("total", 0)
    for k in ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing"):
        pct = 100 * scores.get(k, 0)
        lines.append(f"  {k}: {pct:.1f}% ({int(pct*total/100)}/{total})")
    return "\n".join(lines)


def _format_failure_examples(verdicts: list[dict], trajectories: list[dict], rubric_key: str, n: int = 2) -> str:
    """Return a text snippet showing N trajectory summaries that failed 'rubric_key'."""
    from nonfncall_judge import STRATEGY_KEYS
    import importlib.util
    from pathlib import Path

    _judge_py = Path(__file__).parent.parent / "funclocalize_judge" / "judge.py"
    spec = importlib.util.spec_from_file_location("_judge", _judge_py)
    jmod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(jmod)  # type: ignore

    # Build id -> trajectory mapping
    traj_by_id = {t["instance_id"]: t for t in trajectories}

    failed = [v for v in verdicts if v.get(rubric_key) is False][:n]
    if not failed:
        return f"  (no failures for {rubric_key})"

    lines = []
    for v in failed:
        iid = v["instance_id"]
        row = traj_by_id.get(iid)
        if not row:
            continue
        events = jmod.extract_localization_phase(row.get("history") or [])
        summary = jmod.trajectory_summary(events)[:1500]
        lines.append(f"--- instance {iid} ---\n{summary}\n  Judge note: {v.get('notes', '')}")
    return "\n\n".join(lines) if lines else f"  (no trajectory data for {rubric_key})"


def _template_based_initial_prompt() -> str:
    """Return a rule-engineered initial prompt based on rubrics (no LLM needed).

    Falls back to default_loc_strategy.j2 which already encodes the 3 rubrics.
    """
    loc_strategy_j2 = (
        Path(__file__).parent.parent.parent
        / "benchmarks"
        / "hybridgym_funclocalize"
        / "prompts"
        / "default_loc_strategy.j2"
    )
    return loc_strategy_j2.read_text()


def write_initial_prompt(api_key: str, model: str = WRITER_MODEL) -> str:
    """Write the first iteration prompt based on rubrics (starting from default.j2).

    Falls back to default_loc_strategy.j2 if the LLM API is unavailable.
    """
    client = _make_client(api_key)

    user_msg = f"""\
Here are the rubrics (desired behaviors) for the agent's localization phase:

{RUBRICS_TEXT}

Here is the current (baseline) prompt template that the agent uses:

{DEFAULT_TEMPLATE}

Rewrite this template to incorporate the 3 rubric behaviors as EXPLICIT step-by-step
instructions. Make the instructions concrete and actionable:
- Tell the agent to start with broad repo-wide searches BEFORE opening any files.
- Tell the agent to do multiple rounds of search refinement (≥2 rounds).
- Tell the agent NOT to read full file contents until it has narrowed to ≤3 candidates.

Keep the same 5-step workflow (EXPLORATION, UNDERSTANDING, RECHECK, GENERATION, REVIEW)
but expand the EXPLORATION step with the above localization strategy.

Return ONLY the Jinja2 template. No prose. No fences.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return _call_llm(client, model, messages)
    except Exception as e:
        print(f"  [LLM prompt writer failed: {e}] Using template-based fallback.", flush=True)
        return _template_based_initial_prompt()


def revise_prompt(
    api_key: str,
    current_template: str,
    baseline_scores: dict,
    prev_iter_scores: list[dict],
    baseline_verdicts: list[dict],
    baseline_trajectories: list[dict],
    iter_verdicts: list[dict],
    iter_trajectories: list[dict],
    iteration: int,
    model: str = WRITER_MODEL,
) -> str:
    """Revise the prompt based on rubric compliance scores and failure examples."""
    client = _make_client(api_key)

    # Format score history
    score_history = _format_scores(baseline_scores, "baseline (default prompt)")
    for i, s in enumerate(prev_iter_scores, 1):
        score_history += "\n" + _format_scores(s, f"iter {i}")

    # Find rubrics that still need improvement
    failing_rubrics = [
        k for k in ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing")
        if (prev_iter_scores[-1] if prev_iter_scores else baseline_scores).get(k, 0) < 0.9
    ]

    # Get failure examples for the worst rubric
    failure_text = ""
    if failing_rubrics and iter_verdicts and iter_trajectories:
        worst = min(failing_rubrics, key=lambda k: (prev_iter_scores[-1] if prev_iter_scores else baseline_scores).get(k, 0))
        failure_text = f"\n\nExample trajectories that FAILED '{worst}':\n"
        failure_text += _format_failure_examples(iter_verdicts, iter_trajectories, worst)

    user_msg = f"""\
We are iteratively improving an agent prompt for a Python function localization task.
This is revision iteration {iteration}.

RUBRICS (desired behaviors):
{RUBRICS_TEXT}

COMPLIANCE SCORE HISTORY:
{score_history}

Rubrics still below 90%: {', '.join(failing_rubrics) if failing_rubrics else 'none'}{failure_text}

CURRENT PROMPT TEMPLATE (iteration {iteration - 1}):
{current_template}

Based on the above scores and failure examples:
1. Identify which rubric instructions are not being followed.
2. Make the failing rubric instructions MORE explicit, with concrete examples of correct behavior.
3. If the agent is reading files before narrowing, add a warning/reminder not to do so.
4. If the agent is not doing multi-round search, add an example of good search refinement.

Return ONLY the revised Jinja2 template. No prose. No fences.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return _call_llm(client, model, messages)
    except Exception as e:
        print(f"  [LLM revise failed: {e}] Returning current template unchanged.", flush=True)
        return current_template
