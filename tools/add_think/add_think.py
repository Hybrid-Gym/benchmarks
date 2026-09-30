"""Insert LLM-synthesized `think` steps into function-localization trajectories that have none.

Why: students trained on func_localize_claude45_1457i (opus-4.5, a think step every ~7 actions) lose
accuracy when the think steps are removed; func_localize_gpt5mini_1346i has no think steps at all and
only 38 % of func_localize_claude47_1467i's trajectories have one. This tool adds think steps written
by an LLM, placed and phrased the way opus-4.5 uses them (analysis in tools/add_think/README.md).

Checkpoints (each becomes one `think` call + its "Your thought has been logged." result; two that
fall at the same position are merged into one thought):
  strategy  after the first action's result: read the description, decide where / what to search
  verify    after the first view of the target file: check its code against the description
  compose   right before the first edit: settle what the docstring says and how it is formatted
  review    right before `finish`: check the edit against the description
The target file is the file of the last successful edit (as in training/add_plan/add_plan.py).

Each thought opens with words drawn from opus-4.5's own openers for its role (openers.json), so the
openings are as varied as the original data's (a fixed prompt made 98 % of the strategy thoughts
open "The description ..."), and is written from the run UP TO its position plus the agent's very next call (its own
decision, so the thought can lead into it): nothing later is shown, so the thought cannot use what
the agent has not yet seen. A thought that names an identifier the run only shows later, that speaks
about the writing task instead of as the agent, or that is far off the usual length is asked for
again once, then dropped.

`--fact-check` then has an LLM check every thought against the run up to its position (the full tool
outputs, which the writer saw clipped): a thought that states a wrong name, parameter, default, path,
line number or behaviour (12 % of gpt5mini's, 8 % of claude47's in a sample; 0 of 60 of opus-4.5's own)
is written again with the checker's finding and longer tool results, at most twice, and dropped if it
is still wrong. Later
thoughts are checked with the corrected earlier ones in their run. Records: <dataset>.think.checked.jsonl.

Usage:
  python tools/add_think/add_think.py --hf synthetic-code-training/func_localize_gpt5mini_1346i \
      --out-dir eval_outputs/add_think [--checkpoints strategy,verify,compose,review] [--limit 3]
  python tools/add_think/add_think.py --hf ... --out-dir eval_outputs/add_think --fact-check --build --push
Records: <out-dir>/<dataset>.think.jsonl, one line per trajectory (resumable); the built dataset is
pushed as <org>/<dataset with the model tag>_add_think_<rows>i.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verbosity_rephrase"))

from fill_prose import leaks  # noqa: E402
from llm import (  # noqa: E402
    FORBIDDEN,
    AdaptiveLimiter,
    Messages,
    Rephraser,
    meta_hits,
    parse_versions,
)
from rephrase import load_api_key  # noqa: E402
from tokens import count_tokens  # noqa: E402


CHECKPOINTS = ("strategy", "verify", "compose", "review")
PURPOSE = {
    "strategy": "read the function description closely, note what kind of code it points to, and decide where and how to search first",
    "verify": "check the code it has just viewed against the description, point by point, and decide whether this is the target (and whether it lacks a docstring)",
    "compose": "settle what the docstring will say and how it is formatted (purpose, parameters, return value, style of the surrounding code) before making the edit",
    "review": "check the edit it has made against the description and the file's conventions and decide that the task is complete",
}
WORDS = {
    "strategy": (40, 130),
    "verify": (80, 300),
    "compose": (80, 300),
    "review": (80, 260),
}
MIN_WORDS, MAX_WORDS = 30, 450
TASK_CHARS = 4000
RESULT_CHARS = 1500
CALL_CHARS = 1200
THINK_RESULT = "EXECUTION RESULT of [function]:\nYour thought has been logged."
FACT_CHARS = 200_000  # run text shown to the fact checker; a longer run keeps its start and its latest steps
FACT_ROUNDS = 3  # checks per thought: the original and up to two rewrites
FACT_RESULT_CHARS = 8000  # a rewrite sees longer tool results: most wrong facts were about code clipped at RESULT_CHARS
FACT_NEXT_CHARS = (
    4000  # the next action as the checker sees it (an edit carries the whole docstring)
)
FUNC = re.compile(r"<function=([a-zA-Z_]+)>")
PARAM = re.compile(r"<parameter=([A-Za-z_]+)>(.*?)</parameter>", re.DOTALL)
EDIT = re.compile(
    r"<function=file_editor>\s*<parameter=command>(str_replace|insert)</parameter>\s*<parameter=path>(.*?)</parameter>",
    re.DOTALL,
)
VIEW = re.compile(
    r"<function=file_editor>\s*<parameter=command>view</parameter>\s*<parameter=path>(.*?)</parameter>",
    re.DOTALL,
)

SYSTEM_PROMPT = """You write the `think` steps of a software-engineering agent into a recorded run of that agent. The agent's task is to find, in a repository, the function or class that a description refers to and to write its docstring; it works with a terminal, a file editor and a `think` tool that only logs a thought.

The run below shows the task, every step the agent has taken so far with its tool results (long parts clipped), and then the agent's NEXT step. Write the thought the agent logs between the last result and that next step.

How this agent thinks (from runs of a strong agent on the same task):
- Early on it reads the description closely, names the kind of code it points to, and picks where to search.
- After a search or listing it weighs the candidates it sees, notes which already have docstrings, and picks what to inspect.
- After viewing code it checks the candidate against the description point by point ("it does X - yes, lines 12-15 ...") and decides whether this is the target.
- Before editing it settles the docstring's content and format from the code and the file's conventions.
- Before finishing it re-reads its edit against the description.
Style: first person, present tense, plain prose with a short numbered or bulleted list where it helps; it quotes the description's words and names the concrete files, functions and line numbers it has seen; it ends with the decision that leads into the next step. Its openings vary ("Looking at ...", "I found ...", "Let me ...", "Now ...", "Based on ...", "Both candidates ...", "Interesting - ..."); it never opens two thoughts of one run the same way.

Rules:
- Use only what the run shows up to this point. Never mention a file, name, line number, result or fact that the agent has not seen yet; do not invent code or behaviour.
- Be consistent with the NEXT step: the thought ends by deciding to do what that step does, in the agent's own words, without quoting the call.
- Write as the agent, in the moment. Do not refer to "the run", "the next step", these instructions, or "the agent"; no tool-call markup, no XML tags.
- {words_lo}-{words_hi} words.

Respond with ONLY a JSON object {{"summary": "<about 10 words naming what this thought does>", "thought": "..."}}."""


FACT_PROMPT = """Below is the start of a recorded run of a coding agent (its task, its actions and their tool outputs), a think step inserted at this point (its summary and thought), and the agent's next action. Check the think step.

Flag wrong_fact only for an error that would mislead a reader: a wrong identifier, parameter name, default value, file path, return value or code behaviour, or a claim about what a search or view showed that the run contradicts or clearly does not contain. Ignore line numbers off by one or two, loose wording, hedged guesses ("likely", "probably"), plans, and paraphrases of the task description.

Flag next_mismatch only when the thought commits to something the next action clearly does not do and is not a first step toward: it decides to revise or add something before finishing but the next action is `finish`; it settles a docstring format or content that the edit does not use; it picks one file, function or search and the next action goes to a different one. A next action that is a reasonable first step of the plan is fine.

# Run so far
{prefix}

# THINK STEP
summary: {summary}
thought:
{thought}

# Next action (after the think step)
{next}

Reply with JSON only: {{"wrong_fact": true|false, "next_mismatch": true|false, "detail": "<each problem: what the step says and what the run or the next action shows; empty if none>"}}"""


def clip(s: str, limit: int, tail: int = 300) -> str:
    if len(s) <= limit:
        return s
    return (
        s[: limit - tail] + f"\n[... {len(s) - limit} chars clipped ...]\n" + s[-tail:]
    )


def tool_of(content: str) -> str:
    m = FUNC.search(content)
    return m.group(1) if m else "(none)"


def render(messages: list[dict], upto: int, result_chars: int = RESULT_CHARS) -> str:
    """The task, then every step before message `upto` with its result, numbered."""
    out = []
    step = 0
    i = 1  # messages[0] is the system prompt
    while i < upto:
        m = messages[i]
        if m["role"] == "user":
            out.append(
                ("TASK:\n" if step == 0 else f"RESULT {step}:\n")
                + clip(m["content"], TASK_CHARS if step == 0 else result_chars)
            )
        else:
            step += 1
            out.append(f"STEP {step}:\n{clip(m['content'], CALL_CHARS)}")
        i += 1
    return "\n\n".join(out)


def checkpoints(messages: list[dict], wanted: tuple[str, ...]) -> dict[int, list[str]]:
    """Insertion index (a think goes right BEFORE messages[idx]) -> the checkpoints that fall there."""
    asst = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(asst) < 3 or tool_of(messages[asst[-1]]["content"]) != "finish":
        return {}
    target = None
    first_edit = None
    for j, i in enumerate(asst):
        m = EDIT.search(messages[i]["content"])
        if (
            m
            and i + 1 < len(messages)
            and "has been edited" in messages[i + 1]["content"]
        ):
            target = os.path.basename(m.group(2).strip())
            if first_edit is None:
                first_edit = j
    if target is None or first_edit is None:
        return {}
    first_view = None
    for j, i in enumerate(asst[:first_edit]):
        v = VIEW.search(messages[i]["content"])
        if v and os.path.basename(v.group(1).strip()) == target:
            first_view = j
            break
    at: dict[int, list[str]] = {}
    where = {
        "strategy": asst[1] if first_edit > 1 else None,
        "verify": asst[first_view + 1]
        if first_view is not None and first_view + 1 <= first_edit
        else None,
        "compose": asst[first_edit],
        "review": asst[-1],
    }
    for name in CHECKPOINTS:
        idx = where[name]
        if name in wanted and idx is not None:
            at.setdefault(idx, []).append(name)
    return at


def think_message(summary: str, thought: str) -> dict:
    return {
        "role": "assistant",
        "content": f"<function=think>\n<parameter=summary>{summary}</parameter>\n<parameter=thought>\n{thought}\n</parameter>\n</function>",
    }


def ask_prompt(
    messages: list[dict],
    idx: int,
    names: list[str],
    exemplars: dict[str, Any],
    result_chars: int = RESULT_CHARS,
) -> tuple[str, str]:
    """(system prompt, user prompt) asking for the thought of checkpoint(s) `names` before messages[idx]."""
    lo = min(WORDS[n][0] for n in names)
    hi = max(WORDS[n][1] for n in names)
    purpose = "; and ".join(PURPOSE[n] for n in names)
    rng = random.Random(
        f"{messages[1]['content'][:200]}|{idx}"
    )  # per-trajectory choice, reproducible
    stems = exemplars.get("_openers", {}).get(names[0], [])
    opener = (
        rng.choices([st for st, _ in stems], [w for _, w in stems])[0] if stems else ""
    )
    shots = "\n\n".join(
        f"Example ({ex['type']}; summary: {ex['summary']}):\n{ex['thought']}"
        for n in names
        for ex in rng.sample(exemplars.get(n, []), min(2, len(exemplars.get(n, []))))
    )
    user = (
        f"{render(messages, idx, result_chars)}\n\nNEXT STEP (the agent's next call, after the thought):\n"
        f"{clip(messages[idx]['content'], CALL_CHARS)}\n\n"
        f"At this point the agent pauses to {purpose}. Write that thought now"
        + (
            f', beginning with the words "{opener}" and continuing naturally from them.'
            if opener
            else "."
        )
        + "\n\n"
        f"Examples of such thoughts from other runs on other tasks (style only; their facts do not apply here, and "
        f"do not copy their openings or phrasing - vary how you begin, e.g. from what you have just seen, from the "
        f"description, from a doubt, or from a comparison of candidates):\n\n{shots}"
    )
    return SYSTEM_PROMPT.format(words_lo=lo, words_hi=hi), user


def problems(messages: list[dict], idx: int, thought: str) -> str:
    """Why `thought` is unusable at insertion index `idx`, or ""."""
    if any(f in thought for f in FORBIDDEN) or "<function" in thought:
        return "it contained tool-call markup; write prose only"
    n = len(thought.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return f"it had {n} words; keep to the requested length"
    hits = meta_hits(thought, "", "thought")
    if hits:
        return f'it spoke about the writing task ("{hits[0]}") instead of as the agent'
    leaked = leaks(messages, idx, thought)
    if leaked:
        return (
            f"it named {', '.join(leaked[:3])}, which the agent has not seen yet; use only what "
            "the run shows so far"
        )
    return ""


def write_thought(
    rp: Rephraser,
    messages: list[dict],
    idx: int,
    names: list[str],
    exemplars: dict,
    usage: dict,
    note: str = "",
    result_chars: int = RESULT_CHARS,
) -> tuple[dict | None, list[dict]]:
    """(the thought record, the attempt log) for the checkpoint(s) `names` before messages[idx];
    `note` says what was wrong with an earlier thought for this position."""
    system, user = ask_prompt(messages, idx, names, exemplars, result_chars)
    log: list[dict] = []
    for rnd in range(3):
        ask: Messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user
                + (f"\n\nYour previous reply was unusable: {note}." if note else ""),
            },
        ]
        raw, u = rp.complete(
            ask, temperature=None if rnd == 0 else 0.7, max_tokens=1200
        )
        usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += u.get(k) or 0
        got = parse_versions(raw, ["summary", "thought"])
        thought = got.get("thought", "").strip()
        summary = re.sub(r"\s+", " ", got.get("summary", "").strip())[:120]
        why = (
            "it was not a JSON object with summary and thought"
            if not thought
            else problems(messages, idx, thought)
        )
        log.append(
            {"round": rnd + 1, "words": len(thought.split()), "why": why or None}
        )
        if not why:
            return {
                "checkpoints": names,
                "insert_before": idx,
                "summary": summary or "Thinking through the next step",
                "thought": thought,
                "words": len(thought.split()),
                "rounds": rnd + 1,
            }, log
        note = why
    return None, log


def process_row(
    rp: Rephraser, row: dict, wanted: tuple[str, ...], exemplars: dict
) -> dict:
    t0 = time.time()
    messages = list(row["messages"])
    at = checkpoints(messages, wanted)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    thoughts: list[dict] = []
    dropped: list[dict] = []
    logs: list[dict] = []
    shift = 0  # inserted messages so far shift later original indices
    err = None
    for idx in sorted(at):
        try:
            rec, log = write_thought(
                rp, messages, idx + shift, at[idx], exemplars, usage
            )
        except Exception as e:  # noqa: BLE001
            err = f"api at {idx}: {type(e).__name__}: {str(e)[:200]}"
            break
        logs += log
        if rec is None:
            dropped.append({"checkpoints": at[idx], "insert_before": idx})
            continue
        rec["insert_before"] = idx  # index in the ORIGINAL messages
        thoughts.append(rec)
        messages[idx + shift : idx + shift] = [
            think_message(rec["summary"], rec["thought"]),
            {"role": "user", "content": THINK_RESULT},
        ]
        shift += 2
    out = {
        "instance_id": row["instance_id"],
        "checkpoints": {str(k): v for k, v in at.items()},
        "thoughts": thoughts,
        "dropped": dropped,
        "usage": usage,
        "log": logs,
        "model": rp.model,
        "elapsed": round(time.time() - t0, 2),
    }
    if err:
        out["error"] = err
    return out


def wrong_fact(
    judge: Rephraser,
    messages: list[dict],
    idx: int,
    summary: str,
    thought: str,
    usage: dict,
) -> str:
    """What the think step (`summary`, `thought`) inserted before messages[idx] gets wrong about the run so far
    or about the next action, or ""."""
    prefix = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages[1:idx])
    if len(prefix) > FACT_CHARS:
        prefix = (
            prefix[:8000]
            + "\n[... middle of the run omitted ...]\n"
            + prefix[-(FACT_CHARS - 8000) :]
        )
    ask: Messages = [
        {
            "role": "user",
            "content": FACT_PROMPT.format(
                prefix=prefix,
                next=clip(messages[idx]["content"], FACT_NEXT_CHARS),
                summary=summary,
                thought=thought,
            ),
        }
    ]
    for _ in range(3):
        raw, u = judge.complete(ask)
        usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += u.get(k) or 0
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            verdict = json.loads(m.group(0)) if m else None
        except ValueError:
            verdict = None
        if isinstance(verdict, dict) and "wrong_fact" in verdict:
            if not (verdict["wrong_fact"] or verdict.get("next_mismatch")):
                return ""
            return str(verdict.get("detail") or "unspecified").strip()[:600]
    raise RuntimeError("the fact checker gave no verdict")


def fact_check_row(
    rp: Rephraser, judge: Rephraser, row: dict, rec: dict, exemplars: dict
) -> dict:
    """`rec` with every thought checked against the run up to its position and rewritten (or dropped) if wrong."""
    t0 = time.time()
    messages = list(row["messages"])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    kept: list[dict] = []
    dropped = list(rec["dropped"])
    checks: list[dict] = []
    shift = 0
    try:
        for t in sorted(rec["thoughts"], key=lambda r: r["insert_before"]):
            idx = t["insert_before"] + shift
            cur: dict | None = t
            flags: list[str] = []
            for rnd in range(FACT_ROUNDS):
                assert cur is not None
                why = wrong_fact(
                    judge, messages, idx, cur["summary"], cur["thought"], judge_usage
                )
                if not why:
                    break
                flags.append(why)
                if rnd == FACT_ROUNDS - 1:
                    cur = None
                    break
                cur, _ = write_thought(
                    rp,
                    messages,
                    idx,
                    t["checkpoints"],
                    exemplars,
                    usage,
                    note=f"it disagreed with the run or with the next step ({why}); state only facts the run shows "
                    "and end with the decision the next step carries out",
                    result_chars=FACT_RESULT_CHARS,
                )
                if cur is None:
                    break
            checks.append(
                {
                    "checkpoints": t["checkpoints"],
                    "insert_before": t["insert_before"],
                    "flags": flags,
                    "outcome": "ok"
                    if not flags
                    else ("rewritten" if cur is not None else "dropped"),
                }
            )
            if cur is None:
                dropped.append(
                    {
                        "checkpoints": t["checkpoints"],
                        "insert_before": t["insert_before"],
                        "why": "fact check",
                    }
                )
                continue
            cur["insert_before"] = t["insert_before"]
            kept.append(cur)
            messages[idx:idx] = [
                think_message(cur["summary"], cur["thought"]),
                {"role": "user", "content": THINK_RESULT},
            ]
            shift += 2
    except Exception as e:  # noqa: BLE001
        return {
            "instance_id": rec["instance_id"],
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    return {
        **rec,
        "thoughts": kept,
        "dropped": dropped,
        "fact_check": checks,
        "fact_usage": {"writer": usage, "checker": judge_usage},
        "fact_model": judge.model,
        "fact_elapsed": round(time.time() - t0, 2),
    }


def load_records(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("error"):
                out[r["instance_id"]] = r
    return out


def insert_all(messages: list[dict], thoughts: list[dict]) -> list[dict]:
    out = list(messages)
    for rec in sorted(thoughts, key=lambda r: r["insert_before"], reverse=True):
        out[rec["insert_before"] : rec["insert_before"]] = [
            think_message(rec["summary"], rec["thought"]),
            {"role": "user", "content": THINK_RESULT},
        ]
    return out


def has_think(messages: list[dict]) -> bool:
    return any(
        m["role"] == "assistant" and "<function=think>" in m["content"]
        for m in messages
    )


def derive_repo(base: str, n_rows: int) -> str:
    """synthetic-code-training/func_localize_gpt5mini_1346i -> .../func_localize_gpt5mini_add_think_1346i."""
    stem = base.rsplit("_", 1)[0] if re.search(r"_\d+i$", base) else base
    return f"{stem}_add_think_{n_rows}i"


def card(base: str, repo: str, model: str, stats: dict, wanted: tuple[str, ...]) -> str:
    w = stats["words"]
    return f"""---
license: mit
---
# {repo.split("/")[-1]}

[`{base}`](https://huggingface.co/datasets/{base}) with LLM-synthesized `think` steps added to every trajectory that had none
(`tools/add_think/add_think.py` in the `benchmarks` repo; model `{model}`). Trajectories that already used the think tool are unchanged.

The steps are placed where opus-4.5 (`func_localize_claude45_1457i`) uses its think tool: {", ".join(wanted)}
(after the first action's result; after the first view of the target file; right before the first edit; right before `finish`;
two at the same position are merged). Each thought was written from the run up to that point plus the agent's next call only,
so it cannot use anything the agent had not yet seen; thoughts naming an identifier that appears only later, or speaking about
the writing task, were re-asked once and otherwise dropped.

| | value |
|---|---|
| rows | {stats["rows"]} |
| trajectories given think steps | {stats["rows_added"]} |
| trajectories that already had think steps (unchanged) | {stats["rows_had"]} |
| trajectories without a usable target edit (unchanged) | {stats["rows_none"]} |
| think steps added: count / per trajectory | {stats["added"]} / {stats["added"] / max(stats["rows_added"], 1):.2f} |
| checkpoints dropped after re-asks | {stats["dropped"]} |
| thought words: mean / median | {statistics.mean(w) if w else 0:.0f} / {statistics.median(w) if w else 0:.0f} |
| thought tokens (Qwen3 tokenizer): mean | {statistics.mean(stats["tokens"]) if stats["tokens"] else 0:.0f} |
""" + (
        f"""
Every thought was then checked by an LLM against the run up to its position (full tool outputs): one that stated a wrong
name, parameter, default, path, line number or behaviour was written again with the checker's finding (at most twice) and
dropped if still wrong. Checked {stats["checked"]}, flagged {stats["flagged"]}, rewritten {stats["rewritten"]}, dropped {stats["fact_dropped"]}.
"""
        if stats["checked"]
        else ""
    )


def fact_check(
    args: argparse.Namespace,
    rows: list[dict],
    done: dict[str, dict],
    path: Path,
    exemplars: dict,
) -> dict[str, dict]:
    """Fact-check the records in `done` not yet in `path` (resumable); return the checked records."""
    checked = load_records(path)
    by_id = {r["instance_id"]: r for r in rows}
    todo = [rec for iid, rec in done.items() if iid not in checked and iid in by_id]
    n_thoughts = sum(len(r["thoughts"]) for r in todo)
    print(
        f"fact check: {len(done)} records, {len(checked)} checked, {len(todo)} to check ({n_thoughts} thoughts, "
        f"checker {args.judge_model})",
        file=sys.stderr,
    )
    if not todo:
        return checked
    key = load_api_key(args.api_key)
    if not key:
        sys.exit("error: no API key")
    limiter = AdaptiveLimiter(
        start=max(args.workers_min, args.workers // 2),
        lo=args.workers_min,
        hi=args.workers,
    )
    rp = Rephraser(
        key,
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=1200,
        extra_body=json.loads(args.extra_body),
        limiter=limiter,
    )
    judge = Rephraser(
        key,
        args.base_url,
        args.judge_model,
        temperature=0.0,
        max_tokens=8000,
        extra_body=json.loads(args.judge_extra_body),
        limiter=limiter,
    )
    t0 = time.time()
    n_err = n_flag = n_all = 0
    with (
        path.open("a") as fh,
        concurrent.futures.ThreadPoolExecutor(args.workers) as ex,
    ):
        futs = [
            ex.submit(
                fact_check_row, rp, judge, by_id[rec["instance_id"]], rec, exemplars
            )
            for rec in todo
        ]
        try:
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                res = fut.result()
                n_err += bool(res.get("error"))
                for c in res.get("fact_check", []):
                    n_all += 1
                    n_flag += bool(c["flags"])
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-6) * 60
                    print(
                        f"  {i}/{len(todo)} err={n_err} flagged={n_flag}/{n_all} {rate:.1f} rows/min "
                        f"eta={(len(todo) - i) / max(rate, 1e-6):.0f} min",
                        file=sys.stderr,
                    )
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    return load_records(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--checkpoints", default=",".join(CHECKPOINTS))
    p.add_argument(
        "--exemplars",
        default=str(Path(__file__).with_name("exemplars.json")),
        help="opus-4.5 thoughts by checkpoint type, shown as style examples",
    )
    p.add_argument(
        "--openers",
        default=str(Path(__file__).with_name("openers.json")),
        help="opus-4.5's opening words per role with their frequencies; one is drawn per thought",
    )
    p.add_argument(
        "--model",
        default=os.environ.get(
            "REPHRASE_MODEL", "nvidia/deepseek-ai/deepseek-v4-flash"
        ),
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com/v1"),
    )
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--extra-body", default='{"chat_template_kwargs":{"thinking":false}}'
    )
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--workers-min", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="first N rows (debug)")
    p.add_argument(
        "--fact-check",
        action="store_true",
        help="check every thought against the run so far and rewrite / drop wrong ones (<dataset>.think.checked.jsonl); "
        "--build then uses the checked records",
    )
    p.add_argument(
        "--judge-model",
        default="nvidia/deepseek-ai/deepseek-v4-flash",
        help="fact checker (reasoning on)",
    )
    p.add_argument("--judge-extra-body", default="{}")
    p.add_argument(
        "--build", action="store_true", help="build the dataset from the records"
    )
    p.add_argument("--push", action="store_true")
    p.add_argument("--org", default="synthetic-code-training")
    p.add_argument("--repo", default=None, help="override the derived output repo name")
    args = p.parse_args()
    wanted = tuple(c for c in args.checkpoints.split(",") if c)
    if any(c not in CHECKPOINTS for c in wanted):
        sys.exit(f"error: --checkpoints must be among {CHECKPOINTS}")
    exemplars = json.load(open(args.exemplars))
    exemplars["_openers"] = json.load(open(args.openers))
    ds = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(ds, Dataset)
    if args.limit:
        ds = ds.select(range(min(args.limit, ds.num_rows)))
    rows = list(cast(Any, ds))
    label = args.hf.split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"{label}.think.jsonl"
    done = load_records(rec_path)
    todo = [
        r for r in rows if not has_think(r["messages"]) and r["instance_id"] not in done
    ]
    print(
        f"{label}: {len(rows)} rows, {sum(has_think(r['messages']) for r in rows)} already have think steps, "
        f"{len(done)} done, {len(todo)} to write (checkpoints={','.join(wanted)}, model={args.model})",
        file=sys.stderr,
    )
    if todo:
        key = load_api_key(args.api_key)
        if not key:
            sys.exit("error: no API key")
        rp = Rephraser(
            key,
            args.base_url,
            args.model,
            temperature=args.temperature,
            max_tokens=1200,
            extra_body=json.loads(args.extra_body),
            limiter=AdaptiveLimiter(
                start=max(args.workers_min, args.workers // 2),
                lo=args.workers_min,
                hi=args.workers,
            ),
        )
        t0 = time.time()
        n_err = 0
        with (
            rec_path.open("a") as fh,
            concurrent.futures.ThreadPoolExecutor(args.workers) as ex,
        ):
            futs = [ex.submit(process_row, rp, r, wanted, exemplars) for r in todo]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                res = fut.result()
                n_err += bool(res.get("error"))
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-6) * 60
                    print(
                        f"  {i}/{len(todo)} err={n_err} {rate:.1f} rows/min eta={(len(todo) - i) / max(rate, 1e-6):.0f} min",
                        file=sys.stderr,
                    )
        done = load_records(rec_path)
    if args.fact_check:
        done = fact_check(
            args, rows, done, out_dir / f"{label}.think.checked.jsonl", exemplars
        )
    if not (args.build or args.push):
        return
    missing = [
        r["instance_id"]
        for r in rows
        if not has_think(r["messages"]) and r["instance_id"] not in done
    ]
    if missing:
        sys.exit(
            f"error: {len(missing)} trajectories have no record yet; rerun without --build first"
        )
    stats: dict = {
        "rows": len(rows),
        "rows_added": 0,
        "rows_had": 0,
        "rows_none": 0,
        "added": 0,
        "dropped": 0,
        "words": [],
        "tokens": [],
        "checked": 0,
        "flagged": 0,
        "rewritten": 0,
        "fact_dropped": 0,
    }
    built = []
    for r in rows:
        msgs = r["messages"]
        if has_think(msgs):
            stats["rows_had"] += 1
        else:
            rec = done[r["instance_id"]]
            if rec["thoughts"]:
                stats["rows_added"] += 1
            else:
                stats["rows_none"] += 1
            stats["added"] += len(rec["thoughts"])
            stats["dropped"] += len(rec["dropped"])
            for c in rec.get("fact_check", []):
                stats["checked"] += 1
                stats["flagged"] += bool(c["flags"])
                stats["rewritten"] += c["outcome"] == "rewritten"
                stats["fact_dropped"] += c["outcome"] == "dropped"
            for t in rec["thoughts"]:
                stats["words"].append(t["words"])
                stats["tokens"].append(count_tokens(t["thought"]))
            msgs = insert_all(msgs, rec["thoughts"])
        built.append({**r, "messages": msgs})
    repo = args.repo or f"{args.org}/{derive_repo(label, len(built))}"
    model = next(iter(done.values()))["model"] if done else args.model
    readme = card(args.hf, repo, model, stats, wanted)
    out = Dataset.from_list(built)
    out.save_to_disk(str(out_dir / repo.split("/")[-1]))
    (out_dir / f"{repo.split('/')[-1]}.README.md").write_text(readme)
    print(
        f"{repo}: rows={stats['rows']} added={stats['added']} to {stats['rows_added']} trajectories "
        f"(had think: {stats['rows_had']}, no target: {stats['rows_none']}, dropped checkpoints: {stats['dropped']}) "
        f"words mean={statistics.mean(stats['words']) if stats['words'] else 0:.0f}"
        + (
            f" | fact check: {stats['flagged']}/{stats['checked']} flagged, {stats['rewritten']} rewritten, "
            f"{stats['fact_dropped']} dropped"
            if stats["checked"]
            else ""
        ),
        file=sys.stderr,
    )
    if args.push:
        if args.limit:
            sys.exit("refusing to --push a --limit build")
        out.push_to_hub(repo, split=args.hf_split, private=False)
        HfApi().upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"  pushed {repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
