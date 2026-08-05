"""LLM-as-judge for funclocalize trajectories' compliance with the 3 localization strategies.

Strategies (see prompts/default_loc_strategy.j2):
  1. broad_then_narrow         — broad searches first, then narrow
  2. multi_round_refinement    — multiple distinct keywords, refined across rounds
  3. read_after_narrowing      — read full files only after narrowing to a few candidates

Two input shapes are supported:
  --src PATH      a rollout output.jsonl, whose rows carry SDK `history` events
  --hf REPO       a pushed HF dataset, whose rows carry non-fncall `messages`

Usage:
  python tools/funclocalize_judge/judge.py \\
      --src eval_outputs/.../baseline/output.jsonl \\
      --src eval_outputs/.../experiment/output.jsonl \\
      --out-dir /tmp/verdicts \\
      --filter-ids /tmp/funclocalize_1500_sample300_seed42_ids.txt

  python tools/funclocalize_judge/judge.py \\
      --hf synthetic-code-training/func_localize_claude45_1457i \\
      --out-dir eval_outputs/funclocalize_judge \\
      --model nvidia/deepseek-ai/deepseek-v4-flash

A single source needs --out PATH instead of --out-dir.
The judge call goes to the same OpenAI-compatible gateway as the rollouts.

Verdicts are appended as they complete and already-judged instance ids are skipped
on restart, so an interrupted run resumes instead of re-paying for the same calls.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


MAX_TRAJECTORY_CHARS = 10_000
MAX_TASK_CHARS = 500
JUDGE_MAX_OUTPUT_TOKENS = 4000  # reasoning models eat most of this internally
EDIT_COMMANDS = {"create", "str_replace", "insert"}
STRATEGY_KEYS = ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing")

# The gateway sits behind an AWS WAF per-IP rate limiter that 429s in bursts, so a
# judge call that fails once will usually succeed a few seconds later. Without this
# a long run silently converts transient 429s into permanent `error` rows.
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 2.0


JUDGE_SYSTEM = (
    "You are an evaluator. You assess whether an agent's localization phase "
    "followed three specific search strategies. Be strict but fair, and "
    "respond ONLY with the requested JSON."
)

JUDGE_USER_TEMPLATE = """\
The agent's task: locate a python function/class from a behavior description, then add a docstring.

Below is a condensed log of the agent's actions and observations BEFORE its first edit.

# Task given to the agent
{task_description}

# Localization trajectory
{trajectory_summary}

# Strategies to score

1. BROAD-THEN-NARROW
   Compliant if the agent's first 1-2 search/grep actions cover the WHOLE repository
   (e.g. ripgrep over the tree, find . -name, grep -r), BEFORE opening any specific file,
   AND later actions narrow (subdirectory or specific file).
   Non-compliant if the agent opens a specific file before any broad search, or never narrows.

2. MULTI-ROUND REFINEMENT
   Compliant if the agent tries >=2 DISTINCT keywords/patterns while localizing the file or
   function, with later searches refining earlier ones (a different term drawn from the
   description, a restricted path, a different surrounding pattern) based on what it learned
   from prior results.
   Non-compliant if the agent only ever searches one keyword, or commits to a single candidate
   after just one search. Re-running the same keyword unchanged does not count as a second one.

3. READ AFTER NARROWING
   Compliant if "full" file reads (file_editor view with no view_range OR with a large range
   like more than ~100 lines) happen ONLY after the candidate set has been narrowed to <=3
   files/functions.
   Non-compliant if the agent reads full files early, before narrowing.

Output ONLY this JSON, no prose, no markdown:
{{
  "broad_then_narrow": <true|false>,
  "multi_round_refinement": <true|false>,
  "read_after_narrowing": <true|false>,
  "notes": "<one short sentence summarizing your reasoning for all three>"
}}
"""


# ---- trajectory extraction ----------------------------------------------


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def extract_localization_phase(history: list[dict]) -> list[dict]:
    """Return events up to (and not including) the first file-modifying edit."""
    events = []
    for ev in history:
        if ev.get("kind") == "ActionEvent":
            tc = ev.get("tool_call") or {}
            if (tc.get("name") or ev.get("tool_name")) == "file_editor":
                if _parse_args(tc.get("arguments")).get("command") in EDIT_COMMANDS:
                    break
        events.append(ev)
    return events


def extract_task_description(history: list[dict]) -> str:
    for ev in history:
        if ev.get("kind") == "MessageEvent":
            msg = ev.get("llm_message") or {}
            if msg.get("role") == "user":
                return _flatten(msg.get("content", ""))[:MAX_TASK_CHARS]
    return ""


def summarize_event(ev: dict) -> str | None:
    """One-line condensed summary of an event for the judge."""
    kind = ev.get("kind")
    if kind == "MessageEvent":
        msg = ev.get("llm_message") or {}
        if msg.get("role") == "user":
            return None
        content = _flatten(msg.get("content", ""))
        return f"[think] {content[:300]}" if content.strip() else None

    if kind == "ActionEvent":
        tc = ev.get("tool_call") or {}
        name = tc.get("name") or ev.get("tool_name") or ""
        args = _parse_args(tc.get("arguments"))
        if name == "terminal":
            return f"[terminal] {args.get('command', '')[:240]}"
        if name == "file_editor":
            cmd = args.get("command", "view")
            path = args.get("path", "")
            if cmd == "view":
                vr = args.get("view_range")
                return f"[file_editor view {path}{f' range={vr}' if vr else ' (whole file)'}]"
            return f"[file_editor {cmd} {path}]"
        if name == "think":
            return f"[think] {args.get('thought', '')[:200]}"
        if name == "task_tracker":
            return f"[task_tracker] {args.get('command', '')}"
        if name == "finish":
            return f"[finish] {args.get('message', '')[:120]}"
        return None

    if kind == "ObservationEvent":
        content = _flatten((ev.get("observation") or {}).get("content", ""))
        return f"[result] {content[:280]}" if content else None

    return None


def trajectory_summary(events: list[dict]) -> str:
    lines = [s for s in (summarize_event(ev) for ev in events) if s]
    return _join_and_truncate(lines)


def _join_and_truncate(lines: list[str]) -> str:
    body = "\n".join(lines)
    if len(body) > MAX_TRAJECTORY_CHARS:
        half = MAX_TRAJECTORY_CHARS // 2
        body = body[:half] + "\n[... trajectory truncated ...]\n" + body[-half:]
    return body


# ---- non-fncall `messages` extraction ------------------------------------
#
# Trajectories pushed to the Hub are stored in the non-fncall chat format, where a
# tool call is text inside the assistant turn rather than a structured field:
#
#   <function=terminal>
#   <parameter=command>grep -rn "excepthook" .</parameter>
#   </function>
#
# and the result comes back as a user turn prefixed "EXECUTION RESULT of [function]:".
# The summaries below deliberately mirror summarize_event()'s output so the judge
# prompt sees the same shape regardless of which source the trajectory came from.

_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>", re.DOTALL
)
_RESULT_PREFIX = "EXECUTION RESULT of [function]:"
_THINK_SPLIT_RE = re.compile(r"</think>", re.IGNORECASE)


def parse_function_blocks(content: str) -> list[tuple[str, dict[str, str]]]:
    """Extract (tool_name, params) for each non-fncall block in an assistant turn."""
    return [
        (m.group(1), {k: v.strip() for k, v in _PARAMETER_RE.findall(m.group(2))})
        for m in _FUNCTION_RE.finditer(content)
    ]


def _summarize_call(name: str, args: dict[str, str]) -> str | None:
    if name == "terminal":
        return f"[terminal] {args.get('command', '')[:240]}"
    if name == "file_editor":
        cmd = args.get("command", "view")
        path = args.get("path", "")
        if cmd == "view":
            vr = args.get("view_range")
            return (
                f"[file_editor view {path}{f' range={vr}' if vr else ' (whole file)'}]"
            )
        return f"[file_editor {cmd} {path}]"
    if name == "think":
        return f"[think] {args.get('thought', '')[:200]}"
    if name == "task_tracker":
        return f"[task_tracker] {args.get('command', '')}"
    if name == "finish":
        return f"[finish] {args.get('message', '')[:120]}"
    return None


def summarize_messages(messages: list[dict]) -> tuple[str, str]:
    """Return (task_description, condensed pre-edit trajectory) for a messages row."""
    task = ""
    lines: list[str] = []
    stop = False

    for msg in messages:
        if stop:
            break
        role, content = msg.get("role"), (msg.get("content") or "")

        if role == "system":
            continue

        if role == "user":
            if content.startswith(_RESULT_PREFIX):
                body = content[len(_RESULT_PREFIX) :].strip()
                if body:
                    lines.append(f"[result] {body[:280]}")
            elif not task:
                # The first non-result user turn is the task prompt.
                task = content[:MAX_TASK_CHARS]
            continue

        if role != "assistant":
            continue

        calls = parse_function_blocks(content)
        # Reasoning models emit "<thought></think>" before the call; keep the prose
        # so the judge can see the agent's stated search intent, not just commands.
        prose = _THINK_SPLIT_RE.split(_FUNCTION_RE.sub("", content))[-1].strip()
        if prose:
            lines.append(f"[think] {prose[:300]}")

        for name, args in calls:
            if name == "file_editor" and args.get("command") in EDIT_COMMANDS:
                stop = True  # localization phase ends at the first edit
                break
            summary = _summarize_call(name, args)
            if summary:
                lines.append(summary)

    return task, _join_and_truncate(lines)


# ---- LLM judge -----------------------------------------------------------


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw).strip()


def _call_with_retry(client: OpenAI, model: str, prompt: str) -> str:
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 - any gateway failure is retryable
            last = e
            if attempt < MAX_ATTEMPTS - 1:
                # Jittered, so a burst of workers that all got 429'd does not
                # synchronise and re-trip the per-IP limiter together.
                time.sleep(BACKOFF_BASE_S * (2**attempt) * (0.5 + random.random()))
    raise last if last else RuntimeError("no attempts made")


def judge_one(client: OpenAI, model: str, row: dict) -> dict:
    iid = row["instance_id"]
    if row.get("messages") is not None:
        task, summary = summarize_messages(row["messages"])
    else:
        history = row.get("history") or []
        task = extract_task_description(history)
        summary = trajectory_summary(extract_localization_phase(history))
    if not summary.strip():
        return {"instance_id": iid, "error": "no localization actions"}

    prompt = JUDGE_USER_TEMPLATE.format(
        task_description=task, trajectory_summary=summary
    )
    try:
        raw = _strip_fences(_call_with_retry(client, model, prompt))
    except Exception as e:
        return {"instance_id": iid, "error": f"api: {e}"}

    try:
        verdict = json.loads(raw)
    except Exception as e:
        return {"instance_id": iid, "error": f"parse: {e}", "raw": raw[:300]}

    out: dict = {"instance_id": iid}
    for k in STRATEGY_KEYS:
        v = verdict.get(k)
        out[k] = v if isinstance(v, bool) else None
    out["notes"] = verdict.get("notes", "")
    return out


# ---- driver --------------------------------------------------------------


_LABEL_TS_RE = re.compile(r"-funclocalize-\d{8}T\d{6}Z$")


def short_label(dirname: str) -> str:
    """Strip the boilerplate prefix and timestamp suffix from an eval_outputs dir name."""
    run_note = dirname.split("_N_", 1)[1] if "_N_" in dirname else dirname
    return _LABEL_TS_RE.sub("", run_note) or run_note


def load_rows(path: Path, keep: set[str] | None, limit: int) -> list[dict]:
    rows = []
    for line in path.open():
        r = json.loads(line)
        if r.get("error"):
            continue
        if keep is not None and r["instance_id"] not in keep:
            continue
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def load_hf_rows(
    repo: str, split: str, keep: set[str] | None, limit: int
) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    rows: list[dict] = []
    for raw in ds:
        # Iterating a Dataset is typed as yielding the column-slice union rather
        # than a row mapping, so narrow it before subscripting.
        row: dict = dict(raw)  # type: ignore[arg-type]
        if keep is not None and row["instance_id"] not in keep:
            continue
        rows.append({"instance_id": row["instance_id"], "messages": row["messages"]})
        if limit and len(rows) >= limit:
            break
    return rows


def load_done_ids(path: Path) -> set[str]:
    """Instance ids already judged, so a restart skips them."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            v = json.loads(line)
        except Exception:
            continue  # a torn final line from a killed run
        # Errored rows are retried; only successful verdicts count as done.
        if not v.get("error") and v.get("instance_id"):
            done.add(v["instance_id"])
    return done


def aggregate(verdicts: list[dict]) -> dict:
    total = errs = 0
    counts = dict.fromkeys(STRATEGY_KEYS, 0)
    for v in verdicts:
        if v.get("error"):
            errs += 1
            continue
        total += 1
        for k in STRATEGY_KEYS:
            if v.get(k) is True:
                counts[k] += 1
    return {"total": total, "errors": errs, **counts}


def print_aggregate(label: str, agg: dict) -> None:
    t = agg["total"]
    print(f"\n=== {label} ===")
    print(f"  judged: {t}   errors: {agg['errors']}")
    if t == 0:
        return
    for k in STRATEGY_KEYS:
        c = agg[k]
        print(f"  {k:30s} {c:4d} / {t}   ({100 * c / t:5.1f}%)")


def print_comparison(labels: list[str], aggs: list[dict]) -> None:
    if len(aggs) < 2:
        return
    print("\n=== comparison ===")
    print(
        f"  {'strategy':30s}"
        + "".join(f"  {lbl:>20s}" for lbl in labels)
        + "    Δ(2nd-1st)"
    )
    for k in STRATEGY_KEYS:
        pcts = [100 * a[k] / max(a["total"], 1) for a in aggs]
        cells = "".join(f"  {p:>18.1f}%" for p in pcts)
        delta = pcts[1] - pcts[0]
        print(f"  {k:30s}{cells}    {delta:+.1f}pp")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--src",
        action="append",
        default=[],
        help="Rollout output.jsonl; repeat for head-to-head comparison",
    )
    p.add_argument(
        "--hf",
        action="append",
        default=[],
        help="HF dataset repo of pushed trajectories (messages format); repeatable",
    )
    p.add_argument("--hf-split", default="train", help="Split for --hf sources")
    p.add_argument("--out", help="Output verdicts JSONL (use with a single source)")
    p.add_argument(
        "--out-dir",
        help="Directory for verdicts JSONLs (use with multiple sources); filenames derived from each source's label",
    )
    p.add_argument(
        "--filter-ids",
        help="Optional file of instance IDs (one per line) to restrict judging",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", "openai/openai/gpt-5-mini"),
    )
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com/v1"),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=0, help="Per-source trajectory cap (debug)"
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-judge every trajectory instead of skipping ids already in the output",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    sources = [("src", s) for s in args.src] + [("hf", h) for h in args.hf]
    if not sources:
        sys.exit("error: at least one --src or --hf is required")
    if not args.api_key:
        sys.exit("error: --api-key required (or set LLM_API_KEY)")
    if len(sources) == 1 and not (args.out or args.out_dir):
        sys.exit("error: --out or --out-dir required")
    if len(sources) > 1 and not args.out_dir:
        sys.exit("error: --out-dir required when multiple sources are given")

    keep: set[str] | None = None
    if args.filter_ids:
        keep = {
            ln.strip()
            for ln in Path(args.filter_ids).read_text().splitlines()
            if ln.strip()
        }
        print(
            f"Filtering to {len(keep)} instance IDs from {args.filter_ids}",
            file=sys.stderr,
        )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    labels: list[str] = []
    aggs: list[dict] = []

    for kind, ref in sources:
        if kind == "src":
            label = short_label(Path(ref).parent.name)
            rows = load_rows(Path(ref), keep, args.limit)
        else:
            label = ref.split("/")[-1]
            rows = load_hf_rows(ref, args.hf_split, keep, args.limit)
        labels.append(label)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.out_dir) / f"{label}.verdicts.jsonl"

        done = set() if args.no_resume else load_done_ids(out_path)
        if args.no_resume and out_path.exists():
            out_path.unlink()
        pending = [r for r in rows if r["instance_id"] not in done]
        if done:
            print(f"Resuming {label}: {len(done)} already judged", file=sys.stderr)
        print(
            f"\nJudging {len(pending)} trajectories from {label} "
            f"(model={args.model}, workers={args.workers})",
            file=sys.stderr,
        )

        # Append as results land: a multi-thousand-row run over a rate-limited
        # gateway takes hours, and buffering to the end loses everything on a kill.
        write_lock = threading.Lock()
        with out_path.open("a") as fh:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(judge_one, client, args.model, r) for r in pending]
                for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                    with write_lock:
                        fh.write(json.dumps(fut.result()) + "\n")
                        fh.flush()
                    if i % 25 == 0:
                        print(f"  {i}/{len(pending)}", file=sys.stderr)

        verdicts = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln]
        print(f"Wrote {len(verdicts)} verdicts → {out_path}", file=sys.stderr)

        agg = aggregate(verdicts)
        aggs.append(agg)
        print_aggregate(label, agg)

    print_comparison(labels, aggs)


if __name__ == "__main__":
    main()
