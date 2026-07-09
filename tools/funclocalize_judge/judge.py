"""LLM-as-judge for funclocalize trajectories' compliance with the 3 localization strategies.

Strategies (see prompts/default_loc_strategy.j2):
  1. broad_then_narrow         — broad searches first, then narrow
  2. multi_round_refinement    — iterative search, multiple rounds
  3. read_after_narrowing      — read full files only after narrowing to a few candidates

Usage:
  python tools/funclocalize_judge/judge.py \\
      --src eval_outputs/.../baseline/output.jsonl \\
      --src eval_outputs/.../experiment/output.jsonl \\
      --out-dir /tmp/verdicts \\
      --filter-ids /tmp/funclocalize_1500_sample300_seed42_ids.txt

A single --src needs --out PATH instead of --out-dir.
The judge call goes to the same OpenAI-compatible gateway as the rollouts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


MAX_TRAJECTORY_CHARS = 10_000
MAX_TASK_CHARS = 500
JUDGE_MAX_OUTPUT_TOKENS = 4000  # reasoning models eat most of this internally
EDIT_COMMANDS = {"create", "str_replace", "insert"}
STRATEGY_KEYS = ("broad_then_narrow", "multi_round_refinement", "read_after_narrowing")


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
   Compliant if the agent runs >=2 distinct search commands where later ones clearly refine
   earlier ones (different keywords / restricted path / different pattern) based on what it
   learned from prior results.
   Non-compliant if the agent commits to a single candidate after just one search.

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
    body = "\n".join(lines)
    if len(body) > MAX_TRAJECTORY_CHARS:
        half = MAX_TRAJECTORY_CHARS // 2
        body = body[:half] + "\n[... trajectory truncated ...]\n" + body[-half:]
    return body


# ---- LLM judge -----------------------------------------------------------


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw).strip()


def judge_one(client: OpenAI, model: str, row: dict) -> dict:
    iid = row["instance_id"]
    events = extract_localization_phase(row.get("history") or [])
    summary = trajectory_summary(events)
    if not summary.strip():
        return {"instance_id": iid, "error": "no localization actions"}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        task_description=extract_task_description(
                            row.get("history") or []
                        ),
                        trajectory_summary=summary,
                    ),
                },
            ],
            max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        return {"instance_id": iid, "error": f"api: {e}"}

    raw = _strip_fences((resp.choices[0].message.content or "").strip())
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
        required=True,
        help="Rollout output.jsonl; repeat for head-to-head comparison",
    )
    p.add_argument("--out", help="Output verdicts JSONL (use with a single --src)")
    p.add_argument(
        "--out-dir",
        help="Directory for verdicts JSONLs (use with multiple --src); filenames derived from each src's parent dir",
    )
    p.add_argument(
        "--filter-ids",
        help="Optional file of instance IDs (one per line) to restrict judging",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("JUDGE_MODEL", "openai/openai/gpt-5-mini"),
    )
    p.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY"))
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com/v1"),
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=0, help="Per-src trajectory cap (debug)"
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.api_key:
        sys.exit("error: --api-key required (or set NVIDIA_API_KEY)")
    if len(args.src) == 1 and not args.out:
        sys.exit("error: --out required when a single --src is given")
    if len(args.src) > 1 and not args.out_dir:
        sys.exit("error: --out-dir required when multiple --src are given")

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

    for src_str in args.src:
        src = Path(src_str)
        label = short_label(src.parent.name)
        labels.append(label)

        rows = load_rows(src, keep, args.limit)
        print(
            f"\nJudging {len(rows)} trajectories from {label} (model={args.model}, workers={args.workers})",
            file=sys.stderr,
        )

        verdicts: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(judge_one, client, args.model, r) for r in rows]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                verdicts.append(fut.result())
                if i % 25 == 0:
                    print(f"  {i}/{len(rows)}", file=sys.stderr)

        if args.out:
            out_path = Path(args.out)
        else:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.out_dir) / f"{label}.verdicts.jsonl"
        out_path.write_text("".join(json.dumps(v) + "\n" for v in verdicts))
        print(f"Wrote {len(verdicts)} verdicts → {out_path}", file=sys.stderr)

        agg = aggregate(verdicts)
        aggs.append(agg)
        print_aggregate(label, agg)

    print_comparison(labels, aggs)


if __name__ == "__main__":
    main()
