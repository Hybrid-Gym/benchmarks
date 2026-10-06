"""Write the missing prose of a dataset's empty assistant turns, one LLM call per trajectory.

For the prose-coverage ablation (build_coverage.py): every assistant turn other than think /
task_tracker whose prose before the tool call is empty gets a comment. The model sees the whole
trajectory at once (task, every step with its comment or call, every tool result, clipped) with
the empty steps marked, and returns the comments of all marked steps in one JSON object; steps it
leaves out or answers with unusable text are asked again (only those) up to --max-rounds times.

There is no length target: the comments are asked to match the agent's own comments in the same
trajectory. Because the model sees the steps after a marked one, the prompt forbids using
anything the agent could not know yet (the call's own result, later steps), and a comment that
still names an identifier the run first shows after its step (0.7 % of the first 8 216) is
written again from the run cut off right after that step's call, where it cannot leak.

Output: <out> jsonl, one line per trajectory (a later line for the same trajectory wins):
  {"instance_id", "targets": [msg_idx...], "fills": {msg_idx: text}, "missing": [msg_idx...],
   "refilled": [msg_idx...], "leak_checked": true, "rounds", "usage", "elapsed", "model", "error"?}
Rows already present without "error" are skipped on restart, except that rows without
"leak_checked" get the leak pass.

Usage:
  python tools/verbosity_rephrase/fill_prose.py --hf synthetic-code-training/func_localize_claude45_1457i \
      --out eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.fill.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import (  # noqa: E402
    AdaptiveLimiter,
    Messages,
    Rephraser,
    _clip,
    parse_versions,
)
from rephrase import load_api_key  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import build_skeleton  # noqa: E402


TASK_CHARS = 4000  # the task message, head + tail beyond this
RESULT_CHARS = 1500  # each tool result
CALL_CHARS = 1200  # each tool call
THINK_CHARS = 1500  # each think / task_tracker step
MAX_OUT_PER_STEP = 150  # max_tokens budget per requested comment
WRITE = "[WRITE COMMENT]"
# backticked text, dotted / slashed paths, snake_case and CamelCase names
IDENT = re.compile(
    r"`([^`]{3,80})`"
    r"|\b([A-Za-z_][A-Za-z0-9_]*(?:[._/][A-Za-z0-9_]+)+|[a-z]+_[a-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]+)\b"
)

SYSTEM_PROMPT = """You add the natural-language comments that a software-engineering agent writes before its tool calls to a recorded run of that agent.

The run below shows the task, then every agent step (its COMMENT and its tool CALL) and every tool RESULT, in order; long parts are clipped. Some steps have the comment `[WRITE COMMENT]`: the agent wrote nothing before that call. Write the comment for every such step.

Rules:
- Write what the agent would say just before that call, in its own voice: first person, present tense ("Let me ...", "Now I'll ...", "I see the issue ..."). Match the tone and length of the agent's own comments in this run; most are one or two sentences.
- Use only what the agent knows at that moment: the task and the steps and results BEFORE the marked step. Never mention or hint at the result of the marked call or anything that happens after it.
- Say what the call is for and why it is the natural next step given the results so far. Do not invent facts, file contents or findings.
- Each comment stands on its own: do not refer to other comments you write.
- Plain prose only: no code blocks, no XML tags, no tool-call markup, do not quote the call.

Respond with ONLY a JSON object mapping each marked step number to its comment, for example {"3": "...", "7": "..."}."""


def render(messages: list[dict], targets: set[int]) -> tuple[str, dict[str, int]]:
    """(the run as text with numbered steps, step number -> msg_idx of the marked steps)."""
    sk = build_skeleton(messages, think="keep", plan="keep")
    lines: list[str] = []
    steps: dict[str, int] = {}
    n = 0
    for idx, m in enumerate(messages):
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            if not lines:
                lines.append(
                    f"=== TASK ===\n{_clip(m['content'].strip(), TASK_CHARS, 1000)}"
                )
            else:
                lines.append(
                    f"=== RESULT ===\n{_clip(m['content'].strip(), RESULT_CHARS, 500)}"
                )
            continue
        n += 1
        t = sk.turns.get(idx)
        if t is None:  # think / task_tracker step, kept as it is
            lines.append(
                f"=== STEP {n} ===\n{_clip(m['content'].strip(), THINK_CHARS, 500)}"
            )
            continue
        comment = WRITE if idx in targets else (t.text or "(none)")
        if idx in targets:
            steps[str(n)] = idx
        lines.append(
            f"=== STEP {n} ===\nCOMMENT: {comment}\nCALL:\n{_clip(t.call, CALL_CHARS, 300)}"
        )
    return "\n\n".join(lines), steps


def fill_messages(run: str, ask: list[str]) -> Messages:
    tail = (
        f"Write the comments for steps {', '.join(ask)}."
        if len(ask) < 40
        else f"Write the comments for all {len(ask)} marked steps."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{run}\n\n{tail}"},
    ]


def add_usage(rec: dict, u: dict) -> None:
    rec["usage"]["calls"] += 1
    for k in ("prompt_tokens", "completion_tokens"):
        rec["usage"][k] += u.get(k) or 0


def leaks(messages: list[dict], idx: int, text: str) -> list[str]:
    """Identifiers `text` names that the run shows only after step `idx` (not in the task, earlier steps or the call)."""
    before = "\n".join(m["content"] for m in messages[: idx + 1])
    after = "\n".join(m["content"] for m in messages[idx + 1 :])
    names = {a or b for a, b in IDENT.findall(text)} - {""}
    return sorted(w for w in names if w not in before and w in after)


def fix_leaks(rp: Rephraser, messages: list[dict], rec: dict) -> dict:
    """Rewrite each leaking comment from the run cut off right after its step's call; drop it if that fails.

    Without the later steps in view the rewrite cannot leak, so it is taken as it comes.
    """
    rec = {
        **rec,
        "fills": dict(rec["fills"]),
        "missing": list(rec["missing"]),
        "usage": dict(rec["usage"]),
    }
    refilled: list[int] = []
    for k, text in list(rec["fills"].items()):
        idx = int(k)
        if not leaks(messages, idx, text):
            continue
        run, steps = render(messages[: idx + 1], {idx})
        ask = list(steps)
        for rnd in range(2):
            raw, u = rp.complete(
                fill_messages(run, ask),
                temperature=None if rnd == 0 else 0.7,
                max_tokens=max(rp.max_tokens, 400 + MAX_OUT_PER_STEP),
            )
            add_usage(rec, u)
            got = parse_versions(raw, ask)
            if got:
                rec["fills"][k] = got[ask[0]]
                refilled.append(idx)
                break
        else:
            del rec["fills"][k]
            rec["missing"].append(idx)
    rec["refilled"] = refilled
    rec["leak_checked"] = True
    return rec


def fill_row(rp: Rephraser, row: dict, max_rounds: int) -> dict:
    t0 = time.time()
    msgs = row["messages"]
    sk = build_skeleton(msgs, think="keep", plan="keep")
    targets = sorted(i for i, t in sk.turns.items() if not t.text and t.kind != "text")
    out: dict = {
        "instance_id": row["instance_id"],
        "targets": targets,
        "fills": {},
        "missing": [],
        "refilled": [],
        "leak_checked": True,
        "rounds": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "model": rp.model,
    }
    if not targets:
        out["elapsed"] = 0.0
        return out
    run, steps = render(msgs, set(targets))
    ask = list(steps)
    for rnd in range(1, max_rounds + 1):
        if not ask:
            break
        out["rounds"] = rnd
        try:
            raw, u = rp.complete(
                fill_messages(run, ask),
                temperature=None if rnd == 1 else 0.7,
                max_tokens=max(rp.max_tokens, 400 + MAX_OUT_PER_STEP * len(ask)),
            )
        except Exception as e:  # noqa: BLE001
            out["error"] = f"api round {rnd}: {type(e).__name__}: {str(e)[:200]}"
            break
        add_usage(out, u)
        got = parse_versions(raw, ask)
        if u.get("finish_reason") == "length" and got:
            got.popitem()  # the last value was cut off
        for step, text in got.items():
            out["fills"][str(steps[step])] = text
        ask = [s for s in ask if s not in got]
    out["missing"] = [steps[s] for s in ask]
    if "error" not in out:
        out = fix_leaks(rp, msgs, out)
    out["elapsed"] = round(time.time() - t0, 2)
    return out


def load_fills(path: Path) -> dict[str, dict]:
    """Error-free records by instance_id; a later line for the same trajectory wins."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.open():
            try:
                v = json.loads(line)
            except ValueError:
                continue  # torn last line of a killed run
            if not v.get("error"):
                out[v["instance_id"]] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nvidia/deepseek-ai/deepseek-v4-flash")
    p.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    p.add_argument(
        "--extra-body", default='{"chat_template_kwargs":{"thinking":false}}'
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--workers-min", type=int, default=3)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument(
        "--limit", type=int, default=0, help="first N rows only (smoke test)"
    )
    args = p.parse_args()
    key = load_api_key(None)
    if not key:
        sys.exit("error: no API key")
    rows = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(rows, Dataset)
    if args.limit:
        rows = rows.select(range(args.limit))
    out_path = Path(args.out)
    done = load_fills(out_path)
    all_rows = list(cast(Iterable[dict[str, Any]], rows))
    pending = [r for r in all_rows if r["instance_id"] not in done]
    unchecked = [
        r
        for r in all_rows
        if r["instance_id"] in done and not done[r["instance_id"]].get("leak_checked")
    ]
    print(
        f"{len(all_rows)} rows, {len(done)} done ({len(unchecked)} of them still need the leak pass), "
        f"{len(pending)} pending",
        file=sys.stderr,
    )
    rp = Rephraser(
        key,
        args.base_url,
        args.model,
        extra_body=json.loads(args.extra_body),
        limiter=AdaptiveLimiter(
            start=args.workers, lo=args.workers_min, hi=args.workers
        ),
    )
    t0 = time.time()
    n_err = n_missing = n_filled = n_refilled = 0
    total = len(pending) + len(unchecked)
    with (
        out_path.open("a") as fh,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex,
    ):
        futs = [ex.submit(fill_row, rp, r, args.max_rounds) for r in pending]
        futs += [
            ex.submit(fix_leaks, rp, r["messages"], done[r["instance_id"]])
            for r in unchecked
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001 - a failed leak pass: the row keeps no checked record, the next pass retries it
                n_err += 1
                print(f"  error: {str(e)[:200]}", file=sys.stderr)
                continue
            n_err += bool(r.get("error"))
            n_missing += len(r["missing"])
            n_filled += len(r["fills"])
            n_refilled += len(r.get("refilled", []))
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == total:
                rate = i / max(time.time() - t0, 1e-6) * 60
                print(
                    f"  {i}/{total} rows, filled {n_filled}, rewritten for leaking {n_refilled}, "
                    f"missing {n_missing}, errors {n_err}, {rate:.1f} rows/min, eta {(total - i) / rate:.0f} min",
                    file=sys.stderr,
                )
    print(
        f"Done -> {out_path} (errors {n_err}: rerun to retry them); {token_stats(out_path)}",
        file=sys.stderr,
    )


def token_stats(path: Path) -> str:
    """Mean / median tokens of the written comments, and how many were rewritten for leaking (for the log)."""
    recs = load_fills(path).values()
    n_refilled = sum(len(r.get("refilled", [])) for r in recs)
    toks = sorted(count_tokens(t) for r in recs for t in r["fills"].values())
    return (
        f"{len(toks)} comments, mean {sum(toks) / max(len(toks), 1):.1f} median "
        f"{toks[len(toks) // 2] if toks else 0} tokens, {n_refilled} rewritten for leaking"
    )


if __name__ == "__main__":
    main()
