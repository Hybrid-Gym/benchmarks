"""Rephrase every kept assistant turn of a trajectory dataset at four text lengths.

Each row is reduced to its skeleton (think / task_tracker turns and their results dropped, see
trajectory.py) and every remaining assistant turn becomes one work unit. One request asks for
the prose at ~300 / ~100 / ~50 / ~20 student-model tokens (llm.py); versions outside the ±25 %
band are re-requested with the measured count as feedback, for at most --max-rounds rounds.
A version still outside the band after that falls back to the turn's original text.

Output: <out-dir>/<dataset>.rephrase.jsonl, one line per unit:
  {"instance_id", "msg_idx", "tool", "orig_text", "orig_tokens", "rounds",
   "versions": {"t300": {"text", "tokens", "ok", "round"}, ...},  # ok=False: fell back to orig_text
   "fallback": [keys that fell back], "usage", "rounds_log", "elapsed", "model", "error"?}
Units already present without "error" are skipped on restart, so a killed run resumes.

Usage:
  python tools/verbosity_rephrase/rephrase.py \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --out-dir eval_outputs/verbosity_rephrase --model nvidia/deepseek-ai/deepseek-v4-flash --workers 6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import (
    DEFAULT_TOLERANCE,
    KEY_ORDER,
    Rephraser,
    band,
    first_round_messages,
    parse_versions,
    retry_messages,
)  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import build_skeleton  # noqa: E402


RETRY_TEMPERATURE = 0.7  # warmer than round 1 so the two retry candidates differ
_SENT_END = re.compile(r"(?<=[.!?:])\s+")


def dataset_units(rows, limit: int = 0, sample: int = 0, seed: int = 0) -> list[dict]:
    """Flatten rows into work units; `limit` caps the rows, `sample` picks random units (probing)."""
    units: list[dict] = []
    for n_rows, row in enumerate(rows, 1):
        sk = build_skeleton(row["messages"])
        for idx, t in sk.turns.items():
            units.append(
                {
                    "instance_id": row["instance_id"],
                    "msg_idx": idx,
                    "tool": t.tool,
                    "text": t.text,
                    "call": t.call,
                }
            )
        if n_rows == limit:
            break
    if sample and sample < len(units):
        units = random.Random(seed).sample(units, sample)
    return units


def trim_to_band(text: str, lo: int, hi: int) -> str | None:
    """Drop trailing sentences of an over-long attempt until it fits the band; None if impossible."""
    if count_tokens(text) <= hi:
        return None
    parts = _SENT_END.split(text.strip())
    while len(parts) > 1:
        parts.pop()
        cand = " ".join(parts).strip()
        n = count_tokens(cand)
        if lo <= n <= hi:
            return cand
        if n < lo:
            return None
    return None


def _source(text: str, versions: dict[str, dict]) -> str:
    """What a retry condenses or expands: the accepted t300, else the original prose, else the best t300 attempt."""
    v = versions.get("t300")
    if v and v["ok"]:
        return v["text"]
    return text or (v["text"] if v else "(no usable source - write from the TOOL CALL)")


def process_unit(rp: Rephraser, unit: dict, max_rounds: int, tolerance: float) -> dict:
    t0 = time.time()
    text, call = unit["text"], unit["call"]
    versions: dict[str, dict] = {}  # best attempt per key so far
    failing: dict[str, tuple[int, int]] = {
        k: (0, 0) for k in KEY_ORDER
    }  # key -> (tokens, words) of that attempt
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    rounds_log: list[dict] = []
    rounds = 0
    err = None
    for rnd in range(1, max_rounds + 1):
        if not failing:
            break
        rounds = rnd
        base_keys = [k for k in KEY_ORDER if k in failing]
        if rnd == 1:
            msgs = first_round_messages(text, call, tolerance)
            keys = base_keys
        else:
            msgs = retry_messages(
                text, call, _source(text, versions), failing, tolerance
            )
            keys = [f"{k}_{c}" for k in base_keys for c in "ab"]
        try:
            raw, usage = rp.complete(
                msgs, temperature=None if rnd == 1 else RETRY_TEMPERATURE
            )
        except Exception as e:  # noqa: BLE001
            err = f"api round {rnd}: {type(e).__name__}: {str(e)[:200]}"
            break
        usage_total["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage_total[k] += usage.get(k) or 0
        got = parse_versions(raw, keys)
        cand_tokens: dict[str, list[int]] = {}
        rounds_log.append(
            {
                "round": rnd,
                "asked": keys,
                "got": sorted(got),
                "finish": usage.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "cand_tokens": cand_tokens,
                "raw_head": None if got else (raw or "")[:300],
            }
        )
        for k in base_keys:
            cands = [got[c] for c in (k, f"{k}_a", f"{k}_b") if c in got]
            if not cands:
                failing[k] = (0, 0)
                continue
            lo, hi = band(k, tolerance)
            mid = (lo + hi) / 2
            for v in cands:
                n, trimmed = count_tokens(v), False
                if (
                    n > hi
                ):  # an overshoot can be cut at a sentence boundary; an undershoot cannot
                    cut = trim_to_band(v, lo, hi)
                    if cut is not None:
                        v, n, trimmed = cut, count_tokens(cut), True
                ok = lo <= n <= hi
                cand_tokens.setdefault(k, []).append(n)
                prev = versions.get(k)
                # keep the attempt closest to the band centre; an in-band one beats any miss
                if prev is None or (ok, -abs(n - mid)) > (
                    prev["ok"],
                    -abs(prev["tokens"] - mid),
                ):
                    versions[k] = {
                        "text": v,
                        "tokens": n,
                        "ok": ok,
                        "round": rnd,
                        "trimmed": trimmed,
                    }
            if versions[k]["ok"]:
                failing.pop(k)
            else:
                failing[k] = (versions[k]["tokens"], len(versions[k]["text"].split()))
    out = {
        "instance_id": unit["instance_id"],
        "msg_idx": unit["msg_idx"],
        "tool": unit["tool"],
        "orig_text": text,
        "orig_tokens": count_tokens(text),
        "rounds": rounds,
        "versions": {},
        "fallback": [],
        "usage": usage_total,
        "rounds_log": rounds_log,
        "elapsed": round(time.time() - t0, 2),
        "model": rp.model,
    }
    if err:
        out["error"] = err
        return out
    for k in KEY_ORDER:
        v = versions.get(k)
        if v and v["ok"]:
            out["versions"][k] = v
        else:  # keep the original text; the best miss stays for inspection
            out["versions"][k] = {
                "text": text,
                "tokens": out["orig_tokens"],
                "ok": False,
                "round": None,
                "best_miss": v,
            }
            out["fallback"].append(k)
    return out


def load_done(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            try:
                v = json.loads(line)
            except ValueError:
                continue  # torn last line of a killed run
            if not v.get("error"):
                done.add((v["instance_id"], v["msg_idx"]))
    return done


def load_api_key(explicit: str | None) -> str | None:
    """--api-key, else LLM_API_KEY, else the gateway key in config.toml (never printed)."""
    if explicit or os.environ.get("LLM_API_KEY"):
        return explicit or os.environ["LLM_API_KEY"]
    try:
        import tomllib

        with open(Path(__file__).resolve().parents[2] / "config.toml", "rb") as fh:
            return tomllib.load(fh)["llm"]["nvidia_claude_opus47"]["api_key"]
    except Exception:  # noqa: BLE001
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--hf",
        action="append",
        default=[],
        required=True,
        help="HF dataset repo; repeatable",
    )
    p.add_argument("--hf-split", default="train")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--model",
        default=os.environ.get(
            "REPHRASE_MODEL", "nvidia/deepseek-ai/deepseek-v4-flash"
        ),
    )
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com/v1"),
    )
    p.add_argument(
        "--extra-body",
        default="{}",
        help='JSON passed as extra_body, e.g. \'{"chat_template_kwargs":{"thinking":false}}\'',
    )
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=2500)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument("--limit", type=int, default=0, help="per-dataset row cap (debug)")
    p.add_argument(
        "--sample", type=int, default=0, help="random unit sample per dataset (probing)"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    key = load_api_key(args.api_key)
    if not key:
        sys.exit("error: no API key (--api-key / LLM_API_KEY / config.toml)")
    from datasets import load_dataset

    rp = Rephraser(
        key,
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=json.loads(args.extra_body),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo in args.hf:
        label = repo.split("/")[-1]
        out_path = out_dir / f"{label}.rephrase.jsonl"
        if args.no_resume and out_path.exists():
            out_path.unlink()
        units = dataset_units(
            load_dataset(repo, split=args.hf_split),
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
        done = load_done(out_path)
        pending = [u for u in units if (u["instance_id"], u["msg_idx"]) not in done]
        print(
            f"\n{label}: {len(units)} turns, {len(done)} done, {len(pending)} pending "
            f"(model={args.model}, workers={args.workers}, rounds<={args.max_rounds}, tol={args.tolerance})",
            file=sys.stderr,
        )
        n_ok = n_err = n_fb = 0
        t0 = time.time()
        with (
            out_path.open("a") as fh,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex,
        ):
            futs = [
                ex.submit(process_unit, rp, u, args.max_rounds, args.tolerance)
                for u in pending
            ]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                res = fut.result()
                n_err += bool(res.get("error"))
                n_ok += not res.get("error")
                n_fb += bool(res["fallback"])
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(pending):
                    rate = i / max(time.time() - t0, 1e-6) * 60
                    print(
                        f"  {label} {i}/{len(pending)} ok={n_ok} err={n_err} with_fallback={n_fb} "
                        f"{rate:.0f} turns/min eta={(len(pending) - i) / rate:.0f} min",
                        file=sys.stderr,
                    )
        print(
            f"Done {label}: ok={n_ok} err={n_err} with_fallback={n_fb} -> {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
