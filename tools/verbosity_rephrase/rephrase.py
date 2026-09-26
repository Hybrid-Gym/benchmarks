"""Rephrase every non-think assistant turn of a trajectory dataset at several text lengths.

Each row is reduced to its skeleton (trajectory.py) and every assistant turn other than think /
task_tracker becomes one work unit. One request asks for all versions of a family (llm.py):
  --family fixed    ~300 / ~100 / ~50 / ~20 student-model tokens
  --family scaled   0.5x / 2x / 4x / 8x of the turn's own prose length (multiples whose target
                    is under 4 or over 2000 tokens, and every multiple of an empty turn, are not
                    requested), plus 32x written afterwards in parts (cap 8000 tokens)
Versions outside the ±25 % band are re-requested with the measured count as feedback, for at
most --max-rounds rounds. A version still outside the band after that falls back to the
turn's original text. A version written in parts (32x) instead grows until it reaches the
band: each request sees the text so far and writes the next part; an overshoot is trimmed at
a sentence boundary.

Output: <out-dir>/<dataset>.<family>[.<keys>].jsonl, one line per unit:
  {"instance_id", "msg_idx", "tool", "orig_text", "orig_tokens", "family", "targets", "rounds",
   "versions": {key: {"text", "tokens", "ok", "round"}, ...},  # ok=False: fell back to orig_text
   "fallback": [keys that fell back], "skipped": [keys not requested],
   "usage", "rounds_log", "elapsed", "model", "error"?}
Units already present without "error" are skipped on restart, so a killed run resumes.

Usage:
  python tools/verbosity_rephrase/rephrase.py --family scaled \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --out-dir eval_outputs/verbosity_rephrase --model nvidia/deepseek-ai/deepseek-v4-flash --workers 6
  # add the 32x rung to a finished scaled run (its ladder rungs are the sources; records are merged):
  python tools/verbosity_rephrase/rephrase.py --family scaled --keys x32 \
      --prior eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.jsonl ...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import (
    DEFAULT_TOLERANCE,
    FORBIDDEN,
    PART_MAX,
    PARTS_STOP,
    SPECS,
    AdaptiveLimiter,
    Rephraser,
    ScaledLengths,
    Spec,
    band,
    extract_partial,
    first_round_messages,
    parse_versions,
    parts_messages,
    retry_messages,
)  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import build_skeleton  # noqa: E402


RETRY_TEMPERATURE = 0.7  # warmer than round 1 so the two retry candidates differ
OUTPUT_HEADROOM = 2.0  # max_tokens per request = this x the tokens asked for + overhead
PARTS_MAX_BAD = 3  # consecutive unusable part replies before a parts version gives up
PARTS_GAIN = (
    0.7,
    2.5,
)  # the next part's ask is scaled by how far the last one missed its chunk, within these bounds
PARTS_MIN_PLAIN = 20  # tokens; a reply without the JSON wrapper counts as the part when it is at least this long
_SENT_END = re.compile(r"(?<=[.!?:])\s+")
_SENT_END_STRICT = re.compile(r"(?<=[.!?])\s+")  # a cut part must not end on a colon
_PART_LABEL = re.compile(
    r"^\s*(?:\*\*)?part\s*\d+\s*(?:of\s*\d+)?\s*(?:\*\*)?\s*[:.\-\u2013\u2014]\s*",
    re.IGNORECASE,
)
_PLAIN_LEAD = re.compile(
    r"^(?:here(?:'s| is)[^\n]{0,80}:|```[a-z]*)\s*\n", re.IGNORECASE
)


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
    text = text.strip()
    if count_tokens(text) <= hi:
        return None
    for m in reversed(
        list(_SENT_END.finditer(text))
    ):  # cut at sentence ends, latest first
        cand = text[: m.start()].rstrip()
        n = count_tokens(cand)
        if lo <= n <= hi:
            return cand
        if n < lo:
            return None
    return None


def cut_at_sentence(text: str) -> str | None:
    """A truncated reply keeps its complete sentences; None if it has none."""
    ends = list(_SENT_END_STRICT.finditer(text))
    if ends:
        return text[: ends[-1].start()].rstrip()
    return text.rstrip() if re.search(r"[.!?]$", text.rstrip()) else None


def part_of_reply(raw: str, finish: str | None, sofar: str) -> tuple[str | None, str]:
    """(the usable part of a reply, why it is unusable): parsed, cut at a sentence if truncated, cleaned."""
    got = parse_versions(raw, ["part"]).get("part")
    if got is None:  # cut off, or stopped without closing the object
        got = extract_partial(raw, "part")
    if got is None and not re.search(r'"part"\s*:\s*"', raw or ""):
        got = plain_part(raw)
    if got is None:
        if any(f in (raw or "") for f in FORBIDDEN):
            return (
                None,
                "it quoted tool-call markup such as <function= or <parameter=; write prose only, never those tags",
            )
        if not re.search(r'"part"\s*:\s*"', raw or ""):
            return None, 'it was not a JSON object with one "part" string'
        return None, "it was empty"
    if finish == "length":  # cut off: keep its complete sentences
        got = cut_at_sentence(got)
        if got is None:
            return None, "it had no complete sentence before it was cut off"
    got = clean_part(got, sofar)
    if got is None:
        return None, "it restarted the text instead of continuing it"
    return got, ""


def clean_part(part: str, sofar: str) -> str | None:
    """Strip a 'Part N:' label and any re-emitted tail of the text so far; None for an empty part or a restart.

    The model often repeats the last sentence or two before continuing; that overlap is dropped.
    A part that begins with text from earlier in `sofar` and never reaches its end is a restart.
    """
    part = _PART_LABEL.sub("", part).strip()
    if not part:
        return None
    if sofar:
        for n in (100, 60, 40):
            tail = sofar[-n:]
            q = part.find(tail)
            if 0 <= q < 400:
                part = part[q + len(tail) :].strip()
                break
        else:
            if len(part) > 80 and part[:80] in sofar:
                return None
    return part or None


def plain_part(raw: str) -> str | None:
    """A reply that skipped the JSON wrapper and is just the prose (common on long continuations)."""
    s = re.sub(r"\n?```$", "", (raw or "").strip())
    s = _PLAIN_LEAD.sub("", s).strip()
    if not s or s.startswith("{") or any(f in s for f in FORBIDDEN):
        return None
    return s if count_tokens(s) >= PARTS_MIN_PLAIN else None


def write_in_parts(
    rp: Rephraser,
    spec: ScaledLengths,
    key: str,
    target: int,
    orig_tokens: int,
    text: str,
    call: str,
    source_key: str | None,
    source: str,
    tolerance: float,
) -> tuple[dict, dict, list[dict], str | None]:
    """Grow one version part by part until it reaches its band (or the calls run out).

    Each request asks for the next `chunk` tokens, an even share of what is left in parts of at most
    PART_MAX; a truncated reply keeps its complete sentences; an unusable reply is re-asked warmer.
    Returns (version, usage, log, api error).
    """
    lo, hi = band(target, tolerance)
    parts: list[str] = []
    total = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    log: list[dict] = []
    max_calls = math.ceil(target / PART_MAX) + 3
    bad = 0
    gain = 1.0  # ask / chunk, adapted to how the model has been missing its chunks for this turn
    note = ""
    err = None
    while total < round(target * PARTS_STOP) and usage["calls"] < max_calls:
        remaining = target - total
        n_left = max(1, math.ceil(remaining / PART_MAX))
        chunk = max(1, round(remaining / n_left))
        ask = max(1, round(chunk * gain))
        final = n_left == 1
        sofar = "\n\n".join(parts)
        msgs = parts_messages(
            spec,
            key,
            target,
            orig_tokens,
            text,
            call,
            source_key,
            source,
            sofar,
            total,
            ask,
            final,
            note,
        )
        try:
            raw, u = rp.complete(
                msgs,
                temperature=RETRY_TEMPERATURE if bad else None,
                max_tokens=max(rp.max_tokens, round(OUTPUT_HEADROOM * ask) + 300),
            )
        except Exception as e:  # noqa: BLE001
            err = f"api {key} part {len(parts) + 1}: {type(e).__name__}: {str(e)[:200]}"
            break
        usage["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += u.get(k) or 0
        got, why = part_of_reply(raw, u.get("finish_reason"), sofar)
        n = count_tokens(got) if got else 0
        log.append(
            {
                "key": key,
                "part": len(parts) + 1,
                "chunk": chunk,
                "ask": ask,
                "final": final,
                "tokens": n,
                "finish": u.get("finish_reason"),
                "completion_tokens": u.get("completion_tokens"),
                "why": why or None,
                "raw_head": None if got else (raw or "")[:400],
            }
        )
        if not got:
            bad += 1
            note = f"Your previous reply was unusable: {why}."
            if bad >= PARTS_MAX_BAD:
                break
            continue
        bad = 0
        parts.append(got)
        total = count_tokens("\n\n".join(parts))
        gain = min(PARTS_GAIN[1], max(PARTS_GAIN[0], gain * chunk / max(n, 1)))
        note = (
            f"Your previous part came out at {n} tokens, short of the {chunk} asked; this one must reach its length."
            if n < 0.6 * chunk
            else ""
        )
    joined = "\n\n".join(parts)
    trimmed = False
    if total > hi:
        cut = trim_to_band(joined, lo, hi)
        if cut is not None:
            joined, total, trimmed = cut, count_tokens(cut), True
    ok = err is None and lo <= total <= hi
    version = {
        "text": joined,
        "tokens": total,
        "ok": ok,
        "round": 1 if ok else None,
        "trimmed": trimmed,
        "parts": len(parts),
        "calls": usage["calls"],
        "source": source_key,
    }
    return version, usage, log, err


def process_unit(
    rp: Rephraser,
    spec: Spec,
    unit: dict,
    max_rounds: int,
    tolerance: float,
    keys: tuple[str, ...] | None = None,
) -> dict:
    """Write the unit's versions: the ladder keys in one reply (with retries), then the parts keys.

    `keys` restricts the work to a subset of the family's keys; `unit["prior"]` (a record of an
    earlier run) supplies the ladder rungs the parts keys grow from and is merged into the output.
    """
    t0 = time.time()
    text, call = unit["text"], unit["call"]
    orig_tokens = count_tokens(text)
    targets = spec.targets(orig_tokens)
    if keys is not None:
        targets = {k: t for k, t in targets.items() if k in keys}
    ladder = {k: t for k, t in targets.items() if k in spec.LADDER}
    prior: dict | None = unit.get("prior")
    versions: dict[str, dict] = {
        k: v for k, v in (prior or {}).get("versions", {}).items() if v["ok"]
    }  # accepted rungs of the prior run, then the best attempt per key so far
    failing: dict[str, tuple[int, int]] = {
        k: (0, 0) for k in ladder
    }  # key -> (tokens, words) of that attempt
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    rounds_log: list[dict] = []
    rounds = 0
    err = None
    for rnd in range(1, max_rounds + 1):
        if not failing:
            break
        rounds = rnd
        base_keys = [k for k in ladder if k in failing]
        if rnd == 1:
            msgs = first_round_messages(spec, ladder, text, call, tolerance)
            keys_asked = base_keys
        else:
            msgs = retry_messages(
                spec,
                ladder,
                text,
                call,
                spec.source(text, versions),
                failing,
                tolerance,
            )
            keys_asked = [f"{k}_{c}" for k in base_keys for c in "ab"]
        asked = sum(ladder[k] for k in base_keys) * (1 if rnd == 1 else 2)
        try:
            raw, usage = rp.complete(
                msgs,
                temperature=None if rnd == 1 else RETRY_TEMPERATURE,
                max_tokens=max(rp.max_tokens, round(OUTPUT_HEADROOM * asked) + 300),
            )
        except Exception as e:  # noqa: BLE001
            err = f"api round {rnd}: {type(e).__name__}: {str(e)[:200]}"
            break
        usage_total["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            usage_total[k] += usage.get(k) or 0
        got = parse_versions(raw, keys_asked)
        if usage.get("finish_reason") == "length" and got:
            got.popitem()  # the last value was cut off
        cand_tokens: dict[str, list[int]] = {}
        rounds_log.append(
            {
                "round": rnd,
                "asked": keys_asked,
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
            lo, hi = band(ladder[k], tolerance)
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
    for k in spec.PARTS:
        if k not in targets or err or not isinstance(spec, ScaledLengths):
            continue
        source_key, source = spec.parts_source(text, versions)
        versions[k], usage, log, err = write_in_parts(
            rp,
            spec,
            k,
            targets[k],
            orig_tokens,
            text,
            call,
            source_key,
            source,
            tolerance,
        )
        usage_total["calls"] += usage["calls"]
        for f in ("prompt_tokens", "completion_tokens"):
            usage_total[f] += usage[f]
        rounds_log += log
    out = {
        "instance_id": unit["instance_id"],
        "msg_idx": unit["msg_idx"],
        "tool": unit["tool"],
        "orig_text": text,
        "orig_tokens": orig_tokens,
        "family": spec.name,
        "targets": targets,
        "rounds": rounds,
        "versions": {},
        "fallback": [],
        "skipped": [k for k in spec.keys if k not in targets],
        "usage": usage_total,
        "rounds_log": rounds_log,
        "elapsed": round(time.time() - t0, 2),
        "model": rp.model,
    }
    if err:
        out["error"] = err
        return out
    for k in targets:
        v = versions.get(k)
        if v and v["ok"]:
            out["versions"][k] = v
        else:  # keep the original text; the best miss stays for inspection
            out["versions"][k] = {
                "text": text,
                "tokens": orig_tokens,
                "ok": False,
                "round": None,
                "best_miss": v,
            }
            out["fallback"].append(k)
    if prior:  # one record per unit: the prior run's rungs plus this run's
        out["targets"] = {**prior["targets"], **out["targets"]}
        out["versions"] = {**prior["versions"], **out["versions"]}
        out["fallback"] = prior["fallback"] + out["fallback"]
        out["skipped"] = [k for k in spec.keys if k not in out["targets"]]
        out["rounds"] = prior["rounds"]
        out["rounds_log"] = prior["rounds_log"] + out["rounds_log"]
        out["elapsed"] = round(prior["elapsed"] + out["elapsed"], 2)
        for f in ("prompt_tokens", "completion_tokens", "calls"):
            out["usage"][f] += prior["usage"][f]
    return out


def load_records(path: Path) -> dict[tuple[str, int], dict]:
    """Error-free records by unit; a later line for the same unit wins."""
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            try:
                v = json.loads(line)
            except ValueError:
                continue  # torn last line of a killed run
            if not v.get("error"):
                out[(v["instance_id"], v["msg_idx"])] = v
    return out


def load_done(path: Path) -> set[tuple[str, int]]:
    return set(load_records(path))


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
    p.add_argument("--family", choices=list(SPECS), default="fixed")
    p.add_argument(
        "--keys",
        default=None,
        help="comma-separated subset of the family's keys to write (default: all); "
        "the output file gets the keys as a suffix",
    )
    p.add_argument(
        "--prior",
        default=None,
        help="rephrase jsonl of an earlier run of this family: its accepted rungs are the sources "
        "for keys written in parts, and its records are merged into the output",
    )
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
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2500,
        help="floor for the per-request max_tokens (raised for long targets)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="thread pool size = most requests in flight",
    )
    p.add_argument(
        "--workers-min",
        type=int,
        default=0,
        help="floor of the adaptive in-flight cap (0: fixed concurrency = --workers); "
        "the cap starts at --workers-start, drops to 3/4 on a 429, grows after clean calls",
    )
    p.add_argument("--workers-start", type=int, default=0, help="default: --workers")
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

    spec = SPECS[args.family]
    keys = tuple(k for k in args.keys.split(",") if k) if args.keys else None
    if keys and any(k not in spec.keys for k in keys):
        sys.exit(f"error: --keys must be among {spec.keys}")
    prior = load_records(Path(args.prior)) if args.prior else None
    limiter = (
        AdaptiveLimiter(
            start=args.workers_start or args.workers,
            lo=args.workers_min,
            hi=args.workers,
        )
        if args.workers_min
        else None
    )
    rp = Rephraser(
        key,
        args.base_url,
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=json.loads(args.extra_body),
        limiter=limiter,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for repo in args.hf:
        label = repo.split("/")[-1]
        suffix = f".{'+'.join(keys)}" if keys else ""
        out_path = out_dir / f"{label}.{spec.name}{suffix}.jsonl"
        if args.no_resume and out_path.exists():
            out_path.unlink()
        units = dataset_units(
            load_dataset(repo, split=args.hf_split),
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
        if prior is not None:
            missing = 0
            for u in units:
                u["prior"] = prior.get((u["instance_id"], u["msg_idx"]))
                missing += u["prior"] is None
            if missing:
                sys.exit(
                    f"error: {missing} turns have no record in --prior {args.prior}"
                )
        done = load_done(out_path)
        pending = [u for u in units if (u["instance_id"], u["msg_idx"]) not in done]
        print(
            f"\n{label}: {len(units)} turns, {len(done)} done, {len(pending)} pending "
            f"(family={spec.name}, keys={','.join(keys) if keys else 'all'}, model={args.model}, "
            f"workers={args.workers}"
            + (
                f" adaptive[{args.workers_min}..{args.workers}, start {limiter.cap}]"
                if limiter
                else ""
            )
            + f", rounds<={args.max_rounds}, tol={args.tolerance})",
            file=sys.stderr,
        )
        n_ok = n_err = n_fb = n_skip = 0
        t0 = time.time()
        with (
            out_path.open("a") as fh,
            concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex,
        ):
            futs = [
                ex.submit(
                    process_unit, rp, spec, u, args.max_rounds, args.tolerance, keys
                )
                for u in pending
            ]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                res = fut.result()
                n_err += bool(res.get("error"))
                n_ok += not res.get("error")
                n_fb += not res.get("error") and any(
                    keys is None or k in keys for k in res["fallback"]
                )  # this run's keys only; a merged prior record carries its own
                n_skip += not res["targets"]
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(pending):
                    rate = i / max(time.time() - t0, 1e-6) * 60
                    print(
                        f"  {label} {i}/{len(pending)} ok={n_ok} err={n_err} with_fallback={n_fb} "
                        f"all_skipped={n_skip} {rate:.0f} turns/min eta={(len(pending) - i) / rate:.0f} min",
                        file=sys.stderr,
                    )
        print(
            f"Done {label}: ok={n_ok} err={n_err} with_fallback={n_fb} all_skipped={n_skip} -> {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
