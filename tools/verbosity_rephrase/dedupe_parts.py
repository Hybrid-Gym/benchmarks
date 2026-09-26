"""Remove verbatim-repeated sentences from the versions of one rung of a finished rephrase run.

Versions written in parts before the duplicate filter (rephrase.drop_repeats) repeat earlier
sentences: family B's x32 rung has repeats in 10.5 % (claude45) / 15.0 % (r2egym) of its versions.
Each accepted version loses its repeated sentences (40+ characters, outside code blocks); one that
falls under its band is topped up with further parts (continuing the deduplicated text, with the
filter) until it is back in band, or keeps the deduplicated text flagged `below_band` if the calls
run out. The version's record gains `dedup: {tokens_before, removed_tokens, topped_up_tokens?,
below_band?}`; untouched records are copied as they are.

Output is a new jsonl (the input is left alone), written as records complete, resumable.

Usage:
  python tools/verbosity_rephrase/dedupe_parts.py --family scaled --key x32 \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.x32.jsonl \
      --out eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.x32.dedup.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import (
    DEFAULT_TOLERANCE,
    SPECS,
    AdaptiveLimiter,
    Rephraser,
    ScaledLengths,
    band,
)  # noqa: E402
from rephrase import (  # noqa: E402
    FAMILIES,
    dataset_units,
    drop_repeats,
    load_api_key,
    load_records,
    unit_key,
    write_in_parts,
)
from tokens import count_tokens  # noqa: E402


def top_up(
    rp: Rephraser, r: dict, key: str, kept: str, call: str, tolerance: float
) -> dict:
    """The record with its `key` version grown from `kept` back into its band (or flagged below_band)."""
    spec = SPECS[r["family"]]
    assert isinstance(spec, ScaledLengths)
    v = r["versions"][key]
    n = count_tokens(kept)
    info = {"tokens_before": v["tokens"], "removed_tokens": v["tokens"] - n}
    grown, usage, log, err = write_in_parts(
        rp,
        spec,
        key,
        r["targets"][key],
        r["orig_tokens"],
        r["orig_text"],
        call,
        None,
        "",
        tolerance,
        start=kept,
    )
    if err:
        raise RuntimeError(err)
    if grown["ok"]:
        text, tokens = grown["text"], grown["tokens"]
        info["topped_up_tokens"] = tokens - n
    else:
        text, tokens = kept, n
        info["below_band"] = True
    out = {
        **r,
        "versions": {
            **r["versions"],
            key: {**v, "text": text, "tokens": tokens, "dedup": info},
        },
    }
    out["usage"] = {
        f: r["usage"][f] + usage[f]
        for f in ("prompt_tokens", "completion_tokens", "calls")
    }
    out["rounds_log"] = r["rounds_log"] + log
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--hf",
        required=True,
        help="base dataset repo (for the tool calls shown to the model)",
    )
    p.add_argument("--family", choices=FAMILIES, default="scaled")
    p.add_argument("--key", default="x32")
    p.add_argument("--rephrase", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nvidia/deepseek-ai/deepseek-v4-flash")
    p.add_argument("--base-url", default="https://inference-api.nvidia.com/v1")
    p.add_argument(
        "--extra-body", default='{"chat_template_kwargs":{"thinking":false}}'
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--workers-min", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = p.parse_args()
    from datasets import load_dataset

    records = load_records(Path(args.rephrase))
    out_path = Path(args.out)
    done = load_records(out_path)
    calls = {
        unit_key(u): u["call"]
        for u in dataset_units(load_dataset(args.hf, split="train"), args.family)
    }
    jobs: list[tuple[dict, str]] = []
    n_same = n_deduped = 0
    with out_path.open("a") as fh:
        for k, r in records.items():
            if k in done:
                continue
            v = r["versions"].get(args.key)
            kept = drop_repeats(v["text"], "") if v and v["ok"] else None
            if kept is None or kept == v["text"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_same += 1
                continue
            n = count_tokens(kept)
            if n >= band(r["targets"][args.key], args.tolerance)[0]:
                info = {"tokens_before": v["tokens"], "removed_tokens": v["tokens"] - n}
                r = {
                    **r,
                    "versions": {
                        **r["versions"],
                        args.key: {**v, "text": kept, "tokens": n, "dedup": info},
                    },
                }
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_deduped += 1
            else:
                jobs.append((r, kept))
        fh.flush()
        print(
            f"{len(records)} records, {len(done)} already done: {n_same} unchanged, {n_deduped} deduplicated "
            f"in band, {len(jobs)} to top up",
            file=sys.stderr,
        )
        key = load_api_key(None)
        if not key:
            sys.exit("error: no API key")
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
        n_ok = n_below = n_err = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    top_up, rp, r, args.key, kept, calls[unit_key(r)], args.tolerance
                )
                for r, kept in jobs
            ]
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    r = fut.result()
                except Exception as e:  # noqa: BLE001 - left out of the output, so a rerun retries it
                    n_err += 1
                    print(f"  error: {str(e)[:200]}", file=sys.stderr)
                    continue
                below = r["versions"][args.key]["dedup"].get("below_band", False)
                n_below += below
                n_ok += not below
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0 or i == len(jobs):
                    print(
                        f"  topped up {i}/{len(jobs)}: back in band {n_ok}, below band {n_below}, errors {n_err} "
                        f"({(time.time() - t0) / 60:.0f} min)",
                        file=sys.stderr,
                    )
    print(f"Done -> {out_path} (errors {n_err}: rerun to retry them)", file=sys.stderr)


if __name__ == "__main__":
    main()
