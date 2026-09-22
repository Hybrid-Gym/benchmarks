"""Assemble the verbosity variants of a dataset from rephrase.py output and (optionally) push them.

Variants, one HF dataset each, named <base>_text<N>:
  text0                     every kept assistant turn is its tool call only (no LLM involved)
  text20/50/100/300         every kept assistant turn is <rephrased prose at ~N tokens> + the tool call;
                            a turn whose N-version missed its band keeps its original prose
In all five, think / task_tracker turns and their result turns are removed; everything else
(system prompt, task, tool results, the tool calls themselves) is byte-identical to the base.

Columns follow the sibling variants: instance_id, resolved, messages. The dataset card records
the construction and per-turn token statistics.

Usage:
  python tools/verbosity_rephrase/build_variants.py \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.rephrase.jsonl \
      --out-dir eval_outputs/verbosity_rephrase/variants [--push] [--variants text0,text20,...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import DEFAULT_TOLERANCE, TARGETS, band  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import (  # noqa: E402
    FUNCTION_BLOCK,
    assemble,
    build_skeleton,
    split_content,
)


VARIANTS = ["text0", "text20", "text50", "text100", "text300"]
KEY_OF = {"text20": "t20", "text50": "t50", "text100": "t100", "text300": "t300"}


def rows_of(ds: Dataset) -> Iterable[dict[str, Any]]:
    """`Dataset.__iter__` is untyped; every row is a plain dict."""
    return cast(Iterable[dict[str, Any]], ds)


def load_rephrase(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with path.open() as fh:
        for line in fh:
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if not v.get("error"):
                out[(v["instance_id"], v["msg_idx"])] = (
                    v  # a later line for the same unit wins
                )
    return out


def build_one(row: dict, variant: str, reph: dict, stats: dict) -> list[dict]:
    sk = build_skeleton(row["messages"])
    texts: dict[int, str] = {}
    for idx, t in sk.turns.items():
        if variant == "text0":
            texts[idx] = ""
            n = count_tokens(t.text) if t.kind == "text" else 0
        else:
            r = reph.get((row["instance_id"], idx))
            if r is None:
                raise KeyError(f"no rephrase output for {row['instance_id']} msg {idx}")
            v = r["versions"][KEY_OF[variant]]
            texts[idx], n = v["text"], v["tokens"]
            stats["fallback"] += not v["ok"]
        stats["turns"] += 1
        stats["tokens"].append(n)
        stats["empty"] += not texts[idx] and t.kind != "text"
    stats["dropped"] += len(sk.dropped)
    return assemble(sk, texts)


def validate(base_msgs: list[dict], new_msgs: list[dict], variant: str) -> list[str]:
    """Structural checks: non-assistant turns unchanged (minus dropped results), calls unchanged."""
    problems: list[str] = []
    sk = build_skeleton(base_msgs)
    if len(new_msgs) != len(sk.keep):
        return [f"turn count {len(new_msgs)} != kept {len(sk.keep)}"]
    for new, i in zip(new_msgs, sk.keep, strict=True):
        old = base_msgs[i]
        if new["role"] != old["role"]:
            problems.append(f"role mismatch at {i}")
        if i not in sk.turns:
            if new["content"] != old["content"]:
                problems.append(f"non-assistant turn {i} changed")
            continue
        text, call, _tool, kind = split_content(new["content"])
        if call != sk.turns[i].call:
            problems.append(f"call part changed at {i}")
        if variant == "text0" and text and kind != "text":
            problems.append(f"text0 turn {i} still has prose")
        if FUNCTION_BLOCK.search(text):
            problems.append(f"prose contains a function block at {i}")
    if any(m["role"] == "assistant" and not m["content"].strip() for m in new_msgs):
        problems.append("empty assistant turn")
    if any(
        "<function=think>" in m["content"] or "<function=task_tracker>" in m["content"]
        for m in new_msgs
    ):
        problems.append("think/task_tracker survived")
    return problems


def card(
    base_repo: str,
    variant: str,
    model: str | None,
    n_rows: int,
    stats: dict,
    tolerance: float,
) -> str:
    toks = stats["tokens"]
    if variant == "text0":
        what = "every assistant turn is its tool call only: the prose before the call is removed."
    else:
        key = KEY_OF[variant]
        lo, hi = band(key, tolerance)
        what = (
            f"the prose before every tool call is rewritten by `{model}` to about {TARGETS[key]} "
            f"tokens (accepted band {lo}-{hi} tokens of the Qwen3 tokenizer, up to 3 rounds; "
            f"{stats['fallback']} of {stats['turns']} turns missed the band and keep their original prose)."
        )
    return f"""---
license: mit
---
# {base_repo.split("/")[-1]}_{variant}

Verbosity-ablation variant of [`{base_repo}`](https://huggingface.co/datasets/{base_repo}): {what}

Construction (shared by all `_text*` siblings): `think` and `task_tracker` turns and their result turns are
removed (trajectories contain only real tool calls); the system prompt, task, tool calls and tool results are
byte-identical to the base. The rephraser saw only the current turn (its prose + its tool call); the ~300-token
version was written first and condensed to 100/50/20 in the same response so the four lengths share one meaning.
Turns with no original prose received prose explaining their tool call. Malformed tool calls (garbled
`<tool_call>` JSON / `<invoke>`) are kept verbatim as the call part.

| | value |
|---|---|
| rows | {n_rows} |
| assistant turns | {stats["turns"]} |
| think/task_tracker turns removed | {stats["dropped"]} |
| prose tokens per turn: mean / median | {statistics.mean(toks):.1f} / {statistics.median(toks):.0f} |
| turns with empty prose | {stats["empty"]} |
| turns kept original prose (band missed) | {stats["fallback"]} |

Built with `tools/verbosity_rephrase` in the `benchmarks` repo.
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True, help="base dataset repo")
    p.add_argument("--hf-split", default="train")
    p.add_argument(
        "--rephrase",
        help="rephrase.jsonl for this dataset (required unless only text0)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    p.add_argument(
        "--push",
        action="store_true",
        help="push each variant to <base>_<variant> on HF",
    )
    p.add_argument("--org", default="synthetic-code-training")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only the first N rows (debug; never push with this)",
    )
    args = p.parse_args()
    if args.limit and args.push:
        sys.exit("refusing to --push a --limit build")

    base = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(base, Dataset)
    if args.limit:
        base = base.select(range(min(args.limit, base.num_rows)))
    label = args.hf.split("/")[-1]
    variants = [v for v in args.variants.split(",") if v]
    reph: dict = {}
    model = None
    if any(v != "text0" for v in variants):
        if not args.rephrase:
            sys.exit("--rephrase required for text20/50/100/300")
        reph = load_rephrase(Path(args.rephrase))
        model = next(iter(reph.values()))["model"] if reph else None
        missing = sum(
            1
            for row in rows_of(base)
            for i in build_skeleton(row["messages"]).turns
            if (row["instance_id"], i) not in reph
        )
        if missing:
            sys.exit(
                f"error: {missing} kept turns have no rephrase output; finish rephrase.py first"
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        stats = {"turns": 0, "dropped": 0, "empty": 0, "fallback": 0, "tokens": []}
        rows = []
        n_problems = 0
        for row in rows_of(base):
            msgs = build_one(row, variant, reph, stats)
            probs = validate(row["messages"], msgs, variant)
            if probs:
                n_problems += 1
                print(f"  {variant} {row['instance_id']}: {probs[:3]}", file=sys.stderr)
            rows.append(
                {
                    "instance_id": row["instance_id"],
                    "resolved": row["resolved"],
                    "messages": msgs,
                }
            )
        if n_problems:
            sys.exit(f"error: {n_problems} rows failed validation for {variant}")
        ds = Dataset.from_list(rows)
        local = out_dir / f"{label}_{variant}"
        ds.save_to_disk(str(local))
        readme = card(args.hf, variant, model, len(rows), stats, args.tolerance)
        (out_dir / f"{label}_{variant}.README.md").write_text(readme)
        toks = stats["tokens"]
        print(
            f"{label}_{variant}: rows={len(rows)} turns={stats['turns']} dropped={stats['dropped']} "
            f"prose tok mean={statistics.mean(toks):.1f} median={statistics.median(toks):.0f} "
            f"empty={stats['empty']} fallback={stats['fallback']} -> {local}",
            file=sys.stderr,
        )
        if args.push:
            repo = f"{args.org}/{label}_{variant}"
            ds.push_to_hub(
                repo, split=args.hf_split, private=False
            )  # the siblings are public
            HfApi().upload_file(
                path_or_fileobj=readme.encode(),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
            )
            print(f"  pushed {repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
