"""Assemble the verbosity variants of a dataset from rephrase.py output and (optionally) push them.

Two families, one HF dataset per variant, named <base>_<variant>:
  fixed   think / task_tracker turns and their results removed
    text0                  every assistant turn is its tool call only (no LLM involved)
    text20/50/100/300      prose rephrased to ~N tokens + the tool call
  scaled  think / task_tracker turns kept verbatim
    text0x                 every other assistant turn is its tool call only (no LLM involved)
    text0.5x/2x/4x/8x      prose rephrased to N x its own length + the tool call
A turn whose version missed its band, or was not requested (empty prose, target outside the
floor/cap), keeps its original prose. Everything else (system prompt, task, tool results, the
tool calls themselves) is byte-identical to the base.

Columns follow the sibling variants: instance_id, resolved, messages. The dataset card records
the construction and per-turn token statistics.

Usage:
  python tools/verbosity_rephrase/build_variants.py --family scaled \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.jsonl \
      --out-dir eval_outputs/verbosity_rephrase/variants [--push] [--variants text2x,text8x]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import DEFAULT_TOLERANCE, MAX_TARGET, MIN_TARGET, SPECS, band  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import (  # noqa: E402
    FUNCTION_BLOCK,
    assemble,
    build_skeleton,
    split_content,
)


@dataclass(frozen=True)
class Family:
    variants: dict[
        str, str | None
    ]  # variant name -> rephrase key (None: prose removed)
    keep_think: bool


FAMILIES = {
    "fixed": Family(
        {
            "text0": None,
            "text20": "t20",
            "text50": "t50",
            "text100": "t100",
            "text300": "t300",
        },
        keep_think=False,
    ),
    "scaled": Family(
        {
            "text0x": None,
            "text0.5x": "x0.5",
            "text2x": "x2",
            "text4x": "x4",
            "text8x": "x8",
        },
        keep_think=True,
    ),
}


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


def build_one(
    row: dict, key: str | None, keep_think: bool, reph: dict, stats: dict
) -> list[dict]:
    sk = build_skeleton(row["messages"], keep_think)
    texts: dict[int, str] = {}
    for idx, t in sk.turns.items():
        if key is None:
            texts[idx] = ""
            n = count_tokens(t.text) if t.kind == "text" else 0
        else:
            r = reph.get((row["instance_id"], idx))
            if r is None:
                raise KeyError(f"no rephrase output for {row['instance_id']} msg {idx}")
            v = r["versions"].get(key)
            if v is None:  # not requested: keeps the original prose
                texts[idx], n = t.text, r["orig_tokens"]
                stats["skipped"] += 1
            else:
                texts[idx], n = v["text"], v["tokens"]
                stats["fallback"] += not v["ok"]
                if v["ok"] and r["orig_tokens"]:
                    stats["ratios"].append(n / r["orig_tokens"])
        stats["turns"] += 1
        stats["tokens"].append(n)
        stats["empty"] += not texts[idx] and t.kind != "text"
    stats["dropped"] += len(sk.dropped)
    stats["think"] += sum(
        1
        for i in sk.keep
        if i not in sk.turns and sk.messages[i]["role"] == "assistant"
    )
    return assemble(sk, texts)


def validate(
    base_msgs: list[dict], new_msgs: list[dict], key: str | None, keep_think: bool
) -> list[str]:
    """Structural checks: unsplit turns unchanged (minus dropped results), calls unchanged."""
    problems: list[str] = []
    sk = build_skeleton(base_msgs, keep_think)
    if len(new_msgs) != len(sk.keep):
        return [f"turn count {len(new_msgs)} != kept {len(sk.keep)}"]
    for new, i in zip(new_msgs, sk.keep, strict=True):
        old = base_msgs[i]
        if new["role"] != old["role"]:
            problems.append(f"role mismatch at {i}")
        if i not in sk.turns:
            if new["content"] != old["content"]:
                problems.append(f"verbatim turn {i} changed")
            continue
        text, call, _tool, kind = split_content(new["content"])
        if call != sk.turns[i].call:
            problems.append(f"call part changed at {i}")
        if key is None and text and kind != "text":
            problems.append(f"text0 turn {i} still has prose")
        if FUNCTION_BLOCK.search(text):
            problems.append(f"prose contains a function block at {i}")
    if any(m["role"] == "assistant" and not m["content"].strip() for m in new_msgs):
        problems.append("empty assistant turn")
    if not keep_think and any(
        "<function=think>" in m["content"] or "<function=task_tracker>" in m["content"]
        for m in new_msgs
    ):
        problems.append("think/task_tracker survived")
    return problems


def card(
    base_repo: str,
    family: str,
    variant: str,
    model: str | None,
    n_rows: int,
    stats: dict,
    tolerance: float,
) -> str:
    key = FAMILIES[family].variants[variant]
    toks = stats["tokens"]
    pct = round(tolerance * 100)
    if key is None and family == "fixed":
        what = "every assistant turn is its tool call only: the prose before the call is removed."
    elif key is None:
        what = (
            "every assistant turn other than `think` / `task_tracker` is its tool call only: the prose before "
            "the call is removed, while the thinking/planning steps stay."
        )
    elif family == "fixed":
        target = SPECS["fixed"].TARGETS[key]
        lo, hi = band(target, tolerance)
        what = (
            f"the prose before every tool call is rewritten by `{model}` to about {target} "
            f"tokens (accepted band {lo}-{hi} tokens of the Qwen3 tokenizer, up to 3 rounds; "
            f"{stats['fallback']} of {stats['turns']} turns missed the band and keep their original prose)."
        )
    else:
        mult = SPECS["scaled"].MULT[key]
        what = (
            f"the prose before every tool call is rewritten by `{model}` to {mult:g} times its own length "
            f"in Qwen3 tokens (accepted band ±{pct} %, up to 3 rounds). Of {stats['turns']} turns, "
            f"{stats['skipped']} were not rephrased (empty prose, or a target outside {MIN_TARGET}-{MAX_TARGET} "
            f"tokens) and {stats['fallback']} missed the band; both keep their original prose."
        )
    if family == "fixed":
        construction = (
            "Construction (shared by all `_text*` siblings): `think` and `task_tracker` turns and their result turns are\n"
            "removed (trajectories contain only real tool calls); the system prompt, task, tool calls and tool results are\n"
            "byte-identical to the base. The rephraser saw only the current turn (its prose + its tool call); the ~300-token\n"
            "version was written first and condensed to 100/50/20 in the same response so the four lengths share one meaning.\n"
            "Turns with no original prose received prose explaining their tool call."
        )
        table_extra = ""
    else:
        construction = (
            "Construction (shared by the `_text0x/0.5x/2x/4x/8x` siblings): `think` and `task_tracker` turns are kept\n"
            "verbatim, as are the system prompt, task, tool calls and tool results. For the rephrased siblings the\n"
            "rephraser saw only the current turn (its prose + its tool call) and wrote the four versions in one response\n"
            "as a ladder (0.5x shortens the original, 2x elaborates it, 4x elaborates the 2x, 8x elaborates the 4x), so\n"
            "the four lengths share one meaning. A turn's targets are multiples of its own prose length, so empty turns\n"
            "stay empty."
        )
        ratios = stats["ratios"]
        table_extra = (
            f"| prose tokens / base prose tokens, rephrased turns: mean / median | "
            f"{statistics.mean(ratios):.2f} / {statistics.median(ratios):.2f} |\n"
            f"| turns not rephrased (empty prose or target outside {MIN_TARGET}-{MAX_TARGET} tokens) | {stats['skipped']} |\n"
            if key
            else ""
        )
    think_row = (
        f"think/task_tracker turns removed | {stats['dropped']}"
        if family == "fixed"
        else f"think/task_tracker turns kept verbatim | {stats['think']}"
    )
    return f"""---
license: mit
---
# {base_repo.split("/")[-1]}_{variant}

Verbosity-ablation variant of [`{base_repo}`](https://huggingface.co/datasets/{base_repo}): {what}

{construction}
Malformed tool calls (garbled `<tool_call>` JSON / `<invoke>`) are kept verbatim as the call part.

| | value |
|---|---|
| rows | {n_rows} |
| assistant turns (excluding think/task_tracker) | {stats["turns"]} |
| {think_row} |
| prose tokens per turn: mean / median | {statistics.mean(toks):.1f} / {statistics.median(toks):.0f} |
{table_extra}| turns with empty prose | {stats["empty"]} |
| turns kept original prose (band missed) | {stats["fallback"]} |

Built with `tools/verbosity_rephrase` in the `benchmarks` repo.
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True, help="base dataset repo")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--family", choices=list(FAMILIES), default="fixed")
    p.add_argument(
        "--rephrase",
        help="rephrase jsonl for this dataset and family (required unless only text0)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--variants", help="comma-separated subset (default: the family's)")
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
    family = FAMILIES[args.family]
    variants = (
        [v for v in args.variants.split(",") if v]
        if args.variants
        else list(family.variants)
    )
    unknown = [v for v in variants if v not in family.variants]
    if unknown:
        sys.exit(f"error: {unknown} are not {args.family} variants")

    base = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(base, Dataset)
    if args.limit:
        base = base.select(range(min(args.limit, base.num_rows)))
    label = args.hf.split("/")[-1]
    reph: dict = {}
    model = None
    if any(family.variants[v] for v in variants):
        if not args.rephrase:
            sys.exit("--rephrase required for rephrased variants")
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
                f"error: {missing} turns have no rephrase output; finish rephrase.py first"
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        key = family.variants[variant]
        stats = {
            "turns": 0,
            "dropped": 0,
            "empty": 0,
            "fallback": 0,
            "skipped": 0,
            "think": 0,
            "tokens": [],
            "ratios": [],
        }
        rows = []
        n_problems = 0
        for row in rows_of(base):
            msgs = build_one(row, key, family.keep_think, reph, stats)
            probs = validate(row["messages"], msgs, key, family.keep_think)
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
        readme = card(
            args.hf, args.family, variant, model, len(rows), stats, args.tolerance
        )
        (out_dir / f"{label}_{variant}.README.md").write_text(readme)
        toks = stats["tokens"]
        ratio = (
            f" ratio mean={statistics.mean(stats['ratios']):.2f}"
            if stats["ratios"]
            else ""
        )
        print(
            f"{label}_{variant}: rows={len(rows)} turns={stats['turns']} dropped={stats['dropped']} "
            f"prose tok mean={statistics.mean(toks):.1f} median={statistics.median(toks):.0f}{ratio} "
            f"empty={stats['empty']} skipped={stats['skipped']} fallback={stats['fallback']} -> {local}",
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
