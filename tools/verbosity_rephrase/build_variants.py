"""Assemble the verbosity variants of a dataset from rephrase.py output and (optionally) push them.

Three families, one HF dataset per variant, named <base>_<variant>:
  fixed   think / task_tracker turns and their results removed
    text0                  every assistant turn is its tool call only (no LLM involved)
    text20/50/100/300      prose rephrased to ~N tokens + the tool call
  scaled  think / task_tracker turns kept verbatim
    text0x                 every other assistant turn is its tool call only (no LLM involved)
    text0.5x/2x/4x/8x/32x  prose rephrased to N x its own length + the tool call (32x written
                           in parts from the 8x version; its rephrase jsonl is the x32 output)
  think   task_tracker turns kept verbatim; think turns rephrased too (the scaled family + thoughts)
    text0x_think0x         every other assistant turn is its tool call only, think turns removed
    text<N>x_think<N>x     the scaled family's text<N>x, plus each think turn's prose and thought
                           rephrased to N x their own lengths (--rephrase: the scaled x32 jsonl
                           and the think jsonl)
A text whose version missed its band, or was not requested (empty prose, target outside the
floor/cap), keeps its original wording. Everything else (system prompt, task, tool results, the
tool calls apart from a rephrased thought) is byte-identical to the base.

Columns follow the sibling variants: instance_id, resolved, messages. The dataset card records
the construction and per-turn token statistics.

Usage:
  python tools/verbosity_rephrase/build_variants.py --family scaled \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.jsonl \
      --out-dir eval_outputs/verbosity_rephrase/variants [--push] [--variants text2x,text8x]
  python tools/verbosity_rephrase/build_variants.py --family think \
      --hf synthetic-code-training/func_localize_claude45_1457i \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.x32.jsonl \
      --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.think.x32.jsonl \
      --out-dir eval_outputs/verbosity_rephrase/variants [--push]
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

from llm import (  # noqa: E402
    DEFAULT_TOLERANCE,
    MIN_TARGET,
    PART_MAX,
    SPECS,
    THOUGHT_LADDER_MAX,
    THOUGHT_X32_CLAMP,
    ScaledLengths,
    band,
)
from tokens import count_tokens  # noqa: E402
from trajectory import (  # noqa: E402
    FUNCTION_BLOCK,
    assemble,
    build_skeleton,
    split_content,
    thought_of,
    with_thought,
)


@dataclass(frozen=True)
class Family:
    variants: dict[
        str, str | None
    ]  # variant name -> rephrase key (None: prose removed)
    think: str  # build_skeleton policy for think turns: drop / keep / split
    plan: str  # and for task_tracker turns: drop / keep

    def policy(self, key: str | None) -> tuple[str, str]:
        """(think, plan) of a variant; a split family's prose-free variant drops the think turns (0 x a thought)."""
        if key is None and self.think == "split":
            return "drop", self.plan
        return self.think, self.plan


FAMILIES = {
    "fixed": Family(
        {
            "text0": None,
            "text20": "t20",
            "text50": "t50",
            "text100": "t100",
            "text300": "t300",
        },
        think="drop",
        plan="drop",
    ),
    "scaled": Family(
        {
            "text0x": None,
            "text0.5x": "x0.5",
            "text2x": "x2",
            "text4x": "x4",
            "text8x": "x8",
            "text32x": "x32",
        },
        think="keep",
        plan="keep",
    ),
    "think": Family(
        {
            "text0x_think0x": None,
            "text0.5x_think0.5x": "x0.5",
            "text2x_think2x": "x2",
            "text4x_think4x": "x4",
            "text8x_think8x": "x8",
            "text32x_think32x": "x32",
        },
        think="split",
        plan="keep",
    ),
}


def rows_of(ds: Dataset) -> Iterable[dict[str, Any]]:
    """`Dataset.__iter__` is untyped; every row is a plain dict."""
    return cast(Iterable[dict[str, Any]], ds)


def load_rephrase(paths: list[Path]) -> dict[tuple[str, int, str], dict]:
    """Records of all files by (instance_id, msg_idx, field); a later line or file for the same unit wins."""
    out: dict[tuple[str, int, str], dict] = {}
    for path in paths:
        with path.open() as fh:
            for line in fh:
                try:
                    v = json.loads(line)
                except ValueError:
                    continue
                if not v.get("error"):
                    out[(v["instance_id"], v["msg_idx"], v.get("field", "text"))] = v
    return out


def new_stats() -> dict:
    return {
        "turns": 0,
        "fallback": 0,
        "skipped": 0,
        "deduped": 0,
        "below_band": 0,
        "tokens": [],
        "ratios": [],
    }


def version(reph: dict, unit: tuple[str, int, str], key: str, stats: dict) -> str:
    """The unit's `key` version, or its original text when that version was not requested."""
    r = reph.get(unit)
    if r is None:
        raise KeyError(f"no rephrase output for {unit}")
    v = r["versions"].get(key)
    if v is None:
        if key in SPECS[r["family"]].targets(r["orig_tokens"]):
            raise KeyError(f"{unit} has no {key} version; run rephrase.py for {key}")
        stats["skipped"] += 1
        text, n = r["orig_text"], r["orig_tokens"]
    else:
        text, n = v["text"], v["tokens"]
        stats["fallback"] += not v["ok"]
        if "dedup" in v:  # dedupe_parts.py removed repeated sentences
            stats["deduped"] += 1
            stats["below_band"] += v["dedup"].get("below_band", False)
        if v["ok"] and r["orig_tokens"]:
            stats["ratios"].append(n / r["orig_tokens"])
    stats["turns"] += 1
    stats["tokens"].append(n)
    return text


def build_one(
    row: dict, key: str | None, family: Family, reph: dict, stats: dict
) -> list[dict]:
    think, plan = family.policy(key)
    sk = build_skeleton(row["messages"], think, plan)
    texts: dict[int, str] = {}
    thoughts: dict[int, str] = {}
    for idx, t in sk.turns.items():
        if key is None:
            texts[idx] = ""
            stats["turns"] += 1
            stats["tokens"].append(count_tokens(t.text) if t.kind == "text" else 0)
        else:
            texts[idx] = version(reph, (row["instance_id"], idx, "text"), key, stats)
            if t.tool == "think":
                thoughts[idx] = version(
                    reph, (row["instance_id"], idx, "thought"), key, stats["thought"]
                )
        stats["empty"] += not texts[idx] and t.kind != "text"
    stats["dropped"] += len(sk.dropped)
    stats["think"] += sum(
        1
        for i in sk.keep
        if i not in sk.turns and sk.messages[i]["role"] == "assistant"
    )
    return assemble(sk, texts, thoughts)


def validate(
    base_msgs: list[dict], new_msgs: list[dict], key: str | None, family: Family
) -> list[str]:
    """Structural checks: unsplit turns unchanged (minus dropped results), calls unchanged apart from a rephrased thought."""
    problems: list[str] = []
    think, plan = family.policy(key)
    sk = build_skeleton(base_msgs, think, plan)
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
        text, call, tool, kind = split_content(new["content"])
        if tool == "think" and key is not None:
            if (
                not thought_of(call)
                or with_thought(sk.turns[i].call, thought_of(call)) != call
            ):
                problems.append(f"think call changed beyond its thought at {i}")
        elif call != sk.turns[i].call:
            problems.append(f"call part changed at {i}")
        if key is None and text and kind != "text":
            problems.append(f"text0 turn {i} still has prose")
        if FUNCTION_BLOCK.search(text):
            problems.append(f"prose contains a function block at {i}")
    if any(m["role"] == "assistant" and not m["content"].strip() for m in new_msgs):
        problems.append("empty assistant turn")
    for tool, how in (("think", think), ("task_tracker", plan)):
        if how == "drop" and any(
            f"<function={tool}>" in m["content"] for m in new_msgs
        ):
            problems.append(f"{tool} survived")
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
    th = stats["thought"]
    if key is None and family == "think":
        what = (
            "every assistant turn other than `task_tracker` is its tool call only, and the `think` turns are removed "
            "with their result turns (0 x their thought): the planning steps stay, the prose and the thinking steps go."
        )
    elif family == "think":
        assert key is not None
        thought = SPECS["thought"]
        mult = thought.MULT[key]
        what = (
            f"the prose before every tool call and the thought of every `think` call are rewritten by `{model}` "
            f"to {mult:g} times their own length in Qwen3 tokens (accepted band ±{pct} %); `task_tracker` turns "
            f"are kept verbatim. Prose: of {stats['turns']} turns, {stats['skipped']} were not rephrased (empty, or a "
            f"target outside {MIN_TARGET}-{SPECS['scaled'].cap(key)} tokens) and {stats['fallback']} missed the band. "
            f"Thoughts: of {th['turns']}, {th['skipped']} were not rephrased (a target outside "
            f"{MIN_TARGET}-{thought.cap(key)} tokens) and {th['fallback']} missed the band. All of them keep their "
            "original wording."
            + (
                f" A thought whose {mult:g}x would exceed {THOUGHT_X32_CLAMP} tokens is rewritten to {THOUGHT_X32_CLAMP} "
                "tokens instead (longer versions degenerate into filler)."
                if key == "x32"
                else ""
            )
        )
    elif key is None and family == "fixed":
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
        scaled = SPECS["scaled"]
        assert isinstance(scaled, ScaledLengths)
        mult, cap = scaled.MULT[key], scaled.cap(key)
        how = (
            f"written in consecutive parts of at most {PART_MAX} tokens, each request continuing the text so far, "
            f"starting from the accepted 8x version as its source"
            if key in scaled.PARTS
            else "up to 3 rounds"
        )
        what = (
            f"the prose before every tool call is rewritten by `{model}` to {mult:g} times its own length "
            f"in Qwen3 tokens (accepted band ±{pct} %, {how}). Of {stats['turns']} turns, "
            f"{stats['skipped']} were not rephrased (empty prose, or a target outside {MIN_TARGET}-{cap} "
            f"tokens) and {stats['fallback']} missed the band; both keep their original prose."
        )
    if family == "think":
        construction = (
            "Construction (shared by the `_text<N>x_think<N>x` siblings): the same as the `_text0x/0.5x/2x/4x/8x/32x`\n"
            "siblings - the prose of every non-think turn is the very same rewritten text - except for the `think` turns:\n"
            "their prose and the `thought` argument of the call are rewritten to the same multiple (`_text0x_think0x`\n"
            "removes them), while `task_tracker` turns stay verbatim. The system prompt, task, tool results and every\n"
            "other tool-call argument are byte-identical to the base. The thought rephraser saw only the thought and\n"
            f"the call's summary. 0.5x and the versions of up to {THOUGHT_LADDER_MAX} tokens were written in one response; every\n"
            f"longer version was written in consecutive parts of at most {PART_MAX} tokens, elaborating the version\n"
            "below it (4x from 2x, 8x from 4x, 32x from 8x). A text's targets are multiples of its own length, so\n"
            "empty prose stays empty."
        )
        table_extra = ""
        if key:
            table_extra = (
                f"| prose tokens / base prose tokens, rephrased turns: mean / median | "
                f"{statistics.mean(stats['ratios']):.2f} / {statistics.median(stats['ratios']):.2f} |\n"
                f"| thoughts: rephrased / not rephrased / kept original (band missed) | "
                f"{th['turns'] - th['skipped'] - th['fallback']} / {th['skipped']} / {th['fallback']} |\n"
                f"| thought tokens: mean / median | {statistics.mean(th['tokens']):.1f} / {statistics.median(th['tokens']):.0f} |\n"
                f"| thought tokens / base thought tokens, rephrased thoughts: mean / median | "
                f"{statistics.mean(th['ratios']):.2f} / {statistics.median(th['ratios']):.2f} |\n"
            )
    elif family == "fixed":
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
            "Construction (shared by the `_text0x/0.5x/2x/4x/8x/32x` siblings): `think` and `task_tracker` turns are kept\n"
            "verbatim, as are the system prompt, task, tool calls and tool results. For the rephrased siblings the\n"
            "rephraser saw only the current turn (its prose + its tool call) and wrote the four versions in one response\n"
            "as a ladder (0.5x shortens the original, 2x elaborates it, 4x elaborates the 2x, 8x elaborates the 4x), so\n"
            "the four lengths share one meaning; the 32x version was written afterwards by elaborating the 8x version\n"
            "further, in parts. A turn's targets are multiples of its own prose length, so empty turns stay empty."
        )
        ratios = stats["ratios"]
        cap = SPECS["scaled"].cap(key) if key else 0
        table_extra = (
            f"| prose tokens / base prose tokens, rephrased turns: mean / median | "
            f"{statistics.mean(ratios):.2f} / {statistics.median(ratios):.2f} |\n"
            f"| turns not rephrased (empty prose or target outside {MIN_TARGET}-{cap} tokens) | {stats['skipped']} |\n"
            if key
            else ""
        )
    dedup_row = (
        f"| versions whose verbatim-repeated sentences were removed afterwards (dedupe_parts.py) / of them left "
        f"under the band | {stats['deduped']} / {stats['below_band']} |\n"
        if stats["deduped"]
        else ""
    )
    turns_row = "assistant turns (excluding think/task_tracker)"
    if family == "fixed":
        think_row = f"think/task_tracker turns removed | {stats['dropped']}"
    elif family == "scaled":
        think_row = f"think/task_tracker turns kept verbatim | {stats['think']}"
    else:
        think_row = f"think turns removed / task_tracker turns kept verbatim | {stats['dropped']} / {stats['think']}"
        turns_row = "assistant turns (excluding task_tracker)"
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
| {turns_row} | {stats["turns"]} |
| {think_row} |
| prose tokens per turn: mean / median | {statistics.mean(toks):.1f} / {statistics.median(toks):.0f} |
{table_extra}{dedup_row}| turns with empty prose | {stats["empty"]} |
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
        action="append",
        default=[],
        help="rephrase jsonl for this dataset and family (required unless only text0); repeatable, "
        "the think family takes the scaled family's x32 jsonl and its own",
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
        reph = load_rephrase([Path(r) for r in args.rephrase])
        model = next(iter(reph.values()))["model"] if reph else None
        think, plan = family.think, family.plan
        missing = sum(
            1
            for row in rows_of(base)
            for i, t in build_skeleton(row["messages"], think, plan).turns.items()
            for field in ("text", "thought")
            if (field == "text" or t.tool == "think")
            and (row["instance_id"], i, field) not in reph
        )
        if missing:
            sys.exit(
                f"error: {missing} turns have no rephrase output; finish rephrase.py first"
            )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        key = family.variants[variant]
        stats = {**new_stats(), "dropped": 0, "empty": 0, "think": 0}
        stats["thought"] = new_stats()
        rows = []
        n_problems = 0
        for row in rows_of(base):
            msgs = build_one(row, key, family, reph, stats)
            probs = validate(row["messages"], msgs, key, family)
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
        th = stats["thought"]
        if th["turns"]:
            ratio += (
                f" | thoughts={th['turns']} tok mean={statistics.mean(th['tokens']):.1f} "
                f"ratio mean={statistics.mean(th['ratios'] or [0]):.2f} skipped={th['skipped']} fallback={th['fallback']}"
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
