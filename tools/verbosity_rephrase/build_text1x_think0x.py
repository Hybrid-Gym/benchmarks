"""Build `<base>_text1x_think0x`: the base trajectories with their `think` turns (and results) removed and
everything else, including the prose before every tool call and the `task_tracker` turns, kept verbatim.

The think family's `_text0x_think0x` removes the think turns AND every turn's prose, so it moves two things
at once; this variant is the control that removes only the think turns (yiqing, 2026-09-28). No LLM.

Usage:
  python tools/verbosity_rephrase/build_text1x_think0x.py --hf synthetic-code-training/func_localize_claude45_1457i \
      --out-dir eval_outputs/verbosity_rephrase/variants [--push]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokens import count_tokens  # noqa: E402
from trajectory import assemble, build_skeleton  # noqa: E402


def strip_think(messages: list[dict]) -> list[dict]:
    return assemble(build_skeleton(messages, "drop", "keep"), {})


def validate(base: list[dict], new: list[dict]) -> list[str]:
    """`new` == `base` minus every think turn and the user turn right after it; nothing else changes."""
    expect: list[dict] = []
    skip = False
    for m in base:
        if skip:
            skip = False
            continue
        if m["role"] == "assistant" and "<function=think>" in m["content"]:
            skip = True
            continue
        expect.append(m)
    return [] if expect == new else ["differs from the base minus its think pairs"]


def card(base_repo: str, name: str, stats: dict) -> str:
    return f"""---
license: mit
---
# {name}

[`{base_repo}`](https://huggingface.co/datasets/{base_repo}) with every `think` turn removed together with its
"Your thought has been logged." result; every other message, the prose before each tool call included, and the
`task_tracker` turns are byte-identical to the base. The control for the `_text<N>x_think<N>x` series: `_text0x_think0x`
removes the think turns and all prose, this dataset removes the think turns only.

| | value |
|---|---|
| rows | {stats["rows"]} |
| think turns removed | {stats["think"]} (in {stats["rows_with_think"]} rows) |
| assistant turns: base / here | {stats["turns_base"]} / {stats["turns"]} |
| tokens per trajectory without the system prompt (Qwen3 tokenizer), mean: base / here | {statistics.mean(stats["tok_base"]):.0f} / {statistics.mean(stats["tok"]):.0f} |

Built with `tools/verbosity_rephrase/build_text1x_think0x.py` in the `benchmarks` repo (no LLM involved).
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--push", action="store_true")
    p.add_argument("--org", default="synthetic-code-training")
    args = p.parse_args()
    base = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(base, Dataset)
    label = args.hf.split("/")[-1]
    name = f"{label}_text1x_think0x"
    stats: dict[str, Any] = {
        "rows": 0,
        "think": 0,
        "rows_with_think": 0,
        "turns_base": 0,
        "turns": 0,
        "tok_base": [],
        "tok": [],
    }
    rows = []
    for row in cast(Iterable[dict[str, Any]], base):
        new = strip_think(row["messages"])
        probs = validate(row["messages"], new)
        if probs:
            sys.exit(f"error: {row['instance_id']}: {probs}")
        n_think = sum(
            1
            for m in row["messages"]
            if m["role"] == "assistant" and "<function=think>" in m["content"]
        )
        stats["rows"] += 1
        stats["think"] += n_think
        stats["rows_with_think"] += n_think > 0
        stats["turns_base"] += sum(
            1 for m in row["messages"] if m["role"] == "assistant"
        )
        stats["turns"] += sum(1 for m in new if m["role"] == "assistant")
        stats["tok_base"].append(
            sum(count_tokens(m["content"]) for m in row["messages"][1:])
        )
        stats["tok"].append(sum(count_tokens(m["content"]) for m in new[1:]))
        rows.append({**row, "messages": new})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_dir / name))
    readme = card(args.hf, name, stats)
    (out_dir / f"{name}.README.md").write_text(readme)
    print(
        f"{name}: rows={stats['rows']} think removed={stats['think']} turns {stats['turns_base']}->{stats['turns']} "
        f"tokens mean {statistics.mean(stats['tok_base']):.0f}->{statistics.mean(stats['tok']):.0f}",
        file=sys.stderr,
    )
    if args.push:
        repo = f"{args.org}/{name}"
        ds.push_to_hub(repo, split=args.hf_split, private=False)
        HfApi().upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"  pushed {repo}", file=sys.stderr)


if __name__ == "__main__":
    main()
