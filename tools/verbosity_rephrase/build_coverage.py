"""Build the prose-coverage ablation: the share of assistant turns that carry prose, 0 % to 100 %.

The population is every assistant turn other than think / task_tracker (those, like the system
prompt, task, tool calls and tool results, stay byte-identical to the base). In
func_localize_claude45_1457i 40 % of these turns have prose before their tool call and 60 % have
none. Levels, as shares of all turns with prose:
  0 %            all prose removed (the same trajectories as `_text0x`)
  20 %           a random half of the prose turns lose their prose
  40 %           the base as it is
  60 / 80 / 100 % a random 1/3, 2/3 or all of the empty turns get the comment written for them by
                 fill_prose.py (one LLM call per trajectory)
The random orders are drawn once (seed), so the levels are nested: a turn with prose at one level
has the same prose at every higher level. Empty turns fill_prose.py could not fill are left out of
the fill order.

Usage:
  python tools/verbosity_rephrase/build_coverage.py --hf synthetic-code-training/func_localize_claude45_1457i \
      --fills eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.fill.jsonl \
      --out-dir eval_outputs/verbosity_rephrase/variants [--levels 0,20,40] [--push]
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, str(Path(__file__).resolve().parent))

from fill_prose import load_fills  # noqa: E402
from tokens import count_tokens  # noqa: E402
from trajectory import assemble, build_skeleton, split_content  # noqa: E402


LEVELS = (0, 20, 40, 60, 80, 100)
Unit = tuple[str, int]  # (instance_id, msg_idx)


def rows_of(ds: Dataset) -> Iterable[dict[str, Any]]:
    return cast(Iterable[dict[str, Any]], ds)


def orders(
    ds: Dataset, fills: dict[Unit, str], seed: int
) -> tuple[list[Unit], list[Unit], int]:
    """(prose turns in drop order, fillable empty turns in fill order, empty turns without a fill)."""
    prose: list[Unit] = []
    empty: list[Unit] = []
    for row in rows_of(ds):
        for idx, t in build_skeleton(row["messages"], "keep", "keep").turns.items():
            (prose if t.text else empty).append((row["instance_id"], idx))
    rng = random.Random(seed)
    prose.sort()
    empty.sort()
    rng.shuffle(prose)
    rng.shuffle(empty)
    fillable = [u for u in empty if u in fills]
    return prose, fillable, len(empty) - len(fillable)


def changes(
    level: int, prose: list[Unit], fillable: list[Unit]
) -> dict[Unit, str | None]:
    """Turns whose prose changes at `level`: None = removed, "fill" = gets its filled comment."""
    if level == 0:
        return dict.fromkeys(prose)
    if level == 20:
        return dict.fromkeys(prose[: round(len(prose) / 2)])
    if level == 40:
        return {}
    n = len(fillable)
    n_fill = {60: round(n / 3), 80: round(2 * n / 3), 100: n}[level]
    return {u: "fill" for u in fillable[:n_fill]}


def build_row(
    row: dict, change: dict[Unit, str | None], fills: dict[Unit, str]
) -> list[dict]:
    sk = build_skeleton(row["messages"], "keep", "keep")
    texts: dict[int, str] = {}
    for idx in sk.turns:
        u = (row["instance_id"], idx)
        if u in change:
            texts[idx] = "" if change[u] is None else fills[u]
    return assemble(sk, texts)


def validate(
    base: list[dict],
    new: list[dict],
    change_idx: dict[int, str | None],
    fills_of: dict[int, str],
) -> list[str]:
    """Same turns, every message outside the changed turns byte-identical, changed turns = call + the new prose."""
    if len(base) != len(new):
        return ["turn count changed"]
    problems = []
    for i, (b, n) in enumerate(zip(base, new, strict=True)):
        if i not in change_idx:
            if b != n:
                problems.append(f"turn {i} changed")
            continue
        text, call, _, _ = split_content(n["content"])
        if call != split_content(b["content"])[1]:
            problems.append(f"call changed at {i}")
        want = "" if change_idx[i] is None else fills_of[i].strip()
        if text != want:
            problems.append(f"prose at {i} is not the intended one")
    return problems


def card(
    base_repo: str,
    level: int,
    model: str | None,
    stats: dict,
    seed: int,
    n_missing: int,
) -> str:
    what = {
        0: "every assistant turn other than `think` / `task_tracker` is its tool call only (the same trajectories as `_text0x`).",
        20: "a random half of the assistant turns that carry prose before their tool call lose it.",
        40: "unchanged: this is the base dataset, the reference point of the series.",
    }.get(
        level,
        f"a random {({60: 'third', 80: 'two thirds'}).get(level, 'all')} of the assistant turns that have no prose before "
        f"their tool call get a comment written by `{model}`, which saw the whole trajectory (clipped) and wrote the "
        "comments of all such turns of a trajectory in one request, matching the agent's own comments and using only "
        "what the agent knew at that step.",
    )
    toks = stats["tokens"]
    added = stats["added"]
    return f"""---
license: mit
---
# {base_repo.split("/")[-1]}_textcov{level}

Prose-coverage ablation of [`{base_repo}`](https://huggingface.co/datasets/{base_repo}): {what}

The series `_textcov0/20/40/60/80/100` varies only the share of assistant turns (other than `think` /
`task_tracker`) that carry natural-language prose before their tool call: 40 % in the base, 0 / 20 % by
removing prose, 60 / 80 / 100 % by adding it to 1/3, 2/3 or all of the empty turns. The random choices are
nested (seed {seed}): a turn with prose at one level has the same prose at every higher level. `think` /
`task_tracker` turns, the system prompt, task, tool calls and tool results are byte-identical to the base.

| | value |
|---|---|
| rows | {stats["rows"]} |
| assistant turns (excluding think/task_tracker) | {stats["turns"]} |
| turns with prose | {stats["with_prose"]} ({stats["with_prose"] / stats["turns"]:.1%}) |
| prose tokens per turn with prose: mean / median | {statistics.mean(toks) if toks else 0:.1f} / {statistics.median(toks) if toks else 0:.0f} |
| added comments: count / mean / median tokens | {len(added)} / {statistics.mean(added) if added else 0:.1f} / {statistics.median(added) if added else 0:.0f} |
| empty turns the model left unfilled (never selected) | {n_missing} |

Built with `tools/verbosity_rephrase` (`fill_prose.py`, `build_coverage.py`) in the `benchmarks` repo.
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hf", required=True)
    p.add_argument("--hf-split", default="train")
    p.add_argument(
        "--fills", help="fill_prose.py output (required for levels above 40)"
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--levels", default=",".join(map(str, LEVELS)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push", action="store_true")
    p.add_argument("--org", default="synthetic-code-training")
    args = p.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x]
    if any(lv not in LEVELS for lv in levels):
        sys.exit(f"error: levels must be among {LEVELS}")
    base = load_dataset(args.hf, split=args.hf_split)
    assert isinstance(base, Dataset)
    fills: dict[Unit, str] = {}
    model = None
    if args.fills:
        recs = load_fills(Path(args.fills))
        if any(not r.get("leak_checked") for r in recs.values()):
            sys.exit(
                "error: some fills have not had the leak pass; rerun fill_prose.py"
            )
        for r in recs.values():
            model = r["model"]
            for k, v in r["fills"].items():
                fills[(r["instance_id"], int(k))] = v
    elif any(lv > 40 for lv in levels):
        sys.exit("--fills required for levels above 40")
    prose, fillable, n_missing = orders(base, fills, args.seed)
    n_empty = len(fillable) + n_missing
    if any(lv > 40 for lv in levels) and n_missing > 0.01 * n_empty:
        sys.exit(
            f"error: {n_missing} of {n_empty} empty turns have no fill; finish fill_prose.py first"
        )
    label = args.hf.split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        change = changes(level, prose, fillable)
        by_row: dict[str, dict[int, str | None]] = collections.defaultdict(dict)
        for (j, i), c in change.items():
            by_row[j][i] = c
        stats: dict = {
            "rows": 0,
            "turns": 0,
            "with_prose": 0,
            "tokens": [],
            "added": [],
        }
        rows, n_bad = [], 0
        for row in rows_of(base):
            iid = row["instance_id"]
            new = build_row(row, change, fills)
            change_idx = by_row.get(iid, {})
            fills_of = {
                i: fills[(iid, i)] for i, c in change_idx.items() if c is not None
            }
            probs = validate(row["messages"], new, change_idx, fills_of)
            if probs:
                n_bad += 1
                print(f"  {iid}: {probs[:3]}", file=sys.stderr)
            sk = build_skeleton(new, "keep", "keep")
            for idx, t in sk.turns.items():
                stats["turns"] += 1
                if t.text:
                    stats["with_prose"] += 1
                    stats["tokens"].append(count_tokens(t.text))
                    if idx in fills_of:
                        stats["added"].append(stats["tokens"][-1])
            stats["rows"] += 1
            rows.append(
                {"instance_id": iid, "resolved": row["resolved"], "messages": new}
            )
        if n_bad:
            sys.exit(f"error: {n_bad} rows failed validation at level {level}")
        name = f"{label}_textcov{level}"
        ds = Dataset.from_list(rows)
        ds.save_to_disk(str(out_dir / name))
        readme = card(args.hf, level, model, stats, args.seed, n_missing)
        (out_dir / f"{name}.README.md").write_text(readme)
        print(
            f"{name}: rows={stats['rows']} turns={stats['turns']} with_prose={stats['with_prose']} "
            f"({stats['with_prose'] / stats['turns']:.1%}) added={len(stats['added'])}",
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
