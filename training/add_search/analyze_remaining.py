"""
Analyze what action types appear before the first file-editing action in the
reference dataset that are NOT being added by add_search.py
(i.e., not keyword searches and not directory listings).
"""

import re
import sys
from collections import Counter, defaultdict

from datasets import load_dataset

sys.path.insert(0, "/home/yiqingxi/benchmarks/training/add_search")
from add_search import (
    FILE_EDIT_PATTERN,
    KEYWORD_SEARCH_PATTERN,
    DIR_LIST_PATTERN,
    FILE_VIEW_OR_EDIT_PATTERN,
    is_file_editing,
    is_keyword_search,
    is_dir_listing,
    build_action_result_pairs,
)

TERMINAL_PATTERN = re.compile(r"<function=terminal>", re.DOTALL)
COMMAND_PATTERN = re.compile(r"<parameter=command>(.*?)</parameter>", re.DOTALL)

BASE_DATASET = "synthetic-code-training/func_localize_claude45_1457i_min_explore"
REF_DATASET  = "synthetic-code-training/func_localize_claude45_1457i"


def first_token(command: str) -> str:
    """Return the first non-flag token (the tool/command name)."""
    tokens = command.strip().split()
    for tok in tokens:
        if not tok.startswith("-"):
            return tok.split("/")[-1]  # strip path prefix
    return tokens[0] if tokens else ""


def categorize(msg: dict) -> str:
    content = msg.get("content", "")
    if FILE_VIEW_OR_EDIT_PATTERN.search(content):
        return "file_editor"
    if is_keyword_search(msg):
        return "keyword_search"
    if is_dir_listing(msg):
        return "dir_listing"
    if TERMINAL_PATTERN.search(content):
        m = COMMAND_PATTERN.search(content)
        if m:
            return f"terminal:{first_token(m.group(1))}"
        return "terminal:?"
    return "other"


def main():
    print(f"Loading reference: {REF_DATASET}")
    ref_ds = load_dataset(REF_DATASET, split="train")
    print(f"  {len(ref_ds)} trajectories\n")

    # Per-category counts of assistant actions before first file-edit in reference
    category_counts: Counter = Counter()          # how many actions total
    instance_counts: Counter = Counter()          # how many instances have ≥1 of this category
    examples: defaultdict = defaultdict(list)     # up to 3 example commands per category

    for row in ref_ds:
        messages = row["messages"]

        first_file_edit = None
        for i, msg in enumerate(messages):
            if is_file_editing(msg):
                first_file_edit = i
                break
        cutoff = first_file_edit if first_file_edit is not None else len(messages)

        seen_in_this = set()
        for i, msg in enumerate(messages[:cutoff]):
            if msg.get("role") != "assistant":
                continue
            cat = categorize(msg)
            category_counts[cat] += 1
            seen_in_this.add(cat)
            if len(examples[cat]) < 3:
                m = COMMAND_PATTERN.search(msg.get("content", ""))
                cmd = m.group(1).strip()[:120] if m else msg.get("content", "")[:120]
                examples[cat].append(cmd)

        for cat in seen_in_this:
            instance_counts[cat] += 1

    total = len(ref_ds)
    print(f"{'Category':<35} {'#actions':>8}  {'#instances':>10}  {'% instances':>12}")
    print("-" * 72)
    for cat, cnt in category_counts.most_common():
        inst = instance_counts[cat]
        added = cat in ("keyword_search", "dir_listing")
        tag = "  [ADDED]" if added else "  [remaining]"
        print(f"{cat:<35} {cnt:>8}  {inst:>10}  {inst/total*100:>11.1f}%{tag}")

    print("\n--- Example commands for REMAINING categories ---")
    for cat, cnt in category_counts.most_common():
        if cat in ("keyword_search", "dir_listing"):
            continue
        print(f"\n[{cat}]  ({cnt} actions)")
        for ex in examples[cat]:
            print(f"  {ex!r}")


if __name__ == "__main__":
    main()
