"""
Compute the percentage of examples that have NO verification stage —
i.e., the finish action immediately follows the last file-editing action's
environment feedback, with no intermediate steps.

Usage:
    python training/analysis/count_no_verify.py
    python training/analysis/count_no_verify.py --datasets ds1 ds2 ds3
"""

import argparse
import re

from datasets import load_dataset

FINISH_PATTERN = re.compile(r"<function=finish[\s>]")
FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|create_file|insert)</parameter>"
)


def is_finish_action(message: dict) -> bool:
    return (
        message.get("role") == "assistant"
        and FINISH_PATTERN.search(message.get("content", "")) is not None
    )


def is_file_editing_action(message: dict) -> bool:
    return (
        message.get("role") == "assistant"
        and FILE_EDIT_PATTERN.search(message.get("content", "")) is not None
    )


def has_verification_steps(messages: list[dict]) -> bool | None:
    """
    Return:
      True  — there ARE verification steps between last file edit and finish
      False — finish immediately follows last file edit (no verification stage)
      None  — trajectory has no finish or no file editing action (skip)
    """
    # Find finish action index (last one)
    finish_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if is_finish_action(messages[i]):
            finish_idx = i
            break

    if finish_idx is None:
        return None

    # Find the last file-editing action before finish
    last_edit_idx = None
    for i in range(finish_idx - 1, -1, -1):
        if is_file_editing_action(messages[i]):
            last_edit_idx = i
            break

    if last_edit_idx is None:
        return None

    # The environment feedback immediately follows the file-editing action
    feedback_idx = last_edit_idx + 1
    if feedback_idx >= finish_idx:
        # Finish immediately follows the edit (possibly without feedback) — no verify
        return False

    if messages[feedback_idx].get("role") != "user":
        # Unexpected structure; treat as no verification
        return False

    cut_start = feedback_idx + 1
    cut_end = finish_idx

    steps_between = cut_end - cut_start
    return steps_between > 0


def analyze_dataset(dataset_name: str, resolved: bool) -> dict:
    print(f"\nLoading: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    total = len(ds)
    print(f"  {total} examples")

    no_finish = 0
    no_edit = 0
    no_verify_count = 0
    has_verify_count = 0

    for row in ds:
        messages = row.get("messages", [])
        if resolved and not row.get("resolved", False):
            continue
        result = has_verification_steps(messages)
        if result is None:
            # Check which case
            finish_present = any(is_finish_action(m) for m in messages)
            if not finish_present:
                no_finish += 1
            else:
                no_edit += 1
        elif result:
            has_verify_count += 1
        else:
            no_verify_count += 1

    eligible = total - no_finish - no_edit
    pct = no_verify_count / eligible * 100 if eligible > 0 else float("nan")

    print(f"  Skipped (no finish action):    {no_finish}")
    print(f"  Skipped (no file edit):        {no_edit}")
    print(f"  Eligible:                      {eligible}")
    print(f"  No verification stage:         {no_verify_count}  ({pct:.1f}%)")
    print(f"  Has verification stage:        {has_verify_count}  ({100 - pct:.1f}%)")

    return {
        "dataset": dataset_name,
        "total": total,
        "eligible": eligible,
        "no_verify": no_verify_count,
        "has_verify": has_verify_count,
        "pct_no_verify": pct,
    }


# DEFAULT_DATASETS = [
#     "synthetic-code-training/swegym_gpt5mini_1500i",
#     "synthetic-code-training/swegym_kimi25_1359i",
#     "synthetic-code-training/swegym_qwen80b_1500i",
#     "synthetic-code-training/swegym_opus45_1495i",
# ]

DEFAULT_DATASETS = [
    "synthetic-code-training/r2egym_opus45_1502i",
    "synthetic-code-training/r2egym_converted_1054i",
    "synthetic-code-training/r2egym_qwen3next80b_1500i",
    "synthetic-code-training/r2egym_gpt5mini_1500i",
    "synthetic-code-training/r2egym_deepseek_v4_flash_1390i",
]


def main():
    parser = argparse.ArgumentParser(
        description="Compute %% of examples with no verification stage across datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="HuggingFace dataset names to analyze",
    )
    parser.add_argument(
        "--resolved",
        action="store_true",
        help="Only count examples that have been resolved",
    )
    args = parser.parse_args()

    results = []
    for ds_name in args.datasets:
        stats = analyze_dataset(ds_name, args.resolved)
        results.append(stats)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    header = f"{'Dataset':<50} {'Eligible':>8} {'No-verify':>10} {'%':>7}"
    print(header)
    print("-" * 70)
    for r in results:
        name = r["dataset"].split("/")[-1]
        print(
            f"{name:<50} {r['eligible']:>8} {r['no_verify']:>10} {r['pct_no_verify']:>6.1f}%"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
