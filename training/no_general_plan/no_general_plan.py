"""
Remove task-tracker actions and their results that appear before the first
non-task-tracker action.

All task-tracker action+result pairs that occur before the first
non-task-tracker assistant action are stripped.  Everything from the first
non-task-tracker action onward is left untouched.

The resulting trajectory therefore contains:

  [system / user preamble]
  <everything from the first non-task-tracker action onward, unchanged>

An example is skipped entirely if there is no non-task-tracker assistant
action in the trajectory.

Definitions
-----------
- task-tracker action : assistant message calling <function=task_tracker>
- non-task-tracker action : any assistant message that does NOT call
                            task_tracker
"""

import argparse
import ast
import json
import re

from datasets import Dataset, load_dataset


# ── Regex patterns ────────────────────────────────────────────────────────────
TASK_TRACKER_PATTERN = re.compile(r"<function=task_tracker[\s>]")

_TASK_LIST_PARAM_RE = re.compile(
    r'(<parameter=task_list>)(.*?)(</parameter>)',
    re.DOTALL
)


def fix_task_list_json(content: str) -> str:
    """Convert Python single-quote task_list values to JSON double-quote syntax."""
    def _replace(m: re.Match) -> str:
        raw = m.group(2)
        try:
            parsed = ast.literal_eval(raw)
            return m.group(1) + json.dumps(parsed) + m.group(3)
        except (ValueError, SyntaxError):
            return m.group(0)
    return _TASK_LIST_PARAM_RE.sub(_replace, content)


def fix_messages_task_list(messages: list[dict]) -> tuple[list[dict], bool]:
    """Fix task_list JSON syntax in all messages. Returns (fixed_messages, was_changed)."""
    changed = False
    result = []
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str) and "<parameter=task_list>" in content:
            new_content = fix_task_list_json(content)
            if new_content != content:
                changed = True
                msg = {**msg, "content": new_content}
        result.append(msg)
    return result, changed


# ── Message classifiers ───────────────────────────────────────────────────────

def is_task_tracker(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and TASK_TRACKER_PATTERN.search(msg.get("content", "")) is not None
    )


# ── Pair builder ──────────────────────────────────────────────────────────────

def build_action_result_pairs(messages: list[dict]) -> list[tuple[int, int | None]]:
    """
    Build (action_idx, result_idx) pairs for the full message list.

    Consecutive assistant messages form a batch; they are paired with the
    immediately following user messages in the same order.  result_idx is
    None when no result message exists for an action.
    """
    pairs: list[tuple[int, int | None]] = []
    i = 0
    while i < len(messages):
        if messages[i]["role"] == "assistant":
            j = i
            while j < len(messages) and messages[j]["role"] == "assistant":
                j += 1
            n = j - i  # batch size

            results: list[int | None] = []
            k = j
            while (
                k < len(messages)
                and messages[k]["role"] == "user"
                and len(results) < n
            ):
                results.append(k)
                k += 1
            results.extend([None] * (n - len(results)))

            for p in range(n):
                pairs.append((i + p, results[p]))
            i = k
        else:
            i += 1
    return pairs


# ── Main processing ───────────────────────────────────────────────────────────

def no_general_plan_messages(
    messages: list[dict],
) -> tuple[list[dict] | None, dict]:
    """
    Remove task-tracker action+result pairs before the first non-task-tracker
    action.

    Returns (filtered_messages, stats_dict).
    Returns (None, stats_dict) when the example should be skipped.
    """
    stats = {
        "skipped_no_non_tt_action": False,
        "tt_pairs_removed": 0,
    }

    all_pairs = build_action_result_pairs(messages)

    # Collect indices of task-tracker actions and their results
    tt_indices: set[int] = set()
    for ai, ri in all_pairs:
        if is_task_tracker(messages[ai]):
            tt_indices.add(ai)
            if ri is not None:
                tt_indices.add(ri)

    # Find the index of the first non-task-tracker assistant action
    first_non_tt: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and i not in tt_indices:
            first_non_tt = i
            break

    if first_non_tt is None:
        stats["skipped_no_non_tt_action"] = True
        return None, stats

    # Count how many TT pairs appear before first_non_tt
    tt_pairs_removed = sum(
        1
        for ai, ri in all_pairs
        if is_task_tracker(messages[ai]) and ai < first_non_tt
    )
    stats["tt_pairs_removed"] = tt_pairs_removed

    # Preamble: all non-assistant messages before first_non_tt
    preamble = [msg for msg in messages[:first_non_tt] if msg.get("role") != "assistant"]

    # Keep everything from first_non_tt onward unchanged
    tail = messages[first_non_tt:]

    filtered = preamble + tail
    return filtered, stats


# ── Dataset processing ────────────────────────────────────────────────────────

def derive_hub_repo(dataset_name: str) -> str:
    base = dataset_name.split(":")[0]
    return f"{base}_no_general_plan"


def process_dataset(dataset_name: str, dry_run: bool = False):
    hub_repo = derive_hub_repo(dataset_name)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    skipped_no_non_tt_action = 0
    total_tt_pairs_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1
        result, stats = no_general_plan_messages(messages)

        if result is None:
            if stats["skipped_no_non_tt_action"]:
                skipped_no_non_tt_action += 1
            continue

        total_tt_pairs_removed += stats["tt_pairs_removed"]
        total_msgs_before += len(messages)
        total_msgs_after += len(result)
        processed_rows.append({**row, "messages": result})

    kept = len(processed_rows)
    total = len(ds)
    skipped = total - kept

    print("\nResults:")
    print(f"  Total trajectories:              {total}")
    print(f"  Kept:                            {kept}")
    print(f"  Skipped total:                   {skipped}")
    print(f"    No non-task-tracker action:    {skipped_no_non_tt_action}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
        print(f"  Total TT pairs removed:          {total_tt_pairs_removed}")
        print(f"  Avg TT pairs removed per traj:   {total_tt_pairs_removed / kept:.2f}")
    print(f"  Task-list JSON fixed:            {task_list_fixed_count} / {total}")
    print(f"  Output repo:                     {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return kept, skipped

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return kept, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="synthetic-code-training/func_localize_claude45_1457i",
        help="HuggingFace dataset name (output repo will be {dataset}_no_general_plan)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report stats without pushing to HuggingFace Hub",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_name=args.dataset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
