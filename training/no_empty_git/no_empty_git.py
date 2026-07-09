"""
Remove git-diff and git-status actions (and their feedback) that appear before
the first successful file-editing action.

Agents often run `git diff` / `git status` early in a trajectory to inspect the
workspace before making any edits.  These read-only terminal calls add noise
without teaching useful behaviour.

This script strips every such pair that occurs strictly before the first
*successful* file-editing action.  Everything from that successful edit onward
is left untouched.

Definitions
-----------
- git-status/diff action : assistant message calling <function=terminal> whose
    command parameter contains `git diff` or `git status`
- file-editing action : assistant message calling file_editor with
    command = str_replace | create_file | insert
- successful : the immediately following user feedback does NOT contain "ERROR"
- pair : (action message, its result/feedback message)

An example is kept unchanged when there is no successful file-editing action
(nothing to remove before it).
"""

import argparse
import ast
import json
import re

from datasets import Dataset, load_dataset


# ── Regex patterns ────────────────────────────────────────────────────────────

TERMINAL_CMD_PATTERN = re.compile(
    r"<function=terminal>\s*<parameter=command>(.*?)</parameter>",
    re.DOTALL,
)
GIT_CMD_RE = re.compile(r"\bgit\s+(diff|status)\b")

FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|create_file|insert)</parameter>"
)
ERROR_PATTERN = re.compile(r"\bERROR\b", re.IGNORECASE)

_TASK_LIST_PARAM_RE = re.compile(
    r'(<parameter=task_list>)(.*?)(</parameter>)',
    re.DOTALL,
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

def is_git_status_diff(msg: dict) -> bool:
    """True if msg is an assistant terminal call containing git diff or git status."""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content", "")
    for m in TERMINAL_CMD_PATTERN.finditer(content):
        if GIT_CMD_RE.search(m.group(1)):
            return True
    return False


def is_file_editing_action(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and FILE_EDIT_PATTERN.search(msg.get("content", "")) is not None
    )


def is_error_feedback(msg: dict) -> bool:
    """True if a user feedback message contains an error indicator."""
    return (
        msg.get("role") == "user"
        and ERROR_PATTERN.search(msg.get("content", "")) is not None
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

def no_empty_git_messages(
    messages: list[dict],
) -> tuple[list[dict], dict]:
    """
    Remove git-diff/status action+result pairs before the first successful
    file-editing action.

    Returns (filtered_messages, stats_dict).
    The messages are returned unchanged when there is no successful file edit.
    """
    stats = {
        "no_successful_edit": False,
        "git_pairs_removed": 0,
    }

    all_pairs = build_action_result_pairs(messages)

    # Find the action index of the first successful file-editing action
    first_successful_edit: int | None = None
    for ai, ri in all_pairs:
        if is_file_editing_action(messages[ai]):
            # Successful = feedback exists and has no ERROR
            if ri is not None and not is_error_feedback(messages[ri]):
                first_successful_edit = ai
                break

    if first_successful_edit is None:
        stats["no_successful_edit"] = True
        return messages, stats

    # Collect indices to remove: git diff/status pairs before first_successful_edit
    remove_indices: set[int] = set()
    for ai, ri in all_pairs:
        if ai >= first_successful_edit:
            break
        if is_git_status_diff(messages[ai]):
            remove_indices.add(ai)
            if ri is not None:
                remove_indices.add(ri)

    stats["git_pairs_removed"] = sum(
        1 for ai, _ in all_pairs
        if ai < first_successful_edit and is_git_status_diff(messages[ai])
    )

    if not remove_indices:
        return messages, stats

    filtered = [msg for i, msg in enumerate(messages) if i not in remove_indices]
    return filtered, stats


# ── Dataset processing ────────────────────────────────────────────────────────

def derive_hub_repo(dataset_name: str, dataset_size: int) -> str:
    base = dataset_name.split(":")[0]
    base_size = dataset_name.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"no_empty_git_{dataset_size}i")
    else:
        return base + f"_no_empty_git_{dataset_size}i"


def process_dataset(dataset_name: str, dry_run: bool = False):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    no_successful_edit_count = 0
    affected_trajectories = 0
    total_git_pairs_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1

        filtered, stats = no_empty_git_messages(messages)

        if stats["no_successful_edit"]:
            no_successful_edit_count += 1

        if stats["git_pairs_removed"] > 0:
            affected_trajectories += 1
            total_git_pairs_removed += stats["git_pairs_removed"]

        total_msgs_before += len(messages)
        total_msgs_after += len(filtered)
        processed_rows.append({**row, "messages": filtered})

    kept = len(processed_rows)
    total = len(ds)
    hub_repo = derive_hub_repo(dataset_name, dataset_size=kept)

    print("\nResults:")
    print(f"  Total trajectories:                    {total}")
    print(f"  No successful file edit (unchanged):   {no_successful_edit_count}")
    print(f"  Affected (git pairs removed):          {affected_trajectories}")
    if affected_trajectories > 0:
        print(f"  Total git pairs removed:               {total_git_pairs_removed}")
        print(f"  Avg git pairs removed per affected:    {total_git_pairs_removed / affected_trajectories:.2f}")
    print(f"  Avg messages before:                   {total_msgs_before / total:.2f}")
    print(f"  Avg messages after:                    {total_msgs_after / total:.2f}")
    print(f"  Task-list JSON fixed:                  {task_list_fixed_count} / {total}")
    print(f"  Output repo:                           {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return kept, affected_trajectories

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return kept, affected_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="synthetic-code-training/func_localize_gpt55_1477i",
        help="HuggingFace dataset name (output repo replaces size suffix with no_empty_git_{size}i)",
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
