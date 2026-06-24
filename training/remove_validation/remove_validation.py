"""
Remove validation steps between the finish action and the last file-editing action.

After an agent completes its last file edit it sometimes runs tests, linters, or
other validation commands before calling finish.  These intermediate steps add
noise without teaching useful editing behaviour.

This script:
  1. Finds the finish action (assistant message containing <function=finish>).
  2. Finds the last file-editing action before the finish action.
     File-editing actions are assistant messages calling file_editor with
       command=str_replace, create_file, or insert.
  3. Keeps everything up to and including that file-editing action and its
     immediately following user feedback (environment result).
  4. Removes all steps between that feedback and the finish action.
  5. Keeps the finish action and any following user feedback.
"""

import argparse
import ast
import json
import re

from datasets import Dataset, load_dataset


FINISH_PATTERN = re.compile(r"<function=finish[\s>]")
FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|create_file|insert)</parameter>"
)

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


def remove_validation_steps(messages: list[dict]) -> tuple[list[dict], int]:
    """
    Remove validation steps between the last file-editing action and the finish action.

    Returns (filtered_messages, steps_removed).
    """
    # Find finish action index
    finish_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if is_finish_action(messages[i]):
            finish_idx = i
            break

    if finish_idx is None:
        return messages, 0

    # Find the last file-editing action before finish
    last_edit_idx = None
    for i in range(finish_idx - 1, -1, -1):
        if is_file_editing_action(messages[i]):
            last_edit_idx = i
            break

    if last_edit_idx is None:
        return messages, 0

    # The environment feedback immediately follows the file-editing action
    feedback_idx = last_edit_idx + 1
    if feedback_idx >= finish_idx:
        # Nothing to remove — finish immediately follows the edit feedback
        return messages, 0

    # Verify the feedback message is a user message
    if messages[feedback_idx].get("role") != "user":
        # Unexpected structure; skip to be safe
        return messages, 0

    cut_start = feedback_idx + 1  # first message to remove
    cut_end = finish_idx  # exclusive end (keep finish onwards)

    steps_removed = cut_end - cut_start
    if steps_removed <= 0:
        return messages, 0

    filtered = messages[:cut_start] + messages[cut_end:]
    return filtered, steps_removed


def derive_hub_repo(dataset_name: str, dataset_size: int) -> str:
    base = dataset_name.split(":")[0]
    base_size = dataset_name.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"remove_validation_{dataset_size}i")
    else:
        return base + f"_remove_validation_{dataset_size}i"


def process_dataset(dataset_name: str, dry_run: bool = False):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    affected_trajectories = 0
    total_steps_removed = 0
    skipped_no_edit = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1
        if not any(is_file_editing_action(m) for m in messages):
            skipped_no_edit += 1
            continue

        filtered_messages, steps_removed = remove_validation_steps(messages)

        if steps_removed > 0:
            affected_trajectories += 1
            total_steps_removed += steps_removed

        processed_rows.append({**row, "messages": filtered_messages})

    hub_repo = derive_hub_repo(dataset_name, dataset_size=len(processed_rows))
    print("\nResults:")
    print(f"  Skipped (no file edit pattern):  {skipped_no_edit} / {len(ds)}")
    print(
        f"  Affected trajectories:           {affected_trajectories} / {len(ds) - skipped_no_edit}"
    )
    if affected_trajectories > 0:
        print(
            f"  Avg steps removed:               {total_steps_removed / affected_trajectories:.2f}"
        )
    print("    (steps between last file edit feedback and finish action)")
    print(f"  Task-list JSON fixed:            {task_list_fixed_count} / {len(ds)}")
    print(f"  Output repo:             {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return (
            affected_trajectories,
            total_steps_removed / affected_trajectories
            if affected_trajectories > 0
            else 0,
        )

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return (
        affected_trajectories,
        total_steps_removed / affected_trajectories if affected_trajectories > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="synthetic-code-training/func_localize_claude45_1457i",
        help="HuggingFace dataset name (output repo will be {dataset}_remove_validation)",
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
