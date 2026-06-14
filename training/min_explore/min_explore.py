"""
Minimize the exploration phase of trajectories.

The exploration phase is the range of steps between the first non-task-tracker
action and the first file-editing action (exclusive).

Within this range, only keep:
  1. The "file searching action" whose result contains the fewest files and
     (target file = the file edited in the last successful file-editing action).
  2. Any actions after that searching action that view the target file directly.

An example is skipped (excluded from the output dataset) if no file searching
action in the exploration range leads to the target file.

Definitions
-----------
- task-tracker action: assistant message calling <function=task_tracker>
- file-editing action: assistant message calling file_editor with
    command=str_replace, create_file, or insert
- file searching action: an assistant bash action running a keyword search tool
    (grep, find, rg, ag, ack, fzf) whose result contains the target file path
- view of target file: file_editor with command=view and path=<target_file>
"""

import argparse
import os
import re

from datasets import Dataset, load_dataset


TASK_TRACKER_PATTERN = re.compile(r"<function=task_tracker[\s>]")
FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|insert)</parameter>"
)
# Matches bash executions of keyword searching tools (grep, find, rg, ag, ack, fzf)
KEYWORD_SEARCH_PATTERN = re.compile(
    r"<function=terminal>.*?<parameter=command>[^\n]*\b(grep|find|rg|ag|ack|fzf)\b",
    re.DOTALL,
)
FILE_VIEW_PATH_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>view</parameter>\n<parameter=path>(.*?)</parameter>",
    re.DOTALL,
)
FILE_EDIT_PATH_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(?:str_replace|insert)</parameter>\n<parameter=path>(.*?)</parameter>",
    re.DOTALL,
)
EDIT_SUCCESS_PATTERN = re.compile(r"has been edited")


def is_task_tracker(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and TASK_TRACKER_PATTERN.search(msg.get("content", "")) is not None
    )


def is_file_editing(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and FILE_EDIT_PATTERN.search(msg.get("content", "")) is not None
    )


def get_file_edit_path(msg: dict) -> str | None:
    m = FILE_EDIT_PATH_PATTERN.search(msg.get("content", ""))
    return m.group(1).strip() if m else None


def get_file_view_path(msg: dict) -> str | None:
    m = FILE_VIEW_PATH_PATTERN.search(msg.get("content", ""))
    return m.group(1).strip() if m else None


def is_edit_successful(result_msg: dict | None) -> bool:
    if result_msg is None:
        return False
    return EDIT_SUCCESS_PATTERN.search(result_msg.get("content", "")) is not None


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
            # Collect consecutive assistant messages
            j = i
            while j < len(messages) and messages[j]["role"] == "assistant":
                j += 1
            n = j - i  # batch size

            # Collect up to n following user messages
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


def min_explore_messages(
    messages: list[dict],
) -> tuple[list[dict] | None, int]:
    """
    Minimize the exploration phase of a trajectory.

    Returns (filtered_messages, steps_removed) or (None, 0) when the example
    should be skipped.
    """
    all_pairs = build_action_result_pairs(messages)

    # 1. Find the target file: from the last successful file-editing action
    target_file: str | None = None
    for act_idx, res_idx in reversed(all_pairs):
        if is_file_editing(messages[act_idx]):
            result_msg = messages[res_idx] if res_idx is not None else None
            if is_edit_successful(result_msg):
                path = get_file_edit_path(messages[act_idx])
                if path:
                    target_file = path
                    break

    if target_file is None:
        return messages, 0

    target_basename = os.path.basename(target_file)

    # 2. Find the first non-task-tracker assistant action
    first_non_tt: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and not is_task_tracker(msg):
            first_non_tt = i
            break

    if first_non_tt is None:
        return messages, 0

    # 3. Find the first file-editing action
    first_file_edit: int | None = None
    for i, msg in enumerate(messages):
        if is_file_editing(msg):
            first_file_edit = i
            break

    if first_file_edit is None or first_file_edit <= first_non_tt:
        return messages, 0

    # 4. Collect (action, result) pairs in the exploration range
    explore_pairs = [
        (ai, ri)
        for ai, ri in all_pairs
        if first_non_tt <= ai < first_file_edit
    ]

    # 5. Helpers
    def is_target_view(act_idx: int) -> bool:
        """True if action is a file_editor view of the target file (matched by basename)."""
        path = get_file_view_path(messages[act_idx])
        return path is not None and os.path.basename(path) == target_basename

    def result_has_target(res_idx: int | None) -> bool:
        """True if the result message mentions the target file (matched by basename)."""
        if res_idx is None:
            return False
        content = messages[res_idx].get("content", "")
        return target_basename in content

    def is_keyword_search(act_idx: int) -> bool:
        """True if action is a bash execution of a keyword searching tool (grep, find, rg, ...)."""
        return KEYWORD_SEARCH_PATTERN.search(messages[act_idx].get("content", "")) is not None

    # 6. Find file searching actions: in range, not a direct target view,
    #    a keyword search command, and result mentions the target file
    searching_pairs = [
        (ai, ri)
        for ai, ri in explore_pairs
        if not is_target_view(ai) and is_keyword_search(ai) and result_has_target(ri)
    ]

    # 7. Skip if no searching action leads to the target file
    if not searching_pairs:
        # return None, 0 
        return messages, 0

    def count_result_files(res_idx: int | None) -> int:
        """Count unique file paths in a keyword search result (grep/find output)."""
        if res_idx is None:
            return 0
        paths: set[str] = set()
        for line in messages[res_idx].get("content", "").splitlines():
            line = line.strip()
            if not line:
                continue
            # grep: "path/to/file:linenum:content", find: "path/to/file"
            candidate = line.split(":")[0]
            if "/" in candidate:
                paths.add(candidate)
        return len(paths)

    # 8. Take the searching action whose result contains the fewest files
    last_search_ai, last_search_ri = min(searching_pairs, key=lambda p: count_result_files(p[1]))

    # 9. Collect view-of-target-file pairs AFTER the last searching action
    view_pairs = [
        (ai, ri)
        for ai, ri in explore_pairs
        if ai > last_search_ai and is_target_view(ai)
    ]

    # 10. Build the set of message indices to keep within the exploration range
    keep: set[int] = set()
    keep.add(last_search_ai)
    if last_search_ri is not None and first_non_tt <= last_search_ri < first_file_edit:
        keep.add(last_search_ri)
    for ai, ri in view_pairs:
        keep.add(ai)
        if ri is not None and first_non_tt <= ri < first_file_edit:
            keep.add(ri)

    # Handle results that fall outside the exploration range (batched edge case)
    extra: list[dict] = []
    if last_search_ri is not None and last_search_ri >= first_file_edit:
        extra.append(messages[last_search_ri])
    for ai, ri in view_pairs:
        if ri is not None and ri >= first_file_edit:
            extra.append(messages[ri])

    pre = messages[:first_non_tt]
    mid = [messages[i] for i in range(first_non_tt, first_file_edit) if i in keep]
    post = messages[first_file_edit:]

    filtered = pre + mid + extra + post
    steps_removed = (first_file_edit - first_non_tt) - len(mid)

    return filtered, steps_removed


def derive_hub_repo(dataset_name: str) -> str:
    base = dataset_name.split(":")[0]
    return f"{base}_min_explore"


def process_dataset(dataset_name: str, dry_run: bool = False):
    hub_repo = derive_hub_repo(dataset_name)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    skipped_no_search = 0
    skipped_no_edit = 0
    affected_trajectories = 0
    total_steps_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]

        # Skip examples with no file-editing action at all
        if not any(is_file_editing(m) for m in messages):
            skipped_no_edit += 1
            continue

        result, steps_removed = min_explore_messages(messages)

        if result is None:
            # No searching action leads to the target file — skip example
            skipped_no_search += 1
            continue

        if steps_removed > 0:
            affected_trajectories += 1
            total_steps_removed += steps_removed

        total_msgs_before += len(messages)
        total_msgs_after += len(result)
        processed_rows.append({**row, "messages": result})

    kept = len(processed_rows)
    print("\nResults:")
    print(f"  Skipped (no file-edit action):   {skipped_no_edit} / {len(ds)}")
    print(f"  Skipped (no search → target):    {skipped_no_search} / {len(ds)}")
    print(f"  Kept trajectories:               {kept}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
    print(f"  Affected (steps removed):        {affected_trajectories} / {kept}")
    if affected_trajectories > 0:
        print(
            f"  Avg steps removed:               {total_steps_removed / affected_trajectories:.2f}"
        )
    print("    (steps between first exploration action and first file edit)")
    print(f"  Output repo:                     {hub_repo}")

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
        help="HuggingFace dataset name (output repo will be {dataset}_min_explore)",
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
