"""
Minimize the file exploration phase of trajectories.

In the exploration range [first_non_tt_action, first_file_edit_action) we keep:

  (1) The *last* action whose result mentions the target file basename
      (the last successful file-localization action).
      If no such action exists, nothing is kept.

  (2) After that last file-localization action, all remaining actions
      (function-localization steps).

This preserves the function-localization steps as faithfully as possible while
discarding earlier redundant file-search steps.

Definitions
-----------
- task-tracker action   : assistant message calling <function=task_tracker>
- file-editing action   : assistant message calling file_editor with
                          command=str_replace or insert
- target file           : file edited in the last successful file-editing action
"""

import argparse
import ast
import json
import os
import re

from datasets import Dataset, load_dataset


TASK_TRACKER_PATTERN = re.compile(r"<function=task_tracker[\s>]")
FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|insert)</parameter>"
)
FILE_EDIT_PATH_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(?:str_replace|insert)</parameter>\n<parameter=path>(.*?)</parameter>",
    re.DOTALL,
)
EDIT_SUCCESS_PATTERN = re.compile(r"has been edited")
FUNCTION_CALL_PATTERN = re.compile(r"<function=\w+>.*?</function>", re.DOTALL)

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


def _result_contains(messages: list[dict], res_idx: int | None, text: str) -> bool:
    if res_idx is None:
        return False
    return text in messages[res_idx].get("content", "")


def strip_text_from_action(msg: dict) -> dict:
    """Return a copy of the message with only function-call blocks; prose stripped."""
    content = msg.get("content", "")
    calls = FUNCTION_CALL_PATTERN.findall(content)
    return {**msg, "content": "\n".join(calls)}


def _add_pair_to_keep(
    keep: set[int],
    extra: list[dict],
    messages: list[dict],
    ai: int,
    ri: int | None,
    range_start: int,
    range_end: int,
) -> None:
    """Add action and its result to the keep set or extra list."""
    keep.add(ai)
    if ri is not None:
        if range_start <= ri < range_end:
            keep.add(ri)
        elif ri >= range_end:
            extra.append(messages[ri])


def min_file_explore(
    messages: list[dict],
) -> tuple[list[dict], int, bool, bool]:
    """
    Minimize the file exploration phase of a trajectory.

    Returns (filtered_messages, steps_removed, file_loc_matched, func_loc_kept).
    Only the action messages are kept (results/tool-outputs are stripped).

    In [first_non_tt, first_file_edit) we keep:
      (1) The last action whose result mentions the target file basename.
      (2) After that last file-localization action, all remaining actions
          (function-localization steps).
    """
    all_pairs = build_action_result_pairs(messages)

    # 1. Find the target file: last successful file-editing action
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
        return messages, 0, False, False

    target_basename = os.path.basename(target_file)

    # 2. Find first non-task-tracker assistant action
    first_non_tt: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and not is_task_tracker(msg):
            first_non_tt = i
            break

    if first_non_tt is None:
        return messages, 0, False, False

    # 3. Find first file-editing action
    first_file_edit: int | None = None
    for i, msg in enumerate(messages):
        if is_file_editing(msg):
            first_file_edit = i
            break

    if first_file_edit is None or first_file_edit <= first_non_tt:
        return messages, 0, False, False

    # 4. All (action, result) pairs in the exploration range
    explore_pairs = [
        (ai, ri)
        for ai, ri in all_pairs
        if first_non_tt <= ai < first_file_edit
    ]

    keep: set[int] = set()
    extra: list[dict] = []
    file_loc_matched = False
    func_loc_kept = False

    # (1) Last action whose result mentions target_basename
    last_file_loc: tuple[int, int | None] | None = None
    for ai, ri in explore_pairs:
        if _result_contains(messages, ri, target_basename):
            last_file_loc = (ai, ri)

    if last_file_loc is not None:
        _add_pair_to_keep(
            keep, extra, messages,
            last_file_loc[0], last_file_loc[1],
            first_non_tt, first_file_edit,
        )
        file_loc_matched = True

        # (2) After last_file_loc: keep all actions
        phase2_pairs = [(ai, ri) for ai, ri in explore_pairs if ai > last_file_loc[0]]
        for ai, ri in phase2_pairs:
            _add_pair_to_keep(
                keep, extra, messages,
                ai, ri,
                first_non_tt, first_file_edit,
            )
            func_loc_kept = True

    pre = messages[:first_non_tt]
    mid = [
        strip_text_from_action(messages[i]) if messages[i]["role"] == "assistant" else messages[i]
        for i in range(first_non_tt, first_file_edit)
        if i in keep
    ]
    post = messages[first_file_edit:]

    filtered = pre + mid + extra + post
    steps_removed = (first_file_edit - first_non_tt) - len(mid)

    return filtered, steps_removed, file_loc_matched, func_loc_kept

def derive_hub_repo(base_dataset: str, dataset_size: int) -> str:
    base = base_dataset.split(":")[0]
    base_size = base_dataset.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"min_file_explore_{dataset_size}i")
    else:
        return base + f"_min_file_explore_{dataset_size}i"


def process_dataset(dataset_name: str, dry_run: bool = False):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    skipped_no_edit = 0
    affected_trajectories = 0
    total_steps_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    file_loc_matched_count = 0
    func_loc_kept_count = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1

        # Skip examples with no file-editing action at all
        if not any(is_file_editing(m) for m in messages):
            skipped_no_edit += 1
            continue

        result, steps_removed, p1_matched, p2_matched = min_file_explore(messages)

        if steps_removed > 0:
            affected_trajectories += 1
            total_steps_removed += steps_removed
        if p1_matched:
            file_loc_matched_count += 1
        if p2_matched:
            func_loc_kept_count += 1

        total_msgs_before += len(messages)
        total_msgs_after += len(result)
        processed_rows.append({**row, "messages": result})
        
    hub_repo = derive_hub_repo(dataset_name, len(processed_rows))

    kept = len(processed_rows)
    print("\nResults:")
    print(f"  Skipped (no file-edit action):   {skipped_no_edit} / {len(ds)}")
    print(f"  Kept trajectories:               {kept}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
    print(f"  File-loc action found:           {file_loc_matched_count} / {kept}")
    print(f"  Func-loc steps kept:             {func_loc_kept_count} / {kept}")
    print(f"  Affected (steps removed):        {affected_trajectories} / {kept}")
    if affected_trajectories > 0:
        print(
            f"  Avg steps removed:               {total_steps_removed / affected_trajectories:.2f}"
        )
    print("    (steps between first exploration action and first file edit)")
    print(f"  Task-list JSON fixed:            {task_list_fixed_count} / {len(ds)}")
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
        help="HuggingFace dataset name (output repo will be {dataset}_min_file_explore)",
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
