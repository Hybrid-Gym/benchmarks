"""
Minimize the exploration phase of trajectories (v2).

The exploration phase is split into two sub-phases, separated by the first
action that *locates* the target file (i.e. the target file's basename appears
in the action content itself):

  Phase 1 – [first_non_tt_action, first_locating_action)
    Only keep the *last* action whose result mentions the target file basename.
    If no such action exists, skip this step (keep nothing in phase 1).

  Phase 2 – [first_locating_action, first_file_edit_action)
    Only keep the *last* action whose result mentions the target
    class/function name.
    If no such action exists, skip this step (keep nothing in phase 2).

If no action in the exploration range locates the target file, Phase 1 is
applied to the entire exploration range and Phase 2 is omitted.

Definitions
-----------
- task-tracker action   : assistant message calling <function=task_tracker>
- file-editing action   : assistant message calling file_editor with
                          command=str_replace or insert
- target file           : file edited in the last successful file-editing action
- locating action       : last action in the exploration range whose own
                          content contains the basename of the target file
- target func/class name: first def/class name found in the old_str of the
                          last successful str_replace on the target file
"""

import argparse
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
FILE_EDIT_OLD_STR_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>str_replace</parameter>\n.*?<parameter=old_str>(.*?)</parameter>",
    re.DOTALL,
)
FUNC_CLASS_NAME_PATTERN = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE
)
EDIT_SUCCESS_PATTERN = re.compile(r"has been edited")
FUNCTION_CALL_PATTERN = re.compile(r"<function=\w+>.*?</function>", re.DOTALL)


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


def get_target_func_name(
    messages: list[dict], all_pairs: list[tuple[int, int | None]], target_basename: str
) -> str | None:
    """
    Extract the target function/class name from the last successful str_replace
    action on the target file.  Returns None if not found.
    """
    for act_idx, res_idx in reversed(all_pairs):
        if not is_file_editing(messages[act_idx]):
            continue
        result_msg = messages[res_idx] if res_idx is not None else None
        if not is_edit_successful(result_msg):
            continue
        path = get_file_edit_path(messages[act_idx])
        if path is None or os.path.basename(path) != target_basename:
            continue
        m = FILE_EDIT_OLD_STR_PATTERN.search(messages[act_idx].get("content", ""))
        if m:
            name_m = FUNC_CLASS_NAME_PATTERN.search(m.group(1))
            if name_m:
                return name_m.group(1)
    return None


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


def min_explore_messages_v2(
    messages: list[dict],
) -> tuple[list[dict], int, bool, bool]:
    """
    Minimize the exploration phase of a trajectory (v2 strategy).

    Returns (filtered_messages, steps_removed, phase1_matched, phase2_matched).
    Only the action messages are kept (results/tool-outputs are stripped).
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

    # 2. Find target function/class name
    target_func_name = get_target_func_name(messages, all_pairs, target_basename)

    # 3. Find first non-task-tracker assistant action
    first_non_tt: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and not is_task_tracker(msg):
            first_non_tt = i
            break

    if first_non_tt is None:
        return messages, 0, False, False

    # 4. Find first file-editing action
    first_file_edit: int | None = None
    for i, msg in enumerate(messages):
        if is_file_editing(msg):
            first_file_edit = i
            break

    if first_file_edit is None or first_file_edit <= first_non_tt:
        return messages, 0, False, False

    # 5. All (action, result) pairs in the exploration range
    explore_pairs = [
        (ai, ri)
        for ai, ri in all_pairs
        if first_non_tt <= ai < first_file_edit
    ]

    # 6. Find first action in the exploration range that *locates* the target
    #    file (target_basename appears in the action content itself).
    first_locates_ai: int | None = None
    for ai, ri in explore_pairs:
        if target_basename in messages[ai].get("content", ""):
            first_locates_ai = ai
            break

    keep: set[int] = set()
    extra: list[dict] = []
    phase1_matched = False
    phase2_matched = False

    if first_locates_ai is not None:
        # ---- Phase 1: [first_non_tt, first_locates_ai) ----
        phase1_pairs = [(ai, ri) for ai, ri in explore_pairs if ai < first_locates_ai]

        # Keep the *last* action whose result contains the target basename
        matched_p1: tuple[int, int | None] | None = None
        for ai, ri in phase1_pairs:
            if _result_contains(messages, ri, target_basename):
                matched_p1 = (ai, ri)  # keep updating to get the last one

        if matched_p1 is not None:
            _add_pair_to_keep(
                keep, extra, messages,
                matched_p1[0], matched_p1[1],
                first_non_tt, first_file_edit,
            )
            phase1_matched = True
        else:
            # No match — keep all steps in phase 1
            for ai, ri in phase1_pairs:
                _add_pair_to_keep(
                    keep, extra, messages,
                    ai, ri,
                    first_non_tt, first_file_edit,
                )

        # ---- Phase 2: [first_locates_ai, first_file_edit) ----
        phase2_pairs = [(ai, ri) for ai, ri in explore_pairs if ai >= first_locates_ai]

        if target_func_name:
            # Keep the *first* action whose result contains the target func name
            matched_p2: tuple[int, int | None] | None = None
            for ai, ri in phase2_pairs:
                if _result_contains(messages, ri, target_func_name):
                    matched_p2 = (ai, ri)  # keep updating to get the last one 
                    
            if matched_p2 is not None:
                _add_pair_to_keep(
                    keep, extra, messages,
                    matched_p2[0], matched_p2[1],
                    first_non_tt, first_file_edit,
                )
                phase2_matched = True
            else:
                # No function name extracted — keep all steps in phase 2
                for ai, ri in phase2_pairs:
                    _add_pair_to_keep(
                        keep, extra, messages,
                        ai, ri,
                        first_non_tt, first_file_edit,
                    )

    else:
        # No locating action found: apply phase-1 rule to entire exploration range
        matched: tuple[int, int | None] | None = None
        for ai, ri in explore_pairs:
            if _result_contains(messages, ri, target_basename):
                matched = (ai, ri)  # last match

        if matched is not None:
            _add_pair_to_keep(
                keep, extra, messages,
                matched[0], matched[1],
                first_non_tt, first_file_edit,
            )
            phase1_matched = True
        else:
            # No match — keep all steps in the exploration range
            for ai, ri in explore_pairs:
                _add_pair_to_keep(
                    keep, extra, messages,
                    ai, ri,
                    first_non_tt, first_file_edit,
                )

    pre = messages[:first_non_tt]
    mid = [
        strip_text_from_action(messages[i]) if messages[i]["role"] == "assistant" else messages[i]
        for i in range(first_non_tt, first_file_edit)
        if i in keep
    ]
    post = messages[first_file_edit:]

    filtered = pre + mid + extra + post
    steps_removed = (first_file_edit - first_non_tt) - len(mid)

    return filtered, steps_removed, phase1_matched, phase2_matched


def derive_hub_repo(dataset_name: str) -> str:
    base = dataset_name.split(":")[0]
    return f"{base}_min_explore_v2"


def process_dataset(dataset_name: str, dry_run: bool = False):
    hub_repo = derive_hub_repo(dataset_name)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    skipped_no_edit = 0
    affected_trajectories = 0
    total_steps_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    phase1_matched_count = 0
    phase2_matched_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]

        # Skip examples with no file-editing action at all
        if not any(is_file_editing(m) for m in messages):
            skipped_no_edit += 1
            continue

        result, steps_removed, p1_matched, p2_matched = min_explore_messages_v2(messages)

        if steps_removed > 0:
            affected_trajectories += 1
            total_steps_removed += steps_removed
        if p1_matched:
            phase1_matched_count += 1
        if p2_matched:
            phase2_matched_count += 1

        total_msgs_before += len(messages)
        total_msgs_after += len(result)
        processed_rows.append({**row, "messages": result})

    kept = len(processed_rows)
    print("\nResults:")
    print(f"  Skipped (no file-edit action):   {skipped_no_edit} / {len(ds)}")
    print(f"  Kept trajectories:               {kept}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
    print(f"  Phase 1 action found:            {phase1_matched_count} / {kept}")
    print(f"  Phase 2 action found:            {phase2_matched_count} / {kept}")
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
        help="HuggingFace dataset name (output repo will be {dataset}_min_explore_v2)",
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
