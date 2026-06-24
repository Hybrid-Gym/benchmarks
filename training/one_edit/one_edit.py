"""
Filter trajectories to keep only the file-editing action that edits the target function.

The target function is extracted from a reference dataset (by instance_id) using the
same method as min_explore_v2: the first def/class name found in the old_str of the
last successful str_replace on the target file.

Among all file-editing actions in each base trajectory, only the one whose old_str
starts with the target function/class definition is kept.  All other file-editing
actions and their results are removed.  Everything else (exploration actions,
task-tracker actions, finish actions, etc.) is left intact.

An example is skipped if:
  - the instance_id has no match in the reference dataset
  - the target function/class name cannot be extracted from the reference trajectory
  - no file-editing action in the base trajectory matches the target function

Definitions
-----------
- file-editing action  : assistant message calling file_editor with
                         command=str_replace or insert
- target file          : file edited in the last successful file-editing action
                         of the *reference* trajectory
- target func/class    : first def/class name found in the old_str of the last
                         successful str_replace on the target file in the reference
                         trajectory
- matching edit        : the last successful str_replace in the base trajectory
                         whose old_str's first def/class name equals the target
                         func/class name
"""

import argparse
import ast
import json
import os
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


# ── Message classifiers ───────────────────────────────────────────────────────

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
            n = j - i

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


# ── Target extraction (same logic as min_explore_v2) ─────────────────────────

def get_target_func_name(
    messages: list[dict],
    all_pairs: list[tuple[int, int | None]],
    target_basename: str,
) -> str | None:
    """
    Extract the target function/class name from the last successful str_replace
    on the target file.  Returns None if not found.
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


def extract_ref_target(ref_messages: list[dict]) -> tuple[str | None, str | None]:
    """
    Extract (target_basename, target_func_name) from a reference trajectory.
    Returns (None, None) if either cannot be determined.
    """
    ref_pairs = build_action_result_pairs(ref_messages)

    # Find target file: last successful file-editing action
    target_file: str | None = None
    for ai, ri in reversed(ref_pairs):
        if is_file_editing(ref_messages[ai]):
            result_msg = ref_messages[ri] if ri is not None else None
            if is_edit_successful(result_msg):
                path = get_file_edit_path(ref_messages[ai])
                if path:
                    target_file = path
                    break

    if target_file is None:
        return None, None

    target_basename = os.path.basename(target_file)
    target_func_name = get_target_func_name(ref_messages, ref_pairs, target_basename)
    return target_basename, target_func_name


# ── Main filtering logic ──────────────────────────────────────────────────────

def one_edit_messages(
    base_messages: list[dict],
    target_func_name: str,
) -> tuple[list[dict] | None, dict]:
    """
    Filter base trajectory to keep only the file-editing action that edits the
    target function.

    Returns (filtered_messages, stats_dict).
    Returns (None, stats_dict) when the example should be skipped.
    """
    stats = {
        "skipped_no_matching_edit": False,
        "edits_removed": 0,
    }

    base_pairs = build_action_result_pairs(base_messages)

    # Find the last successful str_replace whose old_str's first def/class name
    # matches the target function name.
    target_edit_ai: int | None = None
    target_edit_ri: int | None = None
    for ai, ri in reversed(base_pairs):
        if not is_file_editing(base_messages[ai]):
            continue
        result_msg = base_messages[ri] if ri is not None else None
        if not is_edit_successful(result_msg):
            continue
        m = FILE_EDIT_OLD_STR_PATTERN.search(base_messages[ai].get("content", ""))
        if m:
            name_m = FUNC_CLASS_NAME_PATTERN.search(m.group(1))
            if name_m and name_m.group(1) == target_func_name:
                target_edit_ai = ai
                target_edit_ri = ri
                break

    if target_edit_ai is None:
        stats["skipped_no_matching_edit"] = True
        return None, stats

    # Collect indices of all file-editing actions (and their results) that are
    # NOT the target edit.
    indices_to_remove: set[int] = set()
    for ai, ri in base_pairs:
        if is_file_editing(base_messages[ai]) and ai != target_edit_ai:
            indices_to_remove.add(ai)
            if ri is not None:
                indices_to_remove.add(ri)
            stats["edits_removed"] += 1

    filtered = [
        msg for i, msg in enumerate(base_messages) if i not in indices_to_remove
    ]
    return filtered, stats


# ── Dataset processing ────────────────────────────────────────────────────────

def derive_hub_repo(base_dataset: str, dataset_size: int) -> str:
    base = base_dataset.split(":")[0]
    base_size = base_dataset.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"one_edit_{dataset_size}i")
    else:
        return base + f"_one_edit_{dataset_size}i"


def process_datasets(
    base_dataset: str,
    ref_dataset: str,
    output_repo: str | None = None,
    dry_run: bool = False,
):
    print(f"Loading base dataset:      {base_dataset}")
    base_ds = load_dataset(base_dataset, split="train")
    print(f"  {len(base_ds)} trajectories")

    print(f"Loading reference dataset: {ref_dataset}")
    ref_ds = load_dataset(ref_dataset, split="train")
    print(f"  {len(ref_ds)} trajectories")

    # Build lookup: instance_id -> reference messages
    ref_by_id: dict[str, list[dict]] = {}
    for row in ref_ds:
        iid = row.get("instance_id")
        if iid is not None:
            ref_by_id[iid] = row["messages"]

    base_ids = {row.get("instance_id") for row in base_ds}
    overlap = base_ids & set(ref_by_id.keys())
    print(f"\nOverlapping instance_ids:  {len(overlap)} / {len(base_ids)} base")

    skipped_no_overlap = 0
    skipped_no_func_name = 0
    skipped_no_matching_edit = 0
    affected = 0
    total_edits_removed = 0
    total_msgs_before = 0
    total_msgs_after = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in base_ds:
        iid = row.get("instance_id")
        messages, tl_changed = fix_messages_task_list(row["messages"])
        if tl_changed:
            task_list_fixed_count += 1

        if iid not in ref_by_id:
            skipped_no_overlap += 1
            continue

        ref_messages = ref_by_id[iid]
        target_basename, target_func_name = extract_ref_target(ref_messages)

        if target_func_name is None:
            skipped_no_func_name += 1
            continue

        result, stats = one_edit_messages(messages, target_func_name)

        if result is None:
            skipped_no_matching_edit += 1
            continue

        if stats["edits_removed"] > 0:
            affected += 1
            total_edits_removed += stats["edits_removed"]

        total_msgs_before += len(messages)
        total_msgs_after += len(result)
        processed_rows.append({**row, "messages": result})

    kept = len(processed_rows)
    total = len(base_ds)
    skipped = total - kept
    hub_repo = output_repo or derive_hub_repo(base_dataset, dataset_size=kept)

    print("\nResults:")
    print(f"  Total trajectories:              {total}")
    print(f"  Kept:                            {kept}")
    print(f"  Skipped total:                   {skipped}")
    print(f"    No overlap with ref dataset:   {skipped_no_overlap}")
    print(f"    No target func in ref:         {skipped_no_func_name}")
    print(f"    No matching edit in base:      {skipped_no_matching_edit}")
    print(f"  Affected (edits removed):        {affected} / {kept}")
    if affected > 0:
        print(f"  Avg edits removed:               {total_edits_removed / affected:.2f}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
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
    parser = argparse.ArgumentParser(
        description="Filter trajectories to keep only the file-editing action that "
                    "edits the target function (extracted from a reference dataset)."
    )
    parser.add_argument(
        "--base-dataset",
        default="synthetic-code-training/func_localize_gpt55_1477i",
        help="HuggingFace dataset name for the base trajectories to filter "
             "(e.g., synthetic-code-training/func_localize_gpt55_1477i)",
    )
    parser.add_argument(
        "--ref-dataset",
        default="synthetic-code-training/func_localize_claude45_1457i",
        help="HuggingFace dataset name for the reference trajectories used to "
             "extract the target function "
             "(e.g., synthetic-code-training/func_localize_claude45_1457i)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo to push the result to "
             "(default: {base_dataset}_one_edit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report stats without pushing to HuggingFace Hub",
    )
    args = parser.parse_args()

    process_datasets(
        base_dataset=args.base_dataset,
        ref_dataset=args.ref_dataset,
        output_repo=args.output_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
