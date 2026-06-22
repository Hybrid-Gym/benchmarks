"""
Minimize trajectories to keep only the essential actions.

All task-tracker actions and their results are removed.  The exploration
phase (everything before the first file-editing action) is minimised to at
most one phase-1 and one phase-2 action-result pair using the same selection
logic as min_explore_v2.  Only one file-editing action is kept: the first
successful edit that adds a docstring to the target file.  Every action after
that edit is discarded, except for the final finish action.

The resulting trajectory therefore contains at most:

  [system / user preamble]
  phase-1 action + result     (optional; skipped if not found)
  phase-2 action + result     (optional; skipped if not found)
  docstring file-edit action + result
  finish action (+ result if present)

An example is skipped entirely if any of the following is missing:
  - a successful file-editing action that adds a docstring
  - a phase-1 action whose result mentions the target-file basename
  - a phase-2 action whose result mentions the target function/class name
  - a finish action

Definitions
-----------
- task-tracker action : assistant message calling <function=task_tracker>
- file-editing action : assistant message calling file_editor with
                        command=str_replace or insert
- docstring edit       : a successful file-editing action whose new_str
                         contains a triple-quoted string (\"\"\" or \'\'\')
- target file          : file edited in the chosen docstring action
- locating action      : last exploration action whose own content contains
                         the basename of the target file
- target func/class    : first def/class name found in the old_str of the
                         last successful str_replace on the target file
- finish action        : assistant message calling <function=finish>
"""

import argparse
import ast
import json
import os
import re

from datasets import Dataset, load_dataset


# ── Regex patterns ────────────────────────────────────────────────────────────
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
FILE_EDIT_NEW_STR_PATTERN = re.compile(
    r"<parameter=new_str>(.*?)</parameter>",
    re.DOTALL,
)
FUNC_CLASS_NAME_PATTERN = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE
)
EDIT_SUCCESS_PATTERN = re.compile(r"has been edited")
DOCSTRING_PATTERN = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
FINISH_PATTERN = re.compile(r"<function=finish[\s>]")
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


# ── Message classifiers ───────────────────────────────────────────────────────

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


def is_finish(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and FINISH_PATTERN.search(msg.get("content", "")) is not None
    )


def get_file_edit_path(msg: dict) -> str | None:
    m = FILE_EDIT_PATH_PATTERN.search(msg.get("content", ""))
    return m.group(1).strip() if m else None


def is_edit_successful(result_msg: dict | None) -> bool:
    if result_msg is None:
        return False
    return EDIT_SUCCESS_PATTERN.search(result_msg.get("content", "")) is not None


def edit_adds_docstring(msg: dict) -> bool:
    """Return True if the file_editor call contains a docstring in new_str."""
    m = FILE_EDIT_NEW_STR_PATTERN.search(msg.get("content", ""))
    if not m:
        return False
    return DOCSTRING_PATTERN.search(m.group(1)) is not None


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


# ── Target helpers ────────────────────────────────────────────────────────────

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


def _result_contains(messages: list[dict], res_idx: int | None, text: str) -> bool:
    if res_idx is None:
        return False
    return text in messages[res_idx].get("content", "")


def strip_text_from_action(msg: dict) -> dict:
    """Return a copy of the message with only function-call blocks; prose stripped."""
    content = msg.get("content", "")
    calls = FUNCTION_CALL_PATTERN.findall(content)
    return {**msg, "content": "\n".join(calls)}


# ── Main processing ───────────────────────────────────────────────────────────

def min_action_messages(
    messages: list[dict],
) -> tuple[list[dict] | None, dict]:
    """
    Minimise a trajectory to the essential actions.

    Returns (filtered_messages, stats_dict).
    Returns (None, stats_dict) when the example should be skipped.
    """
    stats = {
        "phase1_matched": False,
        "phase2_matched": False,
        "skipped_no_docstring_edit": False,
        "skipped_no_finish": False,
        "skipped_no_phase1": False,
        "skipped_no_phase2": False,
    }

    all_pairs = build_action_result_pairs(messages)

    # 1. Remove task-tracker pairs from consideration: collect TT indices
    tt_indices: set[int] = set()
    for ai, ri in all_pairs:
        if is_task_tracker(messages[ai]):
            tt_indices.add(ai)
            if ri is not None:
                tt_indices.add(ri)

    # 2. Find the first successful file-editing action that adds a docstring
    docstring_edit_ai: int | None = None
    docstring_edit_ri: int | None = None
    for ai, ri in all_pairs:
        if ai in tt_indices:
            continue
        if not is_file_editing(messages[ai]):
            continue
        result_msg = messages[ri] if ri is not None else None
        if not is_edit_successful(result_msg):
            continue
        if not edit_adds_docstring(messages[ai]):
            continue
        docstring_edit_ai = ai
        docstring_edit_ri = ri
        break

    if docstring_edit_ai is None:
        stats["skipped_no_docstring_edit"] = True
        return None, stats

    target_file = get_file_edit_path(messages[docstring_edit_ai])
    if target_file is None:
        stats["skipped_no_docstring_edit"] = True
        return None, stats

    target_basename = os.path.basename(target_file)

    # 3. Find the finish action (last assistant finish message)
    finish_ai: int | None = None
    finish_ri: int | None = None
    for ai, ri in all_pairs:
        if is_finish(messages[ai]):
            finish_ai = ai
            finish_ri = ri

    if finish_ai is None:
        stats["skipped_no_finish"] = True
        return None, stats

    # 4. Find target function/class name (from the docstring edit itself)
    target_func_name = get_target_func_name(
        messages, all_pairs, target_basename
    )

    # 5. Identify the preamble: everything before the first non-TT assistant action
    first_non_tt: int | None = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and i not in tt_indices:
            first_non_tt = i
            break

    if first_non_tt is None:
        stats["skipped_no_phase1"] = True
        return None, stats

    # 6. Exploration pairs: from first_non_tt up to (but not including) the
    #    docstring edit action, excluding task-tracker actions.
    explore_pairs = [
        (ai, ri)
        for ai, ri in all_pairs
        if first_non_tt <= ai < docstring_edit_ai and ai not in tt_indices
    ]

    # 7. Find the first locating action in the exploration range
    first_locates_ai: int | None = None
    for ai, ri in explore_pairs:
        if target_basename in messages[ai].get("content", ""):
            first_locates_ai = ai
            break

    # 8. Select phase-1 and phase-2 pairs
    phase1_pair: tuple[int, int | None] | None = None
    phase2_pair: tuple[int, int | None] | None = None

    if first_locates_ai is not None:
        phase1_candidates = [(ai, ri) for ai, ri in explore_pairs if ai < first_locates_ai]
        phase2_candidates = [(ai, ri) for ai, ri in explore_pairs if ai >= first_locates_ai]

        # Phase 1: last action whose result contains target_basename
        for ai, ri in phase1_candidates:
            if _result_contains(messages, ri, target_basename):
                phase1_pair = (ai, ri)

        # Phase 2: last action whose result contains target_func_name
        if target_func_name:
            for ai, ri in phase2_candidates:
                if _result_contains(messages, ri, target_func_name):
                    phase2_pair = (ai, ri)
    else:
        # No locating action: apply phase-1 rule to the entire explore range
        for ai, ri in explore_pairs:
            if _result_contains(messages, ri, target_basename):
                phase1_pair = (ai, ri)
        # phase2 stays None

    # 9. Require both phase-1 and phase-2 matches; skip otherwise
    if phase1_pair is None:
        stats["skipped_no_phase1"] = True
        return None, stats

    if phase2_pair is None:
        stats["skipped_no_phase2"] = True
        return None, stats

    stats["phase1_matched"] = True
    stats["phase2_matched"] = True

    # 10. Build the final message list
    preamble = messages[:first_non_tt]

    def keep_pair(ai: int, ri: int | None) -> list[dict]:
        action_msg = strip_text_from_action(messages[ai])
        result_msgs = [messages[ri]] if ri is not None else []
        return [action_msg] + result_msgs

    p1_msgs = keep_pair(*phase1_pair)
    p2_msgs = keep_pair(*phase2_pair)
    edit_msgs = keep_pair(docstring_edit_ai, docstring_edit_ri)

    # Finish action (keep as-is, with its result if present)
    finish_msgs = keep_pair(finish_ai, finish_ri)

    filtered = preamble + p1_msgs + p2_msgs + edit_msgs + finish_msgs

    return filtered, stats


# ── Dataset processing ────────────────────────────────────────────────────────

def derive_hub_repo(dataset_name: str) -> str:
    base = dataset_name.split(":")[0]
    return f"{base}_min_action"


def process_dataset(dataset_name: str, dry_run: bool = False):
    hub_repo = derive_hub_repo(dataset_name)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    skipped_no_docstring_edit = 0
    skipped_no_finish = 0
    skipped_no_phase1 = 0
    skipped_no_phase2 = 0
    phase1_matched_count = 0
    phase2_matched_count = 0
    total_msgs_before = 0
    total_msgs_after = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1
        result, stats = min_action_messages(messages)

        if result is None:
            if stats["skipped_no_docstring_edit"]:
                skipped_no_docstring_edit += 1
            elif stats["skipped_no_finish"]:
                skipped_no_finish += 1
            elif stats["skipped_no_phase1"]:
                skipped_no_phase1 += 1
            elif stats["skipped_no_phase2"]:
                skipped_no_phase2 += 1
            continue

        if stats["phase1_matched"]:
            phase1_matched_count += 1
        if stats["phase2_matched"]:
            phase2_matched_count += 1

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
    print(f"    No docstring file-edit:        {skipped_no_docstring_edit}")
    print(f"    No finish action:              {skipped_no_finish}")
    print(f"    No phase-1 match:              {skipped_no_phase1}")
    print(f"    No phase-2 match:              {skipped_no_phase2}")
    if kept > 0:
        print(f"  Avg messages before:             {total_msgs_before / kept:.2f}")
        print(f"  Avg messages after:              {total_msgs_after / kept:.2f}")
    print(f"  Phase 1 action found:            {phase1_matched_count} / {kept}")
    print(f"  Phase 2 action found:            {phase2_matched_count} / {kept}")
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
        help="HuggingFace dataset name (output repo will be {dataset}_min_action)",
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
