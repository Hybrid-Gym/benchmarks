"""
Add keyword searching and directory listing actions from a reference dataset into a base dataset.

For each instance that appears in both datasets (matched by instance_id):
  1. From the reference trajectory, collect all keyword searching and directory
     listing action+result pairs that appear before the first file-editing action.
  2. In the base trajectory, locate the insertion point:
       - If there is at least one exploration action (keyword search or dir listing)
         before the first file-view/edit action, insert the reference actions
         immediately before that first exploration action.
       - Otherwise, insert immediately before the first file-view/edit action.
  3. When inserting, skip any reference action that duplicates one already in the base:
       - Keyword search: skip if the same keyword is already searched.
       - Directory listing: skip if the same directory is already listed.

Definitions
-----------
- file-editing action  : assistant message calling file_editor with
    command=str_replace, create_file, or insert
- file-view/edit action: assistant message calling file_editor with any command,
    OR a terminal action that views a specific file (file_editor view)
- keyword searching action: an assistant terminal action running grep/find/rg/ag/ack/fzf
- directory listing action: an assistant terminal action running ls
"""

import argparse
import re

from datasets import Dataset, load_dataset


TASK_TRACKER_PATTERN = re.compile(r"<function=task_tracker[\s>]")
FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|create_file|insert)</parameter>"
)
FILE_VIEW_OR_EDIT_PATTERN = re.compile(
    r"<function=file_editor>"
)
# Matches bash executions of keyword searching tools (grep, find, rg, ag, ack, fzf)
KEYWORD_SEARCH_PATTERN = re.compile(
    r"<function=terminal>.*?<parameter=command>[^\n]*\b(grep|find|rg|ag|ack|fzf)\b",
    re.DOTALL,
)
# Matches bash executions of directory listing (ls)
DIR_LIST_PATTERN = re.compile(
    r"<function=terminal>.*?<parameter=command>[^\n]*\bls\b",
    re.DOTALL,
)
# Extract the keyword/pattern argument from a search command
SEARCH_KEYWORD_PATTERN = re.compile(
    r"<parameter=command>(.*?)</parameter>",
    re.DOTALL,
)


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


def is_file_view_or_edit(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and FILE_VIEW_OR_EDIT_PATTERN.search(msg.get("content", "")) is not None
    )


def is_keyword_search(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and KEYWORD_SEARCH_PATTERN.search(msg.get("content", "")) is not None
    )


def is_dir_listing(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and DIR_LIST_PATTERN.search(msg.get("content", "")) is not None
    )


def extract_search_command(msg: dict) -> str:
    """Extract the full command string from a terminal action."""
    m = SEARCH_KEYWORD_PATTERN.search(msg.get("content", ""))
    return m.group(1).strip() if m else ""


def extract_search_keywords(command: str) -> set[str]:
    """
    Heuristically extract the search keyword(s) from a grep/rg/find/... command.

    We tokenize the command and take the first non-flag, non-tool argument
    after the tool name as the keyword/pattern.
    """
    tokens = command.split()
    if not tokens:
        return set()

    tools = {"grep", "find", "rg", "ag", "ack", "fzf"}
    keywords: set[str] = set()

    i = 0
    # Find the tool
    while i < len(tokens) and tokens[i] not in tools:
        i += 1
    if i >= len(tokens):
        return keywords

    tool = tokens[i]
    i += 1

    if tool == "find":
        # find . -name "*.py" — look for -name/-iname argument
        while i < len(tokens):
            if tokens[i] in ("-name", "-iname") and i + 1 < len(tokens):
                keywords.add(tokens[i + 1].strip("\"'"))
                i += 2
            else:
                i += 1
    else:
        # grep/rg/ag/ack: skip flags, first non-flag arg is the pattern
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("-"):
                # Some flags take a value (e.g., -e pattern, -f file, --include=...)
                if tok in ("-e", "-f", "--include", "--exclude", "-t", "--type") and i + 1 < len(tokens):
                    if tok == "-e":
                        keywords.add(tokens[i + 1].strip("\"'"))
                    i += 2
                elif "=" in tok:
                    i += 1
                else:
                    i += 1
            else:
                keywords.add(tok.strip("\"'"))
                break

    return keywords


def extract_listing_dir(msg: dict) -> str:
    """
    Extract the directory argument from an ls command.
    Returns "" if no explicit directory is given (i.e. listing cwd).
    """
    m = SEARCH_KEYWORD_PATTERN.search(msg.get("content", ""))
    if not m:
        return ""
    tokens = m.group(1).strip().split()
    i = 0
    while i < len(tokens) and tokens[i] != "ls":
        i += 1
    i += 1  # skip 'ls'
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            i += 1
        else:
            return tok.rstrip("/")
    return ""  # no explicit directory


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


def get_ref_exploration_pairs(
    messages: list[dict],
) -> list[tuple[dict, dict | None]]:
    """
    Extract keyword searching and directory listing (action, result) pairs from
    a reference trajectory that occur before the first file-editing action.

    Returns a list of (action_msg, result_msg_or_None) pairs in order.
    """
    # Find first file-editing action index
    first_file_edit: int | None = None
    for i, msg in enumerate(messages):
        if is_file_editing(msg):
            first_file_edit = i
            break

    cutoff = first_file_edit if first_file_edit is not None else len(messages)

    all_pairs = build_action_result_pairs(messages)

    result_pairs = []
    for ai, ri in all_pairs:
        if ai >= cutoff:
            break
        if is_keyword_search(messages[ai]) or is_dir_listing(messages[ai]):
            act_msg = messages[ai]
            res_msg = messages[ri] if ri is not None else None
            result_pairs.append((act_msg, res_msg))

    return result_pairs


def add_search_messages(
    base_messages: list[dict],
    ref_exploration_pairs: list[tuple[dict, dict | None]],
) -> tuple[list[dict], int]:
    """
    Insert reference keyword searches and directory listings into the base trajectory.

    Returns (new_messages, num_inserted).
    """
    # Find the first file-view/edit action index in base
    first_file_view_edit: int | None = None
    for i, msg in enumerate(base_messages):
        if is_file_view_or_edit(msg):
            first_file_view_edit = i
            break

    if first_file_view_edit is None:
        # No file view/edit action; nothing to do
        return base_messages, 0

    # Find keyword search and dir listing actions before first_file_view_edit in base
    all_pairs = build_action_result_pairs(base_messages)
    base_exploration_pairs = [
        (ai, ri)
        for ai, ri in all_pairs
        if ai < first_file_view_edit
        and (is_keyword_search(base_messages[ai]) or is_dir_listing(base_messages[ai]))
    ]

    # Determine insertion point: before the first existing exploration action (if any),
    # else before the first file-view/edit action
    if base_exploration_pairs:
        insert_before = base_exploration_pairs[0][0]
    else:
        insert_before = first_file_view_edit

    # Collect keywords and listed dirs already present in base (to skip duplicates)
    existing_keywords: set[str] = set()
    existing_listing_dirs: set[str] = set()
    for ai, _ in base_exploration_pairs:
        if is_keyword_search(base_messages[ai]):
            cmd = extract_search_command(base_messages[ai])
            existing_keywords |= extract_search_keywords(cmd)
        elif is_dir_listing(base_messages[ai]):
            d = extract_listing_dir(base_messages[ai])
            existing_listing_dirs.add(d)

    # Build messages to insert, skipping duplicates
    to_insert: list[dict] = []
    for act_msg, res_msg in ref_exploration_pairs:
        if is_keyword_search(act_msg):
            cmd = extract_search_command(act_msg)
            kws = extract_search_keywords(cmd)
            if kws & existing_keywords:
                continue  # skip — same keyword already searched in base
        elif is_dir_listing(act_msg):
            d = extract_listing_dir(act_msg)
            if d in existing_listing_dirs:
                continue  # skip — same directory already listed in base
        to_insert.append(act_msg)
        if res_msg is not None:
            to_insert.append(res_msg)

    if not to_insert:
        return base_messages, 0

    new_messages = (
        base_messages[:insert_before]
        + to_insert
        + base_messages[insert_before:]
    )
    return new_messages, len([m for m in to_insert if m.get("role") == "assistant"])


def derive_hub_repo(base_dataset: str) -> str:
    """Derive an output repo name from the base dataset name."""
    base = base_dataset.split(":")[0]
    return f"{base}_add_search"


def process_datasets(
    base_dataset: str,
    ref_dataset: str,
    output_repo: str | None = None,
    dry_run: bool = False,
):
    hub_repo = output_repo or derive_hub_repo(base_dataset)

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

    def count_steps(messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "assistant")

    ref_total_steps = sum(
        count_steps(ref_by_id[iid]) for iid in overlap
    )
    avg_ref_steps = ref_total_steps / len(overlap) if overlap else 0

    skipped_no_overlap = 0
    affected = 0
    total_inserted = 0
    total_steps_before = 0
    total_steps_after = 0
    processed_rows = []

    for row in base_ds:
        iid = row.get("instance_id")
        steps_before = count_steps(row["messages"])
        total_steps_before += steps_before

        if iid not in ref_by_id:
            skipped_no_overlap += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        ref_msgs = ref_by_id[iid]
        ref_exploration_pairs = get_ref_exploration_pairs(ref_msgs)

        if not ref_exploration_pairs:
            # Reference has no exploration actions before editing; keep base as-is
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        new_messages, num_inserted = add_search_messages(
            row["messages"], ref_exploration_pairs
        )

        if num_inserted > 0:
            affected += 1
            total_inserted += num_inserted

        total_steps_after += count_steps(new_messages)
        processed_rows.append({**row, "messages": new_messages})

    kept = len(processed_rows)
    avg_steps_before = total_steps_before / kept if kept > 0 else 0
    avg_steps_after = total_steps_after / kept if kept > 0 else 0
    print("\nResults:")
    print(f"  Passed through (no overlap):     {skipped_no_overlap} / {len(base_ds)}")
    print(f"  Affected (actions inserted):     {affected} / {kept}")
    if affected > 0:
        print(f"  Avg actions inserted:            {total_inserted / affected:.2f}")
    print(f"  Avg steps (reference dataset):   {avg_ref_steps:.2f}")
    print(f"  Avg steps before adding search:  {avg_steps_before:.2f}")
    print(f"  Avg steps after adding search:   {avg_steps_after:.2f}")
    print(f"  Output repo:                     {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return affected, total_inserted / affected if affected > 0 else 0

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return affected, total_inserted / affected if affected > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Add keyword searching actions from a reference dataset into a base dataset."
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="HuggingFace dataset name for the base trajectories to augment "
             "(e.g., synthetic-code-training/func_localize_claude47_1467i_min_explore)",
    )
    parser.add_argument(
        "--ref-dataset",
        required=True,
        help="HuggingFace dataset name for the reference trajectories to pull searches from "
             "(e.g., synthetic-code-training/func_localize_claude45_1457i)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo to push the result to "
             "(default: {base_dataset}_add_search)",
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
