"""
Splice verification steps from a reference dataset into a base dataset.

For each instance that appears in both datasets (matched by instance_id) where
both trajectories are resolved and both edited the same file:

  1. Keep the base trajectory's messages up to (but not including) its first
     file-editing action — i.e. the exploration steps are preserved.
  2. Replace everything from the first file-editing action onward with the
     reference trajectory's tail: its first file-editing action (docstring edit)
     + any verification steps + finish.
  3. Fix workspace directory paths in the transplanted tail so they point to
     the base trajectory's workspace rather than the reference's.

Definitions
-----------
- file-editing action: assistant message calling file_editor with
    command=str_replace, create_file, or insert
- verification steps: any actions between the file-editing action and the
    finish call (e.g. ast.parse syntax checks, file views to confirm the edit)
- workspace dir: the repo-specific directory under /workspace/, extracted from
    the task description in the first user message
    (e.g. "Gallopsled__pwntools__bc93beb5" from
     "repository in the directory Gallopsled__pwntools__bc93beb5")
"""

import argparse
import re

from datasets import Dataset, load_dataset


FILE_EDIT_PATTERN = re.compile(
    r"<function=file_editor>\n<parameter=command>(str_replace|create_file|insert)</parameter>"
)
FILE_PATH_PATTERN = re.compile(r"<parameter=path>(.*?)</parameter>", re.DOTALL)
WORKSPACE_DIR_PATTERN = re.compile(r"\bdirectory (\S+)")


def is_file_editing(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and FILE_EDIT_PATTERN.search(msg.get("content", "")) is not None
    )


def extract_workspace_dir(messages: list[dict]) -> str | None:
    """
    Return the workspace directory name from the task description.

    The first user message (index 1, after the system prompt) contains a line
    like "repository in the directory Gallopsled__pwntools__bc93beb5".
    """
    for msg in messages:
        if msg.get("role") == "user":
            m = WORKSPACE_DIR_PATTERN.search(msg.get("content", ""))
            if m:
                return m.group(1)
    return None


def extract_edited_file_rel(messages: list[dict], workspace_dir: str) -> str | None:
    """
    Return the relative file path edited by the first file-editing action,
    with the workspace prefix stripped.

    e.g. /workspace/Gallopsled__pwntools__bc93beb5/pwnlib/atexception.py
         -> pwnlib/atexception.py
    """
    prefix = f"/workspace/{workspace_dir}/"
    for msg in messages:
        if is_file_editing(msg):
            m = FILE_PATH_PATTERN.search(msg.get("content", ""))
            if m:
                path = m.group(1).strip()
                if path.startswith(prefix):
                    return path[len(prefix):]
                return path
    return None


def substitute_workspace(messages: list[dict], old_dir: str, new_dir: str) -> list[dict]:
    """Return a copy of messages with old_dir replaced by new_dir throughout."""
    old = f"/workspace/{old_dir}"
    new = f"/workspace/{new_dir}"
    return [
        {**msg, "content": msg["content"].replace(old, new)}
        for msg in messages
    ]


def splice_verify_tail(
    base_messages: list[dict],
    ref_messages: list[dict],
) -> tuple[list[dict], bool]:
    """
    Replace everything from the base's first file-editing action onward with
    the reference's first file-editing action onward (path-adjusted).

    Returns (new_messages, was_changed).
    """
    # Locate the first file-editing action in base
    base_edit_idx: int | None = None
    for i, msg in enumerate(base_messages):
        if is_file_editing(msg):
            base_edit_idx = i
            break
    if base_edit_idx is None:
        return base_messages, False

    # Locate the first file-editing action in ref
    ref_edit_idx: int | None = None
    for i, msg in enumerate(ref_messages):
        if is_file_editing(msg):
            ref_edit_idx = i
            break
    if ref_edit_idx is None:
        return base_messages, False

    ref_tail = ref_messages[ref_edit_idx:]

    # Fix workspace paths in the transplanted tail
    base_ws = extract_workspace_dir(base_messages)
    ref_ws = extract_workspace_dir(ref_messages)
    if base_ws and ref_ws and base_ws != ref_ws:
        ref_tail = substitute_workspace(ref_tail, ref_ws, base_ws)

    new_messages = base_messages[:base_edit_idx] + ref_tail
    return new_messages, True


def derive_hub_repo(base_dataset: str, dataset_size: int) -> str:
    base = base_dataset.split(":")[0]
    base_size = base_dataset.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"add_verify_{dataset_size}i")
    else:
        return base + f"_add_verify_{dataset_size}i"


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

    # Build lookup: instance_id -> ref row (only resolved ones)
    ref_by_id: dict[str, dict] = {}
    for row in ref_ds:
        if row.get("resolved") and row.get("instance_id"):
            ref_by_id[row["instance_id"]] = row

    def count_steps(messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "assistant")

    skipped_base_not_resolved = 0
    skipped_no_ref = 0
    skipped_no_base_edit = 0
    skipped_no_ref_edit = 0
    skipped_different_file = 0
    affected = 0
    total_steps_before = 0
    total_steps_after = 0
    processed_rows = []

    for row in base_ds:
        iid = row.get("instance_id")
        base_msgs = row["messages"]
        steps_before = count_steps(base_msgs)
        total_steps_before += steps_before

        # Base must be resolved
        if not row.get("resolved"):
            skipped_base_not_resolved += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        # Ref must exist and be resolved
        if iid not in ref_by_id:
            skipped_no_ref += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        ref_msgs = ref_by_id[iid]["messages"]

        # Both must have a file-editing action
        base_ws = extract_workspace_dir(base_msgs)
        ref_ws = extract_workspace_dir(ref_msgs)

        base_file = extract_edited_file_rel(base_msgs, base_ws or "")
        if base_file is None:
            skipped_no_base_edit += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        ref_file = extract_edited_file_rel(ref_msgs, ref_ws or "")
        if ref_file is None:
            skipped_no_ref_edit += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        # Both must have edited the same relative file path
        if base_file != ref_file:
            skipped_different_file += 1
            total_steps_after += steps_before
            processed_rows.append(row)
            continue

        new_msgs, changed = splice_verify_tail(base_msgs, ref_msgs)

        if changed:
            affected += 1

        steps_after = count_steps(new_msgs)
        total_steps_after += steps_after
        
        if changed:
            processed_rows.append({**row, "messages": new_msgs})

    kept = len(processed_rows)
    eligible = kept - skipped_base_not_resolved - skipped_no_ref - skipped_no_base_edit - skipped_no_ref_edit - skipped_different_file
    avg_before = total_steps_before / kept if kept else 0
    avg_after = total_steps_after / kept if kept else 0
    
    hub_repo = output_repo or derive_hub_repo(base_dataset, dataset_size=kept)

    print("\nResults:")
    print(f"  Skipped (base not resolved):     {skipped_base_not_resolved} / {len(base_ds)}")
    print(f"  Skipped (no ref match):          {skipped_no_ref} / {len(base_ds)}")
    print(f"  Skipped (base has no edit):      {skipped_no_base_edit} / {len(base_ds)}")
    print(f"  Skipped (ref has no edit):       {skipped_no_ref_edit} / {len(base_ds)}")
    print(f"  Skipped (different file):        {skipped_different_file} / {len(base_ds)}")
    print(f"  Eligible for splicing:           {eligible} / {len(base_ds)}")
    print(f"  Affected (tail replaced):        {affected} / {eligible}")
    print(f"  Avg steps before:                {avg_before:.2f}")
    print(f"  Avg steps after:                 {avg_after:.2f}")
    print(f"  Output repo:                     {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return affected, eligible

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return affected, eligible


def main():
    parser = argparse.ArgumentParser(
        description="Splice verification steps from a reference dataset into a base dataset."
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="HuggingFace dataset name for the base trajectories to augment "
             "(e.g., synthetic-code-training/func_localize_claude47_1467i)",
    )
    parser.add_argument(
        "--ref-dataset",
        required=True,
        help="HuggingFace dataset name for the reference trajectories to pull "
             "the edit+verification+finish tail from "
             "(e.g., synthetic-code-training/func_localize_claude45_1457i)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo to push the result to "
             "(default: {base_dataset}_add_verify)",
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
