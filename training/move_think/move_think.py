"""
Remove general planning steps (think actions) and their feedback from trajectories.

Planning steps are identified as assistant messages containing <function=think> tags.
These steps plan how to complete the task without modifying the environment
(no command execution, no file I/O).

For each planning step removed, the immediately following user message
(the execution result of the think function) is also removed.
"""

import argparse
import ast
import json
import re

from datasets import Dataset, load_dataset


THINK_PATTERN = re.compile(r"<function=think>")
THINK_CONTENT_PATTERN = re.compile(r"<function=think>(.*?)</function>", re.DOTALL)

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


def is_planning_step(message: dict) -> bool:
    """Return True if message is a planning-only think step."""
    return (
        message.get("role") == "assistant"
        and THINK_PATTERN.search(message.get("content", "")) is not None
    )


def extract_think_content(message: dict) -> str:
    """Extract the content inside <function=think> tags."""
    match = THINK_CONTENT_PATTERN.search(message.get("content", ""))
    return match.group(1).strip() if match else ""


def remove_planning_steps(messages: list[dict]) -> tuple[list[dict], int]:
    """
    Remove planning steps and their feedback from a message list.
    The think content is moved to the start of the next step (before its action call).

    Returns (filtered_messages, steps_removed) where steps_removed counts
    both the planning step and its execution result feedback.
    """
    filtered = []
    steps_removed = 0
    skip_next = False
    pending_think_content: str | None = None

    for msg in messages:
        if skip_next:
            # This is the execution result following a think step — remove it
            steps_removed += 1
            skip_next = False
            continue

        if is_planning_step(msg):
            steps_removed += 1
            think_content = extract_think_content(msg)
            if think_content:
                pending_think_content = (
                    (pending_think_content + "\n" + think_content)
                    if pending_think_content
                    else think_content
                )
            # The next user message is the execution result for this think step
            skip_next = True
            continue

        if pending_think_content is not None:
            msg = {
                **msg,
                "content": pending_think_content + "\n" + msg.get("content", ""),
            }
            pending_think_content = None

        filtered.append(msg)

    return filtered, steps_removed


def derive_hub_repo(dataset_name: str, dataset_size: int) -> str:
    """Derive the output Hub repo ID by replacing or appending _move_think."""
    # Strip any split suffix (e.g. ":train") before deriving the repo id
    base = dataset_name.split(":")[0]
    base_size = dataset_name.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"move_think_{dataset_size}i")
    else:
        return base + f"_move_think_{dataset_size}i"


def process_dataset(
    dataset_name: str,
    dry_run: bool = False,
) -> None:
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    affected_trajectories = 0
    total_steps_removed = 0
    task_list_fixed_count = 0
    processed_rows = []

    for row in ds:
        messages = row["messages"]
        messages, tl_changed = fix_messages_task_list(messages)
        if tl_changed:
            task_list_fixed_count += 1
        filtered_messages, steps_removed = remove_planning_steps(messages)

        if steps_removed > 0:
            affected_trajectories += 1
            total_steps_removed += steps_removed

        processed_rows.append({**row, "messages": filtered_messages})

    hub_repo = derive_hub_repo(dataset_name, dataset_size=len(processed_rows))
    print("\nResults:")
    print(f"  Affected trajectories:   {affected_trajectories} / {len(ds)}")
    print(
        f"  Total steps removed per trajectory:     {total_steps_removed / affected_trajectories}"
    )
    print("    (includes both planning steps and their execution result feedbacks)")
    print(f"  Task-list JSON fixed:    {task_list_fixed_count} / {len(ds)}")
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
        help="HuggingFace dataset name (output repo will be {dataset}_move_think)",
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
