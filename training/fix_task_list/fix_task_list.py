"""
Fix single-quote Python syntax in 'task tracker' tool call parameter 'task_list'
to JSON double-quote syntax.

For each trajectory in the dataset, scan all messages for
  <parameter=task_list>...</parameter>
and re-serialise any Python-literal value (single quotes, True/False/None, etc.)
as strict JSON using ast.literal_eval + json.dumps.

Outputs the number of trajectories changed and (optionally) pushes the fixed
dataset to the HuggingFace Hub.
"""

import argparse
import ast
import json
import re

from datasets import Dataset, load_dataset


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


def derive_hub_repo(base_dataset: str, dataset_size: int) -> str:
    base = base_dataset.split(":")[0]
    base_size = base_dataset.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"fixed_{dataset_size}i")
    else:
        return base + f"_fixed_{dataset_size}i"


def process_dataset(
    base_dataset: str,
    output_repo: str | None = None,
    dry_run: bool = False,
):
    print(f"Loading dataset: {base_dataset}")
    ds = load_dataset(base_dataset, split="train")
    print(f"  {len(ds)} trajectories")

    changed_count = 0
    processed_rows = []

    for row in ds:
        new_msgs, changed = fix_messages_task_list(row["messages"])
        if changed:
            changed_count += 1
            processed_rows.append({**row, "messages": new_msgs})
        else:
            processed_rows.append(row)

    hub_repo = output_repo or derive_hub_repo(base_dataset, dataset_size=len(processed_rows))

    print(f"\nResults:")
    print(f"  Total trajectories:      {len(ds)}")
    print(f"  Trajectories changed:    {changed_count}")
    print(f"  Output repo:             {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return changed_count

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return changed_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix single-quote task_list parameter syntax to JSON double-quote syntax."
    )
    parser.add_argument(
        "--base-dataset",
        default="synthetic-code-training/func_localize_gpt55_1477i",
        help="HuggingFace dataset name to fix "
             "(e.g., synthetic-code-training/func_localize_gpt55_1477i)",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo to push the result to "
             "(default: derived from base dataset name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report stats without pushing to HuggingFace Hub",
    )
    args = parser.parse_args()

    process_dataset(
        base_dataset=args.base_dataset,
        output_repo=args.output_repo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
