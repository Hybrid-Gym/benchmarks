"""
Generate and insert a synthetic general planning step (task_tracker) at the start
of each trajectory in a base dataset.

This is the reverse of training/no_general_plan/no_general_plan.py:
instead of removing task_tracker planning steps, we generate them using an LLM
and insert them as a synthetic action+feedback pair right after the system and
user preamble.

The format follows the task_tracker planning steps found in datasets like
synthetic-code-training/func_localize_claude45_1457i:

  Action (assistant):
    <function=task_tracker>
    <parameter=command>plan</parameter>
    <parameter=task_list>[{"title": "...", "status": "in_progress"}, ...]</parameter>
    <parameter=summary>Plan tasks for ...</parameter>
    <parameter=security_risk>LOW</parameter>
    </function>

  Feedback (user):
    EXECUTION RESULT of [function]:
    Task list has been updated with N item(s).

Generated results are cached per instance_id in a local cache directory so that
interrupted runs can be resumed without re-calling the LLM.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from openai import OpenAI

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_BASE_DATASET = "synthetic-code-training/func_localize_claude47_1467i"
DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

# ── Regex ──────────────────────────────────────────────────────────────────────
_TASK_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)


# ── LLM client ────────────────────────────────────────────────────────────────

def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


_SYSTEM_PROMPT = """\
You are generating a task-tracker planning step for an AI coding agent.
The agent is about to start working on a software task.
Your job is to output a single function call that the agent would make at the
very start — before any file exploration — to lay out a structured plan.

Output ONLY the function call, with no surrounding text, no markdown, no
explanation.  Use this exact format:

<function=task_tracker>
<parameter=command>plan</parameter>
<parameter=task_list>[{"title": "...", "status": "in_progress"}, {"title": "...", "status": "todo"}, ...]</parameter>
<parameter=summary>Plan tasks for ...</parameter>
<parameter=security_risk>LOW</parameter>
</function>

Rules:
- The task_list must be valid JSON (double quotes, no trailing commas).
- Include 4-6 tasks.  The first task should have status "in_progress"; the
  rest should have status "todo".
- Tasks should reflect the specific function mentioned in the task description.
- Optionally add a "notes" field to some tasks for helpful context.
- The summary should be a short phrase describing the overall goal.
"""

_USER_PROMPT_TEMPLATE = """\
Here is the task the agent received:

{user_message}

Generate the task_tracker planning step now.
"""


def generate_plan(
    client: OpenAI,
    model: str,
    user_message: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str | None:
    """
    Call the LLM and return the generated task_tracker function call string,
    or None on failure.
    """
    prompt = _USER_PROMPT_TEMPLATE.format(user_message=user_message.strip())
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=0.3,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            # Basic validation: must look like a task_tracker call
            if "<function=task_tracker" in content:
                return content
            print(f"  [warn] LLM output didn't contain task_tracker call: {content[:200]}")
        except Exception as exc:
            print(f"  [warn] LLM call failed (attempt {attempt + 1}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    return None


# ── Cache helpers ──────────────────────────────────────────────────────────────

def cache_path(cache_dir: Path, instance_id: str) -> Path:
    # Sanitize instance_id for use as a filename
    safe_id = re.sub(r"[^\w\-.]", "_", instance_id)
    return cache_dir / f"{safe_id}.json"


def load_from_cache(cache_dir: Path, instance_id: str) -> str | None:
    p = cache_path(cache_dir, instance_id)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get("plan_content")
        except Exception:
            return None
    return None


def save_to_cache(cache_dir: Path, instance_id: str, plan_content: str) -> None:
    p = cache_path(cache_dir, instance_id)
    p.write_text(json.dumps({"instance_id": instance_id, "plan_content": plan_content}))


# ── Plan message builders ──────────────────────────────────────────────────────

def count_tasks(plan_content: str) -> int:
    """Count the number of task items in the task_list."""
    m = _TASK_LIST_RE.search(plan_content)
    if not m:
        return 0
    try:
        tasks = json.loads(m.group(0))
        return len(tasks)
    except Exception:
        # Fall back to counting occurrences of "title"
        return plan_content.count('"title"')


def make_action_message(plan_content: str) -> dict:
    return {"role": "assistant", "content": plan_content}


def make_feedback_message(n_tasks: int) -> dict:
    return {
        "role": "user",
        "content": f"EXECUTION RESULT of [function]:\nTask list has been updated with {n_tasks} item(s).",
    }


# ── Trajectory insertion ───────────────────────────────────────────────────────

def insert_plan(messages: list[dict], plan_content: str) -> list[dict]:
    """
    Insert the generated plan action+feedback pair immediately after the
    system message and user message (i.e. at position 2).

    Returns a new messages list.
    """
    n_tasks = count_tasks(plan_content)
    action_msg = make_action_message(plan_content)
    feedback_msg = make_feedback_message(n_tasks)

    # Find insertion point: after initial system+user preamble
    # Preamble = all leading non-assistant messages
    insert_at = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            insert_at = i
            break
    else:
        insert_at = len(messages)

    return messages[:insert_at] + [action_msg, feedback_msg] + messages[insert_at:]


# ── Dataset processing ─────────────────────────────────────────────────────────

def derive_hub_repo(base_dataset: str, dataset_size: int) -> str:
    base = base_dataset.split(":")[0]
    base_size = base_dataset.split("_")[-1]
    if base_size[-1] == "i" and base_size[0].isdigit():
        return base.replace(base_size, f"add_plan_{dataset_size}i")
    else:
        return base + f"_add_plan_{dataset_size}i"


def process_dataset(
    dataset_name: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_repo: str | None = None,
    dry_run: bool = False,
    max_retries: int = 3,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} trajectories")

    client = make_client(base_url=base_url, api_key=api_key)

    cached_hits = 0
    llm_calls = 0
    llm_failures = 0
    processed_rows = []

    for idx, row in enumerate(ds):
        iid = row.get("instance_id", f"row_{idx}")
        messages = row["messages"]

        # Find user message (first user message after the system prompt)
        user_msg_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg_content = m.get("content", "")
                break

        # Try cache first
        plan_content = load_from_cache(cache_dir, iid)
        if plan_content is not None:
            cached_hits += 1
        else:
            llm_calls += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{len(ds)}] Generating plan for {iid} ...")
            plan_content = generate_plan(
                client=client,
                model=model,
                user_message=user_msg_content,
                max_retries=max_retries,
            )
            if plan_content is None:
                llm_failures += 1
                print(f"  [skip] Failed to generate plan for {iid}")
                # Keep original trajectory without a plan
                processed_rows.append(row)
                continue
            save_to_cache(cache_dir, iid, plan_content)

        new_messages = insert_plan(messages, plan_content)
        processed_rows.append({**row, "messages": new_messages})

    kept = len(processed_rows)
    hub_repo = output_repo or derive_hub_repo(dataset_name, dataset_size=kept)
    
    print("\nResults:")
    print(f"  Total trajectories:    {len(ds)}")
    print(f"  Kept:                  {kept}")
    print(f"  Cache hits:            {cached_hits}")
    print(f"  LLM calls:             {llm_calls}")
    print(f"  LLM failures (skipped):{llm_failures}")
    print(f"  Output repo:           {hub_repo}")

    if dry_run:
        print("\n[dry-run] Skipping push to HuggingFace Hub.")
        return kept

    processed_ds = Dataset.from_list(processed_rows)
    print(f"\nPushing to Hub: {hub_repo}")
    processed_ds.push_to_hub(hub_repo, split="train")
    print("Done.")

    return kept


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Insert a synthetic LLM-generated task_tracker planning step "
                    "at the start of each trajectory in a base dataset."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_BASE_DATASET,
        help="HuggingFace dataset name for the base trajectories "
             f"(default: {DEFAULT_BASE_DATASET})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="API key for the LLM endpoint (default: $NVIDIA_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for the LLM endpoint (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Directory to cache generated plans (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--output-repo",
        default=None,
        help="HuggingFace repo to push results to "
             "(default: {dataset}_add_plan)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and report stats without pushing to HuggingFace Hub",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per LLM call on failure (default: 3)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "No API key provided. Set NVIDIA_API_KEY or pass --api-key."
        )

    process_dataset(
        dataset_name=args.dataset,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        cache_dir=Path(args.cache_dir),
        output_repo=args.output_repo,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
