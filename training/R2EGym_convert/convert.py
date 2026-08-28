#!/usr/bin/env python3
"""
Convert R2E-Gym/R2EGym-SFT-Trajectories to our OpenHands SDK non-function-calling format.

Dataset structure (R2E-Gym):
  messages[0] role=system : R2E tool descriptions + agent instructions
  messages[1] role=user   : task description (GitHub issue)
  messages[2] role=assistant: first tool call (XML embedded in text)
  messages[3] role=user   : observation
  ...

Conversion rules:
  1. System message   : replace with OpenHands system prompt (from reference dataset)
  2. execute_bash     : rename to terminal; params unchanged
  3. search           : rename to terminal; convert (search_term, path) → grep command
  4. finish           : rename kept; remap {command,result} → {message}
  5. file_editor      : kept; drop enable_linting / concise params if present
  6. Observation prefix: "Execution output of [X]:" → "EXECUTION RESULT of [X]:"
                         (with X renamed by the same mapping above)
  7. First user message (task description): kept unchanged
"""

import argparse
import json
import os
import re
import sys

from datasets import Dataset, load_dataset
from tqdm import tqdm

# ── Tool name mapping ────────────────────────────────────────────────────────

TOOL_RENAME: dict[str, str] = {
    "execute_bash": "terminal",
    "search": "terminal",
}

# ── Regexes ──────────────────────────────────────────────────────────────────

# Matches one <function=NAME>...</function> block (non-greedy, dotall)
ACTION_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)

# Matches one <parameter=NAME>value</parameter> block (leading whitespace ok)
PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

# Matches the observation prefix at the start of a user message
# R2E format: "Execution output of [TOOL]:\n"
OBS_PREFIX_RE = re.compile(r"^Execution output of \[(\w+)\]:", re.MULTILINE)

# ── Parameter helpers ─────────────────────────────────────────────────────────

def parse_params(params_str: str) -> dict[str, str]:
    """Return an ordered dict of parameter name → value."""
    return {m.group(1): m.group(2) for m in PARAM_RE.finditer(params_str)}


def build_params(params: dict[str, str]) -> str:
    """Serialise a dict back to <parameter=k>v</parameter> blocks."""
    return "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in params.items())


# ── Action conversion ─────────────────────────────────────────────────────────

def _search_to_grep(params: dict[str, str]) -> str:
    """Convert search tool params → a grep command string."""
    term = params.get("search_term", "")
    path = params.get("path", ".")
    # Escape backslashes first, then double-quotes, for shell safety
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    return f'grep -rn "{escaped}" {path}'


def convert_one_action(func_name: str, params_str: str) -> str:
    """Return a converted <function=NEW_NAME>...</function> string."""
    new_name = TOOL_RENAME.get(func_name, func_name)
    params = parse_params(params_str)

    if func_name == "search":
        new_params = build_params({"command": _search_to_grep(params)})

    elif func_name == "execute_bash":
        # Same single 'command' parameter — rebuild to normalise whitespace
        new_params = build_params(params)

    elif func_name == "finish":
        # R2E: finish(command="submit", result="...") → finish(message="...")
        message = params.get("result", "")
        new_params = build_params({"message": message})

    elif func_name == "file_editor":
        params.pop("enable_linting", None)
        params.pop("concise", None)
        new_params = build_params(params)

    else:
        # Unknown tool — pass through unchanged (name + params)
        new_params = params_str

    return f"<function={new_name}>\n{new_params}</function>"


def convert_assistant_content(content: str) -> str:
    """Convert all tool calls embedded in an assistant message."""
    return ACTION_RE.sub(
        lambda m: convert_one_action(m.group(1), m.group(2)),
        content,
    )


# ── Observation conversion ────────────────────────────────────────────────────

def convert_user_content(content: str) -> str:
    """Convert the observation prefix in a (non-first) user message."""
    def _replace(m: re.Match) -> str:
        old_name = m.group(1)
        new_name = TOOL_RENAME.get(old_name, old_name)
        return f"EXECUTION RESULT of [{new_name}]:"

    # Only replace the first occurrence (the prefix); content body is kept as-is
    return OBS_PREFIX_RE.sub(_replace, content, count=1)


# ── Trajectory conversion ─────────────────────────────────────────────────────

def convert_trajectory(
    messages: list[dict],
    system_prompt: str,
) -> list[dict]:
    """
    Convert one R2E trajectory to our format.

    Expected input layout:
      [0] system  – R2E system prompt       → replaced with OpenHands prompt
      [1] user    – task / GitHub issue     → kept unchanged
      [2] assistant – first tool call       → tool names/params converted
      [3] user    – first observation       → prefix converted
      ...
    """
    out: list[dict] = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # Replace R2E system instructions with OpenHands system prompt
            out.append({"role": "system", "content": system_prompt})

        elif role == "user" and i <= 1:
            # messages[1] is the task description — keep verbatim.
            # (Guard with i <= 1 in case role=system is absent in some rows.)
            out.append({"role": "user", "content": content})

        elif role == "user":
            # Observation message — convert prefix only
            out.append({"role": "user", "content": convert_user_content(content)})

        elif role == "assistant":
            out.append({"role": "assistant", "content": convert_assistant_content(content)})

        else:
            out.append(msg)

    return out


# ── System prompt loading ─────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "system_prompt.txt"
)


def load_system_prompt(hf_token: str | None, prompt_file: str | None) -> str:
    """
    Load the OpenHands system prompt.

    Priority:
      1. --system-prompt-file argument
      2. system_prompt.txt next to this script (default)
    """
    path = prompt_file or _DEFAULT_SYSTEM_PROMPT_FILE
    if os.path.exists(path):
        with open(path) as f:
            content = f.read()
        print(f"System prompt loaded from {path} ({len(content)} chars).")
        return content
    else:
        raise RuntimeError(f"System prompt file not found: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace API token (falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        metavar="PATH",
        help="Path to a text file containing the OpenHands system prompt "
             "(avoids fetching the reference dataset on repeated runs).",
    )
    parser.add_argument(
        "--output-repo",
        default="synthetic-code-training/r2egym_converted_3231i",
        metavar="ORG/REPO",
        help="HF dataset repo to push to, e.g. "
             "synthetic-code-training/r2egym_converted_3231i.",
    )
    parser.add_argument(
        "--out-jsonl",
        default=None,
        metavar="PATH",
        help="Local JSONL file to write converted rows to.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 = all).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print the first converted trajectory (first 6 messages) and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert and report stats but do not write files or push to HF.",
    )
    args = parser.parse_args()

    token: str | None = args.hf_token or os.environ.get("HF_TOKEN")

    # ── Load system prompt ───────────────────────────────────────────────────
    system_prompt = load_system_prompt(token, args.system_prompt_file)

    # ── Load source dataset ──────────────────────────────────────────────────
    print("Loading R2E-Gym/R2EGym-SFT-Trajectories …")
    src_ds = load_dataset("R2E-Gym/R2EGym-SFT-Trajectories", split="train")
    print(f"  Source rows: {len(src_ds)}")

    # ── Inspect mode ─────────────────────────────────────────────────────────
    if args.inspect:
        row = src_ds[0]
        converted = convert_trajectory(row["messages"], system_prompt)
        sep = "=" * 70
        for i, msg in enumerate(converted[:6]):
            print(f"\n{sep}\n[{i}] role={msg['role']}")
            print(msg["content"][:800])
            if len(msg["content"]) > 800:
                print(f"… ({len(msg['content'])} chars total)")
        sys.exit(0)

    # ── Convert ───────────────────────────────────────────────────────────────
    limit = args.limit or len(src_ds)
    rows: list[dict] = []

    for i, row in enumerate(tqdm(src_ds, total=limit, desc="converting")):
        if i >= limit:
            break
        converted = convert_trajectory(row["messages"], system_prompt)
        rows.append(
            {
                "instance_id": row.get("instance_id", f"r2egym_{i}"),
                "messages": converted,
            }
        )

    print(f"Converted {len(rows)} rows.")

    if args.dry_run:
        print("Dry run — skipping output.")
        return

    # ── Write local JSONL ────────────────────────────────────────────────────
    if args.out_jsonl:
        out_dir = os.path.dirname(args.out_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_jsonl, "w") as fout:
            for r in rows:
                fout.write(json.dumps(r) + "\n")
        size_mb = os.path.getsize(args.out_jsonl) / 1024 / 1024
        print(f"Wrote {args.out_jsonl} ({size_mb:.1f} MB).")

    # ── Push to HF ───────────────────────────────────────────────────────────
    if args.output_repo:
        print(f"Pushing to hf://datasets/{args.output_repo} …")
        ds_out = Dataset.from_list(rows)
        url = ds_out.push_to_hub(
            repo_id=args.output_repo,
            split="train",
            token=token,
            commit_message="Add converted R2EGym-SFT-Trajectories (OpenHands format)",
        )
        print(f"Pushed: {url}")


if __name__ == "__main__":
    main()
