"""Run convert_data.py's pipeline on output_success.with_completions.jsonl.gz
and push to HF as synthetic-code-training/full_func_localize_<model>_<N>i.

Mirrors evaluation/convert_data.py in Hybrid-Gym, with these differences:
  - Loads fn_call_converter via importlib to bypass the heavy package __init__.
  - Reads our pre-filtered output_success file (all rows already resolved=True).
  - Preserves instance_id and resolved alongside messages (matches the reference
    dataset synthetic-code-training/full_func_localize_claude_2748i schema).
"""

import argparse
import copy
import gzip
import importlib.util
import json
import os
import sys
import types

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


tqdm.pandas()

HG = "/home/gaokaizhang/Hybrid-Gym"


def _load_fncall_converter():
    for pkg in ["openhands", "openhands.core", "openhands.llm"]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [f"{HG}/{pkg.replace('.', '/')}"]
            sys.modules[pkg] = m

    def _load(modname, path):
        spec = importlib.util.spec_from_file_location(modname, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("openhands.core.exceptions", f"{HG}/openhands/core/exceptions.py")
    _load("openhands.llm.tool_names", f"{HG}/openhands/llm/tool_names.py")
    return _load(
        "openhands.llm.fn_call_converter", f"{HG}/openhands/llm/fn_call_converter.py"
    )


fc = _load_fncall_converter()
FunctionCallConversionError = fc.FunctionCallConversionError
convert_fncall_messages_to_non_fncall_messages = (
    fc.convert_fncall_messages_to_non_fncall_messages
)
convert_from_multiple_tool_calls_to_single_tool_call_messages = (
    fc.convert_from_multiple_tool_calls_to_single_tool_call_messages
)


def _convert_messages(messages, tools, failed_counter):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if m.get("content") is None:
            m["content"] = ""
    try:
        converted = convert_fncall_messages_to_non_fncall_messages(
            msgs, tools, add_in_context_learning_example=False
        )
    except FunctionCallConversionError:
        failed_counter["count"] += 1
        return None
    clean = []
    for m in converted:
        c = m["content"]
        if isinstance(c, list) and c and isinstance(c[0], dict):
            m["content"] = c[0]["text"]
        if not isinstance(m["content"], str):
            failed_counter["count"] += 1
            return None
        clean.append(m)
    return clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", required=True, help="Path to output_success.with_completions.jsonl.gz"
    )
    p.add_argument(
        "--repo",
        required=True,
        help="HF dataset repo id, e.g. synthetic-code-training/full_func_localize_claude_opus_4_7_4982i",
    )
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional local jsonl output path (uncompressed)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"Reading {args.src} ...")
    rows = []
    with gzip.open(args.src, "rt") as f:
        for line in tqdm(f, desc="loading"):
            r = json.loads(line)
            rc = r.get("raw_completions") or {}
            msgs = rc.get("messages")
            tools = rc.get("tools")
            if not msgs:
                continue
            rows.append(
                {
                    "instance_id": r["instance_id"],
                    "resolved": True,
                    "messages": msgs,
                    "tools": tools or [],
                    "git_patch": (r.get("test_result") or {}).get("git_patch", ""),
                }
            )
    df = pd.DataFrame(rows)
    print(f"Rows loaded: {len(df)}")

    failed = {"count": 0}
    print("Converting multi-tool-call messages to single-tool-call ...")
    df["converted_messages"] = df["messages"].progress_apply(
        lambda m: convert_from_multiple_tool_calls_to_single_tool_call_messages(
            m, ignore_final_tool_result=True
        )
    )
    print("Converting fncall messages to non-fncall ...")
    df["nonfncall_messages"] = df.progress_apply(
        lambda r: _convert_messages(r["converted_messages"], r["tools"], failed), axis=1
    )

    before = len(df)
    df = df[df["nonfncall_messages"].notna()].reset_index(drop=True)
    print(
        f"After fncall conversion: kept {len(df)}, dropped {before - len(df)} (conversion failures: {failed['count']})"
    )

    # Build the final dataset shape matching the reference (instance_id, resolved, messages)
    upload_rows = (
        df[["instance_id", "resolved", "nonfncall_messages"]]
        .rename(columns={"nonfncall_messages": "messages"})  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .to_dict(orient="records")
    )
    print(f"Final row count: {len(upload_rows)}")

    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in upload_rows:
                f.write(json.dumps(r) + "\n")
        print(
            f"Wrote local jsonl: {args.out_jsonl} ({os.path.getsize(args.out_jsonl) / 1024 / 1024:.1f} MB)"
        )

    if args.dry_run:
        print("Dry run — not pushing to hub.")
        return

    print(f"Pushing to hf://datasets/{args.repo} ...")
    ds = Dataset.from_list(upload_rows)
    url = ds.push_to_hub(
        repo_id=args.repo,
        split="train",
        token=args.token,
        commit_message=f"Add {len(upload_rows)} non-fncall trajectories (claude-opus-4-7 funclocalize)",
    )
    print(f"Pushed: {url}")


if __name__ == "__main__":
    main()
