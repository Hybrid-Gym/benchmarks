"""Push a combined, partial-eval snapshot of several SWE-Gym models to ONE HF dataset.

Unlike convert_and_push.py (one model -> one repo, only run once eval is 100% done),
this is meant to run mid-eval: some instances have been scored, most haven't yet. Every
row gets an explicit `evaluated` flag so "not yet scored" is never confused with
"scored and failed":
  - evaluated=True,  resolved=True/False  -- this instance has a merged eval verdict
  - evaluated=False, resolved=None        -- rollout exists but no eval verdict yet

Rows are tagged with `model` and concatenated into a single split across all models
passed in, so the whole snapshot lives in one dataset repo.

Usage:
    .venv/bin/python benchmarks/swegym/scripts/push_partial_snapshot.py \
        --repo synthetic-code-training/swegym_partial_snapshot \
        --model gpt5mini=<run_dir> --model qwen80b=<run_dir> ...
"""

import argparse
import copy
import gzip
import importlib.util
import json
import os
import sys
import types

from datasets import Dataset
from tqdm import tqdm


HG = "vendor/Hybrid-Gym"


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


def load_model_rows(model: str, run_dir: str) -> list[dict]:
    src = os.path.join(run_dir, "output.with_completions.jsonl.gz")
    report_path = os.path.join(run_dir, "output.report.json")
    if not os.path.exists(src):
        print(f"[{model}] no {src}, skipping")
        return []

    resolved_ids: set[str] = set()
    evaluated_ids: set[str] = set()
    if os.path.exists(report_path):
        rep = json.load(open(report_path))
        resolved_ids = set(rep.get("resolved_ids") or [])
        # completed_ids is the merge script's "actually scored" set; fall back to
        # resolved+unresolved if a bespoke report doesn't carry it.
        evaluated_ids = set(
            rep.get("completed_ids")
            or (rep.get("resolved_ids") or []) + (rep.get("unresolved_ids") or [])
        )
    else:
        print(
            f"[{model}] no output.report.json at {report_path} -- everything marked unevaluated"
        )

    rows = []
    failed = {"count": 0}
    with gzip.open(src, "rt") as f:
        for line in tqdm(f, desc=f"[{model}] loading"):
            r = json.loads(line)
            rc = r.get("raw_completions") or {}
            msgs = rc.get("messages")
            tools = rc.get("tools")
            if not msgs:
                continue
            converted = convert_from_multiple_tool_calls_to_single_tool_call_messages(
                msgs, ignore_final_tool_result=True
            )
            nonfncall = _convert_messages(converted, tools or [], failed)
            if nonfncall is None:
                continue
            iid = r["instance_id"]
            evaluated = iid in evaluated_ids
            rows.append(
                {
                    "instance_id": iid,
                    "model": model,
                    "evaluated": evaluated,
                    "resolved": (iid in resolved_ids) if evaluated else None,
                    "messages": nonfncall,
                    "tools": tools or [],
                    "git_patch": (r.get("test_result") or {}).get("git_patch", ""),
                }
            )
    print(
        f"[{model}] rows={len(rows)} evaluated={sum(r['evaluated'] for r in rows)} "
        f"resolved={sum(bool(r['resolved']) for r in rows)} "
        f"conversion_failures={failed['count']}"
    )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="model=run_dir, repeatable",
    )
    p.add_argument("--repo", required=True)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    all_rows: list[dict] = []
    for spec in args.model:
        model, run_dir = spec.split("=", 1)
        all_rows.extend(load_model_rows(model, run_dir))

    print(f"Total rows: {len(all_rows)}")
    if not all_rows:
        print("Nothing to push.")
        return

    if args.dry_run:
        print("Dry run -- not pushing to hub.")
        return

    ds = Dataset.from_list(all_rows)
    url = ds.push_to_hub(
        repo_id=args.repo,
        split="train",
        token=args.token,
        commit_message=f"Partial snapshot: {len(all_rows)} rows across {len(args.model)} model(s)",
    )
    print(f"Pushed: {url}")


if __name__ == "__main__":
    main()
