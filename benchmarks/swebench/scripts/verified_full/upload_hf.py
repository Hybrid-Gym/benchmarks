#!/usr/bin/env python3
"""Publish a model's SWE-bench_Verified predictions to the Hugging Face Hub.

This machine has no local docker, so grading happens elsewhere. This script
uploads the predictions in the shape the SWE-bench harness expects, plus a
``resolved`` field that is deliberately ``null`` until an evaluation machine
fills it in.

    python upload_hf.py <model>                 # upload predictions
    python upload_hf.py <model> --with-trajectories
    python upload_hf.py <model> --partial       # allow an unfinished run

The token is read from ~/.config/hf/token_gaokai (mode 600) and is never
printed or written into the repo.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN_PATH = Path.home() / ".config" / "hf" / "token_gaokai"
DEFAULT_REPO = "synthetic-code-training/swebench-verified-results"
DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
EXPECTED_TOTAL = 500


def read_token() -> str:
    if not TOKEN_PATH.exists():
        sys.exit(f"missing HF token at {TOKEN_PATH}")
    token = TOKEN_PATH.read_text().strip()
    if not token:
        sys.exit(f"empty HF token at {TOKEN_PATH}")
    return token


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def load_env() -> dict[str, str]:
    """Pull the sweep's paths out of env.sh so both stay in sync."""
    here = Path(__file__).resolve().parent
    out = sh(
        [
            "bash",
            "-c",
            f'source "{here}/env.sh" >/dev/null 2>&1; '
            'echo "$EVAL_OUT_ROOT"; echo "$SDK_SHORT_SHA"; echo "$MAX_ITER"',
        ]
    ).splitlines()
    if len(out) < 3:
        sys.exit("could not source env.sh")
    return {"eval_out_root": out[0], "sdk": out[1], "maxiter": out[2]}


def output_dir(env: dict[str, str], model: str) -> Path:
    return (
        Path(env["eval_out_root"])
        / "princeton-nlp__SWE-bench_Verified-test"
        / "openai"
        / f"{model}_sdk_{env['sdk']}_maxiter_{env['maxiter']}"
    )


def build_predictions(out_jsonl: Path, model: str) -> list[dict]:
    """One record per instance, last attempt wins, resolved left unset."""
    by_id: dict[str, dict] = {}
    for line in out_jsonl.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        iid = d.get("instance_id")
        if not iid:
            continue
        patch = (d.get("test_result") or {}).get("git_patch") or ""
        by_id[iid] = {
            "instance_id": iid,
            "model_name_or_path": model,
            "model_patch": patch,
            # Filled in by the evaluation machine. null == not yet graded.
            "resolved": None,
        }
    return [by_id[k] for k in sorted(by_id)]


def pad_missing(preds: list[dict], model: str) -> list[str]:
    """Add empty-patch rows for instances that never produced a result.

    Keeps the denominator at 500 so models stay comparable, while marking the
    rows so a reader can tell an infrastructure failure from a model that
    genuinely chose not to edit anything.
    """
    from datasets import load_dataset

    have = {p["instance_id"] for p in preds}
    # load_dataset's return type is a union; with an explicit split it is a
    # Dataset of rows, which pyright cannot narrow on its own.
    ds: Any = load_dataset(DATASET, split=SPLIT)
    missing = [str(r["instance_id"]) for r in ds if r["instance_id"] not in have]
    for iid in missing:
        preds.append(
            {
                "instance_id": iid,
                "model_name_or_path": model,
                "model_patch": "",
                "resolved": None,
                "inference_failed": True,
            }
        )
    return missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model save name (basename, no org prefix)")
    ap.add_argument("--repo", default=os.environ.get("HF_RESULTS_REPO", DEFAULT_REPO))
    ap.add_argument(
        "--with-trajectories",
        action="store_true",
        help="also upload the full output.jsonl, gzipped (~140MB/model)",
    )
    ap.add_argument(
        "--partial",
        action="store_true",
        help="upload even if fewer than 500 instances are present",
    )
    ap.add_argument("--public", action="store_true", help="create the repo public")
    ap.add_argument(
        "--final",
        action="store_true",
        help=(
            "publish a run that will not get any more instances: pad the "
            "missing ones with empty patches so the file covers all 500, and "
            "record which instances never produced a result"
        ),
    )
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = read_token()
    env = load_env()
    odir = output_dir(env, args.model)
    out_jsonl = odir / "output.jsonl"
    if not out_jsonl.exists():
        sys.exit(f"no output.jsonl at {out_jsonl}")

    preds = build_predictions(out_jsonl, args.model)
    inferred = len(preds)
    missing: list[str] = []
    if args.final:
        missing = pad_missing(preds, args.model)
        preds.sort(key=lambda r: r["instance_id"])
        print(f"final:      padded {len(missing)} instance(s) that never ran")
    n = len(preds)
    n_patched = sum(1 for p in preds if p["model_patch"].strip())
    print(f"model:      {args.model}")
    print(f"instances:  {n} / {EXPECTED_TOTAL}   (non-empty patches: {n_patched})")
    if n < EXPECTED_TOTAL and not args.partial:
        sys.exit(
            f"run is incomplete ({n}/{EXPECTED_TOTAL}); pass --partial to upload anyway"
        )

    meta = {
        "model": args.model,
        "instances_inferred": inferred,
        "inference_incomplete": bool(missing),
        "missing_instances": missing,
        "dataset": DATASET,
        "split": SPLIT,
        "instances": n,
        "expected_total": EXPECTED_TOTAL,
        "complete": n >= EXPECTED_TOTAL,
        "non_empty_patches": n_patched,
        "sdk_short_sha": env["sdk"],
        "max_iterations": int(env["maxiter"]),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "resolved_status": "unevaluated",
        "note": (
            "resolved is null for every record until an evaluation machine "
            "with docker runs the SWE-bench harness and fills it in."
        ),
        "final_note": (
            "This run is final. The instances in missing_instances never "
            "produced a result -- the remote runtime ended their conversations "
            "with an error on every attempt -- so they carry an empty patch and "
            "inference_failed=true. They are infrastructure failures, not model "
            "output. They are included so the denominator stays 500 and models "
            "remain comparable; they will grade as unresolved."
        )
        if missing
        else None,
    }

    api = HfApi(token=token)
    api.create_repo(
        args.repo, repo_type="dataset", exist_ok=True, private=not args.public
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "predictions.jsonl"
        with p.open("w") as f:
            for rec in preds:
                f.write(json.dumps(rec) + "\n")
        m = td / "metadata.json"
        m.write_text(json.dumps(meta, indent=2) + "\n")

        base = args.model
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"{base}/predictions.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(m),
            path_in_repo=f"{base}/metadata.json",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"uploaded    {args.repo}:{base}/predictions.jsonl")

        if args.with_trajectories:
            gz = td / "output.jsonl.gz"
            with out_jsonl.open("rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
            size_mb = gz.stat().st_size / 1e6
            print(f"trajectories gz: {size_mb:.0f} MB, uploading ...")
            api.upload_file(
                path_or_fileobj=str(gz),
                path_in_repo=f"{base}/output.jsonl.gz",
                repo_id=args.repo,
                repo_type="dataset",
            )
            print(f"uploaded    {args.repo}:{base}/output.jsonl.gz")

    print(f"done: https://huggingface.co/datasets/{args.repo}/tree/main/{args.model}")


if __name__ == "__main__":
    main()
