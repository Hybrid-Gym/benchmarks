#!/usr/bin/env python3
"""Count instances in an output.jsonl that have a critic-PASSING trajectory.

The supervisor's cheap per-loop counter (`grep` for distinct instance_id) counts any
row, but `aggregate_results` writes a row whenever the run did not raise -- including
runs that ended with an empty git patch or without a finish action. Those are not
usable trajectories and must stay eligible for retry, so a run is only really done
when the *critic-passing* count reaches the target.

Parsing every row is slow (~1-2 min on a 1GB file), which is why the supervisor calls
this only at the moment its cheap counter claims completion.

Usage:
    count_critic_passing.py <output.jsonl> [--select <ids.txt>]

Prints a one-line summary to stderr and the passing count to stdout.
"""

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_jsonl")
    ap.add_argument("--select", help="file of instance ids, one per line")
    args = ap.parse_args()

    from benchmarks.utils.critics import evaluate_output
    from benchmarks.utils.models import EvalOutput
    from openhands.sdk.critic import AgentFinishedCritic

    # Judge with the critic the run was configured with, not whatever the default
    # happens to be -- a mode mismatch here would silently disagree with the runner
    # about which instances are done.
    mode = None
    meta_path = os.path.join(os.path.dirname(args.output_jsonl), "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            mode = (json.load(f).get("critic") or {}).get("mode")
    except Exception:
        pass
    critic = AgentFinishedCritic(mode=mode) if mode else AgentFinishedCritic()
    print(f"critic mode={critic.mode}", file=sys.stderr)

    passing: set[str] = set()
    seen: set[str] = set()
    unusable = 0
    try:
        with open(args.output_jsonl, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    seen.add(row["instance_id"])
                    if evaluate_output(critic, EvalOutput.model_validate(row)):
                        passing.add(row["instance_id"])
                    else:
                        unusable += 1
                except Exception:
                    unusable += 1
    except FileNotFoundError:
        print(0)
        return 0

    if args.select:
        with open(args.select, encoding="utf-8") as f:
            sel = {ln.strip() for ln in f if ln.strip()}
        passing &= sel
        seen &= sel

    print(
        f"critic-passing={len(passing)} recorded={len(seen)} unusable_rows={unusable}",
        file=sys.stderr,
    )
    print(len(passing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
