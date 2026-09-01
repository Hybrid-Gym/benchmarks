#!/usr/bin/env bash
# Waits for the live dv4f rollout (swegym-infer) to exit, then runs the batch eval
# scoped to dv4f alone (reusing the existing 1500-selection eval_batches), and once
# that eval is fully done, pushes it via autopush.sh -- unattended end-to-end so the
# push lands without anyone needing to notice dv4f finished.
#
# Launched once on 2026-08-26 alongside the other four models' autopush watcher; see
# resume_all.sh / autopush.sh for the sibling pieces.
set -u
REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1
LOG=eval_outputs/swegym_outputs/dv4f_pipeline.log
say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

say "dv4f pipeline: waiting for rollout (swegym-infer .../dv4f-1500) to exit"
while pgrep -f "swegym-infer.*dv4f-1500" >/dev/null 2>&1; do
  sleep 120
done
n=$(wc -l < eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/deepseek-ai/deepseek-v4-flash_sdk_e212d45_maxiter_100_N_swegym-dv4f-1500/output.jsonl 2>/dev/null || echo 0)
say "dv4f rollout exited with $n/1500 rows -- starting batch eval"

MODELS="dv4f" bash benchmarks/swegym/scripts/batch_eval.sh
say "dv4f batch eval finished -- handing off to autopush"

MODELS="dv4f" POLL=60 bash benchmarks/swegym/scripts/autopush.sh
say "dv4f pipeline: done"
