#!/usr/bin/env bash
# Watches SWE-Gym batch eval for the given models; once a model's eval batches are all
# present, merges the per-batch reports into output.report.json, builds the non-fncall
# training export, and pushes to HF -- unattended, so a finished eval doesn't sit
# waiting on a human to notice before it lands on HF.
#
# Convention (matches r2egym): push ALL rows in output.jsonl (not just critic-passing),
# `resolved` taken per-row from the merged eval report's resolved_ids.
#
# Re-entrant per model: once DONE[$m] is set the model is skipped on later loop passes,
# and a re-run of merge/combine/push for an already-pushed model just overwrites its own
# outputs (push_to_hub with the same repo id is a new commit, not a duplicate dataset).
set -u
REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

MODELS="${MODELS:-gpt5mini qwen80b}"
REPORTS_DIR=eval_outputs/swegym_outputs/eval_reports
BATCH_DIR=eval_outputs/swegym_outputs/eval_batches
LOG=eval_outputs/swegym_outputs/autopush.log
POLL="${POLL:-300}"

say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

declare -A RUN_DIR=(
  [gpt5mini]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/azure/openai/gpt-5-mini_sdk_e212d45_maxiter_60_N_swegym-gpt5mini-1500"
  [qwen80b]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/qwen/qwen3-next-80b-a3b-instruct_sdk_e212d45_maxiter_60_N_swegym-qwen80b-1500"
  [kimi25]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/moonshotai/kimi-k2.5_sdk_e212d45_maxiter_100_N_swegym-kimi25-1500"
  [opus45]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/aws/anthropic/claude-opus-4-5_sdk_e212d45_maxiter_100_N_swegym-opus45-1500"
  [dv4f]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/deepseek-ai/deepseek-v4-flash_sdk_e212d45_maxiter_100_N_swegym-dv4f-1500"
)

declare -A DONE
for m in $MODELS; do DONE[$m]=0; done

say "autopush start: watching models='$MODELS' (poll every ${POLL}s)"

while :; do
  all_done=1
  for m in $MODELS; do
    if [ "${DONE[$m]:-0}" = "1" ]; then
      continue
    fi
    all_done=0

    total=$(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | wc -l)
    have=$(ls "$REPORTS_DIR"/batch_*."$m".json 2>/dev/null | wc -l)
    if [ "$total" -eq 0 ] || [ "$have" -lt "$total" ]; then
      continue
    fi

    say "$m: all $total batches present ($have reports) -- pushing"
    rd="${RUN_DIR[$m]:-}"
    if [ -z "$rd" ] || [ ! -f "$rd/output.jsonl" ]; then
      say "$m: no run dir / output.jsonl at '$rd', skipping"
      DONE[$m]=1
      continue
    fi

    python3 benchmarks/swegym/scripts/merge_batch_reports.py \
      --reports-dir "$REPORTS_DIR" --batch-dir "$BATCH_DIR" --models "$m" 2>&1 | tee -a "$LOG"
    cp "$REPORTS_DIR/report_$m.json" "$rd/output.report.json"

    say "$m: building non-fncall export (this can take a while for 1500 rows)"
    .venv/bin/python benchmarks/utils/post_process_scripts/combine_completions.py \
      "$rd/output.jsonl" >>"$LOG" 2>&1

    src="${rd}/output.with_completions.jsonl.gz"
    if [ ! -f "$src" ]; then
      say "$m: expected $src not found after combine_completions, ABORTING push for $m"
      DONE[$m]=1
      continue
    fi

    n=$(wc -l < "$rd/output.jsonl" | tr -d ' ')
    repo="synthetic-code-training/swegym_${m}_${n}i"
    say "$m: pushing $n rows to $repo"
    .venv/bin/python benchmarks/utils/post_process_scripts/convert_and_push.py \
      --src "$src" --repo "$repo" 2>&1 | tee -a "$LOG"

    say "$m: DONE -> $repo"
    DONE[$m]=1
  done

  if [ "$all_done" = "1" ]; then
    say "autopush: all models pushed, exiting"
    break
  fi
  sleep "$POLL"
done
