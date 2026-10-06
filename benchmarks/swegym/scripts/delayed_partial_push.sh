#!/usr/bin/env bash
# Waits DELAY_SECONDS (default 4h), then pushes a combined snapshot of whatever
# SWE-Gym eval has finished by then -- one HF dataset repo, all models, every row
# tagged `model` + `evaluated` + `resolved` (resolved is null until evaluated=True,
# so "not scored yet" never gets confused with "scored and failed").
#
# One-shot: run this in its own tmux session so it survives the launching shell.
set -u
REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

DELAY_SECONDS="${DELAY_SECONDS:-14400}"
REPORTS_DIR=eval_outputs/swegym_outputs/eval_reports
BATCH_DIR=eval_outputs/swegym_outputs/eval_batches
LOG=eval_outputs/swegym_outputs/delayed_partial_push.log
HF_REPO="${HF_REPO:-synthetic-code-training/swegym_partial_snapshot}"

say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

declare -A RUN_DIR=(
  [gpt5mini]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/azure/openai/gpt-5-mini_sdk_e212d45_maxiter_60_N_swegym-gpt5mini-1500"
  [qwen80b]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/qwen/qwen3-next-80b-a3b-instruct_sdk_e212d45_maxiter_60_N_swegym-qwen80b-1500"
  [kimi25]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/moonshotai/kimi-k2.5_sdk_e212d45_maxiter_100_N_swegym-kimi25-1500"
  [dv4f]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/deepseek-ai/deepseek-v4-flash_sdk_e212d45_maxiter_100_N_swegym-dv4f-1500"
)
MODELS="${MODELS:-gpt5mini qwen80b kimi25 dv4f}"

say "delayed_partial_push: sleeping ${DELAY_SECONDS}s, then snapshotting models='$MODELS' -> $HF_REPO"
sleep "$DELAY_SECONDS"
say "delayed_partial_push: waking up, building snapshot"

model_args=()
for m in $MODELS; do
  rd="${RUN_DIR[$m]:-}"
  if [ -z "$rd" ] || [ ! -f "$rd/output.jsonl" ]; then
    say "$m: no run dir / output.jsonl, skipping entirely"
    continue
  fi

  total=$(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | wc -l)
  have=$(ls "$REPORTS_DIR"/batch_*."$m".json 2>/dev/null | wc -l)
  if [ "$total" -gt 0 ] && [ "$have" -gt 0 ]; then
    say "$m: merging $have/$total available batch reports (partial ok)"
    python3 benchmarks/swegym/scripts/merge_batch_reports.py \
      --reports-dir "$REPORTS_DIR" --batch-dir "$BATCH_DIR" --models "$m" 2>&1 | tee -a "$LOG"
    cp "$REPORTS_DIR/report_$m.json" "$rd/output.report.json"
  else
    say "$m: no eval batches scored yet -- all rows will be evaluated=False"
  fi

  if [ ! -f "$rd/output.with_completions.jsonl.gz" ]; then
    say "$m: building non-fncall export (can take a while)"
    .venv/bin/python benchmarks/utils/post_process_scripts/combine_completions.py \
      "$rd/output.jsonl" >>"$LOG" 2>&1
  fi

  if [ -f "$rd/output.with_completions.jsonl.gz" ]; then
    model_args+=(--model "$m=$rd")
  else
    say "$m: combine_completions failed to produce output.with_completions.jsonl.gz, skipping"
  fi
done

if [ "${#model_args[@]}" -eq 0 ]; then
  say "delayed_partial_push: nothing to push, exiting"
  exit 0
fi

say "delayed_partial_push: pushing snapshot -> $HF_REPO (${#model_args[@]} model(s))"
.venv/bin/python benchmarks/swegym/scripts/push_partial_snapshot.py \
  --repo "$HF_REPO" "${model_args[@]}" 2>&1 | tee -a "$LOG"
say "delayed_partial_push: done"
