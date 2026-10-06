#!/bin/bash
# Auto-continue orchestrator for the funclocalize multi-model pipeline.
#
# Runs each model SEQUENTIALLY (not parallel) so clone bandwidth and disk
# pressure don't compound. For each model:
#   1. Rollout (resumes from existing output dir if present)
#   2. Retry pass (if errors > threshold; backs up critic + clears errors file)
#   3. Combine raw_completions from event history
#   4. Filter to valid-docstring patches
#   5. Push non-fncall trajectories to HF
#
# Each step writes a `.done` stamp so re-running this script after an
# interruption (Claude session dies, host reboots, etc.) skips completed steps.
#
# Designed to live inside a tmux session: `tmux new -s funclocalize-pipeline`
# then run this script. The tmux session survives Claude session lifecycle.

set -uo pipefail

cd /home/gaokaizhang/benchmarks

# ---- Configuration --------------------------------------------------------

# Queue format: tag:config_file:hf_repo_base
# - tag         used in run note, tmux session name (not used here), stamps
# - config_file LLM config under .llm_config/
# - hf_repo_base resolved count gets appended: <base>_<N>i
QUEUE=(
  "gpt55_loc_strategy:anthropic_gpt5_5_funclocalize.json:synthetic-code-training/func_localize_gpt55_loc_strategy:default_loc_strategy.j2"
)
# Positional args override the QUEUE — lets a second tmux session run a different
# queue without editing this file (which would race the in-memory copy of any
# already-running invocation).
if [[ $# -gt 0 ]]; then
  QUEUE=("$@")
fi

DATASET_NAME="synthetic-code-training/swe_doc_gen_locate_1500"
DATASET_SPLIT="train"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
MAX_ROLLOUT_PASSES="${MAX_ROLLOUT_PASSES:-3}"    # rollout + up-to-2 retry passes
OK_THRESHOLD="${OK_THRESHOLD:-1450}"             # stop retrying once we hit this
N_LIMIT="${N_LIMIT:-0}"                          # 0 = no limit; set to cap dataset
# Auto-lower OK_THRESHOLD to ~97% of N_LIMIT when N_LIMIT is set
if (( N_LIMIT > 0 && OK_THRESHOLD > N_LIMIT )); then
  OK_THRESHOLD=$(( (N_LIMIT * 97) / 100 ))
fi
# Push step only; export HF_TOKEN or `huggingface-cli login` before running.
HF_TOKEN="${HF_TOKEN:-}"

PIPELINE_LOG="/tmp/funclocalize-pipeline.log"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$PIPELINE_LOG"
}

# ---- Per-model pipeline ---------------------------------------------------

run_model() {
  local TAG="$1" CFG="$2" HF_REPO_BASE="$3" PROMPT="${4:-}"

  local RUN_NOTE_FILE="/tmp/$TAG-1500.runnote"
  local RUN_NOTE_VAL
  if [[ -f "$RUN_NOTE_FILE" ]]; then
    RUN_NOTE_VAL=$(cat "$RUN_NOTE_FILE")
    log "[$TAG] reusing run note: $RUN_NOTE_VAL"
  else
    RUN_NOTE_VAL="$TAG-1500-funclocalize-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "$RUN_NOTE_VAL" > "$RUN_NOTE_FILE"
    log "[$TAG] new run note: $RUN_NOTE_VAL"
  fi

  # Resolve OUT dir from LLM config's model field
  local MODEL_PATH
  MODEL_PATH=$(python3 -c "import json; print(json.load(open('.llm_config/$CFG'))['model'])")
  local OUT="./eval_outputs/synthetic-code-training__swe_doc_gen_locate_1500-train/${MODEL_PATH}_sdk_e212d45_maxiter_60_N_${RUN_NOTE_VAL}"
  mkdir -p "$OUT"

  local STAMP_DIR="$OUT/.stamps"
  mkdir -p "$STAMP_DIR"

  # ---- Rollout (with retries) --------------------------------------------
  if [[ ! -f "$STAMP_DIR/rollout_done" ]]; then
    local pass=0
    while (( pass < MAX_ROLLOUT_PASSES )); do
      pass=$((pass + 1))
      local OK ERR
      OK=$(_count_ok "$OUT")
      ERR=$(wc -l < "$OUT/output_errors.jsonl" 2>/dev/null || echo 0)
      log "[$TAG] rollout pass $pass/$MAX_ROLLOUT_PASSES (current ok=$OK err=$ERR)"

      if (( OK >= OK_THRESHOLD )); then
        log "[$TAG] ok threshold ($OK_THRESHOLD) reached, skipping further passes"
        break
      fi

      if (( pass > 1 )); then
        # Reclaim disk between passes — safe because all containers from the
        # previous pass have exited and auto-removed; this just clears stale
        # build cache and any dangling images.
        log "[$TAG] mid-pipeline cleanup before retry pass"
        docker container prune -f > /dev/null 2>&1
        docker image prune -f > /dev/null 2>&1
        docker builder prune -f > /dev/null 2>&1

        # Retry pass: back up + filter critic to only the ok IDs so orchestrator re-queues errors
        log "[$TAG] preparing retry pass: filter critic + clear errors file"
        cp -f "$OUT/output.critic_attempt_1.jsonl" "$OUT/output.critic_attempt_1.pre-retry$pass.bak" 2>/dev/null
        cp -f "$OUT/output_errors.jsonl" "$OUT/output_errors.pre-retry$pass.bak" 2>/dev/null
        python3 - "$OUT" <<'PY'
import json, sys, os
OUT = sys.argv[1]
ok_ids = {json.loads(l)["instance_id"] for l in open(f"{OUT}/output.jsonl")}
src = f"{OUT}/output.critic_attempt_1.jsonl"
tmp = src + ".tmp"
with open(tmp, "w") as fo:
    for line in open(src):
        if json.loads(line).get("instance_id") in ok_ids:
            fo.write(line)
os.replace(tmp, src)
open(f"{OUT}/output_errors.jsonl", "w").close()
print(f"critic filtered to {len(ok_ids)} ok IDs; errors file cleared")
PY
      fi

      # Run rollout
      LLM_CONFIG_PATH=".llm_config/$CFG" \
      DATASET_NAME="$DATASET_NAME" \
      DATASET_SPLIT="$DATASET_SPLIT" \
      WORKSPACE_TYPE=docker \
      NUM_WORKERS="$NUM_WORKERS" \
      MAX_ITERATIONS="$MAX_ITERATIONS" \
      N_LIMIT="$N_LIMIT" \
      RUN_NOTE="$RUN_NOTE_VAL" \
      PROMPT_PATH="$PROMPT" \
      bash benchmarks/hybridgym_funclocalize/scripts/test_infer.sh 2>&1 | tee -a "/tmp/$TAG-1500.log"
    done

    touch "$STAMP_DIR/rollout_done"
    log "[$TAG] rollout phase done. final ok=$(_count_ok "$OUT") err=$(wc -l < "$OUT/output_errors.jsonl" 2>/dev/null || echo 0)"
  else
    log "[$TAG] rollout already stamped done (skipping)"
  fi

  # ---- Combine raw_completions from events -------------------------------
  # Scripts live under benchmarks/utils/post_process_scripts/ (upstream relocation;
  # used to be ./scripts/). The orchestrator now also checks the exit code so
  # silent failures don't get a stamp written.
  if [[ ! -f "$STAMP_DIR/combined_done" ]]; then
    log "[$TAG] combine: reconstructing raw_completions from event history"
    set -o pipefail
    if uv run python benchmarks/utils/post_process_scripts/combine_completions.py "$OUT/output.jsonl" 2>&1 | tail -3; then
      touch "$STAMP_DIR/combined_done"
    else
      log "[$TAG] combine FAILED — not stamping (will retry on next invocation)"
      return 1
    fi
    set +o pipefail
  else
    log "[$TAG] combine already done"
  fi

  # ---- Filter to valid-docstring patches --------------------------------
  if [[ ! -f "$STAMP_DIR/filtered_done" ]]; then
    log "[$TAG] filter: applying judge_valid_docstring_patch"
    set -o pipefail
    if uv run python benchmarks/utils/post_process_scripts/funclocalize_filter_valid_docstring.py "$OUT/output.with_completions.jsonl.gz" 2>&1 | tail -3; then
      touch "$STAMP_DIR/filtered_done"
    else
      log "[$TAG] filter FAILED — not stamping"
      return 1
    fi
    set +o pipefail
  else
    log "[$TAG] filter already done"
  fi

  # ---- Convert + push ----------------------------------------------------
  if [[ ! -f "$STAMP_DIR/pushed_done" ]]; then
    local N_RESOLVED
    N_RESOLVED=$(zcat "$OUT/output_success.with_completions.jsonl.gz" | wc -l)
    local HF_REPO="${HF_REPO_BASE}_${N_RESOLVED}i"
    log "[$TAG] push: $N_RESOLVED trajectories → $HF_REPO"
    set -o pipefail
    if HF_TOKEN="$HF_TOKEN" uv run python benchmarks/utils/post_process_scripts/convert_and_push.py \
        --src "$OUT/output_success.with_completions.jsonl.gz" \
        --repo "$HF_REPO" \
        --out-jsonl "$OUT/funclocalize_${TAG}_nonfncall.jsonl" 2>&1 | tail -3; then
      echo "$HF_REPO" > "$STAMP_DIR/pushed_repo"
      touch "$STAMP_DIR/pushed_done"
    else
      log "[$TAG] push FAILED — not stamping"
      return 1
    fi
    set +o pipefail
  else
    log "[$TAG] push already done ($(cat "$STAMP_DIR/pushed_repo" 2>/dev/null))"
  fi

  # ---- Between-runs cleanup ---------------------------------------------
  log "[$TAG] cleanup: pruning stopped containers + dangling images"
  docker container prune -f > /dev/null 2>&1
  docker image prune -f > /dev/null 2>&1
}

_count_ok() {
  python3 - "$1" <<'PY'
import json, sys, os
out = sys.argv[1]
path = f"{out}/output.jsonl"
if not os.path.exists(path):
    print(0); raise SystemExit
n = 0
for line in open(path):
    try:
        if not json.loads(line).get("error"): n += 1
    except Exception: pass
print(n)
PY
}

# ---- Main loop ------------------------------------------------------------

log "=== Pipeline starting; queue length: ${#QUEUE[@]} ==="
log "=== Workers: $NUM_WORKERS  max_iter: $MAX_ITERATIONS  max_passes: $MAX_ROLLOUT_PASSES ==="

for ENTRY in "${QUEUE[@]}"; do
  IFS=':' read -r TAG CFG HF PROMPT <<< "$ENTRY"
  log ""
  log "================================================================"
  log "== Starting model: $TAG ($CFG → $HF)"
  log "================================================================"
  run_model "$TAG" "$CFG" "$HF" "${PROMPT:-}"
done

log ""
log "=== Pipeline complete ==="
