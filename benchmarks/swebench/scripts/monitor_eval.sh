#!/bin/bash
# Monitor inference output files and trigger evaluation at line count thresholds (100, 150, 200, ...)
# Usage: bash monitor_eval.sh

set -euo pipefail

SDK_SHORT_SHA="e212d45"
MAX_ITER="60"
OGMA_USER="yiqingxi"
OGMA_HOST="ogma.lti.cs.cmu.edu"
OGMA_STORAGE_DIR="/projects/ogma3/yiqingxi"
STORAGE_DIR="${STORAGE_DIR:-/data/tir/projects/tir5/users/yiqingxi}"

# Models to monitor: "MODEL_NAME"
MODELS=(
    "qwen25-coder-7b-r2egym-gpt5mini-1500i-multi-search-false250"
    "qwen3-4b-func-localize-claude45-1457i-read-narrow-false250"
)

THRESHOLD_STEP=50
POLL_INTERVAL=60  # seconds between checks

declare -A last_triggered

for model in "${MODELS[@]}"; do
    last_triggered[$model]=50  # start below first threshold so 100 triggers
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_eval() {
    local model=$1
    local line_count=$2

    local OUTPUT_DIR="${STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${model}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"
    local OGMA_OUTPUT_DIR="${OGMA_STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${model}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"

    log "=== Triggering eval for $model at $line_count lines ==="

    log "Rcloning output.jsonl to ogma..."
    rclone copy "$OUTPUT_DIR/output.jsonl" "ogma:$OGMA_OUTPUT_DIR"

    log "Running docker eval on ogma..."
    local MAX_ATTEMPTS=3
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        log "Docker eval attempt $attempt/$MAX_ATTEMPTS..."
        if ssh "$OGMA_USER@$OGMA_HOST" "cd /home/${OGMA_USER}/benchmarks && bash benchmarks/swebench/scripts/docker_eval.sh $model $MAX_ITER"; then
            log "SSH eval command succeeded"
        else
            log "SSH eval command returned non-zero (may still have produced report)"
        fi

        rclone copy "ogma:$OGMA_OUTPUT_DIR/output.report.json" "$OUTPUT_DIR"

        if [ -f "$OUTPUT_DIR/output.report.json" ]; then
            log "Report file received."
            break
        fi

        if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
            log "Report file not found. Retrying..."
        else
            log "Report file not found after $MAX_ATTEMPTS attempts."
        fi
    done

    log "Running extra_eval.py for $model..."
    python benchmarks/swebench/extra_eval.py --split all --input_file "$OUTPUT_DIR/output.jsonl" --total_num -1
}

log "Starting monitor. Models: ${MODELS[*]}"
log "Threshold step: $THRESHOLD_STEP lines (triggers at 100, 150, 200, ...)"

while true; do
    all_done=true

    for model in "${MODELS[@]}"; do
        OUTPUT_DIR="${STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${model}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"
        JSONL="$OUTPUT_DIR/output.jsonl"

        if [ ! -f "$JSONL" ]; then
            log "[$model] output.jsonl not found yet"
            all_done=false
            continue
        fi

        line_count=$(wc -l < "$JSONL")
        prev_threshold=${last_triggered[$model]}
        next_threshold=$(( (prev_threshold / THRESHOLD_STEP + 1) * THRESHOLD_STEP ))
        # Make sure next_threshold is at least 100
        if [ "$next_threshold" -lt 100 ]; then
            next_threshold=100
        fi

        log "[$model] Lines: $line_count, next threshold: $next_threshold"

        if [ "$line_count" -ge "$next_threshold" ]; then
            # Check if this is the final file (500 lines = full dataset)
            run_eval "$model" "$line_count"
            last_triggered[$model]=$next_threshold
        fi

        # Check if inference seems complete (500 is SWE-bench Verified size)
        if [ "$line_count" -lt 500 ]; then
            all_done=false
        fi
    done

    if [ "$all_done" = true ]; then
        log "Both models appear complete (>=500 lines). Exiting monitor."
        break
    fi

    sleep $POLL_INTERVAL
done
