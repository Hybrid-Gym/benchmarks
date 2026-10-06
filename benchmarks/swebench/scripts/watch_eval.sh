#!/bin/bash
# Monitor output.jsonl for a given model and run evaluation at line-count thresholds.
# Usage: bash benchmarks/swebench/scripts/watch_eval.sh <MODEL_NAME>

MODEL_NAME="${1:?Usage: $0 <model_name>}"

SDK_SHORT_SHA="e212d45"
MAX_ITER="60"
OGMA_USER="yiqingxi"
OGMA_HOST="ogma.lti.cs.cmu.edu"
OGMA_STORAGE_DIR="/projects/ogma3/yiqingxi"
STORAGE_DIR="${STORAGE_DIR:-/data/tir/projects/tir5/users/yiqingxi}"

OUTPUT_DIR="${STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"
JSONL="${OUTPUT_DIR}/output.jsonl"
OGMA_OUTPUT_DIR="${OGMA_STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"

EVAL_LOG="/tmp/watch_eval_${MODEL_NAME}_eval.log"
OGMA_LOCK="/tmp/ogma_eval_lock_${MODEL_NAME}"
THRESHOLD_STEP=50
POLL_INTERVAL=30
PROGRESS_INTERVAL=120

# State tracking
last_printed_threshold=0   # last threshold for which extra_eval was printed
last_eval_threshold=0       # last threshold for which eval was triggered
eval_pid=""
eval_was_running=false
last_progress_ts=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

header() {
    local msg="$1"
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    printf "  %-54s\n" "$msg"
    echo "╚══════════════════════════════════════════════════════╝"
}

print_report_summary() {
    local report="${OUTPUT_DIR}/output.report.json"
    if [ -f "$report" ]; then
        python3 -c "
import json
d = json.load(open('$report'))
print(f\"  Instances submitted: {d.get('submitted_instances', '?')}\")
print(f\"  Instances resolved:  {d.get('resolved_instances', '?')}\")
" 2>/dev/null
    fi
}

eval_is_running() {
    [ -n "$eval_pid" ] && kill -0 "$eval_pid" 2>/dev/null
}

# Check whether ogma has an eval in progress via a lock file.
# The lock is created on ogma at the start of trigger_eval and removed when done.
# Using a lock file avoids the pgrep self-match false-positive (pgrep -f 'docker_eval.sh'
# would match the SSH-spawned shell whose cmdline contains that string).
# Returns 0 (true) if locked OR if SSH fails (conservative: block on error).
ogma_eval_is_running() {
    ssh -o ConnectTimeout=10 "$OGMA_USER@$OGMA_HOST" \
        "test -f '$OGMA_LOCK'" 2>/dev/null
    local rc=$?
    if [ "$rc" -eq 255 ]; then
        log "WARNING: SSH to ogma failed; assuming eval is running to avoid double-trigger."
        return 0
    fi
    return "$rc"
}

trigger_eval() {
    local threshold=$1
    > "$EVAL_LOG"
    log "Triggering eval for threshold=$threshold in background..."

    (
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Copying output.jsonl to ogma (threshold=$threshold)..."
        rclone copy "$JSONL" "ogma:$OGMA_OUTPUT_DIR"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Acquiring ogma eval lock..."
        ssh "$OGMA_USER@$OGMA_HOST" "touch '$OGMA_LOCK'" 2>/dev/null

        MAX_ATTEMPTS=3
        for attempt in $(seq 1 $MAX_ATTEMPTS); do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Docker eval attempt $attempt/$MAX_ATTEMPTS..."
            ssh "$OGMA_USER@$OGMA_HOST" \
                "cd /home/${OGMA_USER}/benchmarks && bash benchmarks/swebench/scripts/docker_eval.sh $MODEL_NAME $MAX_ITER"

            rclone copy "ogma:$OGMA_OUTPUT_DIR/output.report.json" "$OUTPUT_DIR"

            if [ -f "${OUTPUT_DIR}/output.report.json" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report file received."
                break
            fi

            if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report not found. Retrying..."
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Report not found after $MAX_ATTEMPTS attempts."
            fi
        done

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Releasing ogma eval lock..."
        ssh "$OGMA_USER@$OGMA_HOST" "rm -f '$OGMA_LOCK'" 2>/dev/null
    ) >> "$EVAL_LOG" 2>&1 &

    eval_pid=$!
    eval_was_running=true
    last_eval_threshold=$threshold
    last_progress_ts=$(date +%s)
}

# ── Startup ────────────────────────────────────────────────────────────────────
# Clear any stale ogma lock left by a previously crashed run.
ssh -o ConnectTimeout=10 "$OGMA_USER@$OGMA_HOST" "rm -f '$OGMA_LOCK'" 2>/dev/null \
    && log "Cleared stale ogma lock (if any)."

log "Model:      $MODEL_NAME"
log "JSONL:      $JSONL"
log "Extra-eval thresholds: 50, 100, 150, ... 450"
log "Full eval thresholds:     100, 150, ... 450 (copy to ogma + docker eval + report back)"
log "Progress reports every ${PROGRESS_INTERVAL}s while eval is running."
echo ""

while true; do
    now=$(date +%s)

    # ── Detect eval completion ─────────────────────────────────────────────────
    if [ "$eval_was_running" = true ] && ! eval_is_running; then
        wait "$eval_pid" 2>/dev/null
        eval_pid=""
        eval_was_running=false

        header "EVAL COMPLETE — $(date '+%Y-%m-%d %H:%M:%S') — threshold=${last_eval_threshold}"
        python benchmarks/swebench/extra_eval.py --total_num -1 --split all \
            --input_file "$JSONL"
        print_report_summary
        echo ""
        last_progress_ts=$now
    fi

    # ── Check line-count thresholds ────────────────────────────────────────────
    if [ ! -f "$JSONL" ]; then
        log "Waiting for $JSONL to appear..."
        sleep $POLL_INTERVAL
        continue
    fi

    line_count=$(wc -l < "$JSONL")

    # Process all newly crossed thresholds in order
    next_threshold=$(( last_printed_threshold + THRESHOLD_STEP ))
    if [ "$next_threshold" -lt 50 ]; then next_threshold=50; fi

    while [ "$line_count" -ge "$next_threshold" ] && [ "$next_threshold" -le 450 ]; do
        header "THRESHOLD ${next_threshold} — $(date '+%Y-%m-%d %H:%M:%S') — lines=${line_count}"

        # Always print extra_eval
        python benchmarks/swebench/extra_eval.py --total_num -1 --split all \
            --input_file "$JSONL"
        print_report_summary

        last_printed_threshold=$next_threshold

        # Trigger full eval at 100, 150, ..., 450
        if [ "$next_threshold" -ge 100 ]; then
            if eval_is_running; then
                log "Eval already running locally (pid=$eval_pid, threshold=${last_eval_threshold}); skipping new trigger."
            elif ogma_eval_is_running; then
                log "Ogma still running a docker_eval.sh; skipping new trigger."
            else
                trigger_eval "$next_threshold"
            fi
        fi

        next_threshold=$(( next_threshold + THRESHOLD_STEP ))
    done

    # ── Report in-progress eval every PROGRESS_INTERVAL ───────────────────────
    if eval_is_running; then
        elapsed=$(( now - last_progress_ts ))
        if [ "$elapsed" -ge "$PROGRESS_INTERVAL" ]; then
            echo ""
            echo "── EVAL IN PROGRESS $(date '+%H:%M:%S') ─────────────────────────────────"
            echo "  Model:     $MODEL_NAME"
            echo "  Pid:       $eval_pid  (threshold=${last_eval_threshold})"
            if [ -f "$EVAL_LOG" ]; then
                echo "  Recent log:"
                tail -6 "$EVAL_LOG" | sed 's/^/    /'
                # Surface tqdm Evaluation: progress if present
                progress=$(grep "Evaluation:" "$EVAL_LOG" 2>/dev/null | tail -1 \
                    | tr '\r' '\n' | grep "Evaluation:" | tail -1)
                [ -n "$progress" ] && echo "  Progress:  $progress"
            fi
            echo "─────────────────────────────────────────────────────────────────"
            last_progress_ts=$now
        fi
    fi

    sleep $POLL_INTERVAL
done
