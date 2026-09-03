#!/usr/bin/env bash
# Run every model in models.txt through FULL SWE-bench_Verified, one at a time.
#
#   bash run_queue.sh [models_file]
#
# For each model: serve (download if needed) -> wait until the tunnel answers
# a real completion -> run inference on all 500 instances -> tear the server
# down -> move on. Progress is recorded in $STATUS_FILE; render it with
# status.sh. Safe to kill and restart — finished models are skipped and
# partially-finished ones resume from output.jsonl.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODELS_FILE="${1:-$SCRIPT_DIR/models.txt}"
PURGE_CHECKPOINTS="${PURGE_CHECKPOINTS:-0}"   # 1 = delete checkpoint after a model finishes
SERVE_READY_TIMEOUT="${SERVE_READY_TIMEOUT:-2400}"  # seconds to wait for READY

# Number of instances that constitutes a complete run. Derived from the
# dataset itself so a DONE verdict can never be based on a stale constant;
# falls back to the known SWE-bench_Verified test size if that lookup fails.
resolve_total_instances() {
    local n
    n="$(cd "$REPO_DIR" && timeout 600 uv run python -c "
from benchmarks.utils.dataset import get_dataset
print(len(get_dataset('$DATASET', '$SPLIT')))
" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [[ -n "$n" && "$n" -gt 0 ]]; then echo "$n"; else echo 500; fi
}
TOTAL_INSTANCES="${TOTAL_INSTANCES:-}"

# Short, append-only narrative of the run. Survives the node being killed and
# is the first thing to read when picking the run back up: one line per event
# plus a progress heartbeat, with the verbose vLLM/infer output kept in the
# per-model logs instead.
RUN_LOG="$RUN_LOG_ROOT/run.log"
log() {
    local line="[$(date '+%F %T')] $*"
    echo "$line"
    echo "$line" >> "$RUN_LOG"
}

# --- status table ---------------------------------------------------------
init_status() {
    if [[ ! -f "$STATUS_FILE" ]]; then
        printf 'model\tstate\tdone\ttotal\tstarted\tupdated\tnote\n' > "$STATUS_FILE"
    fi
}

set_status() {
    # set_status <save_name> <state> <done> <note>
    local name="$1" state="$2" done_n="$3" note="${4:-}"
    local now started tmp
    now="$(date '+%F %T')"
    started="$(awk -F'\t' -v n="$name" '$1==n {print $5}' "$STATUS_FILE" | tail -1)"
    [[ -z "$started" ]] && started="$now"
    tmp="$(mktemp "$TMP_ROOT/status.XXXXXX")"
    awk -F'\t' -v OFS='\t' -v n="$name" '$1!=n' "$STATUS_FILE" > "$tmp"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$state" "$done_n" "$TOTAL_INSTANCES" "$started" "$now" "$note" >> "$tmp"
    mv "$tmp" "$STATUS_FILE"
}

get_state() { awk -F'\t' -v n="$1" '$1==n {print $2}' "$STATUS_FILE" | tail -1; }

count_done() {
    local out="$1/output.jsonl"
    [[ -f "$out" ]] && wc -l < "$out" | tr -d ' ' || echo 0
}

# --- per-model run --------------------------------------------------------
run_one() {
    local hf_name="$1"
    local name; name="$(save_name_of "$hf_name")"
    local log_dir="$RUN_LOG_ROOT/$name"
    local out_dir; out_dir="$(output_dir_of "$name")"
    mkdir -p "$log_dir"

    local done_n; done_n="$(count_done "$out_dir")"
    if [[ "$(get_state "$name")" == "DONE" ]]; then
        log "SKIP $name (already DONE, $done_n/$TOTAL_INSTANCES)"
        return 0
    fi

    log "=== $name ==="
    set_status "$name" "SERVING" "$done_n" "starting vLLM + ngrok"

    local ready_file="$log_dir/base_url.txt"
    rm -f "$ready_file"
    bash "$SCRIPT_DIR/serve.sh" "$hf_name" > "$log_dir/serve.log" 2>&1 &
    local serve_pid=$!
    log "serve pid $serve_pid, log $log_dir/serve.log"

    # Wait for the server to be genuinely usable (READY file written only
    # after a chat completion succeeds through the tunnel).
    local waited=0
    while [[ ! -s "$ready_file" ]]; do
        if ! kill -0 "$serve_pid" 2>/dev/null; then
            log "FAIL $name: serve.sh exited before becoming ready"
            set_status "$name" "FAILED" "$done_n" "serve failed; see serve.log"
            return 1
        fi
        if (( waited >= SERVE_READY_TIMEOUT )); then
            log "FAIL $name: serve did not become ready in ${SERVE_READY_TIMEOUT}s"
            set_status "$name" "FAILED" "$done_n" "serve timeout"
            kill "$serve_pid" 2>/dev/null || true
            return 1
        fi
        sleep 10; waited=$((waited + 10))
    done
    local base_url; base_url="$(cat "$ready_file")"
    log "served at $base_url (${waited}s)"

    set_status "$name" "INFERRING" "$done_n" "$base_url"

    # Background progress ticker: keeps the table live and drops a heartbeat
    # into run.log every 5 min so progress is recoverable after a node kill.
    ( while kill -0 "$serve_pid" 2>/dev/null; do
        sleep 300
        local n; n="$(count_done "$out_dir")"
        set_status "$name" "INFERRING" "$n" "$base_url"
        log "  .. $name $n/$TOTAL_INSTANCES done"
      done ) &
    local ticker_pid=$!

    bash "$SCRIPT_DIR/infer.sh" "$hf_name" > "$log_dir/infer.log" 2>&1
    local rc=$?
    kill "$ticker_pid" 2>/dev/null || true

    log "teardown: stopping serve pid $serve_pid"
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
    pkill -u "$USER" -f "ngrok http $SERVE_PORT" 2>/dev/null || true
    sleep 10

    done_n="$(count_done "$out_dir")"
    if [[ $rc -eq 0 && "$done_n" -ge "$TOTAL_INSTANCES" ]]; then
        set_status "$name" "DONE" "$done_n" "$out_dir"
        log "DONE $name ($done_n/$TOTAL_INSTANCES)"

        # Publish predictions to the Hub for grading elsewhere (no docker here).
        # Never let a Hub hiccup fail the sweep — the local results are the
        # source of truth and the upload can always be re-run by hand.
        local up=(python3 "$SCRIPT_DIR/upload_hf.py" "$name")
        [[ "$UPLOAD_TRAJECTORIES" == "1" ]] && up+=(--with-trajectories)
        if (cd "$REPO_DIR" && "${up[@]}" >> "$log_dir/upload.log" 2>&1); then
            log "uploaded $name to HF ($HF_RESULTS_REPO)"
        else
            log "WARN upload failed for $name; see $log_dir/upload.log (results are safe locally)"
        fi
        if [[ "$PURGE_CHECKPOINTS" == "1" ]]; then
            log "purging checkpoint $CKPT_ROOT/$name"
            rm -rf "${CKPT_ROOT:?}/$name"
        fi
    else
        set_status "$name" "PARTIAL" "$done_n" "rc=$rc; rerun to resume"
        log "PARTIAL $name ($done_n/$TOTAL_INSTANCES, rc=$rc)"
    fi
    return 0
}

# --- main -----------------------------------------------------------------
init_status
if [[ -z "$TOTAL_INSTANCES" ]]; then
    log "resolving instance count for $DATASET/$SPLIT ..."
    TOTAL_INSTANCES="$(resolve_total_instances)"
fi
log "a complete run = $TOTAL_INSTANCES instances"
mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")
log "queue of ${#MODELS[@]} models from $MODELS_FILE"
log "outputs -> $EVAL_OUT_ROOT"
log "status  -> $STATUS_FILE"

# Seed any model that has never been seen before as PENDING.
for m in "${MODELS[@]}"; do
    n="$(save_name_of "$m")"
    if [[ -z "$(get_state "$n")" ]]; then
        set_status "$n" "PENDING" "$(count_done "$(output_dir_of "$n")")" ""
    fi
done

for m in "${MODELS[@]}"; do
    run_one "$m"
done

log "queue finished"
bash "$SCRIPT_DIR/status.sh"
