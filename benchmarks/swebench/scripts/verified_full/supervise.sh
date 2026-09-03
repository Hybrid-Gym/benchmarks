#!/usr/bin/env bash
# Deadline-aware supervisor: runs the remaining models one at a time, uploads
# each to the Hub, and stops cleanly before the Slurm allocation expires.
#
#   bash supervise.sh [models_file]
#
# Why a supervisor rather than just run_queue.sh:
#   * The node is a preemptible Slurm job with a hard EndTime. Starting a model
#     that cannot finish wastes the remaining hours, so each model is gated on
#     having enough time left.
#   * Preemption can kill everything at any moment, so partial results are
#     uploaded periodically — a run cut short still leaves usable predictions
#     on the Hub instead of only on this node's disk.
#   * If the queue dies for any other reason and time remains, it is relaunched;
#     swebench-infer resumes from output.jsonl.
#   * Scripts are snapshotted read-only up front, because bash reads a script
#     by byte offset and editing one mid-run corrupts the running shell (this
#     is what ended the model-1 run with "unexpected EOF" after it logged DONE).
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SRC/env.sh"

MODELS_FILE="$(readlink -f "${1:-$SRC/models.txt}")"

MARGIN_MIN="${MARGIN_MIN:-25}"            # reserved at the end for final uploads
EST_MODEL_H="${EST_MODEL_H:-6.0}"         # model 1 measured 5.83h
MIN_START_H="${MIN_START_H:-2.0}"         # never start a model with less than this
POLL_S="${POLL_S:-300}"
PARTIAL_UPLOAD_S="${PARTIAL_UPLOAD_S:-1800}"

SUP_LOG="$RUN_LOG_ROOT/supervise.log"
mkdir -p "$RUN_LOG_ROOT"

log() { local l="[$(date '+%F %T')] $*"; echo "$l"; echo "$l" >> "$SUP_LOG"; }

# --- deadline -------------------------------------------------------------
deadline_epoch() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local e
        e="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
             | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)"
        if [[ -n "$e" && "$e" != "Unknown" ]]; then
            date -d "$e" +%s 2>/dev/null && return 0
        fi
    fi
    echo "${DEADLINE_EPOCH:-$(( $(date +%s) + 24 * 3600 ))}"
}

hours_left() {   # hours until the deadline, minus the reserved margin
    local dl now
    dl="$(deadline_epoch)"; now="$(date +%s)"
    awk -v d="$dl" -v n="$now" -v m="$MARGIN_MIN" \
        'BEGIN{printf "%.2f", (d - n)/3600.0 - m/60.0}'
}

lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a < b)}'; }   # a < b ?

# --- per-model helpers ----------------------------------------------------
save_name_of_() { local n; n="$(basename "$1")"; echo "${n//./}"; }

done_count() {
    local o; o="$(output_dir_of "$1")/output.jsonl"
    [[ -f "$o" ]] && wc -l < "$o" | tr -d ' ' || echo 0
}

upload() {   # upload <save_name> <partial|full>
    local name="$1" mode="$2" args=("$name")
    [[ "$mode" == "partial" ]] && args+=(--partial)
    [[ "$UPLOAD_TRAJECTORIES" == "1" ]] && args+=(--with-trajectories)
    if (cd "$REPO_DIR" && python3 "$SNAP/upload_hf.py" "${args[@]}") \
            >> "$RUN_LOG_ROOT/upload.log" 2>&1; then
        log "  uploaded $name ($mode, $(done_count "$name")/500)"
    else
        log "  WARN upload failed for $name ($mode); see upload.log"
    fi
}

# --- snapshot -------------------------------------------------------------
SNAP="$RUN_LOG_ROOT/snapshots/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SNAP"
cp "$SRC"/*.sh "$SRC"/*.py "$SRC"/models.txt "$SNAP"/ 2>/dev/null
chmod -w "$SNAP"/*.sh "$SNAP"/*.py 2>/dev/null

# --- preflight ------------------------------------------------------------
mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")
DL="$(deadline_epoch)"
log "=============================================================="
log "supervisor start"
log "  snapshot:  $SNAP"
log "  deadline:  $(date -d "@$DL" '+%F %T')  (margin ${MARGIN_MIN}m)"
log "  hours left after margin: $(hours_left)"
log "  models in list: ${#MODELS[@]}"

PENDING=()
for m in "${MODELS[@]}"; do
    n="$(save_name_of_ "$m")"
    d="$(done_count "$n")"
    if (( d >= 500 )); then
        log "  SKIP  $n (already 500/500)"
    else
        PENDING+=("$m"); log "  QUEUE $n ($d/500)"
    fi
done
FITS="$(awk -v h="$(hours_left)" -v e="$EST_MODEL_H" 'BEGIN{printf "%d", int(h/e)}')"
log "  pending: ${#PENDING[@]}   estimated to fit before deadline: $FITS"
[[ "$FITS" -lt "${#PENDING[@]}" ]] && \
    log "  NOTE only ~$FITS of ${#PENDING[@]} will fit; the rest need a new allocation"

# --- main loop ------------------------------------------------------------
for m in "${PENDING[@]}"; do
    name="$(save_name_of_ "$m")"
    hl="$(hours_left)"

    if lt "$hl" "$MIN_START_H"; then
        log "STOP  only ${hl}h left (< ${MIN_START_H}h): not starting $name"
        break
    fi
    if lt "$hl" "$EST_MODEL_H"; then
        log "WARN  ${hl}h left, a model needs ~${EST_MODEL_H}h — starting $name anyway;"
        log "      it will be partial-uploaded when the deadline margin is hit"
    fi

    log "START $name (${hl}h left before margin)"
    one="$RUN_LOG_ROOT/models_one_${name}.txt"
    echo "$m" > "$one"

    setsid nohup bash "$SNAP/run_queue.sh" "$one" \
        >> "$RUN_LOG_ROOT/queue_stdout.log" 2>&1 < /dev/null &
    sleep 5
    qpid="$(pgrep -u "$USER" -f "$SNAP/run_queue.sh $one" | head -1)"
    log "  queue pid ${qpid:-<unknown>}"

    last_upload="$(date +%s)"
    while :; do
        # queue finished?
        if [[ -z "$(pgrep -u "$USER" -f "$SNAP/run_queue.sh $one" | head -1)" ]]; then
            d="$(done_count "$name")"
            log "  queue exited for $name at $d/500"
            if (( d >= 500 )); then upload "$name" full
            elif (( d > 0 ));  then upload "$name" partial
            fi
            break
        fi

        # deadline reached -> stop everything and save what we have
        hl="$(hours_left)"
        if lt "$hl" 0; then
            log "DEADLINE margin reached (${hl}h): stopping $name"
            pkill -u "$USER" -f "$SNAP/run_queue.sh $one" 2>/dev/null || true
            sleep 10
            pkill -u "$USER" -f "vllm serve" 2>/dev/null || true
            pkill -u "$USER" -f "ngrok http $SERVE_PORT" 2>/dev/null || true
            d="$(done_count "$name")"
            (( d > 0 )) && upload "$name" partial
            log "supervisor exiting at the deadline margin"
            exit 0
        fi

        # periodic partial upload so preemption never loses work
        now="$(date +%s)"
        if (( now - last_upload >= PARTIAL_UPLOAD_S )); then
            d="$(done_count "$name")"
            (( d > 0 )) && upload "$name" partial
            last_upload="$now"
        fi

        sleep "$POLL_S"
    done
done

log "supervisor done; final states:"
bash "$SNAP/status.sh" 2>/dev/null | sed -n '4,12p' | while read -r l; do log "  $l"; done
