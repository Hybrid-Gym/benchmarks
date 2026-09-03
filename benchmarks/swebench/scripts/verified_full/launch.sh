#!/usr/bin/env bash
# Launch the sweep from an immutable snapshot of these scripts.
#
#   bash launch.sh [models_file]
#
# bash reads a script incrementally by byte offset, so editing a script while
# it runs makes the running shell read shifted content and die mid-sweep.
# (That is exactly what ended the model-1 run: an in-place rewrite of
# run_queue.sh caused "unexpected EOF" right after it logged DONE, which is
# why its results never auto-uploaded.) Copying to a timestamped snapshot and
# running that copy means the working tree stays freely editable.
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SRC/env.sh"

MODELS_FILE="${1:-$SRC/models.txt}"
STAMP="$(date +%Y%m%d-%H%M%S)"
SNAP="$RUN_LOG_ROOT/snapshots/$STAMP"
mkdir -p "$SNAP"
cp "$SRC"/*.sh "$SRC"/*.py "$SRC"/models.txt "$SNAP"/ 2>/dev/null
chmod -w "$SNAP"/*.sh "$SNAP"/*.py 2>/dev/null

# Resolve the model list to an absolute path so the snapshot reads the same one.
MODELS_ABS="$(readlink -f "$MODELS_FILE")"

echo "snapshot:   $SNAP"
echo "models:     $MODELS_ABS ($(grep -cvE '^\s*(#|$)' "$MODELS_ABS") models)"
echo "workers:    $NUM_WORKERS   eager=$ENFORCE_EAGER   upload->$HF_RESULTS_REPO"

setsid nohup bash "$SNAP/run_queue.sh" "$MODELS_ABS" \
    > "$RUN_LOG_ROOT/queue_stdout.log" 2>&1 < /dev/null &
sleep 5
PID="$(pgrep -u "$USER" -f "$SNAP/run_queue.sh" | head -1)"
echo "$PID" > "$RUN_LOG_ROOT/queue.pid"
echo "queue pid:  ${PID:-<not found>}"
echo "run log:    $RUN_LOG_ROOT/run.log"
