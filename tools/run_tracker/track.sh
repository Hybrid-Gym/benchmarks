#!/usr/bin/env bash
# Periodic wrapper around status.sh: appends a snapshot to a log every INTERVAL.
#
# Must live in tmux, not a background shell tied to an agent session -- those get
# SIGTERMed when the session ends, which is how earlier trackers silently died:
#   tmux new -s run-tracker "bash tools/run_tracker/track.sh"
#
# Read the latest snapshot with:
#   tail -40 eval_outputs/run_tracker.log
set -u

REPO="${REPO:-/home/gaokaizhang/benchmarks}"
cd "$REPO" || exit 1

INTERVAL="${INTERVAL:-1800}"                       # 30 min
LOG="${LOG:-eval_outputs/run_tracker.log}"
MAX_LOG_MB="${MAX_LOG_MB:-20}"

while :; do
  bash tools/run_tracker/status.sh >> "$LOG" 2>&1
  echo >> "$LOG"
  # Unbounded growth would eventually matter on a shared disk; keep the tail.
  if [ "$(du -m "$LOG" | cut -f1)" -gt "$MAX_LOG_MB" ]; then
    tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
  sleep "$INTERVAL"
done
