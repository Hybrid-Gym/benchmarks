#!/usr/bin/env bash
# Render the sweep status table.  Usage: bash status.sh [--watch]
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
MODELS_FILE="${MODELS_FILE:-$SCRIPT_DIR/models.txt}"

row_for() {
    # Emit the table row for one model save name, with a live instance count.
    local name="$1" state done_n total started updated note out
    IFS=$'\t' read -r _ state done_n total started updated note < <(
        [[ -f "$STATUS_FILE" ]] && awk -F'\t' -v n="$name" '$1==n' "$STATUS_FILE" | tail -1)
    [[ -z "${state:-}" ]] && { state="PENDING"; done_n=0; total=500; started="-"; updated="-"; note=""; }
    out="$(output_dir_of "$name")/output.jsonl"
    [[ -f "$out" ]] && done_n="$(wc -l < "$out" | tr -d ' ')"
    printf '%s\t%s\t%s/%s\t%s\t%s\t%s\n' \
        "${name#qwen25-coder-7b-func-localize-}" "$state" "$done_n" "${total:-500}" \
        "${started:--}" "${updated:--}" "${note:0:46}"
}

render() {
    echo "SWE-bench_Verified (full, 500 instances) — $(date '+%F %T')"
    echo "outputs: $EVAL_OUT_ROOT"
    echo "logs:    $RUN_LOG_ROOT/<model>/{serve,infer}.log"
    echo
    if [[ ! -f "$STATUS_FILE" ]]; then
        echo "(no run started yet — $STATUS_FILE does not exist)"
    fi
    { printf 'MODEL\tSTATE\tDONE\tSTARTED\tUPDATED\tNOTE\n'
      # Render in models.txt queue order, not status-file order.
      while read -r m; do row_for "$(save_name_of "$m")"; done \
          < <(grep -vE '^\s*(#|$)' "$MODELS_FILE")
    } | column -t -s $'\t'
    echo
    echo "disk: /home $(df -h /home/gaokaiz | awk 'NR==2{print $4" free"}')  |  STORAGE_DIR $(df -h "$STORAGE_DIR" | awk 'NR==2{print $4" free"}')"
}

if [[ "${1:-}" == "--watch" ]]; then
    while true; do clear; render; sleep 60; done
else
    render
fi
