#!/usr/bin/env bash
# One-shot status table for every long-running benchmark job on this box.
#
# Runs are discovered from the environment of live run_supervisor.sh processes
# rather than from a hardcoded list, so a newly launched rollout shows up here
# without editing anything. Judge/eval jobs are discovered from their output dirs.
#
# Counts DISTINCT instance ids, never line counts: output_errors.jsonl appends one
# line per failed attempt and a resumed run re-attempts previously-errored
# instances, so `wc -l` has over-reported progress by ~2.8x in practice.
#
# Usage:
#   bash tools/run_tracker/status.sh            # print once
#   watch -n 300 bash tools/run_tracker/status.sh
set -u

REPO="${REPO:-/home/gaokaizhang/benchmarks}"
cd "$REPO" || exit 1

uniq_ids() {
  # cut before grep: trajectory rows are ~1MB each and the id is always near the
  # front, so this turns a multi-GB scan into a cheap one.
  cat "$@" 2>/dev/null | cut -c1-70 | grep -oP '"instance_id":\s*"\K[^"]+' | sort -u | wc -l
}

age() {
  [ -f "$1" ] || { echo "-"; return; }
  local now mt d
  now=$(date +%s); mt=$(stat -c %Y "$1"); d=$(( (now - mt) / 60 ))
  if [ "$d" -lt 60 ]; then echo "${d}m"; else echo "$((d / 60))h$((d % 60))m"; fi
}

echo "===== $(date -u '+%F %T UTC') ====="
echo

# ---- rollouts -------------------------------------------------------------
printf '%-34s %-8s %6s %6s %6s %8s  %s\n' RUN MODEL OK ERR TOTAL IDLE STATE
for pid in $(pgrep -f run_supervisor.sh 2>/dev/null); do
  env_of() { tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -oP "^$1=\K.*"; }
  note=$(env_of RUN_NOTE); [ -n "$note" ] || continue
  dir=$(env_of RUN_DIR); total=$(env_of TOTAL); workers=$(env_of NUM_WORKERS)
  ok=$(uniq_ids "$dir/output.jsonl")
  err=$(uniq_ids "$dir/output_errors.jsonl")
  idle=$(age "$dir/output.jsonl")
  # A rollout whose output.jsonl has not moved in over an hour is worth a look:
  # every historical stall (upstream 529s, WAF 429s, a crashed child) showed up
  # here first, well before the supervisor's own counters noticed.
  state="ok"
  case "$idle" in *h*) state="STALLED?";; esac
  printf '%-34s %-8s %6s %6s %6s %8s  %s\n' \
    "${note:0:34}" "w=$workers" "$ok" "$err" "${total:-?}" "$idle" "$state"
done
echo

# ---- judge / eval side jobs ----------------------------------------------
jdir="eval_outputs/funclocalize_judge"
if [ -d "$jdir" ]; then
  echo "judge verdicts:"
  for f in "$jdir"/*.verdicts.jsonl; do
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    # grep -c prints 0 and exits 1 on no match, so a `|| echo 0` fallback would
    # emit a second 0 and break the row's formatting.
    e=$(grep -c '"error"' "$f" 2>/dev/null); e=${e:-0}
    printf '  %-42s %5s judged  %4s errored  (idle %s)\n' \
      "$(basename "$f" .verdicts.jsonl)" "$n" "$e" "$(age "$f")"
  done
  echo
fi

bdir="eval_outputs/tmp/dv4f_batches"
if [ -d "$bdir" ]; then
  done_n=$(ls "$bdir"/*.report.json 2>/dev/null | wc -l)
  all_n=$(ls "$bdir"/batch_*.txt 2>/dev/null | wc -l)
  echo "dv4f eval batches: $done_n/$all_n reports"
  echo
fi

# ---- shared resources -----------------------------------------------------
echo "resources:"
printf '  disk /home %s free, /mnt/data %s free\n' \
  "$(df -h --output=avail /home | tail -1 | tr -d ' ')" \
  "$(df -h --output=avail /mnt/data 2>/dev/null | tail -1 | tr -d ' ')"
printf '  docker: %s containers, %s agent-server images\n' \
  "$(docker ps -q 2>/dev/null | wc -l)" \
  "$(docker images --format '{{.Repository}}' 2>/dev/null | grep -c eval-agent-server)"

# The gateway sits behind an AWS WAF per-IP limiter (x-amzn-waf-rule:
# CPE_RateLimit_IP). It is shared by every run on this box, so it throttles on
# TOTAL concurrency across rollouts, not per model or per key. Sample it rather
# than trusting a single probe: the block is bursty and one 200 proves nothing.
if [ -f .llm_config/anthropic_deepseek_v4_flash_r2egym.json ]; then
  key=$(python3 -c "import json;print(json.load(open('.llm_config/anthropic_deepseek_v4_flash_r2egym.json'))['api_key'])" 2>/dev/null)
  blocked=0
  for _ in 1 2 3 4 5; do
    c=$(curl -s -o /dev/null -w '%{http_code}' -m 20 \
      -X POST https://inference-api.nvidia.com/v1/chat/completions \
      -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
      -d '{"model":"nvidia/qwen/qwen3.6-35b-a3b","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}')
    [ "$c" = "429" ] && blocked=$((blocked + 1))
    sleep 2
  done
  echo "  gateway WAF: $blocked/5 probes rate-limited"
fi
