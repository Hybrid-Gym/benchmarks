#!/usr/bin/env bash
# Budget- and disk-aware batched eval of the deepseek-v4-flash R2E-Gym trajectories.
#
# This runs CONCURRENTLY with the still-live rollout, which is normally discouraged:
# eval and inference share one Docker Hub pull quota (gaokaiz2 = 200/hr), and on the
# original 1500 run that contention produced 722 spurious "errors" that were really
# `toomanyrequests`. Two things make it safe enough here:
#   1. The rollout is down to retrying ~114 instances whose agent-server images are
#      already built locally, so it is pulling ~4/hr, not continuously.
#   2. MIN_BUDGET below leaves headroom on every batch instead of draining to zero.
# It also guards free disk: the box is shared and another tenant's job has been
# eating ~10GB/h, so we pause rather than help fill it.
#
# Re-entrant: a batch with an existing report is skipped, so it can be restarted.
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

RUN_DIR="eval_outputs/r2egym_outputs/R2E-Gym__R2E-Gym-Lite-train/openai/nvidia/deepseek-ai/deepseek-v4-flash_sdk_e212d45_maxiter_40_N_deepseek-v4-flash-r2egym-1502"
# Frozen slim copy of output.jsonl (instance_id + git_patch for the critic-passing
# set). Reading the live file would let it change under us mid-run.
SRC="$RUN_DIR/eval_snapshot.jsonl"
BATCH_DIR="eval_outputs/tmp/dv4f_batches"
LOG="eval_outputs/r2egym_outputs/dv4f_eval.log"

# Leave >=40 pulls of headroom for the concurrent rollout after an 80-image batch.
MIN_BUDGET="${MIN_BUDGET:-120}"
# Never start a batch that could push a shared disk toward full.
MIN_FREE_GB="${MIN_FREE_GB:-120}"
W="${W:-4}"

say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }

get_remaining() {
  local creds token rem
  creds=$(python3 -c "import json;print(json.load(open('$HOME/.docker/config.json'))['auths']['https://index.docker.io/v1/']['auth'])" 2>/dev/null)
  [ -z "$creds" ] && { echo -1; return; }
  token=$(curl -s -H "Authorization: Basic $creds" \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
  [ -z "$token" ] && { echo -1; return; }
  rem=$(curl -s -I -H "Authorization: Bearer $token" \
    "https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest" 2>/dev/null \
    | grep -i "^ratelimit-remaining" | grep -oE "[0-9]+" | head -1)
  echo "${rem:-0}"
}
free_gb() { df --output=avail -BG /home | tail -1 | tr -dc 0-9; }

say "eval start: src=$SRC batches=$(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | wc -l) W=$W min_budget=$MIN_BUDGET min_free=${MIN_FREE_GB}G"

for b in "$BATCH_DIR"/batch_*.txt; do
  name=$(basename "$b" .txt)
  report="$BATCH_DIR/${name}.report.json"
  if [ -f "$report" ]; then say "$name already done, skipping"; continue; fi

  while true; do
    rem=$(get_remaining); free=$(free_gb)
    # -1 means the budget probe itself failed; treat that as "do not know, do not
    # spend" rather than charging ahead into a quota we cannot see.
    if [ "$rem" -ge "$MIN_BUDGET" ] && [ "$free" -ge "$MIN_FREE_GB" ]; then
      say "$name starting: budget=$rem free=${free}G ids=$(wc -l < "$b")"
      break
    fi
    say "$name waiting: budget=$rem (need $MIN_BUDGET) free=${free}G (need ${MIN_FREE_GB}G)"
    sleep 180
  done

  "$REPO/.venv/bin/r2egym-eval" "$SRC" --select "$b" --workers "$W" --timeout 300 \
    --dataset R2E-Gym/R2E-Gym-Lite --split train --output-file "$report" \
    >> "$LOG" 2>&1
  rc=$?

  if [ -f "$report" ]; then
    python3 - "$report" "$name" <<'PY' >> "$LOG" 2>&1
import json, sys
r = json.load(open(sys.argv[1]))
res = r["results"]
print(f"{sys.argv[2]} DONE: resolved={sum(1 for i in res if i.get('resolved'))} "
      f"errored={sum(1 for i in res if i.get('error'))} total={len(res)}")
PY
  else
    say "$name FAILED rc=$rc, no report written"
  fi
done

say "ALL BATCHES DONE"
