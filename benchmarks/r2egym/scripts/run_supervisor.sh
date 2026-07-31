#!/usr/bin/env bash
# Supervisor for a long R2E-Gym rollout.
#
# Why this exists: `tmux new-session -d -s NAME "cmd"` tears the session down as
# soon as cmd exits, so a mid-run crash is indistinguishable from "never started".
# On 2026-07-29 a deepseek run died 18 min after launch (process-wide chdir race in
# the agent-server image build deleted the CWD) and sat dead for ~9 hours unnoticed.
#
# The rollout is re-entrant: attempt-1 work is derived from output.jsonl, so a
# restart resumes and additionally retries previously-errored instances.
#
# Usage:
#   MODEL_NAME=... RUN_NOTE=... OUTPUT_SUBDIR=<abs path to run dir> \
#     bash run_supervisor.sh
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME}"
RUN_NOTE="${RUN_NOTE:?set RUN_NOTE}"
RUN_DIR="${RUN_DIR:?set RUN_DIR (absolute path to the run output dir)}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
TOTAL="${TOTAL:-1502}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
# Restrict the run to a list of instance ids. Without this, N_LIMIT=0 walks the FULL
# 4578-instance R2E-Gym-Lite split -- TOTAL above would then be wrong and the run
# would spend ~6 days on instances outside the comparison subset.
SELECT="${SELECT:-}"

SUP_LOG="$REPO/eval_outputs/r2egym_outputs/supervisor_${RUN_NOTE}.log"
say() { echo "$(date -u '+%F %T') $*" >> "$SUP_LOG"; }

say "supervisor start: model=$MODEL_NAME note=$RUN_NOTE workers=$NUM_WORKERS total=$TOTAL select=${SELECT:-<none>}"

# Count DISTINCT instance ids, not lines. output_errors.jsonl accumulates one line
# per failed attempt, and a resumed run re-attempts previously-errored instances, so
# `wc -l` over-counts badly (on 2026-07-31: 587 error lines for 293 distinct
# instances). Using lines here made the run look ~300 instances further along than it
# was, and would have fired the COMPLETE branch below with ~500 instances untouched.
uniq_ids() {
  cat "$@" 2>/dev/null | cut -c1-70 | grep -oP '"instance_id":\s*"\K[^"]+' | sort -u | wc -l
}

restarts=0
stalled=0
MAX_STALLED="${MAX_STALLED:-2}"
while :; do
  ok=$(uniq_ids "$RUN_DIR/output.jsonl")
  err=$(uniq_ids "$RUN_DIR/output_errors.jsonl")
  done_n=$(uniq_ids "$RUN_DIR/output.jsonl" "$RUN_DIR/output_errors.jsonl")
  if [ "$ok" -ge "$TOTAL" ]; then
    say "COMPLETE: all $TOTAL instances succeeded. exiting."
    exit 0
  fi
  # Stop on "attempted everything", not "succeeded on everything" -- but only once
  # relaunching stops recovering anything. The runner itself retries (n_critic_runs
  # attempts, each with up to 4 tries per instance); a relaunch re-derives pending
  # work from output.jsonl, so it gives the still-failing instances another pass.
  # Some instances never succeed (broken images, task beyond the model), so bail out
  # after MAX_STALLED consecutive launches that recovered no new successes.
  if [ "$done_n" -ge "$TOTAL" ] && [ "$stalled" -ge "$MAX_STALLED" ]; then
    say "CONVERGED: attempted=$done_n/$TOTAL ok=$ok err=$err; no new successes in $stalled launches. exiting."
    exit 0
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "GIVING UP after $restarts restarts (ok=$ok err=$err attempted=$done_n)."
    exit 1
  fi

  say "launch #$((restarts+1)): ok=$ok err=$err attempted=$done_n/$TOTAL"
  # Always re-enter the repo: the crash mode we guard against deletes the CWD.
  cd "$REPO" || exit 1
  MODEL_NAME="$MODEL_NAME" NUM_WORKERS="$NUM_WORKERS" \
  MAX_ITERATIONS="$MAX_ITERATIONS" N_LIMIT=0 RUN_NOTE="$RUN_NOTE" SELECT="$SELECT" \
    bash benchmarks/r2egym/scripts/test_infer.sh \
    >> "$REPO/eval_outputs/r2egym_outputs/${RUN_NOTE}_run.log" 2>&1
  rc=$?

  new_ok=$(uniq_ids "$RUN_DIR/output.jsonl")
  new_err=$(uniq_ids "$RUN_DIR/output_errors.jsonl")
  say "exited rc=$rc; progressed ok $ok->$new_ok err $err->$new_err"

  if [ "$new_ok" -le "$ok" ]; then
    stalled=$((stalled+1))
    say "no new successes this launch (stalled=$stalled/$MAX_STALLED)"
  else
    stalled=0
  fi

  # No forward progress on a non-zero exit means restarting will just re-crash.
  if [ "$rc" -ne 0 ] && [ "$new_ok" -eq "$ok" ] && [ "$new_err" -eq "$err" ]; then
    say "NO PROGRESS on failed run; backing off 300s to avoid a crash loop."
    sleep 300
  fi

  restarts=$((restarts+1))
  sleep 30
done
