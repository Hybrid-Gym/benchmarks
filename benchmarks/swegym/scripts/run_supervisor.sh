#!/usr/bin/env bash
# Supervisor for a long SWE-Gym rollout.
#
# A near-copy of benchmarks/r2egym/scripts/run_supervisor.sh -- the supervision logic
# is benchmark-agnostic, but the r2egym copy could not be parameterised in place
# because bash reads a script incrementally as it executes, so editing the file under
# a live multi-day supervisor corrupts the loop it is still reading. Fold the two back
# together once no r2egym run is in flight.
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
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
TOTAL="${TOTAL:-1500}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
# Restrict the run to a list of instance ids. Without this, N_LIMIT=0 walks the FULL
# 2438-instance SWE-Gym train split -- TOTAL above would then be wrong and the four
# models would not be comparable, since each would cover a different set.
SELECT="${SELECT:-eval_outputs/swegym_select_1500.txt}"

SUP_LOG="$REPO/eval_outputs/swegym_outputs/supervisor_${RUN_NOTE}.log"
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

# uniq_ids over-counts success: aggregate_results writes a row for any run that did
# not raise, including ones that ended with an empty git patch or no finish action.
# Those are not usable trajectories and still need a retry, so "done" must be judged
# by the critic. Parsing every row costs ~1-2 min on a ~1GB file, so this is called
# ONLY when the cheap counter claims completion, never in the polling path.
critic_pass() {
  local sel_args=()
  [ -n "$SELECT" ] && sel_args=(--select "$SELECT")
  # Benchmark-agnostic (it just parses an output.jsonl against the run's own critic),
  # so it is shared from the r2egym dir rather than duplicated.
  "$REPO/.venv/bin/python3" "$REPO/benchmarks/r2egym/scripts/count_critic_passing.py" \
    "$RUN_DIR/output.jsonl" "${sel_args[@]}" 2>/dev/null | tail -1
}

restarts=0
stalled=0
MAX_STALLED="${MAX_STALLED:-2}"
while :; do
  ok=$(uniq_ids "$RUN_DIR/output.jsonl")
  err=$(uniq_ids "$RUN_DIR/output_errors.jsonl")
  done_n=$(uniq_ids "$RUN_DIR/output.jsonl" "$RUN_DIR/output_errors.jsonl")
  if [ "$ok" -ge "$TOTAL" ]; then
    real=$(critic_pass)
    if [ -n "$real" ] && [ "$real" -ge "$TOTAL" ] 2>/dev/null; then
      say "COMPLETE: all $TOTAL instances have a critic-passing trajectory. exiting."
      exit 0
    fi
    say "records=$ok >= $TOTAL but only ${real:-?} pass the critic; continuing."
  fi
  # Stop on "attempted everything", not "succeeded on everything" -- but only once
  # relaunching stops recovering anything. The runner itself retries (n_critic_runs
  # attempts, each with up to 4 tries per instance); a relaunch re-derives pending
  # work from output.jsonl, so it gives the still-failing instances another pass.
  # Some instances never succeed (broken images, task beyond the model), so bail out
  # after MAX_STALLED consecutive launches that recovered no new successes.
  if [ "$done_n" -ge "$TOTAL" ] && [ "$stalled" -ge "$MAX_STALLED" ]; then
    say "CONVERGED: attempted=$done_n/$TOTAL ok=$ok (critic-passing=$(critic_pass)) err=$err; no new successes in $stalled launches. exiting."
    exit 0
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "GIVING UP after $restarts restarts (ok=$ok err=$err attempted=$done_n)."
    exit 1
  fi

  # Cheap after the first launch: the previous iteration's post-launch count is this
  # one's baseline, so critic_pass runs once per launch, not twice.
  [ -n "${pass_before:-}" ] || pass_before=$(critic_pass)
  say "launch #$((restarts+1)): ok=$ok err=$err attempted=$done_n/$TOTAL critic-passing=${pass_before:-?}"
  # Always re-enter the repo: the crash mode we guard against deletes the CWD.
  cd "$REPO" || exit 1
  MODEL_NAME="$MODEL_NAME" NUM_WORKERS="$NUM_WORKERS" \
  MAX_ITERATIONS="$MAX_ITERATIONS" N_LIMIT=0 RUN_NOTE="$RUN_NOTE" SELECT="$SELECT" \
    bash benchmarks/swegym/scripts/test_infer.sh \
    >> "$REPO/eval_outputs/swegym_outputs/${RUN_NOTE}_run.log" 2>&1
  rc=$?

  new_ok=$(uniq_ids "$RUN_DIR/output.jsonl")
  new_err=$(uniq_ids "$RUN_DIR/output_errors.jsonl")
  pass_after=$(critic_pass)
  # An empty count means the counter itself failed; treat that as "no information"
  # rather than as "no progress", so a broken helper cannot end the run.
  [ -n "$pass_after" ] || pass_after="$pass_before"
  say "exited rc=$rc; progressed ok $ok->$new_ok err $err->$new_err critic-passing ${pass_before:-?}->${pass_after:-?}"

  # Repairing an instance rewrites an existing instance_id, so uniq_ids does NOT move
  # even when the launch genuinely recovered work. Launch #2 of the deepseek run left
  # ok at 1404 while critic-passing went 1363->1381; judging "stalled" on uniq_ids
  # alone counted that as a wasted launch and would have declared CONVERGED with 121
  # instances still unfilled. Only a launch that moved neither counter is stalled.
  if [ "$new_ok" -le "$ok" ] && [ "${pass_after:-0}" -le "${pass_before:-0}" ]; then
    stalled=$((stalled+1))
    say "no new successes this launch (stalled=$stalled/$MAX_STALLED)"
  else
    stalled=0
  fi
  pass_before="$pass_after"

  # No forward progress on a non-zero exit means restarting will just re-crash.
  if [ "$rc" -ne 0 ] && [ "$new_ok" -eq "$ok" ] && [ "$new_err" -eq "$err" ]; then
    say "NO PROGRESS on failed run; backing off 300s to avoid a crash loop."
    sleep 300
  fi

  restarts=$((restarts+1))
  sleep 30
done
