#!/usr/bin/env bash
# Bring the whole SWE-Gym 4-model rollout back up, correctly and idempotently.
#
# Why this exists rather than "just run launch_runs.sh": MAX_ITERATIONS is part of the
# run dir name, and it DIFFERS per model (dv4f and kimi25 were moved to 100 on
# 2026-08-07 because they were burning the 60-turn cap; qwen80b and gpt5mini stay at
# 60). One `launch_runs.sh all` therefore resumes two of the four models into the WRONG
# directory -- a fresh, empty one -- silently discarding their progress. It takes two
# invocations, and this script is the place that remembers which is which.
#
# A hand-rolled relaunch on 2026-08-07 got this wrong in the other direction: it passed
# MAX_ITERATIONS=100 to the supervisor but a stale RUN_DIR pointing at the maxiter_60
# dir, so the supervisor counted progress in a directory the runner was no longer
# writing to, and reported ok=28 forever.
#
# It also carries the per-model worker split and the pause list. A flat NUM_WORKERS for
# all four would, on the next reboot, silently undo both of the tuning decisions made
# while the box was up -- see the WORKERS table below.
#
# Idempotent: launch_runs.sh skips any model whose tmux session already exists, and the
# guards are only started if their session is missing. Safe to run on a live box, and
# safe to run from an @reboot hook.
#
# Usage:
#   bash benchmarks/swegym/scripts/resume_all.sh
#   W_DV4F=2 bash benchmarks/swegym/scripts/resume_all.sh    # un-pause one model
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

# test_infer.sh runs `uv run swegym-infer`, and uv lives in a conda prefix that is NOT
# on the PATH cron gives an @reboot job (/usr/bin:/bin). Without this every rollout
# would die instantly with "uv: command not found" -- and because launch_runs.sh
# backgrounds each one into its own tmux session, it would do so silently. Prepend
# rather than replace so an interactive caller keeps their own PATH.
export PATH="/mnt/taurus/data2/gaokaizhang/tools/miniconda3/bin:$PATH"
command -v uv >/dev/null || { echo "uv not on PATH; refusing to launch" >&2; exit 1; }

# Total gateway workers across ALL live jobs on this box is what the NVIDIA gateway's
# per-IP WAF counts -- not per model, not per key. ~8 total is the observed-safe level
# and 14 produced a 429 storm on 2026-08-05, so these must sum to <= BUDGET.
#
# 0 means PAUSED: the model is skipped entirely rather than launched. deepseek-v4-flash
# is at 0 -- stopped on purpose for the Aug-21 deadline push (2026-08-18): even after
# fixing its separate max_completion_tokens bug (see .llm_config), it has no realistic
# path from 133/1500 critic-passing to 1500 in the time left, so all budget goes to
# kimi25 instead. It also has its OWN older issue (endpoint returning no tool call and
# no content, 348 occurrences on 2026-08-10, 46% of rows unusable) which was never
# confirmed fixed -- check `grep -c "no tool call and no content"` over its logs/ before
# ever re-enabling it, independent of the token-cap fix.
#
# qwen80b and gpt5mini are both converged/complete -- set to 0 (PAUSED/skipped) rather
# than relaunched. Relaunching them was harmless before (their supervisors see
# CONVERGED/COMPLETE and exit immediately) but their nominal WORKERS value still counted
# against the total-vs-BUDGET guard below, and kimi25=8 alone would blow that budget if
# they weren't zeroed too.
#
# kimi25 KILLED (2026-08-19, deadline push): stopped at 1359/1500 and pushed as-is
# (synthetic-code-training/swegym_kimi25_1359i, no eval) to free the whole BUDGET for
# opus45 -- explicit user call: kimi25 was too far from finishing rollout+eval in the
# remaining ~2 days, and every worker instead goes to getting as many opus-4.5
# instances done as possible before Friday. The batch eval process for
# gpt5mini/qwen80b was also killed at 76/125 batches for the same reason (it was
# saturating this box's CPU -- see feedback_gateway_worker_ceiling -- taking cycles
# away from opus45's rollout). Their eval stays at 76/125 until eval is deliberately
# resumed after opus45 wraps.
BUDGET="${BUDGET:-8}"
declare -A WORKERS=(
  [qwen80b]="${W_QWEN80B:-0}"
  [gpt5mini]="${W_GPT5MINI:-0}"
  [kimi25]="${W_KIMI25:-0}"
  [dv4f]="${W_DV4F:-0}"
  [opus45]="${W_OPUS45:-8}"
)

# MAX_ITERATIONS is per-model and feeds the run dir name; dv4f and kimi25 moved to 100
# on 2026-08-07 because they were burning the 60-turn cap. opus45 also started at 100
# (2026-08-19) to avoid truncating a stronger agent's trajectories on an untested
# model. Getting this wrong resumes into a fresh empty dir.
declare -A MAXITER=(
  [qwen80b]=60
  [gpt5mini]=60
  [kimi25]=100
  [dv4f]=100
  [opus45]=100
)

total=0
for k in "${!WORKERS[@]}"; do total=$(( total + WORKERS[$k] )); done
if [ "$total" -gt "$BUDGET" ]; then
  echo "worker total $total exceeds WAF budget $BUDGET; refusing to launch" >&2
  exit 1
fi

# Wait for dockerd: at boot this can run before the daemon accepts connections, and
# every rollout would then die instantly on its first image build.
for _ in $(seq 1 60); do
  docker info >/dev/null 2>&1 && break
  sleep 5
done
if ! docker info >/dev/null 2>&1; then
  echo "docker daemon not ready after 300s; not launching" >&2
  exit 1
fi

echo "== rollouts ($total/$BUDGET workers)"
for key in qwen80b gpt5mini kimi25 dv4f opus45; do
  w="${WORKERS[$key]}"
  if [ "$w" -eq 0 ]; then
    echo "PAUSED $key (workers=0)"
    continue
  fi
  NUM_WORKERS="$w" MAX_ITERATIONS="${MAXITER[$key]}" \
    bash benchmarks/swegym/scripts/launch_runs.sh "$key" 2>&1 | grep -E '^(START|SKIP|      run_dir)'
done

# The disk guard sweeps only below MIN_FREE_GB. Leaving it at 0 (always sweep) drops
# every base image older than an hour, including ones a pending instance still needs,
# which buys disk back at the cost of re-pulls against the 200/hr Docker Hub budget.
# 450G keeps a cushion above the eval's own 300G prune trigger and 150G floor.
start_guard() {
  local session="$1"; shift
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "SKIP $session: already running"
  else
    tmux new-session -d -s "$session" "$@" && echo "START $session"
  fi
}

echo "== guards"
start_guard swegym-diskguard "MIN_FREE_GB=450 bash $REPO/benchmarks/swegym/scripts/disk_guard.sh"
start_guard pullcache-guard  "bash $REPO/tools/pullcache/cache_guard.sh"
# Hands a converged model's workers to whichever rollout still needs them. Without it
# the pool silently runs under budget from the moment the first model finishes.
start_guard swegym-rebalance "BUDGET=$BUDGET bash $REPO/benchmarks/swegym/scripts/rebalance.sh"

echo
echo "live sessions:"; tmux ls 2>/dev/null | grep -E 'swegym|pullcache' || echo "  (none)"
