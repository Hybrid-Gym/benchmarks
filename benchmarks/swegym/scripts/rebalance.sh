#!/usr/bin/env bash
# Keep the SWE-Gym worker pool full without a human in the loop.
#
# The problem it solves: the gateway ceiling is a fixed BUDGET of workers summed across
# every live rollout, and each supervisor reads NUM_WORKERS once at startup. So when a
# model converges its workers do not go anywhere -- they simply stop existing, and the
# pool runs under budget until someone notices. gpt5mini and qwen80b converge within a
# few hours of each other, which is 5 of 8 workers idle, potentially overnight.
#
# What it does every INTERVAL:
#   1. sums NUM_WORKERS over live swegym supervisors  -> allocated
#   2. free = BUDGET - allocated
#   3. gives `free` to the first SINK that is not running and not finished
#   4. if every sink is already running, folds `free` into the last sink by restarting
#      it (rollouts are re-entrant, so a restart costs only in-flight instances)
#
# It only ever ADDS work up to BUDGET; it never raises the total above it, so it cannot
# cause the 429 storm that 14 concurrent workers produced on 2026-08-05.
#
# Usage (must outlive the launching shell, so tmux):
#   tmux new-session -d -s swegym-rebalance "bash benchmarks/swegym/scripts/rebalance.sh"
#   DRY_RUN=1 bash benchmarks/swegym/scripts/rebalance.sh    # print decisions, change nothing
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1
export PATH="/mnt/taurus/data2/gaokaizhang/tools/miniconda3/bin:$PATH"

BUDGET="${BUDGET:-8}"
INTERVAL="${INTERVAL:-300}"
DRY_RUN="${DRY_RUN:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-120}"     # refuse to add workers below this much free disk
RESTART_COOLDOWN="${RESTART_COOLDOWN:-3600}"
MAX_PER_LAUNCH="${MAX_PER_LAUNCH:-3}" # ceiling on a single launch, whatever `free` says
LOG="${LOG:-$REPO/eval_outputs/swegym_outputs/rebalance.log}"

# Why this is paranoid about its own reading of the world (incident 2026-08-12 16:14):
# under heavy load the box stalled for 36 minutes between cycles, and on the cycle that
# followed, the supervisor scan returned NOTHING. allocated=0 => free=8 => it launched
# dv4f with 8 workers on top of the 8 already running: 16 concurrent, twice the WAF
# ceiling that produced a 429 storm on 2026-08-05. The scan failing must never read as
# "the box is idle". Three independent brakes now:
#   1. DISTRUST  - rollout tmux sessions exist but the scan saw 0 workers => skip cycle
#   2. DEBOUNCE  - act only when two consecutive cycles agree on the same `free`
#   3. CAP       - never hand any single launch more than MAX_PER_LAUNCH workers
# Any one of these alone would have prevented the incident; all three are cheap.

# Where freed workers go. dv4f is deliberately NOT here.
#
# An endpoint probe on 2026-08-11 (92 single-shot tool-call requests) showed ~1% dead
# turns and looked recovered. The real agent loop says otherwise: 2.5h of accidental
# dv4f rollout on 2026-08-12 produced 101 instances with 550 dead turns across 96 of
# them, 81% of rows critic-unusable, and ZERO new critic-passing trajectories. The
# probe was not representative -- it used short contexts, and the failure is
# context-length dependent, so it only reproduces under a real condensed trajectory.
# Judge this model by critic-passing rows in a live run, never by a probe.
# dv4pro DROPPED 2026-08-15. It could not finish: measured 1.05 critic-passing per
# worker-hour against 1189 remaining, i.e. ~377h with 118h left before the Aug 20
# deadline. Dropping it hands its 3 workers to kimi25 (which finishes ~2 days earlier
# as a result) and frees the box for the eval. It must NOT be in SINKS -- the top-up
# loop launches any sink that is not running and not finished, so leaving it here
# would silently resurrect it on the next cycle.
SINKS="kimi25"

declare -A MAXITER=([qwen80b]=60 [gpt5mini]=60 [kimi25]=100 [dv4f]=100 [dv4pro]=100)

# Where the freed workers should end up, rather than "whoever is last in SINKS gets
# everything". Sized from measured critic-passing rates (2026-08-13) so both runs land
# before the Aug 20 deadline instead of one finishing early and the other missing it:
#
#   kimi25  1.52/worker-hour, ~981 left at handoff -> 5 workers -> Aug 19 08:00
#   dv4pro  4.32/worker-hour, ~1260 left           -> 3 workers -> Aug 18 00:00
#
# The targets sum to exactly BUDGET, so top-up converges and then stops.
declare -A TARGETS=([kimi25]=8)

last_restart=0

say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }

# Live swegym supervisors only. `pgrep -f run_supervisor.sh` also matches the r2egym
# supervisors, which share the basename -- counting those would make us think the
# budget is spent and never rebalance.
swegym_supervisor_pids() {
  for p in $(pgrep -f run_supervisor.sh 2>/dev/null); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q 'swegym/scripts/run_supervisor.sh' \
      && echo "$p"
  done
}

env_of() { tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | grep -oP "^$2=\K.*" | head -1; }

allocated_workers() {
  local t=0 w
  for p in $(swegym_supervisor_pids); do
    w=$(env_of "$p" NUM_WORKERS)
    case "$w" in ''|*[!0-9]*) w=0 ;; esac
    t=$((t + w))
  done
  echo "$t"
}

pid_for_note() {
  local note="$1" p
  for p in $(swegym_supervisor_pids); do
    [ "$(env_of "$p" RUN_NOTE)" = "$note" ] && { echo "$p"; return; }
  done
}

# A converged run must not be relaunched: its tmux session is gone (the session dies
# with its command), so session-existence alone would say "not running" forever and we
# would restart a finished model every INTERVAL.
is_finished() {
  local log="$REPO/eval_outputs/swegym_outputs/supervisor_swegym-$1-1500.log"
  [ -f "$log" ] || return 1
  local last
  last=$(grep -E 'supervisor start|COMPLETE|CONVERGED' "$log" | tail -1)
  case "$last" in *COMPLETE*|*CONVERGED*) return 0 ;; *) return 1 ;; esac
}

is_running() { tmux has-session -t "swegym-$1" 2>/dev/null; }

# Independent second opinion on whether any rollout is alive. tmux is a different
# mechanism from /proc scanning, so the two failing together is far less likely.
rollout_sessions() {
  tmux ls 2>/dev/null | grep -cE '^swegym-(qwen80b|gpt5mini|kimi25):'
}

launch() {
  local key="$1" workers="$2"
  if [ "$DRY_RUN" = "1" ]; then
    say "DRY_RUN would launch $key with $workers workers"
    return
  fi
  say "LAUNCH $key workers=$workers maxiter=${MAXITER[$key]}"
  NUM_WORKERS="$workers" MAX_ITERATIONS="${MAXITER[$key]}" \
    bash "$REPO/benchmarks/swegym/scripts/launch_runs.sh" "$key" >> "$LOG" 2>&1
}

# Ramping a live model means a restart: the supervisor reads NUM_WORKERS once, and
# launch_runs.sh skips any model whose session exists. Kill the supervisor BEFORE its
# swegym-infer children, or it treats their death as a crash and relaunches them.
restart_with() {
  local key="$1" workers="$2" note="swegym-${key}-1500" p
  if [ "$DRY_RUN" = "1" ]; then
    say "DRY_RUN would restart $key with $workers workers"
    return
  fi
  say "RESTART $key -> workers=$workers"

  # Snapshot the OTHER live rollouts first. On 2026-08-14 00:16 a kimi25 top-up
  # coincided exactly with dv4pro's supervisor dying: its log stopped mid-instance
  # with no exit line (so it was signalled, not converged) and its session was gone
  # by the time launch_runs.sh listed sessions 18s later. The kill paths below are
  # all note-scoped and none of them should touch another run, so the mechanism is
  # unconfirmed -- which is exactly why this guard checks the outcome rather than
  # trusting the reasoning. A silently dead rollout costs a day of deadline before
  # anyone notices.
  local before after k
  before=$(tmux ls 2>/dev/null | grep -oE '^swegym-(qwen80b|gpt5mini|kimi25)' | sort -u)

  p=$(pid_for_note "$note")
  [ -n "$p" ] && kill "$p" 2>/dev/null
  sleep 5
  for c in $(pgrep -f swegym-infer 2>/dev/null); do
    [ "$(env_of "$c" RUN_NOTE)" = "$note" ] && kill "$c" 2>/dev/null
  done
  sleep 5
  tmux kill-session -t "swegym-$key" 2>/dev/null
  sleep 2
  launch "$key" "$workers"
  last_restart=$(date +%s)

  # Anything that was live before and is not live now -- other than the run we just
  # relaunched -- was collateral damage. Bring it back at its target.
  sleep 10
  after=$(tmux ls 2>/dev/null | grep -oE '^swegym-(qwen80b|gpt5mini|kimi25)' | sort -u)
  for k in $before; do
    [ "$k" = "swegym-$key" ] && continue
    if ! echo "$after" | grep -qx "$k"; then
      local lost="${k#swegym-}"
      if is_finished "$lost"; then
        say "note: $lost disappeared during restart but is finished; not relaunching"
        continue
      fi
      say "COLLATERAL: $lost died during the $key restart; relaunching at ${TARGETS[$lost]:-2}"
      launch "$lost" "${TARGETS[$lost]:-2}"
    fi
  done
}

say "rebalancer start: BUDGET=$BUDGET INTERVAL=${INTERVAL}s sinks='$SINKS' dry_run=$DRY_RUN"

prev_free=""
while true; do
  alloc=$(allocated_workers)
  free=$(( BUDGET - alloc ))
  avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  sessions=$(rollout_sessions)

  # Brake 1: the two views disagree in the one direction that is dangerous.
  if [ "$alloc" -eq 0 ] && [ "$sessions" -gt 0 ]; then
    say "DISTRUST: scan saw 0 workers but $sessions rollout session(s) live; skipping cycle"
    prev_free=""
    sleep "$INTERVAL"
    continue
  fi

  # Brake 2: one anomalous reading can never act on its own.
  if [ "$free" -gt 0 ] && [ "$free" != "${prev_free:-}" ]; then
    say "settling: free=$free (previous='${prev_free:-none}'); need two agreeing cycles"
    prev_free="$free"
    sleep "$INTERVAL"
    continue
  fi
  prev_free="$free"

  if [ "$free" -le 0 ]; then
    say "ok: allocated=$alloc/$BUDGET, nothing to do (${avail}GB free)"
  elif [ "$avail" -lt "$MIN_FREE_GB" ]; then
    say "HOLD: $free worker(s) free but only ${avail}GB disk (< ${MIN_FREE_GB}GB); not adding load"
  else
    placed=0
    for key in $SINKS; do
      if is_finished "$key"; then
        say "skip $key: finished"
        continue
      fi
      if ! is_running "$key"; then
        # Brake 3: cap the blast radius of any single launch.
        give=$(( free > MAX_PER_LAUNCH ? MAX_PER_LAUNCH : free ))
        [ "$give" -lt "$free" ] && say "capping launch: free=$free -> $give (MAX_PER_LAUNCH)"
        launch "$key" "$give"
        placed=1
        break
      fi
    done

    if [ "$placed" -eq 0 ]; then
      # Every sink is already running, so the only way to absorb free workers is to
      # restart one with more. Top sinks up toward TARGETS in order, ONE per cycle, so
      # a bad reading can only ever move a single run, and the cooldown keeps a
      # flapping rollout out of a restart loop.
      now=$(date +%s)
      if [ $(( now - last_restart )) -ge "$RESTART_COOLDOWN" ]; then
        topped=0
        for sink in $SINKS; do
          is_running "$sink" || continue
          is_finished "$sink" && continue
          want="${TARGETS[$sink]:-0}"
          cur=$(env_of "$(pid_for_note "swegym-${sink}-1500")" NUM_WORKERS)
          case "$cur" in ''|*[!0-9]*) cur=0 ;; esac
          [ "$cur" -ge "$want" ] && continue

          target=$(( cur + free ))
          [ "$target" -gt "$want" ] && target=$want
          if [ "$target" -gt "$cur" ] && [ "$target" -le "$BUDGET" ]; then
            say "top-up $sink: $cur -> $target (target=$want, free=$free)"
            restart_with "$sink" "$target"
            topped=1
            break
          fi
        done
        [ "$topped" -eq 0 ] && say "waiting: $free free but every sink is already at its target"
      else
        say "waiting: $free free, all sinks running (cooldown $(( RESTART_COOLDOWN - (now - last_restart) ))s)"
      fi
    fi
  fi

  # Nothing left to feed and no live rollouts means the experiment is over.
  if [ -z "$(swegym_supervisor_pids)" ]; then
    done_all=1
    for key in qwen80b gpt5mini kimi25; do
      is_finished "$key" || done_all=0
    done
    if [ "$done_all" = "1" ]; then
      say "all four models finished; rebalancer exiting"
      exit 0
    fi
  fi

  sleep "$INTERVAL"
done
