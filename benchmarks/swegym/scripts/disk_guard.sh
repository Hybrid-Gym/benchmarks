#!/usr/bin/env bash
# Disk guard for the SWE-Gym rollouts. Reclaims leaked images every 10 min while any
# swegym-infer is alive; self-exits once they all finish.
#
# Why this is needed even though run_infer cleans up after each instance: the four
# comparison runs share ONE instance list, so two models routinely hold the same
# image at the same moment. Whoever finishes first gets
#   "conflict: unable to delete <id> - image is being used by running container"
# and nothing ever retries, so that ~5GB image is leaked for the rest of the run. 23
# such failures in the first 20h were enough to put the disk on a ~12h path to full.
#
# It reclaims BOTH image families a SWE-Gym instance creates -- the agent-server image
# and the xingyaoww base it was built FROM. The r2egym guard only matches the former,
# which is why the bases piled up here.
#
# Two guards against deleting an image a live worker still needs, both learned from
# benchmarks/r2egym/scripts/disk_guard_proc.sh:
#   a) match containers in ANY state (docker ps -a), since an image is attached to its
#      container from `create` onward, not just while running;
#   b) only reclaim images whose LOCAL tag is at least ~1h old -- a freshly built or
#      freshly pulled image whose container does not exist yet is invisible to (a),
#      and without the age check it gets deleted out from under a worker mid-startup.
#      Age must come from Metadata.LastTagTime (when this host tagged/pulled it), NOT
#      from CreatedSince: a xingyaoww base image was built upstream months ago, so
#      CreatedSince calls a base image pulled ten seconds ago "2 months old" and the
#      guard would happily delete it mid-build.
#
# Never touches anything outside our two name prefixes: this box also hosts another
# user's swesmith/susvibes runs, so `docker system prune -a` would be destructive.
#
# Usage (needs tmux -- it must outlive the session that starts it):
#   tmux new-session -d -s swegym-diskguard "bash benchmarks/swegym/scripts/disk_guard.sh"
set -u

REPO=/home/gaokaizhang/benchmarks
GUARD_LOG="$REPO/eval_outputs/swegym_outputs/disk_guard.log"
INTERVAL="${INTERVAL:-600}"
MIN_FREE_GB="${MIN_FREE_GB:-0}"   # 0 = always sweep; >0 = only sweep below this

echo "$(date -u '+%F %T') swegym disk guard start" >> "$GUARD_LOG"
miss=0
while true; do
  # Bracket trick keeps pgrep from matching this script's own cmdline.
  if pgrep -f "[s]wegym-infer" >/dev/null 2>&1; then
    miss=0
  else
    miss=$((miss + 1))
    if [ "$miss" -ge 2 ]; then
      echo "$(date -u '+%F %T') exit (no swegym-infer for 2 checks)" >> "$GUARD_LOG"
      exit 0
    fi
  fi

  avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$MIN_FREE_GB" -gt 0 ] && [ "$avail" -gt "$MIN_FREE_GB" ]; then
    echo "$(date -u '+%F %T') skip sweep (${avail}GB free > ${MIN_FREE_GB}GB)" >> "$GUARD_LOG"
    sleep "$INTERVAL"
    continue
  fi

  bc=$(docker builder prune -f --filter until=5m 2>/dev/null | grep -i 'Total:' || true)
  docker image prune -f >/dev/null 2>&1 || true

  ACTIVE_IMGS=$(docker ps -a --format '{{.Image}}' 2>/dev/null || true)
  cutoff=$(( $(date +%s) - 3600 ))
  reclaimed=0
  while read -r img; do
    echo "$ACTIVE_IMGS" | grep -qF "$img" && continue
    tagged=$(docker image inspect -f '{{.Metadata.LastTagTime}}' "$img" 2>/dev/null)
    [ -z "$tagged" ] && continue
    tagged_epoch=$(date -d "$tagged" +%s 2>/dev/null) || continue
    [ "$tagged_epoch" -gt "$cutoff" ] && continue
    docker rmi -f "$img" >/dev/null 2>&1 && reclaimed=$((reclaimed + 1))
  done < <(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
           | grep -E "ghcr\.io/openhands/eval-agent-server:|xingyaoww/sweb\.eval")

  after=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "$(date -u '+%F %T') pruned [$bc] | reclaimed=${reclaimed} | ${avail}GB -> ${after}GB free | workers=$(pgrep -f '[s]wegym-infer' | wc -l)" >> "$GUARD_LOG"
  sleep "$INTERVAL"
done
