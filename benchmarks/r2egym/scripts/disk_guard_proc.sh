#!/usr/bin/env bash
# Process-based disk guard for the PARALLEL free-models runners (multiple tmux
# sessions). Prunes leaked build cache + leaked eval-agent-server images every
# 10 min while ANY r2egym rollout is alive; self-exits when they all finish.
# Bracket trick in pgrep avoids self-match. Never touches neighbors' images.
set -u
GUARD_LOG=/home/gaokaizhang/benchmarks/eval_outputs/r2egym_outputs/disk_guard.log
echo "$(date -u '+%F %T') proc-guard start" >> "$GUARD_LOG"
miss=0
while true; do
  if pgrep -f "[r]2egym-infer" >/dev/null 2>&1; then
    miss=0
  else
    miss=$((miss+1))
    [ "$miss" -ge 2 ] && { echo "$(date -u '+%F %T') proc-guard exit (no r2egym-infer)" >> "$GUARD_LOG"; exit 0; }
  fi

  # 1. Build cache: use 5min filter so completed datalad builds are pruned quickly
  bc=$(docker builder prune -f --filter until=5m 2>/dev/null | grep -i 'Total:' || true)

  # 2. Dangling images
  docker image prune -f >/dev/null 2>&1 || true

  # 3. Leaked eval-agent-server images: our cleanup hook misses the secondary
  #    datala_tag_* build tag. Remove orphaned agent-server images.
  #
  #    Match ANY sha prefix, not a hardcoded one: the tag prefix is the SDK's git
  #    short sha, so any commit to vendor/software-agent-sdk changes it. Pinning
  #    the sha here silently stops reclaiming anything after an SDK bump, which
  #    leaks ~5GB per instance on a shared disk.
  #
  #    Two guards against killing an image a live worker still needs:
  #      a) match containers in ANY state (docker ps -a), not just running --
  #         an image is attached to its container from `create` onward;
  #      b) only reclaim images at least ~1h old. A freshly built image whose
  #         container has not been created yet is invisible to (a), so without
  #         the age check we delete it out from under the worker mid-startup.
  #         That race is rare in steady state but hits EVERY worker at once
  #         when all runs are restarted together.
  ACTIVE_IMGS=$(docker ps -a --format '{{.Image}}' 2>/dev/null || true)
  leaked_count=0
  while IFS='|' read -r img age; do
    case "$age" in *hour*|*day*|*week*|*month*|*year*) ;; *) continue ;; esac
    echo "$ACTIVE_IMGS" | grep -qF "$img" && continue
    docker rmi -f "$img" >/dev/null 2>&1 && leaked_count=$((leaked_count+1))
  done < <(docker images --format '{{.Repository}}:{{.Tag}}|{{.CreatedSince}}' 2>/dev/null \
           | grep "ghcr.io/openhands/eval-agent-server:")

  avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "$(date -u '+%F %T') pruned [$bc] | leaked_imgs=${leaked_count} | ${avail}GB free | workers=$(pgrep -f '[r]2egym-infer' | wc -l)" >> "$GUARD_LOG"
  sleep 600
done
