#!/usr/bin/env bash
# Keep the Docker Hub pull-through cache under a size cap.
#
# The cache (container `pullcache`, 127.0.0.1:5000) proxies docker.io so a repeat pull
# of the same image costs no Docker Hub quota. That matters because the eval deletes
# each image right after using it: without a cache, every retry, top-up, or
# disk-pressure re-fetch spends another pull against the 200/hr account budget.
#
# registry:2 has no built-in size limit. It expires proxied blobs on REGISTRY_PROXY_TTL
# and its scheduler purges them, but nothing bounds the total, so on a 6000-instance
# eval sweep it would grow without limit on a disk we are already tight on.
#
# The cache is pure derived data, so the cheapest correct eviction is to drop all of it
# and let it refill: wipe, restart, carry on. That costs re-pulls only for images
# needed again afterwards. A smarter LRU over blobs would need to walk the manifest
# graph to avoid orphaning layers -- not worth it for a throwaway cache.
#
# Usage (tmux, so it outlives the session that starts it):
#   tmux new-session -d -s pullcache-guard "bash tools/pullcache/cache_guard.sh"
set -u

CACHE_DIR="${CACHE_DIR:-/home/gaokaizhang/pullcache}"
CAP_GB="${CAP_GB:-300}"
INTERVAL="${INTERVAL:-300}"
CONTAINER="${CONTAINER:-pullcache}"
LOG="${LOG:-/home/gaokaizhang/benchmarks/eval_outputs/pullcache_guard.log}"

say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }
say "cache guard start: dir=$CACHE_DIR cap=${CAP_GB}G interval=${INTERVAL}s"

while true; do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    say "container $CONTAINER not running; exiting"
    exit 0
  fi

  used_gb=$(du -sBG "$CACHE_DIR" 2>/dev/null | cut -f1 | tr -dc '0-9')
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')

  if [ -n "$used_gb" ] && [ "$used_gb" -ge "$CAP_GB" ]; then
    say "cache ${used_gb}G >= cap ${CAP_GB}G -> wiping"
    # Delete from INSIDE the container. registry:2 runs as root and its blobs are
    # root-owned, but this guard runs as an unprivileged user, so the host-side
    # `rm -rf "$CACHE_DIR/docker"` this used to do failed with EPERM -- and because
    # the failure went to the tmux pane rather than the log, and the next line
    # reported success unconditionally, the cap silently stopped being enforced.
    # It logged "wiped and restarted; now 670G" while the cap was 300G, and the
    # cache had grown to 670G by 2026-08-10, taking / to 98% full.
    # Delete only the registry's own data subtree, never $CACHE_DIR itself, so a
    # mistyped/empty CACHE_DIR cannot turn this into an rm of something else.
    if ! docker exec "$CONTAINER" rm -rf /var/lib/registry/docker 2>>"$LOG"; then
      say "ERROR: in-container wipe failed; cache still ${used_gb}G (cap ${CAP_GB}G)"
      sleep "$INTERVAL"
      continue
    fi
    docker restart "$CONTAINER" >/dev/null 2>&1
    # Never report success without measuring: that is exactly how this went unnoticed.
    now_gb=$(du -sBG "$CACHE_DIR" 2>/dev/null | cut -f1 | tr -dc '0-9')
    if [ -n "$now_gb" ] && [ "$now_gb" -ge "$CAP_GB" ]; then
      say "ERROR: wipe did not shrink cache (${used_gb}G -> ${now_gb}G, cap ${CAP_GB}G)"
    else
      say "wiped and restarted; ${used_gb}G -> ${now_gb:-?}G free=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')G"
    fi
  else
    say "cache=${used_gb:-?}G / ${CAP_GB}G cap | disk ${free_gb}G free"
  fi

  sleep "$INTERVAL"
done
