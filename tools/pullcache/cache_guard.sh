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
    docker stop "$CONTAINER" >/dev/null 2>&1
    # Delete only the registry's own data subtree, never $CACHE_DIR itself, so a
    # mistyped/empty CACHE_DIR cannot turn this into an rm of something else.
    rm -rf "${CACHE_DIR:?}/docker"
    docker start "$CONTAINER" >/dev/null 2>&1
    say "wiped and restarted; now $(du -sBG "$CACHE_DIR" 2>/dev/null | cut -f1)"
  else
    say "cache=${used_gb:-?}G / ${CAP_GB}G cap | disk ${free_gb}G free"
  fi

  sleep "$INTERVAL"
done
