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
#   b) only reclaim images this guard has been SEEING for at least ~1h -- a freshly
#      built or freshly pulled image whose container does not exist yet is invisible to
#      (a), and without the age check it gets deleted out from under a worker
#      mid-startup.
#
# Where "age" comes from, and why it is not read off the image (fixed 2026-08-11):
#   - CreatedSince is upstream build time. A xingyaoww base pulled ten seconds ago
#     reports "2 months old", so a CreatedSince guard deletes images mid-build.
#   - Metadata.LastTagTime is only set for images this host TAGGED. Pulled images keep
#     the Go zero value "0001-01-01 00:00:00 +0000 UTC", which `date -d` cannot parse
#     (it carries both +0000 and UTC). The original code did `date ... || continue`, so
#     EVERY pulled image fell through the skip and the reclaim loop deleted nothing:
#     223 consecutive sweeps logged reclaimed=0 while 333GB of idle bases sat on disk.
#     Free space only looked healthy because `builder prune` + `image prune -f` were
#     carrying it, until other tenants grew and free fell 617GB -> 292GB in 6h.
#   - So we keep our own first-seen ledger instead. An image absent from the ledger is
#     recorded and spared; it becomes reclaimable one INTERVAL-quantised hour later.
#     That is strictly conservative: a just-pulled image always gets a full grace
#     period, and the worst case after a guard restart is one wasted hour, never an
#     early delete.
#
# Never touches anything outside our two name prefixes: this box also hosts another
# user's swesmith/susvibes runs, so `docker system prune -a` would be destructive.
#
# Usage (needs tmux -- it must outlive the session that starts it):
#   tmux new-session -d -s swegym-diskguard "bash benchmarks/swegym/scripts/disk_guard.sh"
set -u

REPO=/home/gaokaizhang/benchmarks
GUARD_LOG="$REPO/eval_outputs/swegym_outputs/disk_guard.log"
SEEN_DB="$REPO/eval_outputs/swegym_outputs/.image_first_seen"
INTERVAL="${INTERVAL:-300}"
MIN_FREE_GB="${MIN_FREE_GB:-0}"   # 0 = always sweep; >0 = only sweep below this

# Seconds an image must have been visible before rmi. Lowered 3600 -> 1200 on
# 2026-08-15 under disk pressure. The grace only ever protects the gap between an
# image existing and its container being created (a few seconds) -- an image attached
# to a container in ANY state is skipped above, before this check, so a shorter grace
# cannot touch live work. At 3600 a 5-worker rollout finishing ~25 instances/hr kept
# ~125GB of already-finished instances' images pinned, on a box with ~100GB free.
GRACE="${GRACE:-1200}"

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

  # REMOVED 2026-08-16: `docker image prune -f`.
  #
  # It deletes every dangling image box-wide, and an image being pulled or built
  # right now is briefly dangling. Running every 300s against eval batches that take
  # 30-60 minutes, it kept destroying in-flight layers -- the eval failed with
  # `ImageNotFound: No such image: sha256:...` on roughly half its batches, and the
  # failure looked like a harness bug rather than ours. It is also box-wide, so it
  # was reaching other tenants' intermediate layers, which the prefix-filtered rmi
  # loop below deliberately avoids.
  #
  # The targeted reclaim below already frees our own images; this added nothing
  # except a race.

  ACTIVE_IMGS=$(docker ps -a --format '{{.Image}}' 2>/dev/null || true)
  now=$(date +%s)
  cutoff=$(( now - GRACE ))
  touch "$SEEN_DB"
  next_db=$(mktemp)
  reclaimed=0
  while read -r img; do
    # Ledger is rebuilt from the images that still exist, so entries for images we
    # deleted (or that some other path removed) do not accumulate forever.
    first=$(grep -F -m1 "	$img" "$SEEN_DB" 2>/dev/null | cut -f1)
    case "$first" in ''|*[!0-9]*) first=$now ;; esac
    printf '%s\t%s\n' "$first" "$img" >> "$next_db"

    echo "$ACTIVE_IMGS" | grep -qF "$img" && continue
    [ "$first" -gt "$cutoff" ] && continue
    if docker rmi -f "$img" >/dev/null 2>&1; then
      reclaimed=$((reclaimed + 1))
      # Drop the just-deleted image from the ledger we are about to install.
      sed -i "\|	${img}\$|d" "$next_db"
    fi
  done < <(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
           | grep -E "ghcr\.io/openhands/eval-agent-server:|xingyaoww/sweb\.eval")
  mv -f "$next_db" "$SEEN_DB"

  after=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "$(date -u '+%F %T') pruned [$bc] | reclaimed=${reclaimed} | ${avail}GB -> ${after}GB free | workers=$(pgrep -f '[s]wegym-infer' | wc -l)" >> "$GUARD_LOG"
  sleep "$INTERVAL"
done
