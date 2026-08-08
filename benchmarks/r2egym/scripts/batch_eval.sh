#!/usr/bin/env bash
# Budget-, disk- and cache-aware batched R2E-Gym eval.
#
# Generalised from batch_eval_dv4f.sh (which hardcoded one run). Everything that
# differs per run comes from the environment:
#
#   RUN_DIR=<abs or repo-relative run dir>  bash benchmarks/r2egym/scripts/batch_eval.sh
#
# Why batching at all: the eval starts a container per instance and deletes the image
# straight after, so it pulls ~340 images/hr while the Docker Hub account budget is
# 200/hr. Running it flat-out starves both the eval and any live rollout, and the
# failures come back as opaque `toomanyrequests` errors -- 722 spurious ones on an
# earlier run. So each batch waits for real headroom before starting.
#
# Three gates before a batch runs:
#   1. Docker Hub budget >= MIN_BUDGET, leaving room for concurrent rollouts.
#   2. Free disk >= MIN_FREE_GB.
#   3. If disk is below PRUNE_FREE_GB, reclaim our own images first and re-check
#      rather than waiting for someone else to free space.
#
# PREFETCH pulls each batch's images through the local pull-through cache
# (127.0.0.1:5000) and retags them to their canonical names, so `docker run` finds them
# locally and never pulls. A cache hit costs no Docker Hub quota, which is what makes
# the prune-and-refetch loop above affordable: dropping an image no longer means
# paying for it twice.
#
# Re-entrant: a batch whose report exists is skipped, so it can be killed and resumed.
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

RUN_DIR="${RUN_DIR:?set RUN_DIR to the rollout output dir}"
SRC="${SRC:-$RUN_DIR/eval_snapshot.jsonl}"
BATCH_DIR="${BATCH_DIR:?set BATCH_DIR}"
LOG="${LOG:-$RUN_DIR/batch_eval.log}"

DATASET="${DATASET:-R2E-Gym/R2E-Gym-Lite}"
SPLIT="${SPLIT:-train}"
W="${W:-4}"
TIMEOUT="${TIMEOUT:-300}"

MIN_BUDGET="${MIN_BUDGET:-120}"     # keep headroom for concurrent rollouts
MIN_FREE_GB="${MIN_FREE_GB:-150}"   # never start a batch that could fill a shared disk
PRUNE_FREE_GB="${PRUNE_FREE_GB:-300}"
CACHE="${CACHE:-127.0.0.1:5000}"
PREFETCH="${PREFETCH:-1}"

say() { echo "$(date -u '+%F %T') $*" >> "$LOG"; }

get_remaining() {
  local creds token rem
  creds=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.docker/config.json')))['auths']['https://index.docker.io/v1/']['auth'])" 2>/dev/null)
  [ -z "$creds" ] && { echo -1; return; }
  token=$(curl -s --max-time 20 -H "Authorization: Basic $creds" \
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
  [ -z "$token" ] && { echo -1; return; }
  rem=$(curl -s --max-time 20 -I -H "Authorization: Bearer $token" \
    "https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest" 2>/dev/null \
    | grep -i "^ratelimit-remaining" | grep -oE "[0-9]+" | head -1)
  echo "${rem:-0}"
}

free_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

# Reclaim OUR images only -- this box also hosts another tenant's containers, so
# `docker system prune -a` would destroy their work. Skips anything attached to a
# container in any state, and anything tagged locally within the hour (a just-pulled
# image whose container does not exist yet).
prune_ours() {
  local active cutoff n tagged epoch
  active=$(docker ps -a --format '{{.Image}}' 2>/dev/null || true)
  cutoff=$(( $(date +%s) - 3600 ))
  n=0
  while read -r img; do
    echo "$active" | grep -qF "$img" && continue
    tagged=$(docker image inspect -f '{{.Metadata.LastTagTime}}' "$img" 2>/dev/null) || continue
    [ -z "$tagged" ] && continue
    epoch=$(date -d "$tagged" +%s 2>/dev/null) || continue
    [ "$epoch" -gt "$cutoff" ] && continue
    docker rmi -f "$img" >/dev/null 2>&1 && n=$((n + 1))
  done < <(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
           | grep -E "namanjain12/|ghcr\.io/openhands/eval-agent-server:|xingyaoww/sweb\.eval")
  docker builder prune -f --filter until=5m >/dev/null 2>&1 || true
  echo "$n"
}

# Pull a batch's images through the cache and retag to canonical, so the eval's
# `docker run` finds them locally. Cache hits cost no Docker Hub quota.
prefetch_batch() {
  local batch="$1" hit=0 miss=0
  while read -r img; do
    [ -z "$img" ] && continue
    docker image inspect "$img" >/dev/null 2>&1 && { hit=$((hit + 1)); continue; }
    if docker pull -q "$CACHE/$img" >/dev/null 2>&1; then
      docker tag "$CACHE/$img" "$img" >/dev/null 2>&1
      docker rmi "$CACHE/$img" >/dev/null 2>&1
      miss=$((miss + 1))
    fi
  done < <(.venv/bin/python3 - "$batch" "$DATASET" "$SPLIT" <<'PY' 2>/dev/null
import sys

# Use the eval's own loader, not load_dataset: R2E-Gym-Lite has NO instance_id
# column -- get_dataset synthesises it -- so mapping ids to images any other way
# silently matches nothing and the prefetch quietly becomes a no-op.
from benchmarks.r2egym.dataset import get_dataset

df = get_dataset(
    dataset_name=sys.argv[2],
    split=sys.argv[3],
    eval_limit=None,
    selected_instances_file=sys.argv[1],
)
for img in df["docker_image"]:
    print(str(img))
PY
  )
  say "  prefetch: $miss fetched via cache, $hit already local"
}

say "eval start: src=$SRC batches=$(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | wc -l) W=$W min_budget=$MIN_BUDGET min_free=${MIN_FREE_GB}G prune_below=${PRUNE_FREE_GB}G prefetch=$PREFETCH"

for b in "$BATCH_DIR"/batch_*.txt; do
  name=$(basename "$b" .txt)
  report="$BATCH_DIR/${name}.report.json"
  if [ -f "$report" ]; then say "$name already done, skipping"; continue; fi

  while true; do
    free=$(free_gb)
    if [ "$free" -lt "$PRUNE_FREE_GB" ]; then
      freed=$(prune_ours)
      say "$name disk ${free}G < ${PRUNE_FREE_GB}G -> pruned $freed images, now $(free_gb)G"
      free=$(free_gb)
    fi
    rem=$(get_remaining)
    # -1 means the budget probe itself failed: treat as "unknown, do not spend"
    # rather than charging into a quota we cannot see.
    if [ "$rem" -ge "$MIN_BUDGET" ] && [ "$free" -ge "$MIN_FREE_GB" ]; then
      say "$name starting: budget=$rem free=${free}G ids=$(wc -l < "$b")"
      break
    fi
    say "$name waiting: budget=$rem (need $MIN_BUDGET) free=${free}G (need ${MIN_FREE_GB}G)"
    sleep 180
  done

  [ "$PREFETCH" = "1" ] && prefetch_batch "$b"

  "$REPO/.venv/bin/r2egym-eval" "$SRC" --select "$b" --workers "$W" --timeout "$TIMEOUT" \
    --dataset "$DATASET" --split "$SPLIT" --output-file "$report" >> "$LOG" 2>&1
  rc=$?

  if [ -f "$report" ]; then
    python3 - "$report" "$name" <<'PY' >> "$LOG" 2>&1
import json, sys
res = json.load(open(sys.argv[1]))["results"]
print(f"{sys.argv[2]} DONE: resolved={sum(1 for i in res if i.get('resolved'))} "
      f"errored={sum(1 for i in res if i.get('error'))} total={len(res)}")
PY
  else
    say "$name FAILED rc=$rc, no report written"
  fi
done

say "ALL BATCHES DONE"
