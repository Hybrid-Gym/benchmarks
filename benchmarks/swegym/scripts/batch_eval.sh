#!/usr/bin/env bash
# Disk- and budget-aware batched SWE-Gym eval, scoring EVERY model per batch.
#
# Adapted from benchmarks/r2egym/scripts/batch_eval.sh. Three things differ and each
# one matters:
#
#  1. The SWE-Bench-Fork harness does NOT delete images. It reports `Unremoved
#     images: N` and leaves them on disk. r2egym's evaluator removed each image right
#     after use, so its batching only had to bound the pull RATE; here the batch
#     script itself must delete, or ~4GB/instance fills a shared disk within two
#     batches and takes the live rollouts down with it.
#
#  2. All models share the SAME 1500-instance selection, so we score every model
#     against a batch while its images are resident, then delete once. Model-at-a-time
#     would pull the same 1500 images once per model -- 4500 pulls and ~30h against
#     Docker Hub's 200/hr instead of 1500 and ~7.5h.
#
#  2b. --cache_level instance is REQUIRED for (2) to work. At the default 'env',
#     should_remove() deletes every `sweb.eval.*` image after the run that used it,
#     so the FIRST model of a batch consumes the images and the second dies with
#     `ImageNotFound: No such image: sha256:...`. Observed on batches 0000/0002/0003:
#     model 1 completes cleanly, model 2 fails, and 11 of 12 images are gone.
#     'instance' is outside the {none,base,env} removal set, so images survive the
#     whole batch and cleanup_batch below is what bounds disk.
#
#  3. Image names are sanitized on the hub (lowercase, `__`->`_s_`) but the harness
#     wants the unsanitised local tag, so every pull is followed by a retag.
#
# Re-entrant: a (batch, model) whose report exists is skipped, so it can be killed and
# resumed at any point.
#
# Usage:
#   MODELS="gpt5mini qwen80b" tmux new-session -d -s swegym-eval \
#     "bash benchmarks/swegym/scripts/batch_eval.sh"
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

SELECT="${SELECT:-eval_outputs/swegym_select_1500.txt}"
BATCH_DIR="${BATCH_DIR:-eval_outputs/swegym_outputs/eval_batches}"
OUT_DIR="${OUT_DIR:-eval_outputs/swegym_outputs/eval_reports}"
LOG="${LOG:-eval_outputs/swegym_outputs/batch_eval.log}"
BATCH_SIZE="${BATCH_SIZE:-20}"

DATASET="${DATASET:-SWE-Gym/SWE-Gym}"
SPLIT="${SPLIT:-train}"
# Raised 4 -> 8 on 2026-08-17. A batch is 12 instances x 2 models = 24 evals, so W=4
# meant six serial waves. The box has 128 cores at load ~30 and 680G free RAM, and the
# eval containers are the same weight as the rollout's, which already runs 8.
W="${W:-8}"
TIMEOUT="${TIMEOUT:-1200}"

MIN_BUDGET="${MIN_BUDGET:-120}"    # leave Docker Hub headroom for the live rollout
MIN_FREE_GB="${MIN_FREE_GB:-90}"   # never start a batch that could fill a shared disk
CACHE="${CACHE:-127.0.0.1:5000}"
PREFETCH="${PREFETCH:-1}"

# Concurrent image pulls. Measured on batches 0004-0006: a batch spent 33-40min in
# prefetch and only 21-32min in the eval itself, because the pulls were serial -- 12
# MONAI images at 12.8GB each is ~154GB down a single 73MB/s stream. Worse, one stuck
# pull blocks all the rest: batch_0007 sat in prefetch for 163 minutes. Pulling in
# parallel both divides the transfer time and stops one bad layer from holding up the
# other eleven.
PAR="${PAR:-6}"
# A pull already in flight can still land ~13GB after its floor check passed, so with
# PAR of them running the plain MIN_FREE_GB floor is not a floor at all. Require enough
# headroom for every in-flight pull to complete before starting another.
PULL_FLOOR=$(( MIN_FREE_GB + PAR * 15 ))
# Ceiling on a single pull. Past this the layer is assumed wedged, not slow; the pull
# is abandoned and the harness fetches that one instance itself.
PULL_TIMEOUT="${PULL_TIMEOUT:-1200}"

# Which rollouts to score. Keys map to run dirs below.
MODELS="${MODELS:-gpt5mini qwen80b kimi25}"

declare -A RUN_DIR=(
  [gpt5mini]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/azure/openai/gpt-5-mini_sdk_e212d45_maxiter_60_N_swegym-gpt5mini-1500"
  [qwen80b]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/qwen/qwen3-next-80b-a3b-instruct_sdk_e212d45_maxiter_60_N_swegym-qwen80b-1500"
  [kimi25]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/moonshotai/kimi-k2.5_sdk_e212d45_maxiter_100_N_swegym-kimi25-1500"
  [opus45]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/aws/anthropic/claude-opus-4-5_sdk_e212d45_maxiter_100_N_swegym-opus45-1500"
  [dv4f]="eval_outputs/swegym_outputs/SWE-Gym__SWE-Gym-train/openai/nvidia/deepseek-ai/deepseek-v4-flash_sdk_e212d45_maxiter_100_N_swegym-dv4f-1500"
)

mkdir -p "$BATCH_DIR" "$OUT_DIR" "$(dirname "$LOG")"
say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }

free_gb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

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

# Fetch one image, preferring the local pull-through cache, and retag to the
# unsanitised name the harness looks for. Writes its verdict to stdout so the parallel
# driver below can tally results without shared counters.
pull_one() {
  local remote="$1" local_tag="$2"
  if timeout "$PULL_TIMEOUT" docker pull -q "$CACHE/$remote" >/dev/null 2>&1; then
    docker tag "$CACHE/$remote" "$local_tag" >/dev/null 2>&1
    docker rmi "$CACHE/$remote" >/dev/null 2>&1
    echo ok
  elif timeout "$PULL_TIMEOUT" docker pull -q "$remote" >/dev/null 2>&1; then
    docker tag "$remote" "$local_tag" >/dev/null 2>&1
    docker rmi "$remote" >/dev/null 2>&1
    echo ok
  else
    echo fail
  fi
}

# Pull a batch's images PAR at a time. Stops at the disk floor rather than filling the
# disk; the harness will pull whatever is missing as it goes.
prefetch_batch() {
  local batch="$1" hit=0 miss=0 fail=0 stopped=0 remote local_tag
  local todo resdir n=0 t0 t1
  todo=$(mktemp); resdir=$(mktemp -d); t0=$(date +%s)

  # Split local-vs-missing first: `docker image inspect` is cheap and doing it up front
  # keeps the parallel section to actual network work.
  while read -r remote local_tag; do
    [ -z "$remote" ] && continue
    if docker image inspect "$local_tag" >/dev/null 2>&1; then
      hit=$((hit + 1))
    else
      printf '%s\t%s\n' "$remote" "$local_tag" >> "$todo"
    fi
  done < <("$REPO/.venv/bin/python3" -m benchmarks.swegym.eval_infer images --select "$batch" 2>/dev/null)

  while IFS=$'\t' read -r remote local_tag; do
    [ -z "$remote" ] && continue
    # Floor is re-checked per slot, so a batch that starts with room but fills up
    # mid-prefetch stops here instead of taking the shared disk down.
    if [ "$(free_gb)" -lt "$PULL_FLOOR" ]; then stopped=1; break; fi
    pull_one "$remote" "$local_tag" > "$resdir/$n" 2>/dev/null &
    n=$((n + 1))
    # Block until a slot frees, which is also what paces the floor check above.
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n 2>/dev/null || break; done
  done < "$todo"
  wait

  miss=$(grep -lx 'ok' "$resdir"/* 2>/dev/null | wc -l)
  fail=$(grep -lx 'fail' "$resdir"/* 2>/dev/null | wc -l)
  rm -rf "$resdir" "$todo"
  t1=$(date +%s)
  say "  prefetch: $miss fetched, $hit already local, $fail unavailable in $(( (t1 - t0) / 60 ))m$([ "$stopped" = 1 ] && echo ', STOPPED at disk floor')"
}

# The harness leaves images behind; drop this batch's so the next one has room.
# Only ever touches `sweb.eval.x86_64.*` tags we created -- other tenants' images
# are never matched.
cleanup_batch() {
  local batch="$1" n=0 remote local_tag
  while read -r remote local_tag; do
    [ -z "$local_tag" ] && continue
    docker rmi -f "$local_tag" >/dev/null 2>&1 && n=$((n + 1))
  done < <("$REPO/.venv/bin/python3" -m benchmarks.swegym.eval_infer images --select "$batch" 2>/dev/null)
  docker container prune -f --filter until=10m >/dev/null 2>&1 || true
  say "  cleanup: removed $n images, free now $(free_gb)G"
}

# --- build batches and per-model prediction files once -------------------------
if ! ls "$BATCH_DIR"/batch_*.txt >/dev/null 2>&1; then
  "$REPO/.venv/bin/python3" -m benchmarks.swegym.eval_infer batches \
    --select "$SELECT" --size "$BATCH_SIZE" --out-dir "$BATCH_DIR" | tee -a "$LOG"
fi

for m in $MODELS; do
  preds="$OUT_DIR/preds_${m}.jsonl"
  [ -f "$preds" ] && continue
  "$REPO/.venv/bin/python3" -m benchmarks.swegym.eval_infer prepare \
    --run-dir "${RUN_DIR[$m]}" --model-name "swegym-${m}-1500" \
    --select "$SELECT" --out "$preds" | tee -a "$LOG"
done

say "eval start: models='$MODELS' batches=$(ls "$BATCH_DIR"/batch_*.txt | wc -l) size=$BATCH_SIZE W=$W min_free=${MIN_FREE_GB}G min_budget=$MIN_BUDGET"

for b in "$BATCH_DIR"/batch_*.txt; do
  name=$(basename "$b" .txt)

  # Skip the batch entirely only if EVERY model already has a report for it.
  all_done=1
  for m in $MODELS; do
    [ -f "$OUT_DIR/${name}.${m}.json" ] || all_done=0
  done
  if [ "$all_done" = "1" ]; then say "$name: all models done, skipping"; continue; fi

  # Gate on disk and Hub budget before spending anything.
  while true; do
    free=$(free_gb); rem=$(get_remaining)
    # -1 means the budget probe failed: treat as "unknown, do not spend".
    if [ "$rem" -ge "$MIN_BUDGET" ] && [ "$free" -ge "$MIN_FREE_GB" ]; then
      say "$name starting: budget=$rem free=${free}G ids=$(wc -l < "$b")"
      break
    fi
    say "$name waiting: budget=$rem (need $MIN_BUDGET) free=${free}G (need ${MIN_FREE_GB}G)"
    sleep 180
  done

  [ "$PREFETCH" = "1" ] && prefetch_batch "$b"

  for m in $MODELS; do
    report="$OUT_DIR/${name}.${m}.json"
    [ -f "$report" ] && continue
    preds="$OUT_DIR/preds_${m}.jsonl"
    run_id="${name}_${m}"

    # The harness writes <model_name_or_path>.<run_id>.json into its CWD, so run it
    # from a scratch dir and move the result to the name we index by.
    # Resolve to absolute BEFORE the subshell cds away. Prefixing $REPO/ instead
    # breaks whenever BATCH_DIR/OUT_DIR are given as absolute paths.
    b_abs=$(realpath "$b"); preds_abs=$(realpath "$preds")
    ids=$(tr '\n' ' ' < "$b_abs")
    if [ -z "${ids// /}" ]; then say "  $name [$m] SKIP: empty batch"; continue; fi
    # Retry within the pass. This box runs other tenants' swebench evals concurrently
    # (e.g. run_id=swebench-verified-500-*), and their docker image operations delete
    # dangling layers box-wide, so our pulls die mid-flight with
    # `ImageNotFound: No such image: sha256:...`. That is transient and unrelated to
    # the prediction being scored, so retry with a fresh prefetch instead of losing
    # the whole (batch, model) to someone else's cleanup.
    produced=""
    for attempt in 1 2 3; do
      tmp=$(mktemp -d)
      ( cd "$tmp" && "$REPO/.venv-swegym-eval/bin/python" -m swebench.harness.run_evaluation \
          --dataset_name "$DATASET" --split "$SPLIT" \
          --predictions_path "$preds_abs" \
          --instance_ids $ids \
          --max_workers "$W" --run_id "${run_id}_a${attempt}" --timeout "$TIMEOUT" \
          --cache_level instance ) >> "$LOG" 2>&1
      produced=$(ls "$tmp"/*."${run_id}_a${attempt}".json 2>/dev/null | head -1)
      [ -n "$produced" ] && break
      rm -rf "$tmp"
      say "  $name [$m] attempt $attempt produced no report; re-prefetching and retrying"
      prefetch_batch "$b"
    done
    if [ -n "$produced" ]; then
      mv "$produced" "$report"
      "$REPO/.venv/bin/python3" - "$report" "$name" "$m" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]} [{sys.argv[3]}] resolved={d.get('resolved_instances')} "
      f"unresolved={d.get('unresolved_instances')} errors={d.get('error_instances')} "
      f"empty={d.get('empty_patch_instances')} completed={d.get('completed_instances')}")
PY
    else
      say "  $name [$m] FAILED: no report produced"
    fi
    rm -rf "$tmp"
  done

  cleanup_batch "$b"
done

say "ALL BATCHES DONE"
