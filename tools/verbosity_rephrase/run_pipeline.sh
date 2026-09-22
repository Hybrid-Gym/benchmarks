#!/usr/bin/env bash
# Verbosity-ablation pipeline: rephrase each source dataset, build its five variants, push.
# Re-entrant: rephrase.py resumes its jsonl; a dataset with a .pushed stamp is skipped.
set -u
cd "$(dirname "$0")/../.."
export OPENHANDS_SUPPRESS_BANNER=1
PY=.venv/bin/python
OUT=${OUT:-eval_outputs/verbosity_rephrase}
MODEL=${MODEL:-nvidia/deepseek-ai/deepseek-v4-flash}
EXTRA_BODY=${EXTRA_BODY:-'{"chat_template_kwargs":{"thinking":false}}'}
WORKERS=${WORKERS:-6}
MAX_TOKENS=${MAX_TOKENS:-2500}
PUSH=${PUSH:-1}
DATASETS=${DATASETS:-"synthetic-code-training/func_localize_claude45_1457i synthetic-code-training/r2egym_qwen3next80b_1500i"}
NOISE='^\[\|sitecustomize\|collision\|Overriding\|run_instance_modal\|ion_modal\|function \[None\]'  # sdk banner lines
mkdir -p "$OUT/variants" "$OUT/logs"
LOG=$OUT/logs/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "START model=$MODEL workers=$WORKERS push=$PUSH"
for REPO in $DATASETS; do
  NAME=${REPO##*/}
  if [ -f "$OUT/$NAME.pushed" ] && [ "$PUSH" = 1 ]; then log "skip $NAME (already pushed)"; continue; fi
  # a pass can leave api-error lines; the next pass redoes them. Done when a pass starts with 0 pending.
  for PASS in 1 2 3 4 5; do
    log "rephrase $NAME pass $PASS"
    $PY tools/verbosity_rephrase/rephrase.py --hf "$REPO" --out-dir "$OUT" --model "$MODEL" \
        --extra-body "$EXTRA_BODY" --workers "$WORKERS" --max-tokens "$MAX_TOKENS" 2>&1 \
        | grep --line-buffered -v "$NOISE" | tee -a "$OUT/logs/$NAME.rephrase.log"
    PENDING=$(grep -a "pending" "$OUT/logs/$NAME.rephrase.log" | tail -1 | sed -E 's/.* ([0-9]+) pending.*/\1/')
    ERRS=$(grep -ac '"error"' "$OUT/$NAME.rephrase.jsonl" 2>/dev/null || true)
    log "rephrase $NAME pass $PASS done (last pending=$PENDING, error lines so far=${ERRS:-0})"
    [ "${PENDING:-1}" = 0 ] && break
    sleep 60
  done
  log "build $NAME"
  ARGS=(--hf "$REPO" --rephrase "$OUT/$NAME.rephrase.jsonl" --out-dir "$OUT/variants")
  [ "$PUSH" = 1 ] && ARGS+=(--push)
  if $PY tools/verbosity_rephrase/build_variants.py "${ARGS[@]}" 2>&1 | grep --line-buffered -v "$NOISE" | tee -a "$OUT/logs/$NAME.build.log"; then
    if grep -q "pushed .*_text300" "$OUT/logs/$NAME.build.log"; then touch "$OUT/$NAME.pushed"; log "PUSHED $NAME"; else log "build $NAME finished (push=$PUSH)"; fi
  else
    log "BUILD FAILED $NAME"
  fi
done
log "PIPELINE_DONE"
