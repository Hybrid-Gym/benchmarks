#!/usr/bin/env bash
# Verbosity-ablation pipeline: rephrase each source dataset, build the family's variants, push.
# FAMILY=fixed (text0/20/50/100/300, think dropped) or scaled (text0.5x/2x/4x/8x, think kept).
# Re-entrant: rephrase.py resumes its jsonl; a dataset with a .pushed stamp is skipped.
set -u
cd "$(dirname "$0")/../.."
export OPENHANDS_SUPPRESS_BANNER=1
PY=.venv/bin/python
OUT=${OUT:-eval_outputs/verbosity_rephrase}
FAMILY=${FAMILY:-fixed}
MODEL=${MODEL:-nvidia/deepseek-ai/deepseek-v4-flash}
EXTRA_BODY=${EXTRA_BODY:-'{"chat_template_kwargs":{"thinking":false}}'}
WORKERS=${WORKERS:-6}
MAX_TOKENS=${MAX_TOKENS:-2500}
PUSH=${PUSH:-1}
DATASETS=${DATASETS:-"synthetic-code-training/func_localize_claude45_1457i synthetic-code-training/r2egym_qwen3next80b_1500i"}
NOISE='^\[\|sitecustomize\|collision\|Overriding\|run_instance_modal\|ion_modal\|function \[None\]'  # sdk banner lines
LAST=$([ "$FAMILY" = scaled ] && echo text8x || echo text300)  # the variant pushed last
mkdir -p "$OUT/variants" "$OUT/logs"
LOG=$OUT/logs/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "START family=$FAMILY model=$MODEL workers=$WORKERS push=$PUSH"
for REPO in $DATASETS; do
  NAME=${REPO##*/}
  TAG=$NAME.$FAMILY
  if [ -f "$OUT/$TAG.pushed" ] && [ "$PUSH" = 1 ]; then log "skip $TAG (already pushed)"; continue; fi
  # a pass can leave api-error lines; the next pass redoes them. Done when a pass starts with 0 pending.
  for PASS in 1 2 3 4 5; do
    log "rephrase $TAG pass $PASS"
    $PY tools/verbosity_rephrase/rephrase.py --family "$FAMILY" --hf "$REPO" --out-dir "$OUT" --model "$MODEL" \
        --extra-body "$EXTRA_BODY" --workers "$WORKERS" --max-tokens "$MAX_TOKENS" 2>&1 \
        | grep --line-buffered -v "$NOISE" | tee -a "$OUT/logs/$TAG.rephrase.log"
    PENDING=$(grep -a "pending" "$OUT/logs/$TAG.rephrase.log" | tail -1 | sed -E 's/.* ([0-9]+) pending.*/\1/')
    ERRS=$(grep -ac '"error"' "$OUT/$TAG.jsonl" 2>/dev/null || true)
    log "rephrase $TAG pass $PASS done (last pending=$PENDING, error lines so far=${ERRS:-0})"
    [ "${PENDING:-1}" = 0 ] && break
    sleep 60
  done
  log "build $TAG"
  ARGS=(--family "$FAMILY" --hf "$REPO" --rephrase "$OUT/$TAG.jsonl" --out-dir "$OUT/variants")
  [ "$PUSH" = 1 ] && ARGS+=(--push)
  if $PY tools/verbosity_rephrase/build_variants.py "${ARGS[@]}" 2>&1 | grep --line-buffered -v "$NOISE" | tee -a "$OUT/logs/$TAG.build.log"; then
    if grep -q "pushed .*_$LAST" "$OUT/logs/$TAG.build.log"; then touch "$OUT/$TAG.pushed"; log "PUSHED $TAG"; else log "build $TAG finished (push=$PUSH)"; fi
  else
    log "BUILD FAILED $TAG"
  fi
done
log "PIPELINE_DONE"
