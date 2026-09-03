#!/usr/bin/env bash
# Run inference for one already-served model on the FULL SWE-bench_Verified set.
#
#   bash infer.sh <hf_repo_id_or_save_name>
#
# Re-running is safe: swebench-infer treats output.jsonl as the source of
# truth and only processes instances that have not completed yet.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_SAVE_NAME="$(save_name_of "${1:?usage: infer.sh <model>}")"
CONFIG_FILE="$REPO_DIR/.llm_config/${MODEL_SAVE_NAME}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: missing $CONFIG_FILE — is the model served?" >&2
    exit 1
fi

echo "model:       $MODEL_SAVE_NAME"
echo "config:      $CONFIG_FILE"
echo "dataset:     $DATASET ($SPLIT, full set — no --select)"
echo "output root: $EVAL_OUT_ROOT"
echo "base_url:    $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_url"])' "$CONFIG_FILE")"

cd "$REPO_DIR"
uv run swebench-infer "$CONFIG_FILE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --workspace remote \
    --num-workers "$NUM_WORKERS" \
    --max-iterations "$MAX_ITER" \
    --output-dir "$EVAL_OUT_ROOT"
rc=$?

OUTPUT_DIR="$(output_dir_of "$MODEL_SAVE_NAME")"
echo "exit code:   $rc"
echo "output dir:  $OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/output.jsonl" ]]; then
    echo "completed:   $(wc -l < "$OUTPUT_DIR/output.jsonl") instances"
fi
exit $rc
