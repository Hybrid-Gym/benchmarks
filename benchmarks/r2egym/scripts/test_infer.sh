#!/usr/bin/env bash
# Smoke-test / small R2E-Gym inference run on the LOCAL docker workspace.
#
# Disk safety: run_infer.py removes each per-instance agent-server image AND its
# R2E-Gym base image right after the instance finishes (cleanup_image=True plus
# the _cleanup_workspace hook), so a long run will not fill the disk.
#
# Usage:
#   bash benchmarks/r2egym/scripts/test_infer.sh                 # opus 4.5, 1 instance
#   N_LIMIT=0 NUM_WORKERS=4 bash benchmarks/r2egym/scripts/test_infer.sh
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-anthropic_opus45_r2egym}"
LLM_CONFIG="${LLM_CONFIG:-.llm_config/${MODEL_NAME}.json}"

DATASET="${DATASET:-R2E-Gym/R2E-Gym-Lite}"
SPLIT="${SPLIT:-train}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
N_LIMIT="${N_LIMIT:-1}"          # 1 = quick smoke; set 0 for the whole split
RUN_NOTE="${RUN_NOTE:-r2egym-opus45-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/r2egym_outputs}"
# Optional file of instance ids to restrict the run to, e.g.
# eval_outputs/r2egym_qwen35_select_1502.txt. Without it N_LIMIT=0 means the FULL
# 4578-instance split, not the 1502 comparison subset the run notes refer to.
SELECT="${SELECT:-}"

# Pin the agent-server image tag prefix to the SDK short sha (matches the built
# images and avoids the content-hash mismatch seen on local docker builds).
export IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-$(git -C vendor/software-agent-sdk rev-parse --short=7 HEAD)}"

echo "LLM_CONFIG=${LLM_CONFIG}"
echo "DATASET=${DATASET} SPLIT=${SPLIT} NUM_WORKERS=${NUM_WORKERS} N_LIMIT=${N_LIMIT}"
echo "IMAGE_TAG_PREFIX=${IMAGE_TAG_PREFIX}"

SELECT_ARGS=()
if [ -n "${SELECT}" ]; then
    echo "SELECT=${SELECT} ($(wc -l < "${SELECT}") instances)"
    SELECT_ARGS=(--select "${SELECT}")
fi

uv run r2egym-infer "${LLM_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --workspace docker \
    --num-workers "${NUM_WORKERS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --n-limit "${N_LIMIT}" \
    --note "${RUN_NOTE}" \
    "${SELECT_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}"
