#!/usr/bin/env bash
# Smoke-test / small SWE-Gym inference run on the local docker workspace.
#
# SWE-Gym instances are SWE-bench-format; run_infer builds each agent-server
# image on demand from docker.io/xingyaoww/sweb.eval.x86_64.<instance_id>, so a
# long run does not need pre-built images.
#
# Usage:
#   bash benchmarks/swegym/scripts/test_infer.sh                 # 1 instance
#   N_LIMIT=0 NUM_WORKERS=8 bash benchmarks/swegym/scripts/test_infer.sh
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-anthropic_opus45_r2egym}"
LLM_CONFIG="${LLM_CONFIG:-.llm_config/${MODEL_NAME}.json}"

DATASET="${DATASET:-SWE-Gym/SWE-Gym}"
SPLIT="${SPLIT:-train}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
N_LIMIT="${N_LIMIT:-1}"          # 1 = quick smoke; set 0 for the whole split
RUN_NOTE="${RUN_NOTE:-swegym-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/swegym_outputs}"

# Pin the agent-server image tag prefix to the SDK short sha (matches the built
# images and avoids the content-hash mismatch seen on local docker builds).
export IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-$(git -C vendor/software-agent-sdk rev-parse --short=7 HEAD)}"

echo "LLM_CONFIG=${LLM_CONFIG}"
echo "DATASET=${DATASET} SPLIT=${SPLIT} NUM_WORKERS=${NUM_WORKERS} N_LIMIT=${N_LIMIT}"
echo "IMAGE_TAG_PREFIX=${IMAGE_TAG_PREFIX}"

uv run swegym-infer "${LLM_CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --workspace docker \
    --num-workers "${NUM_WORKERS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --n-limit "${N_LIMIT}" \
    --note "${RUN_NOTE}" \
    --output-dir "${OUTPUT_DIR}"
