#!/usr/bin/env bash
set -euo pipefail

# Base qwen3.5-9b (NVIDIA gateway) on SWE-bench Verified easy50, LOCAL docker
# workspace. Per-instance images are auto-deleted after each instance
# (cleanup_image=True on DockerWorkspace + agent-server/base image removal in
# SWEBenchEvaluation._cleanup_workspace), so disk stays bounded across the run.

cd /home/gaokaizhang/benchmarks

LLM_CONFIG="${LLM_CONFIG:-.llm_config/anthropic_qwen35_9b_funclocalize.json}"
NUM_WORKERS="${NUM_WORKERS:-8}"
N_LIMIT="${N_LIMIT:-0}"          # 0 = all easy50; set 1 for a smoke test
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
NOTE="${NOTE:-qwen35-9b-base-easy50}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/swe_bench_easy50_outputs}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Align the image-tag prefix with what the LOCAL docker build actually tags.
# The local build tags images as "<sdk_short_sha>-<custom_tag>-<target>", but the
# phased-build lookup otherwise expects an extra Dockerfile content-hash segment
# ("<sdk_short_sha>-<hash>-..."), which never gets built locally. Setting
# IMAGE_TAG_PREFIX explicitly bypasses the content hash so lookup == build.
export IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-$(git -C vendor/software-agent-sdk rev-parse --short=7 HEAD)}"

uv run swebench-infer "${LLM_CONFIG}" \
    --dataset princeton-nlp/SWE-bench_Verified \
    --select benchmarks/swebench/easy50_instances.txt \
    --split test \
    --workspace docker \
    --num-workers "${NUM_WORKERS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --n-limit "${N_LIMIT}" \
    --note "${NOTE}" \
    --output-dir "${OUTPUT_DIR}"
