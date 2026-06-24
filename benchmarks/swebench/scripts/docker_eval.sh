MODEL_NAME="${1:-qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"

export SDK_SHORT_SHA="e212d45"
export IMAGE_TAG_PREFIX="e212d45-35d813f"

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"


export OGMA_STORAGE_DIR="/projects/ogma3/yiqingxi"
export OGMA_OUTPUT_DIR="${OGMA_STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_60"

uv run swebench-eval $OGMA_OUTPUT_DIR/output.jsonl --run-id init --no-modal --timeout 600

