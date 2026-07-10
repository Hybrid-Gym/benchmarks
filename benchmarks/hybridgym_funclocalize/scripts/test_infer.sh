MODEL_NAME="${1:-gpt5_mini}"

export SDK_SHORT_SHA="e212d45"
export IMAGE_TAG_PREFIX="e212d45-35d813f"

export RUNTIME_API_KEY=$REMOTE_KEY
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev" 

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"

PROMPT_NAME="keyword_search"

uv run hybridgym-funclocalize-infer .llm_config/${MODEL_NAME}.json \
    --dataset synthetic-code-training/swe_doc_gen_locate_1500 \
    --split train \
    --workspace remote \
    --num-workers 1 \
    --max-iterations 60 \
    --output-dir $STORAGE_DIR/benchmarks/evaluation_outputs/func_loc_${PROMPT_NAME}_outputs \
    --n-limit 1 

