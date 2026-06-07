MODEL_NAME="${1:-qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"

# Required: Your runtime API key
export RUNTIME_API_KEY=$REMOTE_KEY

# Optional: Override default runtime API URL
# export RUNTIME_API_URL="https://runtime.eval.all-hands.dev" 

uv run swebench-infer .llm_config/${MODEL_NAME}.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --select benchmarks/swebench/easy50_instances.txt \
    --split test \
    --workspace remote \
    --num-workers 16 \
    --max-iterations 60 \
    --output-dir $STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs

    # --n-limit 1