# Required: Your runtime API key
export RUNTIME_API_KEY=$REMOTE_KEY

# Optional: Override default runtime API URL
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev" 

uv run swebench-infer .llm_config/gpt5_mini.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --select benchmarks/swebench/easy50_instances.txt \
    --split test \
    --workspace remote \
    --num-workers 32 \
    --max-iterations 60 \
    --n-limit 1

    # --output-dir $STORAGE_DIR/openhands/evaluation/evaluation_outputs/swe_bench_easy50_outputs \