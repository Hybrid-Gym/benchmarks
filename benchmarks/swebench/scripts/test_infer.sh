MODEL_NAME="${1:-qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"

export SDK_SHORT_SHA="e212d45"
export IMAGE_TAG_PREFIX="e212d45-35d813f"

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export RUNTIME_API_KEY=$REMOTE_KEY

# Optional: Override default runtime API URL
# export RUNTIME_API_URL="https://runtime.eval.all-hands.dev" 

# uv run swebench-infer .llm_config/${MODEL_NAME}.json \
#     --dataset princeton-nlp/SWE-bench_Verified \
#     --select benchmarks/swebench/easy50_instances.txt \
#     --split test \
#     --workspace remote \
#     --num-workers 16 \
#     --max-iterations 60 \
#     --output-dir $STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs

    # --n-limit 1

OUTPUT_DIR=$STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_60

# uv run swebench-eval $OUTPUT_DIR/output.jsonl --run-id init --modal

export OGMA_STORAGE_DIR="/projects/ogma3/yiqingxi"
export OGMA_OUTPUT_DIR="${OGMA_STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_60"

rclone copy $OUTPUT_DIR/output.jsonl ogma:$OGMA_OUTPUT_DIR

OGMA_USER="yiqingxi"
OGMA_HOST="ogma.lti.cs.cmu.edu"

ssh $OGMA_USER@$OGMA_HOST "cd /home/${OGMA_USER}/benchmarks && bash benchmarks/swebench/scripts/docker_eval.sh $MODEL_NAME"

rclone copy ogma:$OGMA_OUTPUT_DIR/output.report.json $OUTPUT_DIR 

python benchmarks/swebench/extra_eval.py $OUTPUT_DIR/output.jsonl --total_num -1
