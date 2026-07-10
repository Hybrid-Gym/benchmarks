MODEL_NAME="${1:-qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"
CONFIG_NAME=$(basename $MODEL_NAME)

echo $MODEL_NAME 
echo $CONFIG_NAME

export SDK_SHORT_SHA="e212d45"
export IMAGE_TAG_PREFIX="e212d45-35d813f"

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export RUNTIME_API_KEY=$REMOTE_KEY

uv run swebench-infer .llm_config/${CONFIG_NAME}.json \
    --dataset princeton-nlp/SWE-bench_Verified \
    --select benchmarks/swebench/easy50_instances.txt \
    --split test \
    --workspace remote \
    --num-workers 16 \
    --max-iterations 60 \
    --output-dir $STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs

    # --n-limit 1

OUTPUT_DIR=$STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_60

# uv run swebench-eval $OUTPUT_DIR/output.jsonl --run-id init --modal

export OGMA_STORAGE_DIR="/projects/ogma3/yiqingxi"
export OGMA_OUTPUT_DIR="${OGMA_STORAGE_DIR}/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_60"

rclone copy $OUTPUT_DIR/output.jsonl ogma:$OGMA_OUTPUT_DIR

OGMA_USER="yiqingxi"
OGMA_HOST="ogma.lti.cs.cmu.edu"

MAX_ATTEMPTS=3
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "Running docker eval (attempt $attempt/$MAX_ATTEMPTS)..."
    ssh $OGMA_USER@$OGMA_HOST "cd /home/${OGMA_USER}/benchmarks && bash benchmarks/swebench/scripts/docker_eval.sh $MODEL_NAME"
    rclone copy ogma:$OGMA_OUTPUT_DIR/output.report.json $OUTPUT_DIR

    if [ -f "$OUTPUT_DIR/output.report.json" ]; then
        echo "Report file exists"
        break
    fi

    if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
        echo "Report file does not exist. Retrying..."
    else
        echo "Report file does not exist after $MAX_ATTEMPTS attempts."
    fi
done

python benchmarks/swebench/extra_eval.py --input_file $OUTPUT_DIR/output.jsonl --total_num -1

uv run python benchmarks/utils/post_process_scripts/combine_completions.py $OUTPUT_DIR/output.jsonl