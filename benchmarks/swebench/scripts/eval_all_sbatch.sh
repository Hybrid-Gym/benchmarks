#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=agent
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --output=/home/yiqingxi/tmp/eval%A.out
#SBATCH --mail-user=yiqingxi@andrew.cmu.edu
#SBATCH --mail-type=END,FAIL

MODEL_HF_NAME="${1:-synthetic-code-training/qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"
NGROK_ACCOUNT_ID="${2:-1}"
MAX_ITER="${3:-60}"

MODEL_NAME=$(basename $MODEL_HF_NAME)
MODEL_NAME=${MODEL_NAME//./}

export USERNAME=yiqingxi
export STORAGE_DIR=/data/tir/projects/tir5/users/yiqingxi
export HOME_DIR=/home/$USERNAME

export SDK_SHORT_SHA="e212d45"
export IMAGE_TAG_PREFIX="e212d45-35d813f"

echo "Node(s): $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=pci.bus_id
nvidia-smi -q -d PAGE_RETIREMENT

OUTPUT_DIR=$STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/${MODEL_NAME}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER} 
REPORT_FILE=$OUTPUT_DIR/output.report.json
OUTPUT_FILE=$OUTPUT_DIR/output.jsonl

check_model_served() {
    uv run python -c "
from benchmarks.utils.llm_config import load_llm_config
from openhands.sdk.llm import Message, TextContent
llm = load_llm_config('${CONFIG_FILE}')
messages = [Message(role='user', content=[TextContent(text='Say hello')])]
response = llm.completion(messages)
print(response.message.content[0].text)
" > /dev/null 2>&1
}

source $HOME_DIR/home_conda_setup.sh openhands
cd $HOME_DIR/benchmarks
export PYTHONPATH=`pwd`

if [ -f "$REPORT_FILE" ] && [ -f "$OUTPUT_FILE" ] && [ $(wc -l < "$OUTPUT_FILE") -gt 495 ]; then
    echo "Report file exists and has more than 495 lines"
    echo "Skipping evaluation block..."
else
    echo "Running evaluation block..."

    CONFIG_FILE=".llm_config/${MODEL_NAME}.json"
    bash benchmarks/utils/vllm_serve_ngrok.sh $MODEL_HF_NAME $NGROK_ACCOUNT_ID $HOME_DIR & 

    echo "Waiting for vLLM server to start..."
    MAX_HEALTH_CHECK_ATTEMPTS=15
    HEALTH_CHECK_ATTEMPT=0
    SERVER_READY=0
    sleep 3m

    while [ $HEALTH_CHECK_ATTEMPT -lt $MAX_HEALTH_CHECK_ATTEMPTS ]; do
        if check_model_served; then
        echo "vLLM server is responding!"
        SERVER_READY=1
        break
        fi
        echo "vLLM server not responding yet... (attempt $((HEALTH_CHECK_ATTEMPT + 1))/$MAX_HEALTH_CHECK_ATTEMPTS)"
        sleep 1m
        HEALTH_CHECK_ATTEMPT=$((HEALTH_CHECK_ATTEMPT + 1))
    done

    if [ $SERVER_READY -eq 0 ]; then
        echo "ERROR: vLLM server failed to respond after $MAX_HEALTH_CHECK_ATTEMPTS attempts"
        echo "Exiting script..."
        exit 1
    fi

    echo "start running run_infer"

    bash benchmarks/swebench/scripts/infer_all.sh $MODEL_NAME $MAX_ITER
fi

python benchmarks/swebench/extra_eval.py --split all --input_file $OUTPUT_DIR/output.jsonl --total_num -1 

if [ -f "$REPORT_FILE" ] && [ -f "$OUTPUT_FILE" ] && [ $(wc -l < "$OUTPUT_FILE") -gt 495 ]; then
    # remove ckpt
    MODEL_SAVE_NAME=$(basename $MODEL_HF_NAME)
    CHECKPOINT_DIR="$HOME_DIR/checkpoints/$MODEL_SAVE_NAME"
    rm -r $CHECKPOINT_DIR
    echo "Removed checkpoint directory: $CHECKPOINT_DIR"

    echo "Completed!"
else
    echo "ERROR: Report file does not exist or has less than 495 lines"
    echo "Exiting script..."
fi