MODEL_HF_NAME="${1:-synthetic-code-training/qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16}"
STORAGE_DIR="${2:-/home/yiqingxi}"

MAX_MODEL_LEN=32768

MODEL_SAVE_NAME=$(basename $MODEL_HF_NAME)
MODEL_SAVE_NAME=${MODEL_SAVE_NAME//./}
CHECKPOINT_DIR="$STORAGE_DIR/checkpoints/$MODEL_SAVE_NAME"
CONFIG_FILE=".llm_config/${MODEL_SAVE_NAME}.json"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Downloading checkpoint $MODEL_HF_NAME to $CHECKPOINT_DIR ..."
  hf download \
    $MODEL_HF_NAME \
    --local-dir "$CHECKPOINT_DIR"
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory $CHECKPOINT_DIR does not exist"
  exit 1
fi


CONFIG_ENTRY="
{
  \"model\": \"openai/$MODEL_SAVE_NAME\",
  \"api_key\": \"api_key\",
  \"base_url\": \"http://127.0.0.1:8000/v1\",
  \"temperature\": 0.0,
  \"native_tool_calling\": false
}
"

echo "Adding configuration for $MODEL_SAVE_NAME to $CONFIG_FILE ..."
echo "$CONFIG_ENTRY" > "$CONFIG_FILE"


CUDA_VISIBLE_DEVICES=0 vllm serve $CHECKPOINT_DIR \
    --api-key "api_key" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --served-model-name $MODEL_SAVE_NAME \
    --enable-prefix-caching \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len $MAX_MODEL_LEN 

