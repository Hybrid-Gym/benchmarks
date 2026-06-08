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



API_KEY="api_key"
PORT=8000
NGROK_API="http://127.0.0.1:4040/api/tunnels"
NGROK_LOG="${STORAGE_DIR}/tmp/ngrok.log"

stop_existing_endpoint() {
    echo "Stopping existing endpoint on port $PORT, if any..."

    # Stop any process currently listening on PORT.
    local pids
    pids="$(lsof -ti tcp:"$PORT" || true)"

    if [[ -n "$pids" ]]; then
        echo "Killing process(es) on port $PORT: $pids"
        kill $pids 2>/dev/null || true
        sleep 2

        # Force kill if still alive.
        pids="$(lsof -ti tcp:"$PORT" || true)"
        if [[ -n "$pids" ]]; then
            echo "Force killing process(es) on port $PORT: $pids"
            kill -9 $pids 2>/dev/null || true
        fi
    fi

    # Stop existing ngrok processes.
    if pgrep -f "ngrok http $PORT" >/dev/null 2>&1; then
        echo "Killing existing ngrok tunnel for port $PORT"
        pkill -f "ngrok http $PORT" || true
    fi
}

start_vllm() {
    echo "Starting vLLM on port $PORT ..."

    CUDA_VISIBLE_DEVICES=0 vllm serve "$CHECKPOINT_DIR" \
        --api-key "$API_KEY" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --tensor-parallel-size 1 \
        --served-model-name "$MODEL_SAVE_NAME" \
        --enable-prefix-caching \
        --dtype bfloat16 \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
}

start_ngrok_tunnel() {
    echo "Starting ngrok tunnel for port $PORT ..."

    ngrok http "$PORT" > "$NGROK_LOG" 2>&1 &

    NGROK_PID=$!
    echo "ngrok PID: $NGROK_PID"
}

get_ngrok_public_url() {
    local public_url=""

    echo "Waiting for ngrok public URL..." >&2

    for _ in {1..30}; do
        public_url="$(
            curl -s "$NGROK_API" | python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
    for tunnel in data.get("tunnels", []):
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            print(url)
            break
except Exception:
    pass
'
        )"

        if [[ -n "$public_url" ]]; then
            echo "$public_url"
            return 0
        fi

        sleep 1
    done

    echo "Failed to get ngrok public URL." >&2
    echo "ngrok log:" >&2
    cat "$NGROK_LOG" >&2 || true
    return 1
}

write_config() {
    local base_url="$1"

    echo "Writing config for $MODEL_SAVE_NAME to $CONFIG_FILE ..."
    echo "base_url: $base_url"

    cat > "$CONFIG_FILE" <<EOF
{
  "model": "openai/$MODEL_SAVE_NAME",
  "api_key": "$API_KEY",
  "base_url": "$base_url",
  "temperature": 0.0,
  "native_tool_calling": false
}
EOF
}

cleanup() {
    echo "Stopping background processes..."

    if [[ -n "${NGROK_PID:-}" ]]; then
        kill "$NGROK_PID" 2>/dev/null || true
    fi

    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
}

main() {
    trap cleanup EXIT

    stop_existing_endpoint

    start_vllm
    start_ngrok_tunnel

    local ngrok_public_url
    ngrok_public_url="$(get_ngrok_public_url)"

    local base_url="${ngrok_public_url}/v1"
    write_config "$base_url"

    echo "Configuration written successfully."
    echo "Public LiteLLM base_url: $base_url"

    wait "$VLLM_PID"
}

main "$@"

