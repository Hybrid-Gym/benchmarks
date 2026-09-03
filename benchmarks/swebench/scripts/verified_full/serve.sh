#!/usr/bin/env bash
# Serve one checkpoint with vLLM + expose it through ngrok.
#
#   bash serve.sh <hf_repo_id>
#
# Downloads the checkpoint to $CKPT_ROOT if missing, starts vLLM on
# $SERVE_PORT, opens an ngrok tunnel, writes .llm_config/<save_name>.json,
# and writes the public base_url to $READY_FILE once the server answers.
# Runs in the foreground until vLLM exits, so the caller can background it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_HF_NAME="${1:?usage: serve.sh <hf_repo_id>}"
MODEL_SAVE_NAME="$(save_name_of "$MODEL_HF_NAME")"
CHECKPOINT_DIR="$CKPT_ROOT/$MODEL_SAVE_NAME"
CONFIG_FILE="$REPO_DIR/.llm_config/${MODEL_SAVE_NAME}.json"

LOG_DIR="$RUN_LOG_ROOT/$MODEL_SAVE_NAME"
mkdir -p "$LOG_DIR" "$REPO_DIR/.llm_config"
NGROK_LOG="$LOG_DIR/ngrok.log"
READY_FILE="$LOG_DIR/base_url.txt"
rm -f "$READY_FILE"

VLLM_PID=""
NGROK_PID=""

log() { echo "[serve $(date '+%H:%M:%S')] $*"; }

download_checkpoint() {
    # Complete iff every shard named by the index file is present.
    if [[ -f "$CHECKPOINT_DIR/model.safetensors.index.json" ]] && \
       python3 - "$CHECKPOINT_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
sys.exit(1 if missing else 0)
PY
    then
        log "checkpoint already complete at $CHECKPOINT_DIR"
        return 0
    fi

    log "downloading $MODEL_HF_NAME -> $CHECKPOINT_DIR"
    hf download "$MODEL_HF_NAME" --local-dir "$CHECKPOINT_DIR" || return 1
}

stop_existing_endpoint() {
    log "clearing port $SERVE_PORT and stale ngrok agents"
    pkill -u "$USER" -f "ngrok http $SERVE_PORT" 2>/dev/null || true
    pkill -u "$USER" -f "ngrok .*--config $NGROK_CONFIG_FILE" 2>/dev/null || true

    local pids
    pids="$(lsof -ti tcp:"$SERVE_PORT" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        log "killing pids on port $SERVE_PORT: $pids"
        kill $pids 2>/dev/null || true
        sleep 5
        pids="$(lsof -ti tcp:"$SERVE_PORT" 2>/dev/null || true)"
        [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null || true
    fi
    sleep 2
}

start_vllm() {
    log "starting vLLM on port $SERVE_PORT (max_model_len=$MAX_MODEL_LEN)"
    local extra=()
    [[ "$ENFORCE_EAGER" == "1" ]] && extra+=(--enforce-eager)
    # Without a tool-call parser vLLM rejects any request carrying
    # tool_choice="auto" with HTTP 400. Measured at ~3.5% of calls even with
    # native_tool_calling=false, so enable the Qwen2.5-family parser.
    [[ "$ENABLE_TOOL_PARSER" == "1" ]] && \
        extra+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$VLLM_BIN" serve "$CHECKPOINT_DIR" \
        --api-key "$SERVE_API_KEY" \
        --host 0.0.0.0 \
        --port "$SERVE_PORT" \
        --tensor-parallel-size 1 \
        --served-model-name "$MODEL_SAVE_NAME" \
        --enable-prefix-caching \
        --dtype bfloat16 \
        "${extra[@]}" \
        --max-model-len "$MAX_MODEL_LEN" &
    VLLM_PID=$!
    log "vLLM pid $VLLM_PID"
}

wait_for_vllm_local() {
    log "waiting for vLLM /health (up to 30 min)"
    for _ in $(seq 1 360); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log "ERROR: vLLM process died during startup"
            return 1
        fi
        if curl -sf -m 5 "http://127.0.0.1:$SERVE_PORT/health" >/dev/null 2>&1; then
            log "vLLM is healthy locally"
            return 0
        fi
        sleep 5
    done
    log "ERROR: vLLM did not become healthy in time"
    return 1
}

start_ngrok_tunnel() {
    log "starting ngrok tunnel for port $SERVE_PORT"
    "$NGROK_BIN" http "$SERVE_PORT" --config "$NGROK_CONFIG_FILE" > "$NGROK_LOG" 2>&1 &
    NGROK_PID=$!
    log "ngrok pid $NGROK_PID"
}

get_ngrok_public_url() {
    for _ in $(seq 1 60); do
        local url
        url="$(curl -s -m 5 "$NGROK_API" 2>/dev/null | python3 -c '
import json, sys
try:
    for t in json.load(sys.stdin).get("tunnels", []):
        u = t.get("public_url", "")
        if u.startswith("https://"):
            print(u); break
except Exception:
    pass
')"
        if [[ -n "$url" ]]; then echo "$url"; return 0; fi
        sleep 2
    done
    log "ERROR: no ngrok public URL; log follows" >&2
    cat "$NGROK_LOG" >&2 || true
    return 1
}

write_config() {
    local base_url="$1"
    log "writing $CONFIG_FILE (base_url=$base_url)"
    cat > "$CONFIG_FILE" <<EOF
{
  "model": "openai/$MODEL_SAVE_NAME",
  "api_key": "$SERVE_API_KEY",
  "base_url": "$base_url",
  "temperature": 0.0,
  "native_tool_calling": false,
  "litellm_extra_body": {"enable_thinking": false}
}
EOF
}

verify_through_tunnel() {
    local base_url="$1"
    log "verifying a chat completion through the tunnel"
    for _ in $(seq 1 20); do
        local body
        body="$(curl -s -m 60 "$base_url/chat/completions" \
            -H "Authorization: Bearer $SERVE_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL_SAVE_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":16,\"temperature\":0}" 2>/dev/null)"
        if echo "$body" | grep -q '"choices"'; then
            log "tunnel OK: $(echo "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:80])' 2>/dev/null)"
            return 0
        fi
        sleep 5
    done
    log "ERROR: chat completion through tunnel failed"
    return 1
}

cleanup() {
    log "cleanup: stopping ngrok and vLLM"
    [[ -n "$NGROK_PID" ]] && kill "$NGROK_PID" 2>/dev/null || true
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true
    sleep 5
    [[ -n "$VLLM_PID" ]] && kill -9 "$VLLM_PID" 2>/dev/null || true
    rm -f "$READY_FILE"
}

main() {
    trap cleanup EXIT INT TERM

    download_checkpoint || { log "FATAL: checkpoint download failed"; exit 1; }
    stop_existing_endpoint
    start_vllm
    wait_for_vllm_local || exit 1
    start_ngrok_tunnel

    local url base_url
    url="$(get_ngrok_public_url)" || exit 1
    base_url="${url}/v1"
    write_config "$base_url"
    verify_through_tunnel "$base_url" || exit 1

    echo "$base_url" > "$READY_FILE"
    log "READY: $base_url"
    wait "$VLLM_PID"
}

main "$@"
