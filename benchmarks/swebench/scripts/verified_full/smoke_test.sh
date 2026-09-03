#!/usr/bin/env bash
# Serving smoke test: bring one model up end-to-end, prove the OpenHands SDK
# can talk to it through the ngrok tunnel, then tear everything down.
#
#   bash smoke_test.sh [hf_repo_id]     (defaults to the first model in models.txt)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

MODEL_HF_NAME="${1:-$(grep -vE '^\s*(#|$)' "$SCRIPT_DIR/models.txt" | head -1)}"
NAME="$(save_name_of "$MODEL_HF_NAME")"
LOG_DIR="$RUN_LOG_ROOT/$NAME"; mkdir -p "$LOG_DIR"
READY_FILE="$LOG_DIR/base_url.txt"
SERVE_LOG="$LOG_DIR/smoke_serve.log"

echo "smoke test model: $MODEL_HF_NAME"
echo "serve log:        $SERVE_LOG"

rm -f "$READY_FILE"
bash "$SCRIPT_DIR/serve.sh" "$MODEL_HF_NAME" > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap 'echo "tearing down serve pid $SERVE_PID"; kill $SERVE_PID 2>/dev/null; sleep 5; pkill -f "ngrok http $SERVE_PORT" 2>/dev/null; true' EXIT

TIMEOUT="${SMOKE_TIMEOUT:-3600}"
waited=0
while [[ ! -s "$READY_FILE" ]]; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "FAIL: serve.sh exited early. Last 40 lines:"; tail -40 "$SERVE_LOG"; exit 1
    fi
    (( waited >= TIMEOUT )) && { echo "FAIL: not ready after ${TIMEOUT}s"; tail -40 "$SERVE_LOG"; exit 1; }
    if (( waited % 60 == 0 )); then echo "  ... waiting (${waited}s): $(tail -1 "$SERVE_LOG")"; fi
    sleep 10; waited=$((waited + 10))
done

BASE_URL="$(cat "$READY_FILE")"
echo
echo "PASS: vLLM + ngrok ready in ${waited}s at $BASE_URL"
echo

echo "--- SDK check (the exact path swebench-infer uses) ---"
cd "$REPO_DIR"
uv run python -c "
from benchmarks.utils.llm_config import load_llm_config
from openhands.sdk.llm import Message, TextContent
llm = load_llm_config('.llm_config/${NAME}.json')
r = llm.completion([Message(role='user', content=[TextContent(text='Reply with exactly: SMOKE_OK')])])
print('SDK response:', r.message.content[0].text.strip()[:120])
"
rc=$?
echo
[[ $rc -eq 0 ]] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED (SDK check rc=$rc)"
exit $rc
