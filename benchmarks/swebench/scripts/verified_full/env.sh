# Shared environment for the full SWE-bench_Verified model sweep.
# Source this; do not execute it.
#
# Everything that can grow without bound lives on /data/user_data/gaokaiz
# (2.0T free) rather than /home/gaokaiz (100G NFS quota).

REPO_DIR="${REPO_DIR:-/home/gaokaiz/benchmarks}"
export STORAGE_DIR="${STORAGE_DIR:-/data/user_data/gaokaiz}"

# --- storage layout -------------------------------------------------------
export CKPT_ROOT="$STORAGE_DIR/checkpoints"
export EVAL_OUT_ROOT="$STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_verified_full_outputs"
export RUN_LOG_ROOT="$STORAGE_DIR/benchmarks/run_logs"
export STATUS_FILE="${STATUS_FILE:-$RUN_LOG_ROOT/status.tsv}"
export TMP_ROOT="$STORAGE_DIR/tmp"

# Keep every large cache off /home.
export HF_HOME="$STORAGE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$STORAGE_DIR/hf_cache/hub"
export VLLM_CACHE_ROOT="$STORAGE_DIR/vllm_cache"
export TRITON_CACHE_DIR="$STORAGE_DIR/vllm_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$STORAGE_DIR/vllm_cache/inductor"
export XDG_CACHE_HOME="$STORAGE_DIR/vllm_cache/xdg"
export TMPDIR="$TMP_ROOT"

mkdir -p "$CKPT_ROOT" "$EVAL_OUT_ROOT" "$RUN_LOG_ROOT" "$TMP_ROOT" \
         "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# --- serving --------------------------------------------------------------
# ngrok accounts we own are named gaokai_<N> and each has its own config file
# /home/gaokaiz/.config/ngrok/ngrok_gaokai_<N>.yml (mode 600, outside the git
# repo so the authtoken can never be committed). Today we own exactly one,
# gaokai_1; more are planned, hence the numbering.
#
#   NGROK_ACCOUNT=gaokai_1   -> ngrok_gaokai_1.yml, serve port 2334, web 4041
#   NGROK_ACCOUNT=gaokai_2   -> ngrok_gaokai_2.yml, serve port 2335, web 4042
#
# When adding gaokai_2, set that file's agent.web_addr to localhost:4042 so the
# two agents do not fight over the local API port.
#
# ngrok.yml in that directory belongs to a COLLABORATOR — never start tunnels on
# it. ngrok1.yml is an older copy of gaokai_1. (ngrok_gaokai.yml is kept as a
# symlink to ngrok_gaokai_1.yml so older scripts and snapshots keep working.)
#
# The free plan gives one static *.ngrok-free.dev domain per account, so ONE
# model per account can be exposed over HTTPS at a time; the queue is sequential
# for that reason. A second account is what would allow a second model in
# parallel — on its own port, with its own vLLM process.
export NGROK_ACCOUNT="${NGROK_ACCOUNT:-gaokai_1}"
export NGROK_CONFIG_FILE="${NGROK_CONFIG_FILE:-/home/gaokaiz/.config/ngrok/ngrok_${NGROK_ACCOUNT}.yml}"
# Trailing number of the account name drives the port offsets, so gaokai_2 can
# never collide with gaokai_1 on the same node.
export NGROK_ACCOUNT_ID="${NGROK_ACCOUNT_ID:-${NGROK_ACCOUNT##*_}}"
[[ "$NGROK_ACCOUNT_ID" =~ ^[0-9]+$ ]] || NGROK_ACCOUNT_ID=1
export SERVE_PORT=$((2333 + NGROK_ACCOUNT_ID))
export NGROK_WEB_PORT=$((4040 + NGROK_ACCOUNT_ID))
export NGROK_API="http://127.0.0.1:${NGROK_WEB_PORT}/api/tunnels"
export SERVE_API_KEY="${SERVE_API_KEY:-api_key}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# CUDA graphs are worth having: eager mode measured ~500 tok/s aggregate decode
# for a 7B on a 96GB Blackwell at batch ~13. Set ENFORCE_EAGER=1 to restore the
# old behaviour if compilation causes trouble.
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Off by default: enabling the parser changes how vLLM returns tool calls, which
# could disturb the prompt-based path this model is trained for. Turn on only
# after checking it does not break tool parsing.
export ENABLE_TOOL_PARSER="${ENABLE_TOOL_PARSER:-0}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
export VLLM_BIN="${VLLM_BIN:-/home/gaokaiz/miniconda3/envs/eval/bin/vllm}"
export NGROK_BIN="${NGROK_BIN:-/home/gaokaiz/.local/bin/ngrok}"

# --- results publishing ---------------------------------------------------
# Grading needs docker, which this node lacks, so each finished model is
# published to the Hub and graded on a machine that has it. Every record
# carries resolved=null until that machine fills it in.
export HF_RESULTS_REPO="${HF_RESULTS_REPO:-synthetic-code-training/swebench-verified-results}"
export HF_TOKEN_FILE="${HF_TOKEN_FILE:-/home/gaokaiz/.config/hf/token_gaokai}"
export UPLOAD_TRAJECTORIES="${UPLOAD_TRAJECTORIES:-0}"

# --- inference ------------------------------------------------------------
export SDK_SHORT_SHA="${SDK_SHORT_SHA:-e212d45}"
export IMAGE_TAG_PREFIX="${IMAGE_TAG_PREFIX:-e212d45-35d813f}"
export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
export EVAL_AGENT_SERVER_IMAGE="ghcr.io/hybrid-gym/eval-agent-server"
# The remote-runtime key lives only in the environment of whoever launched the
# job, so a fresh Slurm allocation starts without it and every instance fails.
# Fall back to the persisted copy.
RUNTIME_KEYS_FILE="${RUNTIME_KEYS_FILE:-/home/gaokaiz/.config/openhands/runtime_keys.env}"
if [[ -z "${REMOTE_KEY:-}" && -r "$RUNTIME_KEYS_FILE" ]]; then
    source "$RUNTIME_KEYS_FILE"
fi
export RUNTIME_API_KEY="${RUNTIME_API_KEY:-${REMOTE_KEY:-}}"
if [[ -z "$RUNTIME_API_KEY" ]]; then
    echo "WARNING: RUNTIME_API_KEY/REMOTE_KEY is empty — the remote workspace will reject inference." >&2
fi
export MAX_ITER="${MAX_ITER:-60}"
# vLLM measured 40.66x max concurrency at 32k context and never once queued a
# request (Waiting: 0 on every scheduler tick) with 16 workers, so the worker
# pool — not the GPU — was the throughput limit. 24 is a deliberate middle
# ground: a 1.5x lift that stays well under the 40x ceiling and keeps the
# remote-runtime API and the single ngrok tunnel from becoming the new
# bottleneck, which is where a stall would come from rather than the GPU.
export NUM_WORKERS="${NUM_WORKERS:-24}"
export DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
export SPLIT="${SPLIT:-test}"

# Save name matches what vllm --served-model-name and .llm_config use.
save_name_of() {
    local n
    n="$(basename "$1")"
    echo "${n//./}"
}

output_dir_of() {
    # $1 = model save name
    echo "$EVAL_OUT_ROOT/princeton-nlp__SWE-bench_Verified-test/openai/${1}_sdk_${SDK_SHORT_SHA}_maxiter_${MAX_ITER}"
}
