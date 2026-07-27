
#!/usr/bin/env bash
set -euo pipefail

LLM_CONFIG_PATH="${LLM_CONFIG_PATH:-.llm_config/anthropic_sonnet4_funclocalize.json}"
DATASET_NAME="${DATASET_NAME:-synthetic-code-training/swe_doc_gen_locate_5000}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
WORKSPACE_TYPE="${WORKSPACE_TYPE:-docker}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
N_LIMIT="${N_LIMIT:-1}"
RUN_NOTE="${RUN_NOTE:-masked5000-opus47-funclocalize}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ "${WORKSPACE_TYPE}" == "remote" ]]; then
    export RUNTIME_API_KEY="${RUNTIME_API_KEY:-${REMOTE_KEY:-}}"
    export RUNTIME_API_URL="${RUNTIME_API_URL:-https://runtime.eval.all-hands.dev}"
    export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="${OPENHANDS_EVAL_AGENT_SERVER_IMAGE:-ghcr.io/yiqingxyq/eval-agent-server}"
else
    export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="${OPENHANDS_EVAL_AGENT_SERVER_IMAGE:-ghcr.io/openhands/agent-server}"
fi

PROMPT_PATH_ARG=()
if [[ -n "${PROMPT_PATH:-}" ]]; then
    PROMPT_PATH_ARG=(--prompt-path "${PROMPT_PATH}")
fi

uv run hybridgym-funclocalize-infer "${LLM_CONFIG_PATH}" \
    --dataset "${DATASET_NAME}" \
    --split "${DATASET_SPLIT}" \
    --workspace "${WORKSPACE_TYPE}" \
    --num-workers "${NUM_WORKERS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --n-limit "${N_LIMIT}" \
    --note "${RUN_NOTE}" \
    "${PROMPT_PATH_ARG[@]}"
