
export RUNTIME_API_KEY=$REMOTE_KEY
export RUNTIME_API_URL="https://runtime.eval.all-hands.dev" 

export OPENHANDS_EVAL_AGENT_SERVER_IMAGE="ghcr.io/yiqingxyq/eval-agent-server"

uv run hybridgym-funclocalize-infer .llm_config/gpt5_mini.json \
    --dataset synthetic-code-training/swe_doc_gen_locate_5000 \
    --split train \
    --workspace remote \
    --num-workers 8 \
    --max-iterations 60 \
    --n-limit 1 

