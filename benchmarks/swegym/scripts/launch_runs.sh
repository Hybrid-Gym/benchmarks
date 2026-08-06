#!/usr/bin/env bash
# Launch supervised SWE-Gym rollouts for one or more models on the SHARED selection.
#
# Every model runs the same instance list (eval_outputs/swegym_select_1500.txt) so the
# results are directly comparable; that file is built once from the instances whose
# xingyaoww image actually exists on Docker Hub.
#
# Worker budget is the thing to get right. The NVIDIA gateway sits behind an AWS WAF
# per-IP rate limiter that counts EVERY concurrent job on this box, not per model or
# per key, so the ceiling is on the SUM of NUM_WORKERS across all live rollouts.
# ~8 total is the observed-safe level; 14 produced a 429 storm on 2026-08-05.
#
# Usage:
#   bash benchmarks/swegym/scripts/launch_runs.sh kimi25 dv4f          # start two
#   NUM_WORKERS=2 bash benchmarks/swegym/scripts/launch_runs.sh all
#
# Re-running for a model that is already live is a no-op, so this is safe to repeat
# when ramping workers up after another run finishes.
set -u

REPO=/home/gaokaizhang/benchmarks
cd "$REPO" || exit 1

NUM_WORKERS="${NUM_WORKERS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
SELECT="${SELECT:-eval_outputs/swegym_select_1500.txt}"
TOTAL="${TOTAL:-$(wc -l < "$SELECT")}"

# key -> llm config basename. The configs carry only model/key/base_url, so the
# _r2egym ones are reused as-is rather than duplicating API keys into new files.
declare -A CONFIG=(
  [qwen80b]=anthropic_qwen3_next_80b_r2egym
  [dv4f]=anthropic_deepseek_v4_flash_r2egym
  [gpt5mini]=anthropic_gpt5_mini_azuredirect_r2egym
  [kimi25]=anthropic_kimi_k25_r2egym
)
ALL="qwen80b dv4f gpt5mini kimi25"

targets="$*"
[ -z "$targets" ] && targets="$ALL"
[ "$targets" = "all" ] && targets="$ALL"

for key in $targets; do
  cfg="${CONFIG[$key]:-}"
  if [ -z "$cfg" ]; then
    echo "unknown model key '$key' (want one of: $ALL)" >&2
    exit 1
  fi
  note="swegym-${key}-1500"
  session="swegym-${key}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "SKIP $key: tmux session $session already exists"
    continue
  fi

  # Ask the code itself where the run dir will be, rather than re-deriving the
  # naming rule here: it embeds the SDK short sha, which changes whenever
  # vendor/software-agent-sdk moves and would silently split a resumed run.
  run_dir=$("$REPO/.venv/bin/python3" - "$cfg" "$note" "$MAX_ITERATIONS" <<'PY'
import json, sys
from benchmarks.utils.evaluation_utils import construct_eval_output_dir
cfg, note, maxiter = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = json.load(open(f".llm_config/{cfg}.json"))["model"]
print(construct_eval_output_dir(
    base_dir="./eval_outputs/swegym_outputs",
    dataset_name="SWE-Gym__SWE-Gym-train",
    model_name=model,
    max_iterations=maxiter,
    eval_note=note,
))
PY
  ) || { echo "FAILED to resolve run dir for $key" >&2; exit 1; }

  echo "START $key  workers=$NUM_WORKERS  note=$note"
  echo "      run_dir=$run_dir"
  tmux new-session -d -s "$session" \
    "MODEL_NAME=$cfg RUN_NOTE=$note RUN_DIR='$run_dir' \
     NUM_WORKERS=$NUM_WORKERS MAX_ITERATIONS=$MAX_ITERATIONS \
     TOTAL=$TOTAL SELECT=$SELECT \
     bash benchmarks/swegym/scripts/run_supervisor.sh"
done

echo
echo "live sessions:"; tmux ls 2>/dev/null | grep swegym || echo "  (none)"
