# SWE-Gym Benchmark - building Docker images

This directory contains implementation for building custom agent server Docker images for SWE-Gym. The primary purpose is to use GitHub workflows for building these images fast and using them to train LLMs as SWE agents.

## Dataset

- **Source**: [Paper](https://arxiv.org/abs/2412.21139)
- **Dataset**: 
  - `SWE-Gym/SWE-Gym` - Full dataset
- **Splits**: `train`

## Usage

### Build Docker Images

You need to build Docker images for the SWE-Gym instances. Each instance requires a specific environment setup based on the repository and issue. **Note that this will consume atleast 5-6TB of disk space. Considering setting `--n-limit` to a smaller value if required**

```bash
uv run python -m benchmarks.swegym.build_images \
  --dataset SWE-Gym/SWE-Gym \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

### Running rollouts

SWE-Gym instances use the SWE-bench schema, so the rollout reuses
`SWEBenchEvaluation` and only swaps the base image resolver to the SWE-Gym image
set (`docker.io/xingyaoww/sweb.eval.x86_64.<instance_id>`). On the local docker
workspace each agent-server image is built on demand, so no pre-build is needed.

```bash
# 1 instance (smoke)
bash benchmarks/swegym/scripts/test_infer.sh

# whole split
N_LIMIT=0 NUM_WORKERS=8 MODEL_NAME=<llm_config> \
  bash benchmarks/swegym/scripts/test_infer.sh
```

Or call the entry point directly:

```bash
uv run swegym-infer .llm_config/<llm_config>.json \
  --dataset SWE-Gym/SWE-Gym --split train --workspace docker \
  --num-workers 8 --max-iterations 60 --n-limit 0
```

Evaluation is the SWE-bench harness (`FAIL_TO_PASS`/`PASS_TO_PASS`) — SWE-Gym
ships no separate reward.

### Multi-model comparison runs

`launch_runs.sh` starts a supervised rollout per model, all on the same instance
list so the models are directly comparable:

```bash
NUM_WORKERS=2 bash benchmarks/swegym/scripts/launch_runs.sh all
NUM_WORKERS=2 bash benchmarks/swegym/scripts/launch_runs.sh kimi25 dv4f   # subset
```

Each model gets its own tmux session (`swegym-<key>`) running
`run_supervisor.sh`, which relaunches the rollout until every selected instance has
a critic-passing trajectory, then exits. Re-running the launcher skips models whose
session is already live, so it doubles as the way to ramp workers up.

Two constraints decide `NUM_WORKERS`:

- **The gateway WAF limit is per-IP and box-wide.** It counts the *sum* of workers
  across every live rollout here, regardless of model or API key — ~8 total is the
  observed-safe level. Set `NUM_WORKERS` to your share of that budget, not to what
  one model could sustain alone.
- **Docker Hub allows 200 pulls/hr**, and a rollout pulls one base image per
  instance. Four runs at ~7 instances/hr/worker stay well inside that; a concurrent
  *eval* does not (it pulls ~340/hr).

The shared selection is `eval_outputs/swegym_select_1500.txt`, rebuilt from scratch
with:

```bash
uv run python benchmarks/swegym/scripts/probe_images.py     # ~25 min, resumable
uv run python benchmarks/swegym/scripts/build_selection.py  # seed 42 -> 1500 ids
```

`probe_images.py` records which instances have a published
`xingyaoww/sweb.eval.x86_64.*` image; 37 of the 2438 train instances have none and
can never run for any model, so `build_selection.py` samples only from the other
2401. It uses hub.docker.com's repository API rather than registry manifest reads,
which would count against the same 200/hr pull budget the evals need. The seed makes
the draw reproducible — re-running yields the identical 1500 ids.

