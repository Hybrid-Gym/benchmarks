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

