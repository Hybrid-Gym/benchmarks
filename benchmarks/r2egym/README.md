# R2E-Gym Benchmark Evaluation

This directory runs [R2E-Gym](https://arxiv.org/abs/2504.07164) inference and
evaluation with OpenHands agents. It mirrors the `benchmarks/swebench/`
implementation and reuses the same shared machinery (workspaces, image build
pipeline, condenser, critics, disk cleanup); only the dataset-specific glue
differs.

## Overview

R2E-Gym is a large (~8.1K problems) execution-based environment for training and
evaluating SWE agents. Like SWE-Bench, each instance is a real repository at a
buggy commit plus a hidden test suite; the agent must produce a patch that makes
the tests pass. R2E-Gym is primarily used to **generate agent trajectories** and
to score them with an execution reward.

### How it differs from SWE-Bench (what "changed on the swebench code")

| | SWE-Bench | R2E-Gym |
|---|---|---|
| Base docker image | derived from `instance_id` (`swebench/sweb.eval.x86_64.<repo>_1776_<name>`) | shipped **in the dataset** as the `docker_image` column (`namanjain12/<repo>_final:<commit>`) |
| Repo field | `repo` = `owner/name` | `repo_name` = bare name (e.g. `aiohttp`) |
| Instance id | `instance_id` column | synthesized as `repo_name__commit_hash` (see `dataset.py`) |
| Base commit | `base_commit` column | none; a base-snapshot commit of the shipped `/testbed` is taken at runtime as the diff base |
| Repo path in image | `/testbed` | `/testbed` (same) |
| Evaluation | official `swebench.harness` | run baked `run_tests.sh`, compare per-test PASS/FAIL to `expected_output_json` (`eval_infer.py`, faithful to R2E-Gym's `_calculate_reward_r2e`) |

## Dataset

- **Source**: `R2E-Gym` org on the Hugging Face Hub
- **Variants**:
  - `R2E-Gym/R2E-Gym-Lite` (curated subset) - **default**
  - `R2E-Gym/R2E-Gym-Subset`
  - `R2E-Gym/R2E-Gym-V1` (full)
- **Split**: `train`

## Usage

### Docker Workspace (Local Evaluation)

R2E-Gym base images are pulled from Docker Hub on demand and the agent-server
layer is built locally per instance - **no pre-build step is required** for the
`docker` workspace.

```bash
uv run r2egym-infer .llm_config/anthropic_opus45_r2egym.json \
    --dataset R2E-Gym/R2E-Gym-Lite \
    --split train \
    --workspace docker \
    --num-workers 4 \
    --max-iterations 60
```

Quick smoke test (1 instance, opus 4.5):

```bash
bash benchmarks/r2egym/scripts/test_infer.sh
# whole split:  N_LIMIT=0 NUM_WORKERS=4 bash benchmarks/r2egym/scripts/test_infer.sh
```

**Disk safety.** Each per-instance agent-server image *and* the R2E-Gym base image
it was built from are removed right after the instance finishes (via
`cleanup_image=True` plus the `_cleanup_workspace` hook in `run_infer.py`). R2E-Gym
base images are large and unique per instance, so this keeps a long run from
filling the disk. Only exact images (by id / exact tag) are removed - unrelated
images on a shared host are never touched.

Resume a run by re-running the same command with the same `--output-dir`;
completed instances are skipped.

**Selecting specific instances:**

```bash
# instance ids are "<repo_name>__<commit_hash>", e.g. aiohttp__f0d74880...
echo "aiohttp__f0d74880deec8fcd982bce639c93c5e130d41198" > instances.txt

uv run r2egym-infer .llm_config/anthropic_opus45_r2egym.json \
    --select instances.txt --workspace docker --num-workers 1
```

### Remote / Apptainer Workspaces (pre-built images)

> Not supported yet for R2E-Gym. The root permission fix runs via
> `docker exec --user root` and only applies to the local `docker` workspace; on
> remote/apptainer the root-owned `/testbed` is unwritable by the `openhands`
> user and every instance fails at the base-snapshot commit. To enable these,
> bake the permission fix (readable uv python + `chown /testbed`) into the pushed
> image. Use `--workspace docker` for now.

`remote` and `apptainer` workspaces pull **pre-built, publicly accessible**
agent-server images from a registry, so build and push them first:

```bash
uv run python -m benchmarks.r2egym.build_images \
  --dataset R2E-Gym/R2E-Gym-Lite \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal \
  --push --max-workers 32
```

Then, for remote:

```bash
export RUNTIME_API_KEY="your-runtime-api-key"
uv run r2egym-infer .llm_config/anthropic_opus45_r2egym.json \
    --workspace remote --num-workers 32 --max-iterations 500
```

The image build reuses the SWE-Bench three-phase pipeline; the only R2E-Gym
change is that base images come from the dataset's `docker_image` column and the
per-instance tag is `"<repo>_final_<commit>"` (see `build_images.extract_custom_tag`).

## Evaluation

`r2egym-eval` scores the patches in an inference `output.jsonl`. It is a
self-contained reimplementation of R2E-Gym's `_calculate_reward_r2e` and does
**not** require the `r2egym` pip package. For each instance it starts a fresh
container from the instance's `docker_image`, reproduces R2E-Gym's `setup_env`,
applies the model patch (`git apply --whitespace=fix` in `/testbed`), runs the
baked `bash /root/run_tests.sh`, parses the pytest summary, and awards reward 1.0
iff the per-test result map matches `expected_output_json` **exactly** (same size
and identical per-test status - a test that is expected to remain `FAILED` must
still fail).

```bash
uv run r2egym-eval eval_outputs/r2egym_outputs/.../output.jsonl
# options:
uv run r2egym-eval output.jsonl --workers 2 --timeout 600 --keep-images
```

Writes `<output>.report.json` with the resolve rate, per-instance results, and
resolved ids. **Disk safety**: like inference, each base image is `docker rmi`'d
after its instance (peak disk is roughly `--workers` x image size); pass `--keep-images`
to disable.

> Validated end-to-end on this box (2026-07-08): opus-4.5 on the aiohttp instance
> `aiohttp__f0d74880...` produced a clean patch and `r2egym-eval` scored it
> **1/1 resolved (reward 1.0, 64/64 tests)**. `run_tests.sh` / `expected_output_json`
> are baked per-image artifacts, so still sanity-check resolve rates on new repos
> against R2E-Gym's reported numbers.

### Runtime notes (R2E-Gym specifics)

- **Root-centric images.** R2E-Gym images run as root: the repo venv interpreter
  lives under `/root` and `/testbed` is root-owned, but the OpenHands agent-server
  runs as `openhands`. `run_infer.py` repairs this at container start
  (`docker exec --user root`: make the uv python readable, `chown` `/testbed`), so
  the prepared venv (already first on PATH) is usable and edits land in `/testbed`.
- **In-place editing.** The agent edits `/testbed` directly (not a `/workspace`
  copy) because the repo is installed editable into `/testbed/.venv`.
- **Clean patch.** `/testbed` ships with uncommitted R2E setup changes and untracked
  harness files; inference takes a base-snapshot commit (no `git reset`) before the
  agent so the diff is only the agent's net changes.

## Files

| File | Purpose |
|---|---|
| `run_infer.py` | inference entrypoint (`r2egym-infer`) |
| `eval_infer.py` | evaluation entrypoint (`r2egym-eval`) |
| `dataset.py` | loads a split and synthesizes `instance_id` |
| `build_images.py` | pre-build agent-server images (remote/apptainer) |
| `config.py` | inference / evaluation defaults |
| `constants.py` | paths, git identity, build target |
| `prompts/default.j2` | agent instruction template |
| `scripts/test_infer.sh` | local docker smoke test |

## References

- [R2E-Gym Paper](https://arxiv.org/abs/2504.07164)
- [R2E-Gym GitHub](https://github.com/R2E-Gym/R2E-Gym)
- [R2E-Gym on Hugging Face](https://huggingface.co/R2E-Gym)
