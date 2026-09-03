# Full SWE-bench_Verified sweep

Runs a queue of vLLM-served checkpoints, one at a time, over the **complete**
SWE-bench_Verified set (500 instances — no `--select`), exposing each model to
the remote runtime through ngrok.

## Files

| File | Purpose |
| --- | --- |
| `models.txt` | The model queue, one HF repo id per line, in run order |
| `env.sh` | Shared paths and serving/inference settings — sourced by everything else |
| `serve.sh` | Download + serve one checkpoint (vLLM + ngrok), write `.llm_config/<model>.json` |
| `infer.sh` | Run `swebench-infer` for one served model over the full set |
| `run_queue.sh` | The orchestrator: serve → verify → infer → tear down → next model |
| `status.sh` | Render the run status table (`--watch` to refresh every 60s) |
| `smoke_test.sh` | Bring one model up, prove the SDK can reach it, tear it down |

## Storage — nothing large touches `/home`

`/home/gaokaiz` is a **100G NFS quota**. Six 7B bf16 checkpoints (~15G each)
plus 6 × 500 trajectories would blow through it, so `env.sh` redirects
everything to `$STORAGE_DIR` (`/data/user_data/gaokaiz`, 2.0T free):

- checkpoints → `$STORAGE_DIR/checkpoints`
- trajectories / `output.jsonl` → `$STORAGE_DIR/benchmarks/evaluation_outputs/swe_bench_verified_full_outputs`
- serve/infer logs → `$STORAGE_DIR/benchmarks/run_logs/<model>/`
- `HF_HOME`, `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`,
  `XDG_CACHE_HOME`, `TMPDIR` → all under `$STORAGE_DIR`

The only thing written under the repo is `.llm_config/<model>.json` (a few
hundred bytes per model).

## Usage

```bash
cd /home/gaokaiz/benchmarks

# One-off serving check before committing to the sweep
bash benchmarks/swebench/scripts/verified_full/smoke_test.sh

# Run the whole queue (long — background it)
nohup bash benchmarks/swebench/scripts/verified_full/run_queue.sh \
  > /data/user_data/gaokaiz/benchmarks/run_logs/queue.log 2>&1 &

# Watch progress
bash benchmarks/swebench/scripts/verified_full/status.sh --watch
```

Set `PURGE_CHECKPOINTS=1` to delete each checkpoint once its model reaches
`DONE` (not needed at current disk headroom — 6 × 15G ≈ 90G of 2.0T).

## Status table

`$STORAGE_DIR/benchmarks/run_logs/status.tsv`, columns
`model / state / done / total / started / updated / note`. States:

- `PENDING` — queued, not started
- `SERVING` — downloading the checkpoint and/or bringing vLLM + ngrok up
- `INFERRING` — server verified, running the 500 instances
- `PARTIAL` — the run stopped early; rerun `run_queue.sh` to resume
- `DONE` — all 500 instances present in `output.jsonl`
- `FAILED` — the server never came up; see `run_logs/<model>/serve.log`

`status.sh` recounts `output.jsonl` on every render, so the `done` column is
live rather than only as fresh as the last 5-minute tick.

## Restartability

`run_queue.sh` can be killed and restarted at any time. `DONE` models are
skipped, and `swebench-infer` treats `output.jsonl` as the source of truth,
so a partially-finished model picks up only the instances it still owes.
A model is marked `DONE` only after a real chat completion succeeded through
the tunnel *and* all 500 instances landed — a server that never answers is
recorded as `FAILED` instead of silently producing an empty run.

## Concurrency ceiling: one model at a time

The ngrok free plan allows **3 concurrent endpoints but only one static
`*.ngrok-free.dev` domain per account**, and today we own a single account
(`gaokai_1`). Every endpoint on an account binds to that one domain, so 3
endpoints cannot serve 3 different models:

- 3 endpoints under one agent all share the domain and ngrok load-balances
  requests across them at random — a request for model A silently answers from
  model B.
- 2 separate agent processes claiming the domain break it outright:
  `ERR_NGROK_6030`, HTTP 400 on every request.

That is why `run_queue.sh` is strictly sequential. There are two ways to scale
past one concurrent model:

1. **A second ngrok account.** Each authtoken gets its own config file named
   `ngrok_gaokai_<N>.yml`, passed with `--config` at call time. `env.sh` selects
   one via `NGROK_ACCOUNT` (default `gaokai_1`) and derives the vLLM port
   (`2333 + N`) and the ngrok `web_addr` port (`4040 + N`) from the trailing
   number, so two accounts run side by side on one node without colliding.
   Adding `gaokai_2` is planned.
2. **A router behind one endpoint** that dispatches on the request's `model`
   field to one of several local vLLM ports.

What does *not* help is more endpoints on the same account, or more GPUs on
their own.

## Not included

Grading (`swebench-eval`) is deliberately out of this sweep: there is no
docker on this host, and the existing `test_infer.sh` grading path shells out
to `ogma` under another user's account. Run grading separately once the
`output.jsonl` files exist.
