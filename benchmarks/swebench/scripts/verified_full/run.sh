#!/usr/bin/env bash
# ============================================================================
#  SWE-bench_Verified sweep — single entry point and runbook.
#
#    bash run.sh              # this runbook (keys, ngrok, serving, storage)
#    bash run.sh status       # what is done / running / left  + node time left
#    bash run.sh go           # START THE SWEEP (the one command you need)
#    bash run.sh stop         # stop everything this user is running
#    bash run.sh serve <m>    # serve one model by hand (debug)
#    bash run.sh smoke        # serving smoke test
#    bash run.sh upload <m>   # (re)upload one model's predictions
#    bash run.sh sbatch       # print an sbatch template for a new allocation
# ============================================================================
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

runbook() {
cat <<'DOC'
============================================================================
 SWE-bench_Verified sweep — runbook
============================================================================

A. KEYS, NGROK, AND SERVING
---------------------------------------------------------------------------
 All three credentials live OUTSIDE the git repo (mode 600) so they can never
 be committed. env.sh wires them up automatically; you should not need to
 export anything by hand.

   ngrok authtoken   /home/gaokaiz/.config/ngrok/ngrok_gaokai_1.yml
                     (account gaokai_1 = GaokaiZhang's own; web_addr :4041)
   HF write token    /home/gaokaiz/.config/hf/token_gaokai
                     (user gaokaiz, write access to synthetic-code-training)
   runtime API key   /home/gaokaiz/.config/openhands/runtime_keys.env
                     (REMOTE_KEY / REMOTE_KEY2 -> RUNTIME_API_KEY)

 ONE CONFIG FILE PER NGROK KEY, named ngrok_gaokai_<N>.yml, selected at call
 time with --config. Today we own exactly one account, gaokai_1; more are
 planned, so use the numbering rather than adding un-numbered files:

     NGROK_ACCOUNT=gaokai_1  -> ngrok_gaokai_1.yml, serve 2334, web_addr :4041
     NGROK_ACCOUNT=gaokai_2  -> ngrok_gaokai_2.yml, serve 2335, web_addr :4042

 To add a key: write the new authtoken into ngrok_gaokai_2.yml (mode 600, same
 shape as gaokai_1) with agent.web_addr localhost:4042, then run with
 NGROK_ACCOUNT=gaokai_2. env.sh derives the config path and both port offsets
 from the trailing number, so two accounts can serve two models on one node
 without colliding. Nothing else needs editing.

 IMPORTANT — ngrok.yml in that directory belongs to a COLLABORATOR: never start
 a tunnel with it. ngrok1.yml is an older copy of gaokai_1. ngrok_gaokai.yml is
 a symlink to ngrok_gaokai_1.yml, kept so older scripts and run snapshots that
 hardcode the old name keep working.

 The runtime key previously existed ONLY in the launching shell's environment.
 It is now persisted in the file above, and env.sh sources that file whenever
 REMOTE_KEY is unset — which is what makes a brand-new Slurm job work.

 How the endpoint is built (serve.sh does all of this):
   1. download the checkpoint to $STORAGE_DIR/checkpoints/<model>
   2. vllm serve on port 2334, --served-model-name <model>, 32k ctx, bf16
   3. ngrok http 2334 --config <ngrok_gaokai_1.yml> -> https://<domain>/v1
   4. write .llm_config/<model>.json with that base_url
   5. verify a real chat completion through the tunnel, then write base_url.txt
 Only after step 5 does inference start, so a dead endpoint can never be
 mistaken for an empty run.

 ngrok CONCURRENCY LIMIT — the free plan gives 3 concurrent endpoints but only
 ONE static *.ngrok-free.dev domain PER ACCOUNT. Every endpoint on an account
 binds to that same domain, so one account serves exactly ONE model at a time
 over HTTPS. A second account (gaokai_2) is what buys a second concurrent
 model; extra endpoints on the same account do not:
   - 3 endpoints on one agent  -> ngrok load-balances at random, so a request
     for model A can be answered by model B (silent, no error)
   - 2 agent processes         -> ERR_NGROK_6030, HTTP 400 on everything
 This is why the sweep is strictly sequential today. More GPUs will NOT help
 on their own. Two ways to parallelise: a second ngrok account (one config file
 per key, as above), or a router behind the single endpoint that dispatches on
 the request's "model" field to several local vLLM ports.

 Toolchain (absolute paths, no module/conda activation needed):
   vllm   /home/gaokaiz/miniconda3/envs/eval/bin/vllm
   hf     /home/gaokaiz/miniconda3/envs/eval/bin/hf
   ngrok  /home/gaokaiz/.local/bin/ngrok
   infer  uv run swebench-infer   (from /home/gaokaiz/benchmarks)

 STORAGE — nothing large goes to /home (100G quota!). env.sh redirects
 checkpoints, trajectories, logs, HF_HOME, VLLM_CACHE_ROOT, TRITON/INDUCTOR
 caches and TMPDIR to $STORAGE_DIR=/data/user_data/gaokaiz (2.0T).

B. WHAT TO RUN / WHAT HAS RUN
---------------------------------------------------------------------------
 The queue is models.txt (6 models, in order). Run `bash run.sh status` for
 the live picture; it reads status.tsv and recounts output.jsonl, so it is
 always accurate rather than remembered.

 A model counts as finished at 500/500 in output.jsonl. Results are published
 to the Hub as they complete:

   synthetic-code-training/swebench-verified-results   (private dataset)
     <model>/predictions.jsonl    instance_id, model_name_or_path,
                                  model_patch, resolved=null
     <model>/metadata.json        instance count, complete flag, sdk sha

 resolved is ALWAYS null on upload. The eval machine (the one with docker)
 fills in true/false. Keep that field and its null default in any new tooling.

 GRADING IS NOT DONE HERE — this node has no docker. Pull predictions.jsonl
 on the eval machine, run the SWE-bench harness, write resolved back.

 TIMING, measured not guessed: model 1 took 5.83h for 500 instances at
 24 workers (astropy is fast, django is slow and is the biggest slice).
 Budget ~6h per model plus ~6min checkpoint download. The node is a
 PREEMPTIBLE 2-day Slurm job, so `run.sh go` starts a supervisor that:
   - reads the real EndTime from scontrol every 5 min
   - refuses to start a model with <2h left
   - partial-uploads every 30 min, so preemption never loses work
   - at the deadline, stops, partial-uploads, exits cleanly
 Whatever does not fit simply stays PENDING for the next allocation; rerun
 `run.sh go` there and it resumes, skipping finished models.

C. STARTING A FRESH ALLOCATION (tomorrow)
---------------------------------------------------------------------------
   bash run.sh sbatch > /tmp/sweep.sbatch && sbatch /tmp/sweep.sbatch
 or, inside an interactive job on a GPU node:
   cd /home/gaokaiz/benchmarks && bash benchmarks/swebench/scripts/verified_full/run.sh go

 Needs 1 GPU with >=40G (a 7B bf16 + 32k ctx; measured ~15G weights and
 KV cache never above 20% of an RTX PRO 6000). A6000 is plenty.
 Your quota: 24 GPUs under preempt_qos, 8 under fnsw_qos.
============================================================================
DOC
}

status() {
    source "$SRC/env.sh" >/dev/null 2>&1
    echo "=== node / allocation ==="
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        local e dl now
        e="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oE 'EndTime=[^ ]+' | head -1 | cut -d= -f2)"
        if [[ -n "$e" ]]; then
            dl="$(date -d "$e" +%s)"; now="$(date +%s)"
            printf "  job %s on %s\n  deadline %s\n  time left %s\n" \
              "$SLURM_JOB_ID" "$(hostname)" "$e" \
              "$(awk -v d="$dl" -v n="$now" 'BEGIN{h=(d-n)/3600; printf "%.2fh (~%d models at 6h)", h, int(h/6)}')"
        fi
    else
        echo "  not inside a Slurm job (SLURM_JOB_ID unset)"
    fi
    echo "=== processes ==="
    pgrep -u "$USER" -af "supervise.sh|run_queue.sh|vllm serve|ngrok http|swebench-infer" \
        | grep -v "bash -c\|pgrep -u" | sed 's/^/  /' || echo "  nothing running"
    echo
    bash "$SRC/status.sh"
    echo "=== supervisor log (last 6) ==="
    tail -6 "$RUN_LOG_ROOT/supervise.log" 2>/dev/null | sed 's/^/  /' || echo "  none"
}

sbatch_template() {
cat <<'DOC'
#!/bin/bash
#SBATCH --job-name=swebench-sweep
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user_data/gaokaiz/benchmarks/run_logs/sbatch-%j.out

# Everything else (keys, storage, caches) comes from env.sh.
cd /home/gaokaiz/benchmarks || exit 1
bash benchmarks/swebench/scripts/verified_full/run.sh go

# Keep the allocation alive while the detached supervisor works.
while pgrep -u "$USER" -f "verified_full/supervise.sh" >/dev/null; do sleep 300; done
DOC
}

case "${1:-help}" in
    help|-h|--help|"") runbook ;;
    status)            status ;;
    go)
        if pgrep -u "$USER" -f "verified_full/supervise.sh" >/dev/null 2>&1; then
            echo "supervisor already running; use 'run.sh status'"; exit 1
        fi
        source "$SRC/env.sh" >/dev/null 2>&1
        cd /home/gaokaiz/benchmarks || exit 1
        setsid nohup bash "$SRC/supervise.sh" "${2:-$SRC/models.txt}" \
            > "$RUN_LOG_ROOT/supervise_stdout.log" 2>&1 < /dev/null &
        sleep 8
        echo "supervisor started; log: $RUN_LOG_ROOT/supervise.log"
        tail -12 "$RUN_LOG_ROOT/supervise.log" 2>/dev/null
        ;;
    stop)
        for p in supervise.sh run_queue.sh swebench-infer "vllm serve" "ngrok http"; do
            pkill -u "$USER" -f "$p" 2>/dev/null && echo "  stopped: $p"
        done
        sleep 3; echo "remaining:"; pgrep -u "$USER" -af "supervise.sh|run_queue.sh|vllm serve" \
            | grep -v "bash -c\|pgrep" || echo "  none"
        ;;
    serve)  bash "$SRC/serve.sh" "${2:?usage: run.sh serve <hf_repo_or_model>}" ;;
    smoke)  bash "$SRC/smoke_test.sh" "${2:-}" ;;
    upload) shift; (cd /home/gaokaiz/benchmarks && python3 "$SRC/upload_hf.py" "$@") ;;
    sbatch) sbatch_template ;;
    *) echo "unknown command: $1"; echo; runbook; exit 1 ;;
esac
