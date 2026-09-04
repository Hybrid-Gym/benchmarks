# Handoff: grade SWE-bench_Verified predictions (eval machine, has docker)

You are the **evaluation** side of a two-machine split. Another machine (a
GPU Slurm node with no docker) runs inference and publishes predictions to the
Hugging Face Hub; you have docker and run the SWE-bench harness on them.

## Where the predictions are

Private HF dataset repo:

    synthetic-code-training/swebench-verified-results

Layout, one directory per model save name:

    <model>/predictions.jsonl    one JSON object per line
    <model>/metadata.json        instance count + completeness

You need an HF token with read access to the `synthetic-code-training` org
(ask the user for it; do not commit it anywhere).

    export HF_TOKEN=<token>
    hf download synthetic-code-training/swebench-verified-results \
        --repo-type dataset --local-dir ./swebench-results

## Record format and the `resolved` contract

    {"instance_id": "...", "model_name_or_path": "<model>",
     "model_patch": "<unified diff or empty string>", "resolved": null}

`resolved` is **always `null` on upload** — it means "not yet graded". Your job
is to fill it with `true`/`false` per instance and push the file back to the
same path. Keep the field and never drop it; empty `model_patch` is a legitimate
value (the model produced no edit) and grades as `false`.

Check `metadata.json` before grading:

    "instances": 500, "expected_total": 500, "complete": true

`complete: false` means the upload is a **partial snapshot of a run still in
progress** (the inference side partial-uploads every 30 min so preemption never
loses work). Grade only `complete: true` models unless the user asks otherwise —
a partial file will be overwritten with more rows later.

## What to grade

Dataset: `princeton-nlp/SWE-bench_Verified`, split `test`, 500 instances.
Standard harness:

    python -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path <model>/predictions.jsonl \
        --max_workers 8 \
        --run_id <model>

Then merge the harness report's resolved instance ids back into
`predictions.jsonl` (`resolved: true` if the id is in `resolved_ids`, else
`false`), and upload the updated file to the same `<model>/predictions.jsonl`
path. Also write a short `<model>/eval_report.json` next to it with the
resolve rate and the harness run id, so the inference side can read results
without re-deriving them.

## The full model list (queue order, all 6)

    1. qwen25-coder-7b-func-localize-claude45-1457i-multi-round-false250
    2. qwen25-coder-7b-func-localize-claude45-1457i-multi-round-true250
    3. qwen25-coder-7b-func-localize-claude45-1457i-read-narrow-false250
    4. qwen25-coder-7b-func-localize-claude47-1467i-read-narrow-false338
    5. qwen25-coder-7b-func-localize-claude47-1467i-multi-round-true338
    6. qwen25-coder-7b-func-localize-claude47-1467i-multi-round-false338

All are Qwen2.5-Coder-7B fine-tunes, served at 32k context, run with the
OpenHands SDK at `max_iterations=60`, 24 workers, remote workspaces.

## Status right now (2026-09-03 22:05)

    qwen25-coder-7b-func-localize-claude45-1457i-multi-round-false250
        COMPLETE 500/500, complete=true, 158 non-empty patches
        -> GRADE THIS ONE NOW
    qwen25-coder-7b-func-localize-claude45-1457i-multi-round-true250
        486/500, complete=false. The run finished its retry attempts (exit 0);
        14 instances never completed because the remote runtime kept ending
        their conversations with an error. They will be resumed in the next
        allocation, after which this becomes complete=true.
        -> DO NOT GRADE YET unless the user says to grade at 486
    qwen25-coder-7b-func-localize-claude45-1457i-read-narrow-false250
        IN PROGRESS 163/500, ETA ~02:50 tonight
        -> partial on the Hub, do not grade
    qwen25-coder-7b-func-localize-claude47-1467i-read-narrow-false338
        not started; expected to fit in the current allocation
    qwen25-coder-7b-func-localize-claude47-1467i-multi-round-true338
        not started; needs a new allocation
    qwen25-coder-7b-func-localize-claude47-1467i-multi-round-false338
        not started; needs a new allocation

The inference node's Slurm allocation ends 2026-09-04 11:29.

## What to do

1. Grade model 1 now and write `resolved` back.
2. **Do not poll or grade the rest on your own.** The user will tell you when a
   model is ready. When they do, re-download the repo, confirm
   `metadata.json.complete == true`, and grade that model the same way.
3. Report per model: resolve rate (resolved/500), how many predictions were
   empty patches, and anything that failed to build or apply.

## Two things worth knowing about the predictions

- Empty patches are common, and the rate differs by model. Measured on
  completed output:

      qwen25-coder-7b-func-localize-claude45-1457i-multi-round-false250
      (complete, 500/500):
          non-empty patch          158/500  31.6%
          touches real source      102/500  20.4%
          pyproject.toml only       56
      qwen25-coder-7b-func-localize-claude45-1457i-multi-round-true250
      (in progress, 368 so far):
          non-empty patch          147/368  39.9%
          touches real source      118/368  32.1%
          pyproject.toml only       29

  So a large minority of instances finish without producing a source edit.
  That is genuine model behaviour, not a harness bug — patch extraction was
  verified working on the inference side. Expect resolve rates bounded well
  below the non-empty rate.
- The gap between "non-empty" and "touches real source" above is a
  `pyproject.toml` setuptools pin diff that SWE-bench's own environment setup
  introduces. It is a **SWE-bench environment artifact, not model output**, and
  it alone cannot resolve an instance. It was deliberately left unfiltered so
  results are not altered — be aware of it if you inspect patches by hand or
  count "attempted" instances.
