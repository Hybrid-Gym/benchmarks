# funclocalize_judge

LLM-as-judge for checking whether a funclocalize agent's localization phase followed the
three strategies in `benchmarks/hybridgym_funclocalize/prompts/default_loc_strategy.j2`:

1. **broad_then_narrow** — broad search across the repo first, then narrow
2. **multi_round_refinement** — iterative search across multiple rounds
3. **read_after_narrowing** — read full files only after narrowing to a few candidates

For each trajectory, the tool condenses the pre-edit action sequence to one line per
action/observation, sends it to an LLM judge, and records a boolean per strategy. Use
it to compare compliance rates between two prompts on the same instances, or to attach
strategy labels to an already-published dataset for data selection.

Two input shapes are supported, and they carry trajectories differently:

| source | flag | trajectory field |
|---|---|---|
| rollout output dir | `--src output.jsonl` | SDK `history` events |
| pushed HF dataset | `--hf org/dataset` | non-fncall `messages` |

Both are condensed to the same one-line-per-action summary before judging, so verdicts
are comparable across sources.

## Usage

```bash
export LLM_API_KEY=sk-...   # same gateway key the rollouts use

# Head-to-head between two prompts (the original use case)
python tools/funclocalize_judge/judge.py \
    --src eval_outputs/.../baseline/output.jsonl \
    --src eval_outputs/.../experiment/output.jsonl \
    --out-dir /tmp/verdicts \
    --filter-ids /tmp/funclocalize_1500_sample300_seed42_ids.txt

# Label published datasets for data selection
python tools/funclocalize_judge/judge.py \
    --hf synthetic-code-training/func_localize_claude45_1457i \
    --hf synthetic-code-training/func_localize_claude47_1467i \
    --hf synthetic-code-training/func_localize_kimi_k25_1431i \
    --out-dir eval_outputs/funclocalize_judge \
    --model nvidia/deepseek-ai/deepseek-v4-flash \
    --workers 3
```

Verdicts are appended as each call returns and already-judged ids are skipped on
restart, so an interrupted multi-thousand-row run resumes instead of re-paying for the
same calls. Pass `--no-resume` to force a clean re-judge. Judge calls retry with
jittered backoff: the gateway sits behind an AWS WAF per-IP limiter that 429s in
bursts, and without retries those transient failures become permanent `error` rows.

Keep `--workers` low (3-4) while rollouts are running — the WAF limit is shared across
every job on the box, so it throttles on *total* concurrency, not per model or key.

The head-to-head form prints a comparison table:

```
=== comparison ===
  strategy                          baseline    experiment    Δ(2nd-1st)
  broad_then_narrow                    97.7%        99.3%       +1.7pp
  multi_round_refinement               93.3%        97.7%       +4.3pp
  read_after_narrowing                 65.7%        86.0%      +20.3pp
```

Single rollout: pass one `--src` plus `--out PATH` instead of `--out-dir`.

## Flags

| flag | default | purpose |
|---|---|---|
| `--src PATH` | — | rollout `output.jsonl`; repeatable |
| `--hf REPO` | — | HF dataset of pushed trajectories; repeatable |
| `--hf-split` | `train` | split for `--hf` sources |
| `--out PATH` | — | verdicts file (single source) |
| `--out-dir DIR` | — | verdicts directory (multiple sources); filenames derived from each source's label |
| `--filter-ids PATH` | none | newline-separated instance IDs to restrict judging |
| `--model` | `openai/openai/gpt-5-mini` | judge model |
| `--api-key` | `$LLM_API_KEY` | gateway key |
| `--base-url` | `$LLM_BASE_URL` or NVIDIA gateway | gateway base URL |
| `--workers` | 8 | concurrent judge calls |
| `--limit N` | 0 (all) | per-source trajectory cap (debug) |
| `--no-resume` | off | re-judge everything instead of skipping ids already in the output |

At least one `--src` or `--hf` is required.

## Verdict schema

```json
{
  "instance_id": "...",
  "broad_then_narrow": true,
  "multi_round_refinement": false,
  "read_after_narrowing": true,
  "notes": "Two broad rg searches then targeted view; no refinement after first hit."
}
```

Errored rows record `error` (and `raw` for parse failures) in place of the booleans.

A retry appends a fresh verdict rather than rewriting the old row, so a verdicts file
can hold several rows per instance. Readers must take the **last** row per
`instance_id` — counting lines over-reports, and counting `error` rows anywhere in the
file reports failures that were later repaired.

## Publishing labels

`push_labels.py` joins a verdicts file back onto the dataset it was judged from and
pushes a new revision with `broad_then_narrow`, `multi_round_refinement`,
`read_after_narrowing` and `judge_notes` added. Existing columns are untouched, so
loaders that only read `messages`/`resolved` keep working.

```bash
python tools/funclocalize_judge/push_labels.py \
    --repo synthetic-code-training/func_localize_claude45_1457i \
    --repo synthetic-code-training/func_localize_claude47_1467i \
    --repo synthetic-code-training/func_localize_kimi_k25_1431i \
    --dry-run          # drop to push; --dest-suffix _labeled writes a copy instead
```

It aborts unless every dataset row has a verdict: a partially labelled dataset reads as
"these ones are False" and would quietly bias any selection built on it.
