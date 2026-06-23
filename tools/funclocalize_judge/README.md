# funclocalize_judge

LLM-as-judge for checking whether a funclocalize agent's localization phase followed the
three strategies in `benchmarks/hybridgym_funclocalize/prompts/default_loc_strategy.j2`:

1. **broad_then_narrow** — broad search across the repo first, then narrow
2. **multi_round_refinement** — iterative search across multiple rounds
3. **read_after_narrowing** — read full files only after narrowing to a few candidates

For each trajectory, the tool condenses the pre-edit action sequence to one line per
action/observation, sends it to an LLM judge, and records a boolean per strategy. Use
it to compare compliance rates between two prompts on the same instances.

## Usage

```bash
export LLM_API_KEY=sk-...   # same gateway key the rollouts use

# Head-to-head (the common case)
python tools/funclocalize_judge/judge.py \
    --src eval_outputs/.../baseline/output.jsonl \
    --src eval_outputs/.../experiment/output.jsonl \
    --out-dir /tmp/verdicts \
    --filter-ids /tmp/funclocalize_1500_sample300_seed42_ids.txt
```

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
| `--src PATH` | required | rollout `output.jsonl`; repeat for head-to-head |
| `--out PATH` | — | verdicts file (single `--src`) |
| `--out-dir DIR` | — | verdicts directory (multiple `--src`); filenames derived from each src's parent dir |
| `--filter-ids PATH` | none | newline-separated instance IDs to restrict judging |
| `--model` | `openai/openai/gpt-5-mini` | judge model |
| `--api-key` | `$LLM_API_KEY` | gateway key |
| `--base-url` | `$LLM_BASE_URL` or NVIDIA gateway | gateway base URL |
| `--workers` | 8 | concurrent judge calls |
| `--limit N` | 0 (all) | per-src trajectory cap (debug) |

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
