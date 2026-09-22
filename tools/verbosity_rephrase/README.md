# verbosity_rephrase

Builds the **verbosity ablation** datasets: the same trajectories with the agent's prose (the
natural-language part of each assistant turn, outside the tool call) removed or rewritten to a
controlled length, so training can be compared across 0 / ~20 / ~50 / ~100 / ~300 tokens of
prose per turn.

Design (Slack thread with Yiqing, 2026-09-20):

1. Strip every assistant turn down to its tool call and remove `think` / `task_tracker` turns
   (with their result turns) entirely, so trajectories contain nothing but real tool calls.
   That is the `text0` variant.
2. Have a fast free gateway LLM rewrite the prose of every turn at ~20 / 50 / 100 / 300 tokens.
   The rephraser sees only the current turn (its prose + its tool call): it rephrases the prose
   or, when there is none, explains the tool call. All four lengths come from one prompt, the
   300-token version first and then condensed, so the four variants carry the same meaning.
   Token counts may float (±25 %: 20 → 15–25 … 300 → 225–375, measured with the Qwen3
   tokenizer the students use); a version outside its band is re-requested with the measured
   count as feedback, at most 3 rounds, and if it still misses the turn keeps its original prose.

Sources: `synthetic-code-training/func_localize_claude45_1457i` and
`synthetic-code-training/r2egym_qwen3next80b_1500i`. Outputs: `<base>_text0`, `_text20`,
`_text50`, `_text100`, `_text300` in the same org, with the sibling-variant schema
(`instance_id`, `resolved`, `messages`).

## Files

| file | role |
|---|---|
| `trajectory.py` | split a turn into prose / call / tool name; positional call↔result pairing; skeleton (think/task_tracker removed); reassembly |
| `tokens.py` | token counting with `Qwen/Qwen3-8B` (same vocabulary as Qwen2.5-Coder) |
| `llm.py` | the prompts (first round + retry), reply parsing, gateway client with backoff |
| `rephrase.py` | the LLM driver: one work unit per kept assistant turn, resumable jsonl output |
| `build_variants.py` | assemble the five variants, validate, save to disk, push with a dataset card |
| `run_pipeline.sh` | tmux runner: rephrase → build → push per dataset, re-entrant |

## Data facts that shaped the code

- Turn format: prose, blank line, one `<function=NAME>…</function>` block; never more than one
  call per turn and never prose after the block. Parallel calls are consecutive assistant turns
  followed by the same number of `EXECUTION RESULT` user turns, paired positionally, which is
  what lets a `think` turn take its "Your thought has been logged." result with it.
- `func_localize_claude45_1457i`: 30 615 assistant turns, 3 738 `think` + 683 `task_tracker`
  dropped, 26 194 kept, of which 15 681 have **no prose at all**. Hence the "explain the tool
  call when TEXT is empty" rule; otherwise the 20-token variant would average ~8 tokens per
  turn. `r2egym_qwen3next80b_1500i`: 30 501 turns, 8 `think`, 30 493 kept, 85 without prose;
  prose is long-tailed (p99 ≈ 545 tokens).
- Malformed calls (4 claude `<invoke>` finishes; 304 garbled qwen `<tool_call>` JSON, each
  answered by a "Please continue working…" nudge) are kept verbatim as the call part and their
  prose is treated like any other. Five r2egym turns are pure prose with no call; they keep
  their text in `text0` (an empty assistant turn is untrainable) and are rephrased elsewhere.

## Model choice (probe of 64 stratified real turns, 2026-09-21)

| candidate | in-band versions /256 | turns with a fallback | calls/turn | s/turn | out tok/call |
|---|---|---|---|---|---|
| deepseek-v4-flash, thinking on | 224 | 13 | 2.0 | 13.3 | 2172 (reasoning burns the budget; 3 turns never answered) |
| deepseek-v4-flash, `thinking:false`, first prompt | 206 | 20 | 2.4 | 4.2 | 332 |
| gpt-oss-120b | 209 | 21 | 2.1 | 12.5 | 1173 |
| qwen3-next-80b-instruct / qwen3.6-35b | undershoot 300 tokens by half | | | | |
| **deepseek-v4-flash, `thinking:false`, final prompt** | **251** | **5** | 2.05 | 5.5 | 565 |

What the final prompt adds over the first one:

- a structure with a floor per length instead of a bare word count ("three paragraphs of
  about 110 words each, never fewer than 280 words" / "one paragraph of five or six
  sentences" / "three sentences" / "one sentence of about 22 words"). A bare word target,
  even inflated 35 %, left every length 15–20 % short; structure + floor doubled round-1 hits
  on the hardest turns (claude45 turns with no prose: t300 8/24 → 17/24, t50 11/24 → 23/24,
  t20 10/24 → 22/24);
- for turns with no prose, an instruction to walk through the call's file / range / pattern /
  command and the expected output;
- retries at temperature 0.7 that quote the attempt's own word and token counts and ask for
  two candidates at different word counts (1.5× for undershoots; the midpoint for overshoots,
  which otherwise drop dense identifiers and land under the band);
- over-band candidates trimmed at a sentence boundary as they arrive;
- lenient JSON extraction (unescaped quotes inside values broke a third of retries).

## Results (2026-09-21, 6 workers, free)

| dataset | kept turns | turns that kept original prose | mean tokens t20 / t50 / t100 / t300 | calls/turn | LLM tokens in / out | wall time |
|---|---|---|---|---|---|---|
| claude45 | 26 194 (4 421 dropped) | 220 (0.84 %): t20 178, t50 8, t300 36 | 19 / 50 / 100 / 296 | 1.35 | 31.8M / 17.8M | 8.8 h |
| r2egym | 30 493 (8 dropped) | 437 (1.43 %): t20 403, t50 42, t100 14, t300 18 | 19 / 50 / 103 / 305 | 1.35 | 41.5M / 19.4M | 10.9 h |

About 70 % of turns were accepted after one round, 25 % after two, 5 % after three; 0 API
errors. The 36 claude45 t300 fallbacks are turns that had no original prose, so they are
call-only in `_text300`. All ten datasets were re-validated fresh from HF: row counts, the
three-column schema, every tool call and every non-assistant turn byte-identical to the base,
no `think` / `task_tracker` left, `text0` prose-free (except r2egym's five pure-text turns).

## Usage

```bash
# everything, in tmux (re-entrant; resumes the jsonl outputs)
tmux new -d -s verbosity 'bash tools/verbosity_rephrase/run_pipeline.sh'

# pieces
.venv/bin/python tools/verbosity_rephrase/rephrase.py --hf synthetic-code-training/func_localize_claude45_1457i \
    --out-dir eval_outputs/verbosity_rephrase --model nvidia/deepseek-ai/deepseek-v4-flash --workers 6
.venv/bin/python tools/verbosity_rephrase/build_variants.py --hf synthetic-code-training/func_localize_claude45_1457i \
    --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.rephrase.jsonl \
    --out-dir eval_outputs/verbosity_rephrase/variants --push
```

The API key is read from `LLM_API_KEY` or, failing that, `config.toml`
(`[llm.nvidia_claude_opus47].api_key`); it is never printed. Keep total workers at ~6–8: the
gateway caps each model at 100 RPM / 100K TPM and throttles the whole box by source IP; the
run above sat at ~85K tokens/min and absorbed the resulting 429 bursts through the client's
backoff.
