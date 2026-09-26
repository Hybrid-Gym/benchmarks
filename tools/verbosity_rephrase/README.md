# verbosity_rephrase

Builds the **verbosity ablation** datasets: the same trajectories with the agent's prose (the
natural-language part of each assistant turn, outside the tool call) removed or rewritten to a
controlled length. Two families:

| family | variants | think / task_tracker turns | prose per turn |
|---|---|---|---|
| A `fixed` | `_text0`, `_text20`, `_text50`, `_text100`, `_text300` | removed, with their result turns | none, or ~20 / 50 / 100 / 300 tokens for every turn |
| B `scaled` | `_text0x`, `_text0.5x`, `_text2x`, `_text4x`, `_text8x`, `_text32x` | kept verbatim | none, or ½ / 2 / 4 / 8 / 32 × the turn's **own** prose length |

Sources: `synthetic-code-training/func_localize_claude45_1457i` and
`synthetic-code-training/r2egym_qwen3next80b_1500i`; outputs are `<base>_<variant>` in the same
org with the sibling-variant schema (`instance_id`, `resolved`, `messages`).

## Design

Family A (Slack thread with Yiqing, 2026-09-20):

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

Family B (2026-09-22): same machinery, two changes, plus `text0x` (2026-09-23, Yiqing: "remove all text but
keep the thinking/planning steps, unlike text0"): every non-think turn is its tool call only, think /
task_tracker turns kept verbatim, no LLM involved.

1. The thinking/planning steps stay: `think` / `task_tracker` turns and their results are kept
   byte-for-byte (623 claude45 think turns carry a one-line lead-in before the call; it stays
   with the step). They are never rephrased.
2. Every other turn's targets are multiples of its own prose length in Qwen3 tokens: ½×, 2×,
   4× and 8×, written in one request as a ladder (½× shortens the original; 2× elaborates it;
   4× elaborates the 2×; 8× elaborates the 4×), same ±25 % band, ≤3 rounds with the original
   as the retry source, original prose on a miss. Condensing from the longest version, as
   family A does, collapsed on long turns (2× of a 300-token turn came back at 140–450 tokens).
   The word ask is inflated for targets above 300 tokens (`LONG_ASK`: ×1.6 at 600, ×2.0 at
   1 000, ×2.6 at 2 000): the first 2 000 turns of the run showed the model writing a median
   97 % of a 150–300-token ask but 68 % of 300–600, 44 % of 600–1 000 and 35 % beyond, and
   with the inflation 13 of 18 probe versions above 600 tokens landed in band instead of 6. A multiple is **not
   requested** when its target is under 4 tokens (halving a 6-token turn) or over 2 000 tokens:
   the model stops well short of anything longer however it is asked (a 3 600-token target got
   700–1 400 tokens across three rounds, whether asked alone, with feedback, or by doubling a
   2× version), so such turns keep their original prose and are counted on the card. Empty
   turns stay empty in every variant (0 × n = 0); in claude45 that is 60 % of turns.
3. `_text32x` (2026-09-24) is a fifth rung added to the finished ladder rather than a re-run:
   `rephrase.py --keys x32 --prior <the scaled jsonl>` writes `<base>.scaled.x32.jsonl`, each
   record the prior one plus the new version. A 32× target exceeds one reply for any turn over
   62 tokens, so the version is **grown in parts**: each request sees the turn, the accepted 8×
   version as SOURCE (first part only; 4× / 2× / the prose if 8× fell back), the text written
   so far, and asks for the next chunk (an even share of what is left, at most 1 500 tokens);
   the parts are concatenated, the text stops once it reaches 90 % of the target, an overshoot
   is trimmed at a sentence boundary, an undershoot gets another part, and after
   ⌈target / 1 500⌉ + 3 requests the turn keeps its original prose. The cap is 8 000 tokens
   (base prose ≤ 250 tokens): the same turns 8× reaches, so `_text8x` and `_text32x` leave the
   same turns untouched. Three things the smoke tests (15 turns, 2–240 tokens) changed: a part
   has its own word ask (`PARTS_WORD_CALIBRATION` 0.9 words per 0.75 token, no `LONG_ASK`),
   because a part written on its own comes out at 1.3–2× its chunk with the ladder's inflated
   ask and 1.5× with the bare one; a reply cut off at the request's token limit keeps its
   complete sentences instead of being discarded (the parser needs a closing brace), which had
   wasted a third of the calls; and a rejected reply is re-asked with the reason (quoted
   tool-call markup, no JSON, a restart instead of a continuation).

## Files

| file | role |
|---|---|
| `trajectory.py` | split a turn into prose / call / tool name; positional call↔result pairing; skeleton (think turns dropped or kept verbatim); reassembly |
| `tokens.py` | token counting with `Qwen/Qwen3-8B` (same vocabulary as Qwen2.5-Coder) |
| `llm.py` | the two length specs, the prompts (first round + retry), reply parsing, gateway client with backoff |
| `rephrase.py` | the LLM driver: one work unit per non-think assistant turn, per-request `max_tokens`, resumable jsonl output; `--keys x32 --prior <jsonl>` adds the parts-written rung to a finished run |
| `build_variants.py` | assemble a family's variants, validate, save to disk, push with a dataset card |
| `run_pipeline.sh` | tmux runner: rephrase → build → push per dataset, `FAMILY=fixed|scaled`, `KEYS=x32` for the added rung, re-entrant |

## Data facts that shaped the code

- Turn format: prose, blank line, one `<function=NAME>…</function>` block; never more than one
  call per turn and never prose after the block. Parallel calls are consecutive assistant turns
  followed by the same number of `EXECUTION RESULT` user turns, paired positionally, which is
  what lets a `think` turn take its "Your thought has been logged." result with it.
- `func_localize_claude45_1457i`: 30 615 assistant turns, 3 738 `think` + 683 `task_tracker`,
  26 194 others, of which 15 681 have **no prose at all**. Hence family A's "explain the tool
  call when TEXT is empty" rule; otherwise the 20-token variant would average ~8 tokens per
  turn. `r2egym_qwen3next80b_1500i`: 30 501 turns, 8 `think`, 30 493 others, 85 without prose;
  prose is long-tailed (p90 ≈ 230, p99 ≈ 545, max 5 070 tokens), which is what the family-B
  cap is about: 8× is out of reach for 2 660 r2egym turns (8.7 %) and 73 claude45 turns (and
  so is 32×, whose 8 000-token cap falls at the same 250-token prose length). Median prose is
  22–23 tokens in both sets, so most 32× versions fit one part; 25 % of r2egym's prose turns
  and 13 % of claude45's need two or more.
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
  t20 10/24 → 22/24). Family B derives the same kind of structure from the target
  (`shape_of` / `struct_of` in `llm.py`);
- for turns with no prose, an instruction to walk through the call's file / range / pattern /
  command and the expected output;
- retries at temperature 0.7 that quote the attempt's own word and token counts and ask for
  two candidates at different word counts (1.5× for undershoots; the midpoint for overshoots,
  which otherwise drop dense identifiers and land under the band);
- over-band candidates trimmed at a sentence boundary as they arrive;
- lenient JSON extraction (unescaped quotes inside values broke a third of retries).

The family-A datasets were produced by the tool at commit `c42f8a9`; the prompt has since
been parameterized for family B and differs, for family A, by one space in the example JSON.

## Results

Family A (2026-09-21, 6 workers, free):

| dataset | kept turns | turns that kept original prose | mean tokens t20 / t50 / t100 / t300 | calls/turn | LLM tokens in / out | wall time |
|---|---|---|---|---|---|---|
| claude45 | 26 194 (4 421 dropped) | 220 (0.84 %): t20 178, t50 8, t300 36 | 19 / 50 / 100 / 296 | 1.35 | 31.8M / 17.8M | 8.8 h |
| r2egym | 30 493 (8 dropped) | 437 (1.43 %): t20 403, t50 42, t100 14, t300 18 | 19 / 50 / 103 / 305 | 1.35 | 41.5M / 19.4M | 10.9 h |

About 70 % of turns were accepted after one round, 25 % after two, 5 % after three; 0 API
errors. The 36 claude45 t300 fallbacks are turns that had no original prose, so they are
call-only in `_text300`. All ten datasets were re-validated fresh from HF: row counts, the
three-column schema, every tool call and every non-assistant turn byte-identical to the base,
no `think` / `task_tracker` left, `text0` prose-free (except r2egym's five pure-text turns).

Family B (2026-09-22/23, 6 workers, free; "rephrased" = turns with a requested multiple):

| dataset | turns / rephrased | not requested (empty, over cap) | fallback ½× / 2× / 4× / 8× | mean ratio ½× / 2× / 4× / 8× | calls/rephrased turn | LLM tokens in / out | wall time |
|---|---|---|---|---|---|---|---|
| claude45 | 26 194 / 10 513 | 15 681 empty; 8× over cap 73, 4× 17, 2× 3 | 0.8 % / 1.2 % / 2.8 % / 3.4 % | 0.52 / 1.93 / 3.92 / 8.14 | 1.85 | 20.9M / 10.1M | 5.3 h |
| r2egym | 30 493 / 30 400 | 85 empty; 8× over cap 2 652, 4× 369, 2× 89; ½× under floor 250 | 0.6 % / 2.5 % / 5.0 % / 4.1 % | 0.52 / 1.91 / 3.87 / 8.05 | 1.98 | 72.0M / 46.1M | 19.1 h |

`_text0x` (no LLM): claude45 26 194 call-only turns + 4 421 think/task_tracker turns kept; r2egym 30 488
call-only turns + 5 pure-text turns keeping their text + 8 think turns kept. Both pushed and validated
2026-09-23.

`_text32x` (2026-09-24/25, written in parts from the 8× rung; "rephrased" = turns with a 32× target,
i.e. base prose 1–250 tokens):

| dataset | turns / rephrased | not requested (empty, over cap) | fallback | ratio mean / median | calls/rephrased turn | LLM tokens in / out | wall time |
|---|---|---|---|---|---|---|---|
| claude45 | 26 194 / 10 440 | 15 681 empty; 73 over cap | 18 (0.2 %) | 33.9 / 33.0 | 1.74 | 25.8M / 15.9M | 7.7 h |
| r2egym | 30 493 / 27 748 | 85 empty; 2 660 over cap | 89 (0.3 %) | 34.0 / 33.1 | 1.89 | 84.7M / 50.6M | 23.8 h |

claude45 fallback by base length: 0 % (1–30 tokens), 0.1 % (31–62), 1.0 % (63–125), 1.7 % (126–250);
parts per turn 1.4 / 1.6 / 2.8 / 3.9 for the same bins; 96 % of the versions grew from the accepted 8×
version, the rest from 4× / 2× / the prose where 8× had fallen back. 2.8 % of replies were unusable and
re-asked (almost all "restarted instead of continuing"). The run is generation-latency bound rather than
token bound: 6 workers reached only 32K tokens/min, 12 reached the ~100K ceiling, but 12 also triggered a
429 storm once another job on the box shared the gateway's per-IP limit; 8 workers (65K tokens/min) ran
clean for the rest.

r2egym fallback by base length: 0 % (1–30 tokens), 0.2 % (31–62), 0.9 % (63–125), 2.2 % (126–250);
parts per turn 1.4 / 1.7 / 2.7 / 4.0 for the same bins; 96 % of the versions grew from the accepted 8×
version (576 from 2×, 419 from 4×, 103 from the prose). 4.2 % of replies were unusable and re-asked
(2 212 of 2 216 "restarted instead of continuing"). The run shared the box's gateway bucket with another
tenant's deepseek-flash job for most of its 23.8 h: any fixed worker count oscillated between a
latency-bound regime (0 429s, 26–40K tokens/min) and 429 storms that persisted until a full stop, so the
second half ran under the adaptive in-flight limiter (`--workers 16 --workers-min 4 --workers-start 8`:
cap ×¾ plus a 20 s cooldown on every 429, +1 after 30 clean calls). The cap cycled 4 → 16 → 4 every few
hundred turns, retries never went deeper than 4/8, throughput held at 16–26 turns/min and the run had
0 errors. Both x32 runs were validated fresh from HF (think turns byte-identical, every rephrased turn in
band, 0 structural mismatches). Note that the gateway now routes `nvidia/deepseek-ai/deepseek-v4-flash`
to a paid deployment first ($0.13 / $0.26 per M tokens in / out), so the two x32 runs cost at most about
$8 (claude45) and $24 (r2egym) if every call hit it.

The 8× fallback is a function of the original length (claude45 / r2egym): 0 % / 0 % for turns of
≤30 tokens, 3 % / 2 % for 31–80, 21 % / 12 % for 81–150 and 44 % / 37 % for 151–250 (targets of
1 200–2 000 tokens); 4× behaves the same one octave up (r2egym: 1 % below 80 tokens, 19 % for
81–250, 26 % for 251–500). Such turns keep their original prose. Turns accepted after one / two /
three rounds: 30 / 55 / 15 % (claude45), 22 / 58 / 20 % (r2egym); 0 API errors in either run. The
r2egym run sat at the gateway's ~96K tokens/min the whole time.

## Usage

```bash
# everything, in tmux (re-entrant; resumes the jsonl outputs)
tmux new -d -s verbosity 'FAMILY=scaled bash tools/verbosity_rephrase/run_pipeline.sh'
# the 32x rung on top of a finished scaled run (reads <base>.scaled.jsonl, pushes _text32x only)
tmux new -d -s verbosity-x32 'FAMILY=scaled KEYS=x32 bash tools/verbosity_rephrase/run_pipeline.sh'

# pieces
.venv/bin/python tools/verbosity_rephrase/rephrase.py --family scaled --hf synthetic-code-training/func_localize_claude45_1457i \
    --out-dir eval_outputs/verbosity_rephrase --model nvidia/deepseek-ai/deepseek-v4-flash --workers 6
.venv/bin/python tools/verbosity_rephrase/build_variants.py --family scaled --hf synthetic-code-training/func_localize_claude45_1457i \
    --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.jsonl \
    --out-dir eval_outputs/verbosity_rephrase/variants --push
.venv/bin/python tools/verbosity_rephrase/rephrase.py --family scaled --keys x32 \
    --prior eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.jsonl \
    --hf synthetic-code-training/func_localize_claude45_1457i --out-dir eval_outputs/verbosity_rephrase --workers 6
.venv/bin/python tools/verbosity_rephrase/build_variants.py --family scaled --variants text32x \
    --hf synthetic-code-training/func_localize_claude45_1457i \
    --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.x32.jsonl \
    --out-dir eval_outputs/verbosity_rephrase/variants --push
```

The API key is read from `LLM_API_KEY` or, failing that, `config.toml`
(`[llm.nvidia_claude_opus47].api_key`); it is never printed. The gateway caps each model at
100 RPM / 100K TPM and throttles the whole box by source IP, and the endpoint's speed swings
with other tenants' load (the 32× run saw 26K–109K tokens/min at a fixed worker count within
one day: idle when slow, 429 storms when fast). `--workers-min N` turns on an adaptive
in-flight cap between N and `--workers` (start `--workers-start`): on a 429 the cap drops to
3/4 and every request waits out a 20 s cooldown, after 30 clean calls it grows by one
(`WORKERS=16 WORKERS_MIN=4 WORKERS_START=8` in the pipeline). Without it, 6–8 workers is the
safe fixed setting; the family-A run sat at ~85K tokens/min and absorbed its 429 bursts through
the client's backoff alone.
