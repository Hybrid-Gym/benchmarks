# verbosity_rephrase

Builds the **verbosity ablation** datasets: the same trajectories with the agent's prose (the
natural-language part of each assistant turn, outside the tool call) removed or rewritten to a
controlled length. Three families, plus a rebuild of A:

| family | variants | think / task_tracker turns | prose per turn |
|---|---|---|---|
| A `fixed` | `_text0`, `_text20`, `_text50`, `_text100`, `_text300` | removed, with their result turns | none, or ~20 / 50 / 100 / 300 tokens for every turn |
| B `scaled` | `_text0x`, `_text0.5x`, `_text2x`, `_text4x`, `_text8x`, `_text32x` | kept verbatim | none, or ½ / 2 / 4 / 8 / 32 × the turn's **own** prose length |
| A+ `withthink` | `_text0_withthink` (= `_text0x`), `_text20_withthink`, `_text50_withthink`, `_text100_withthink`, `_text300_withthink` | kept verbatim | none, or family A's text: ~20 / 50 / 100 / 300 tokens for every turn |
| D `textcov` (claude45 only) | `_textcov0`, `_textcov20`, `_textcov40`, `_textcov60`, `_textcov80`, `_textcov100` | kept verbatim | the original prose, on 0-100 % of the turns (40 % = the base) |
| C `think` | `_text0x_think0x`, `_text0.5x_think0.5x`, `_text2x_think2x`, `_text4x_think4x`, `_text8x_think8x`, `_text32x_think32x` | task_tracker kept verbatim; think turns rephrased (their prose and their `thought`), removed at 0× | family B's prose, plus each think turn's prose and thought at the same multiple of their own lengths |

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

Family C (2026-09-26, Gaokai: "keep think and plan, scale non-tool-call regular text, scale think
but do not scale plan; the only difference with last round is that we also scale thinking
content"): family B plus the `think` turns.

1. Every non-think turn carries **family B's very text** (its `<base>.scaled.x32.jsonl` records),
   so `_text<N>x_think<N>x` differs from `_text<N>x` only in its think turns. `task_tracker`
   turns stay verbatim, as in B.
2. A think turn's prose before the call is rephrased like any other prose (same spec, same caps),
   and the call's `thought` argument is rewritten to the same multiple of its own length; the
   `summary` argument and the whitespace around the thought stay byte-identical. At 0× the think
   turns are removed with their results (a think call with an empty thought is meaningless), so
   `_text0x_think0x` keeps only the tool calls and the planning steps.
3. Thoughts are long where prose is short (claude45: 3 738 thoughts, median 304 tokens, p90 545,
   max 4 134, 1.2M tokens in all; r2egym: 8 thoughts, median 106), so B's caps would leave them
   unscaled: with the one-reply 2 000-token cap, 8× reaches only 36 % of claude45's thoughts, and
   with B's 8 000-token x32 cap 32× also reaches 36 % while the other 64 % stay at 1×, i.e.
   `_text32x_think32x` would carry less thinking in total (~6×) than `_text8x_think8x` (~7.7×).
   So a thought's x0.5 and every rung of at most 600 tokens (`THOUGHT_LADDER_MAX`) are written in
   one reply as B does, and every longer rung is **written in parts** (B's x32 writer) from the
   longest accepted rung below it (x2 from the thought, x4 from x2, x8 from x4, x32 from x8), with
   caps of 8 000 tokens for x2 / x4 / x8 (99 % of thoughts scaled at 8×). x32 is **clamped** at
   16 000 tokens (`THOUGHT_X32_CLAMP`, Gaokai 2026-09-26) instead of capped: thoughts of up to 500
   tokens (87 %) get a full 32×, longer ones are written at 16 000 tokens, so every thought still
   grows from 8× to 32× and the rung carries 29.6× the base thinking in total. Past ~15K tokens the
   model runs out of things to say and pads (see 5.), which is what the clamp avoids; skipping
   over-cap thoughts as B does would leave 64 % of them at 1×, i.e. less thinking in
   `_text32x_think32x` (~6×) than in `_text8x_think8x` (~7.7×). The price is length: claude45
   trajectories (base median 22K tokens) reach a median of 29K at 8× and 53K (p90 87K, max 192K)
   at 32×, versus 30K / 54K / 128K for B's `_text32x`.
4. The thought prompts (`THOUGHT_*` in `llm.py`) replace B's "commentary before a tool call"
   framing with "the reasoning the agent records with its think tool": the rewriter sees only the
   thought and the call's summary, keeps conclusions and concrete references, uses lists and code
   only where the thought does, and reproduces quoted code exactly and at most once.
5. Repeats (added for family C; affects every version written in parts from now on): deep into a
   long version the model sees only the head and tail of the text so far and re-emits earlier
   sentences. Family B's parts writer rejected a part only when it *opened* with earlier text, so
   16.5 % of the sentences of claude45's `_text32x` versions over 4 000 tokens are verbatim repeats
   (6.1 % over all its x32 versions, 9.6 % for r2egym; ≤ 0.2 % for every ladder rung). Each part now
   loses the sentences of 40+ characters (outside code blocks) that the text so far or the part
   itself already has, and a part that loses more than half of its tokens that way is re-asked as
   a restart. On 11 thoughts (80-717 tokens) the x32 rung went from 2 fallbacks and 5.7 % repeated
   sentences to 0 and 0 %, and the share of distinct word 4-grams rose from 0.60-0.83 to 0.85-0.91
   for 5-12K-token versions. It does not rescue the longest ones: past ~15K tokens the model pads
   with short sentences ("I will act now. That is the plan."), 0.70 / 0.46 for the 20K / 22K-token
   x32 versions of 648 / 717-token thoughts.
6. Runs: `rephrase.py --family think` makes two units per think turn (field `text`, spec
   `scaled`; field `thought`, spec `thought`); records are keyed by (instance, message, field).
   Pass 1 writes x0.5 … x8 (`--keys x0.5,x2,x4,x8` → `<base>.think.x0.5+x2+x4+x8.jsonl`), pass 2
   adds x32 on top (`--keys x32 --prior` it → `<base>.think.x32.jsonl`), as B's x32 rung did.

## Files

| file | role |
|---|---|
| `trajectory.py` | split a turn into prose / call / tool name; positional call↔result pairing; skeleton (think / task_tracker turns dropped, kept verbatim, or split); thought extraction and replacement; reassembly |
| `tokens.py` | token counting with `Qwen/Qwen3-8B` (same vocabulary as Qwen2.5-Coder) |
| `llm.py` | the three length specs (fixed, scaled, thought), the prompts (first round + retry + parts), reply parsing, gateway client with backoff and adaptive in-flight cap |
| `rephrase.py` | the LLM driver: one work unit per non-think assistant turn (family think: two per think turn), per-request `max_tokens`, resumable jsonl output; `--keys x32 --prior <jsonl>` adds the parts-written rung to a finished run |
| `build_variants.py` | assemble a family's variants, validate, save to disk, push with a dataset card |
| `fill_prose.py` | write a comment for every empty turn, one request per trajectory (family D) |
| `build_coverage.py` | build / validate / push the `_textcov<N>` levels from the base and the fills |
| `dedupe_parts.py` | remove repeated sentences from one rung of a finished run, topping up versions that fall under their band |
| `run_pipeline.sh` | tmux runner: rephrase → build → push per dataset, `FAMILY=fixed|scaled|think`, `KEYS` for a subset of rungs / the added x32 rung, re-entrant |

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

Family C (2026-09-26, adaptive limiter 16/4/8; "thoughts kept original" = band missed + not requested):

| dataset | think turns | thoughts kept original ½× / 2× / 4× / 8× | thought ratio mean ½× / 2× / 4× / 8× | LLM tokens in / out (pass 1) | wall time |
|---|---|---|---|---|---|
| claude45 | 3 738 (+ 524 lead-ins) | 4 / 11 / 42 / 103 (37 of them over the 8× cap) | 0.53 / 2.06 / 4.23 / 8.37 | 40.3M / 22.8M, 5.8 calls per thought | 9.8 h |
| r2egym | 8 (+ 8 lead-ins) | 0 / 0 / 0 / 0 | 0.52 / 2.03 / 4.3 / 8.9 | 0.1M | 3 min |

All ten `_text{0x..8x}_think*` datasets were validated fresh from HF: every non-think message identical
to family B's same-N dataset (B's `_text0x` minus the think turns for `_text0x_think0x`), every think
call identical to the base apart from its thought. The x32 rung runs as a second pass with the
16 000-token clamp.

Family A+ `withthink` (2026-09-26, Gaokai: add the thinking/planning steps back into text0 … text300,
so that they differ from the `_text<N>x` datasets only in token counts and in the way the prose was
rewritten): no LLM calls, family A's rephrase jsonl rebuilt with `think` / `task_tracker` turns kept
verbatim, pushed as new datasets. Removing their think turns gives back `_text20` … `_text300` byte for
byte, and `_text0` with the think turns kept is byte-identical to `_text0x`; it is published as
`_text0_withthink` too (2026-09-27, Gaokai: complete the series under one naming scheme), and its card
says it is the same data as `_text0x`. Beyond the length rule, A+ and B differ in one more way: A gives turns without prose (60 % of
claude45's) prose explaining the call, B leaves them empty. Family A has no repeat problem (every
version is written in one reply: 0.02 % / 0.01 % repeated sentences in `t300`, none below it).

Family D `textcov` (2026-09-27, Gaokai; `fill_prose.py` + `build_coverage.py`): the share of turns with
prose as the only variable. 40 % of claude45's non-think turns carry prose (10 513 of 26 194; mean 38.7,
median 23 tokens) and 60 % carry none. `_textcov0` removes all of it (the same trajectories as
`_text0x`), `_textcov20` a random half, `_textcov40` is the base, and `_textcov60/80/100` add comments to a
random 1/3, 2/3 or all of the empty turns. The random orders are drawn once (seed 0), so the levels are
nested, and the kept prose is always the original. The comments are written once for every empty turn
(`fill_prose.py`), **one request per trajectory** instead of one per turn: the model sees the whole run
(task, every step with its comment or call, every result; results clipped to 1 500 characters, calls to
1 200, the agent's system prompt left out; median 11K prompt tokens) with the empty steps marked and
returns all their comments as one JSON object, re-asked for any it leaves out. There is no length
target; the prompt asks for the tone and length of the agent's own comments in the run and forbids
anything the agent could not know yet (the call's result, later steps). Smoke test: mean 20 / median 19
tokens, 0 of 34 comments naming an identifier first shown after their step. The endpoint answers such a
request in ~1.5 s, so the 1 457 trajectories take minutes, not hours.

The full run (2026-09-27) showed that seeing the whole run does leak: 0.7 % of the first 8 216 comments
named something the trajectory shows only after their step ("Found the aztec_sandpile function" before
the search that finds it). `fill_prose.py` therefore checks every comment for identifiers (backticked
text, paths, snake_case, CamelCase) that appear only after its step, and writes a flagged one again
from the run cut off right after that step's call, where it cannot leak. Result: 15 681 comments for
15 681 empty turns, 0 missing, 117 (0.75 %) rewritten, mean 17.8 / median 17 tokens (the agent's own
prose: 38.7 / 23). The heuristic still flags 8 of the rewritten ones, all names the model knows from
elsewhere (SageMath, OpenMPI, functions of well-known libraries), not from the trajectory; a leak that
names nothing ("the search found it") is not caught. The six datasets were validated fresh from HF:
coverage 0.0 / 20.1 / 40.1 / 60.1 / 80.0 / 100.0 %, every level's prose contained, unchanged, in the
next, kept prose identical to the base, `_textcov0` = `_text0x`, `_textcov40` = the base, `think` /
`task_tracker` turns and every tool call and result byte-identical.

`dedupe_parts.py` (2026-09-26) removed the repeated sentences from family B's x32 rung after the fact
(see Family C, 5.): each accepted x32 version loses its repeated sentences; one that falls under its
band is topped up with further parts (with the filter) or, when the model has nothing new left to say
and every continuation is itself a repeat, keeps the shorter deduplicated text (~21-23×, flagged
`below_band` and counted on the card). `_text32x` was re-pushed for both datasets; the pre-dedupe
records are kept as `<base>.scaled.x32.predup.jsonl`. The ½×-8× datasets were left alone (≤ 0.2 %
repeats, and students have been trained on them).

Family C x32 (2026-09-27, `THOUGHT_X32_CLAMP` = 16 000, adaptive limiter 16/4/8): claude45 3 580 of
3 738 thoughts rewritten in band (464 of them clamped at 16 000 tokens; 158 kept the original: band
missed or no target), 8 of 8 for r2egym; every non-think message identical to the deduplicated
`_text32x`; 71.4M / 47.8M LLM tokens in / out, ~14 h.

### Reply artifacts and meta-language (found 2026-09-28, fixed with `rephrase.py --redo`)

A scan of every record file for text that is not the agent's own words found two classes of artifact
(rates are the share of accepted versions, in excess of the original text; `llm.META_PATTERNS` /
`llm.RESIDUE_PATTERNS`):

1. **Meta-language about the rewriting task.** The model narrates its instructions instead of thinking
   as the agent: "the thought doesn't mention testing, so I won't plan that", "as the summary says",
   "I'm supposed to keep the same conclusions", "the next part of my reasoning", and in prose "the tool
   call I'm making is a view command". It grows with the number of parts a version is written in.
   claude45 thoughts: ½× 0 %, 2× 4.6 %, 4× 10.0 %, 8× 19.8 %, 32× 45.4 %; family B prose 32×: 4.9 %
   (claude45) / 5.7 % (r2egym), 8× 0.4 %, 4× ≤ 0.2 %, 2× ≤ 0.1 %; family A t300 0.2 % / 0.1 %, t100
   0.1 % / 0.0 %; textcov comments 0 of 15 681. The detector is deliberately narrow: a docstring's
   "summary line", "the next part of the task", a doctest "prompt", "the original wording" of a removed
   docstring, "the source text" a parser reads and "I'm rewriting the condition" are the agent's words and
   pass. Rule echoes in the agent's own voice ("I should not invent details the code doesn't show") are
   not rejected either, only discouraged by the new prompt rule; they occur in ~15 % of the 32× thoughts
   against 0.7 % of the originals.
2. **Reply-format residue.** (a) One-reply versions ending in `"}\n\n{`: the model answered with two
   JSON objects and `_lenient_extract` stripped only one closing quote and brace. Family B 2× / 4×
   prose: 3.3 / 3.4 % (claude45), 2.3 / 2.4 % (r2egym); ½× 0.9 / 1.3 %; 8× 0.5 / 0.3 %; family A t50-t300
   0.3-0.8 %; thoughts 0.4-1.7 %. These are in the pushed `_text2x` / `_text4x` datasets that students
   were trained on. (b) `</think>{"part": "` inside parts-written versions: the model drafted a part,
   emitted `</think>` and restarted the object, and `extract_partial` took everything after the FIRST
   `"part": "`. 32× prose 5.3 % (claude45) / 7.8 % (r2egym), 32× thoughts 8.2 %.

Fixes in the code: `after_think` (a reply is read after its last `</think>`), `extract_partial` takes
the last occurrence of the key, `_lenient_extract` and `strip_json_tail` cut a trailing `"}` / `"}\n\n{`;
a rule in `RULES` / `THOUGHT_RULES` ("write as the agent, in the moment ..."); every ladder candidate and
every part is checked with `meta_hits` / `residue_hits` and rejected with the reason fed back into the
retry ("it spoke about the rewriting task ("the thought") instead of as the agent ..."); and
`build_variants.py` refuses to build a variant whose versions carry either artifact
(`--allow-artifacts` overrides). `rephrase.py --redo <finished jsonl>` rewrites only the versions with
an artifact (a version that merely ends in `"}` is cut and kept when it stays in band) and writes
`<base>.<family>.redo.jsonl`, which the build takes after the original file. Scope of the redo
(versions): claude45 thoughts 2× 173, 4× 369, 8× 719, 32× 1 625 + 295 residue; claude45 prose 32× ~1 000,
8× 46, 4× 18, 2× 9, ½× 4; r2egym prose 32× ~3 500, 8× 123, 4× 73, 2× 17; family A 63 + 66 turns; 717 /
~1 700 (claude45 / r2egym) prose versions and ~90 thoughts only lost their JSON tail. Smoke test on 4
thoughts: every rewritten version clean, 0 fallbacks, 5 meta rejections re-asked successfully; the run
is `eval_outputs/verbosity_rephrase/redo_meta.sh` (tmux `redo-meta`, logs `logs/redo_meta.log`), which
re-pushes every rephrased variant of a (base, family) as soon as its redo finishes.

Outcome for claude45 (2026-09-28): all 18 rephrased datasets re-pushed and validated fresh from HF with 0
structural mismatches, 0 versions with reply markup and 0 with meta-language (`_text{20,50,100,300}`, their
`_withthink` siblings, `_text{0.5x,2x,4x,8x,32x}`, `_text{0.5x,2x,4x,8x,32x}_think*`). Cost of the redo: prose
1 027 units, thoughts 1 982 units (54.5M / 36M tokens in / out, 19K calls, 7 h alone on the gateway),
family A 65 units. A version whose rewrite still failed after the retries keeps the original text: the 32×
thoughts now have 277 of 3 738 originals (158 before; the 119 new ones are mostly "restarted the text"
rejections five times in a row), the 32× prose 49 of 10 440 (18 before). An LLM-judge audit of 40 rewritten
32× prose versions (judge sees the source, the real call and the version) found 0 invented facts and 0
meta-language. Two more lessons: family A's records predate the `family` / `targets` fields (the redo
infers them), and a driver must use `set -o pipefail`, otherwise a build's refusal disappears behind
`| tee` and the log claims a push that never happened. The r2egym datasets were NOT redone: yiqing noted
that r2egym barely uses the think tool, and the effort went to the think family of
`func_localize_qwen35_397b_1299i` instead (below); their artifacts stand (32× prose ~13 % of turns, 2×/4× 2.4 %
JSON tails), and `r2egym_qwen3next80b_1500i.scaled.redo.jsonl` holds 1 635 finished units if it is resumed.

### Family C for `func_localize_qwen35_397b_1299i`, rewritten by its own teacher (started 2026-09-28)

yiqing / Gaokai: the r2egym data barely uses the think tool, so the think family is built instead for the
Qwen3.5-397B func-localize dataset (1 299 rows, 27 assistant turns per row, 2 576 think calls in 92 % of the
rows, 817 task_tracker calls, 54 % of the non-think turns carry prose) and the rewriter is the teacher itself,
`nvidia/qwen/qwen3-5-397b-a17b` (free on the gateway, reasoning off via `chat_template_kwargs.enable_thinking`),
so the rewritten text stays in-family. Rungs x0.5 / x2 / x4 / x8 only (no 32×): variants
`_text{0x,0.5x,2x,4x,8x}_think{same}`. Driver `eval_outputs/verbosity_rephrase/think_family_397b.sh`
(records in `qwen35_397b/`): the scaled family for the non-think turns' prose, the think family for the think
turns. Smoke test on 2 rows: prose 2 calls per unit (the model misses the band on the first reply
more often than deepseek-v4-flash), thoughts 6.3 calls per unit, 0 meta-language or residue, ratios
0.53 / 1.81 / 3.73 / 8.24 (prose) and 0.61 / 2.03 / 3.93 / 7.85 (thoughts). Qwen3.5-397B also passed a
2-row smoke as a rewriter of the claude45 data.

The driver runs in two stages so that every 4× dataset comes first (Gaokai, 2026-09-29):
`think_family_397b.sh 4x` writes the prose at x0.5 / x2 / x4 / x8 (one reply per turn, so all rungs at once)
and the think turns at x2 / x4 only (x4 thoughts over 600 tokens grow in parts from x2), then builds and pushes
`_text0x_think0x`, `_text2x_think2x`, `_text4x_think4x`. `think_family_397b.sh rest` writes the think turns at
x0.5 / x8 with `--prior think.x2+x4.jsonl` (x8 grows in parts from x4; the prior records are merged into
`think.x0.5+x8.jsonl`) and builds `_text0.5x_think0.5x`, `_text8x_think8x`.

The gateway rate-limits per source IP (a probe on an idle model gets instant empty-body 429s while three
jobs run; two jobs on different models are fine), so the gateway jobs run two at a time. `queue5.sh`
orders them 4× first: this family's `4x` stage and the rewriter ablation of `text4x_think4x` by Opus 4.5
(`rewriter_ablation.sh`), then the ablation by Qwen3-Next-80B, then this family's `rest` stage, then the
gpt5mini `add_think` synthesis. Every stage resumes from its records, so a killed job restarts where it
stopped.

Stage `4x` pushed 2026-09-29 18:45Z (think x2/x4: 5 150 units, 0 errors). Validated fresh from HF: rows,
user turns and every call byte-identical to the base (think calls apart from the thought), 0 meta-language;
`_text0x_think0x` checked separately as an ordered subsequence of the base that skips exactly the 2 575 think
turns and their 2 567 results, task_tracker turns kept verbatim.

| dataset | prose ratio | thought ratio | fallback prose / thoughts | residue (from the base) |
|---|---|---|---|---|
| `_text0x_think0x` | 0 | — (think dropped) | — | 1 |
| `_text2x_think2x` | 1.91 | 2.25 | 149 / 4 | 1 |
| `_text4x_think4x` | 3.88 | 4.15 | 143 / 2 | 2 |

The base data has shapes the claude45 data does not, and the first build refused it; `build_variants` and
`trajectory` now handle them without touching the claude45 builds: 63 empty assistant turns (each answered by
"Your last response did not include a function call") are kept like any verbatim turn; 7 think calls without a
`thought` argument (summary only, or `Thought`) keep their call; 2 turns carry a think block inside another
call (`<tool_call><function=file_editor>...</tool_call><function=think>`, a malformed `<tool_call><function=think>`)
and stay verbatim; and 11 runs of parallel calls have fewer results than calls, where the "thought has been
logged" result now goes to the think call (positional pairing gave it to a call of a non-existent tool in 4 runs,
which `_text0x_think0x` would have left answered by a think result).

## Usage

```bash
# everything, in tmux (re-entrant; resumes the jsonl outputs)
tmux new -d -s verbosity 'FAMILY=scaled bash tools/verbosity_rephrase/run_pipeline.sh'
# the 32x rung on top of a finished scaled run (reads <base>.scaled.jsonl, pushes _text32x only)
tmux new -d -s verbosity-x32 'FAMILY=scaled KEYS=x32 bash tools/verbosity_rephrase/run_pipeline.sh'

# family C (think turns on top of the finished family B): x0.5..x8 plus the prose-free variant, then x32
tmux new -d -s verbosity-think 'FAMILY=think KEYS=x0.5,x2,x4,x8 VARIANTS=text0x_think0x,text0.5x_think0.5x,text2x_think2x,text4x_think4x,text8x_think8x WORKERS=16 WORKERS_MIN=4 WORKERS_START=8 bash tools/verbosity_rephrase/run_pipeline.sh && FAMILY=think KEYS=x32 WORKERS=16 WORKERS_MIN=4 WORKERS_START=8 bash tools/verbosity_rephrase/run_pipeline.sh'

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
.venv/bin/python tools/verbosity_rephrase/rephrase.py --family think --keys x0.5,x2,x4,x8 \
    --hf synthetic-code-training/func_localize_claude45_1457i --out-dir eval_outputs/verbosity_rephrase --workers 6
.venv/bin/python tools/verbosity_rephrase/build_variants.py --family think --variants text0x_think0x,text2x_think2x \
    --hf synthetic-code-training/func_localize_claude45_1457i \
    --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.scaled.x32.jsonl \
    --rephrase eval_outputs/verbosity_rephrase/func_localize_claude45_1457i.think.x0.5+x2+x4+x8.jsonl \
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
