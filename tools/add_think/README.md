# add_think: LLM-synthesized `think` steps for function-localization trajectories

Companion of `training/add_plan/add_plan.py` (branch `yiqing`), which adds synthesized `task_tracker`
steps at phase boundaries. Removing the think tool from `func_localize_claude45_1457i` (opus-4.5)
also lowers student accuracy, so this tool adds think steps to trajectories that have none:
`func_localize_gpt5mini_1346i` (no think step at all) and the 914 of 1467 `func_localize_claude47_1467i`
trajectories without one. Everything here is derived from how opus-4.5 uses the tool.

## How opus-4.5 uses `think` (func_localize_claude45_1457i, 1457 trajectories, 3738 think calls)

Analysis scripts: scratchpad `classify_thoughts.py` (deepseek-v4-flash labels each thought's role,
5 per request) and the position statistics in the session log; numbers as of 2026-09-28.

**Frequency.** Every trajectory has at least one think call; 2.57 per trajectory (median 2; 1: 23 %,
2: 35 %, 3: 23 %, 4+: 19 %), i.e. about one per 7 actions. The count grows with the trajectory:
1.7 for trajectories of 5-9 actions, 2.9 at 20-24, 4.6 at 40+. claude47's own think trajectories
have 1.98 calls each and are the longer, search-heavier ones (11.3 vs 7.3 actions, 4.2 vs 2.5 greps).

**Position.** Rarely the first step (7.7 % of trajectories); the first think comes after 2 actions
(median), usually right after a `file_editor view` (79 % of thinks follow a view, 10 % a `cd && ...`
terminal command, 5 % a grep). 18.8 % of trajectories think right before `finish`. 14 % of think
turns carry a sentence of prose before the call; the call has `summary` + `thought` (summary first
in 57 %, thought first in 38 %, no summary in 4 %); the result is always
`EXECUTION RESULT of [function]:\nYour thought has been logged.`

**Roles** (LLM labels; share of the 3738 thoughts, position, length):

| role | share | where | median words | ends with a decision | typical next call |
|---|---|---|---|---|---|
| verify - check the viewed code against the description, decide it is (not) the target | 50 % | middle (67 % of thoughts in the 2nd-3rd fifth) | 213 | 93 % | view 44 %, str_replace 14 % |
| strategy - read the description, decide where to search | 14 % | start (71 % of the first fifth) | 71 | 49 % | view 48 %, cd/find/grep 52 % |
| triage - weigh the candidates a search returned | 14 % | early-middle | 149 | 58 % | view 55 %, grep 17 % |
| compose - settle the docstring's content and format | 13 % | before the edit (39 % of the 4th fifth) | 218 | 98 % | str_replace 68 % |
| review - re-check the edit before finishing | 7 % | end (61 % of the last fifth) | 191 | 100 % | finish 80 % |
| recover / other | 1 % | | | | |

Most common per-trajectory sequences: `verify` (283), `verify, compose` (126), `strategy, verify`
(99), `verify, verify` (92), `triage, verify` (78), `strategy, verify, verify` (48), `verify, review`
(45), `strategy, verify, compose` (40). Form: 91 % use a numbered or bulleted list, 25 % quote code,
12 % use check marks; openers "Looking at ...", "I ...", "Let me ...", "Now ...", "Based on ...".
The thoughts quote the description's phrases and tie them to concrete names and line numbers just
seen; they end with the decision that the next call carries out.

claude47 (1095 thoughts): verify 47 %, triage 30 %, strategy 11 %, compose 6 %, recover 4 %,
review 1 % - it thinks when the search is hard, rarely to plan or review the docstring.

**Student results so far** (SWE-bench Verified, resolved/500; `swebench-verified-results`):

| training data | Qwen2.5-Coder-7B | Qwen3-4B | Qwen3-8B |
|---|---|---|---|
| claude45 (think + tracker, as is) | 68 | 80 | 86 |
| claude45 no_track | 65 | 92 | 81 |
| claude47 (38 % with think, no tracker) | 47 | 61 | 58 |
| claude47 add_plan (synthesized tracker) | 51 | 71 | 88 |
| gpt5mini (no think, no tracker) | 37 | 54 | - |

No graded student of a no-think variant of claude45 exists in the results repo yet.

## Design of the synthesized steps

`add_think.py` follows opus-4.5's four recurring moments as checkpoints (two at the same position
are merged into one thought; `--checkpoints` selects a subset for ablations):

| checkpoint | inserted | opus role it mirrors |
|---|---|---|
| `strategy` | after the first action's result | strategy / triage |
| `verify` | after the first `view` of the target file (the file of the last successful edit, as in add_plan) | verify |
| `compose` | right before the first `str_replace` / `insert` | compose |
| `review` | right before `finish` | review |

In the no-think trajectories the checkpoints exist for 100 % (`strategy`, `compose`, `review`) and
97-100 % (`verify`); `verify` and `compose` coincide (view immediately followed by the edit) in 42 %
of claude47's and 29 % of gpt5mini's trajectories, so most trajectories get 3 thoughts. That is
denser than opus-4.5 (2.6 per 18 actions) because these trajectories are short (7-10 actions), and
matches add_plan's four tracker steps per trajectory.

Each thought is one LLM request (deepseek-v4-flash, thinking off, temperature 0.3): the run up to the
insertion point (task, every step and result, clipped), the agent's NEXT call (its own decision, so the
thought can lead into it), the checkpoint's purpose, and two opus-4.5 thoughts of that role as style
examples, drawn per trajectory from six per role with distinct openers (`exemplars.json`). The first
60 trajectories of the run used the same two examples for everyone and came out formulaic (every
strategy thought opened "The description says", every review "Let me re-read the description"), so
the examples rotate and the prompt asks for varied openings; those 60 were rewritten. Nothing after the next call is shown, so a thought cannot use what the
agent has not seen. Replies are rejected and asked again (up to 3 attempts, then the checkpoint is
dropped) when they contain tool-call markup, speak about the writing task (`llm.META_PATTERNS`), are
under 30 or over 450 words, or name an identifier that appears in the run only later
(`fill_prose.leaks`). The step is inserted as opus-4.5 writes it (summary first, then the thought)
with the standard result message. Trajectories that already have think steps, or whose target file
cannot be determined, are left unchanged.

Smoke test (8 claude47 rows, 4 without think): 3-4 thoughts per trajectory, 55-257 words, one
reply re-asked for naming `sys.stdout` before the agent had seen it; about 10-17K prompt tokens and
3-4 requests per trajectory.

## Usage

```bash
# write the records (resumable; one line per trajectory in eval_outputs/add_think/<dataset>.think.jsonl)
python tools/add_think/add_think.py --hf synthetic-code-training/func_localize_gpt5mini_1346i --out-dir eval_outputs/add_think
# build the dataset and push it as synthetic-code-training/func_localize_gpt5mini_add_think_1346i
python tools/add_think/add_think.py --hf synthetic-code-training/func_localize_gpt5mini_1346i --out-dir eval_outputs/add_think --build --push
```
`eval_outputs/add_think/run_add_think.sh` runs both datasets (tmux `add-think`).
