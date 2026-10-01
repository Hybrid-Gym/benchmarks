# Hybrid-Gym follow-up: execution-stage SWE subtasks — plan

Status: plan, verified by a 4-way review on 2026-09-26 (prior art, Terminal-Bench feasibility,
in-repo infra, methodology); re-checked on 2026-09-28 against Yiqing's brainstorming hints (§0b).
Branch `hybridgym-followup` (worktree `/home/gaokaizhang/benchmarks-hybridgym-followup`, cut from
`funclocalize-judge@ef8396b`).

## TL;DR

- **Question** (from Yiqing): which *individual* SWE-bench subtasks, trained on alone, give a gain on
  SWE-bench Verified **or** Terminal-Bench? No coverage requirement yet.
- **Bet**: Hybrid-Gym trains exploration, reasoning and editing, but almost no **execution /
  verification** — 32% of SWE-bench agent actions (paper Table 6) and nearly all of Terminal-Bench.
  So the new subtasks isolate the execution stages of issue solving.
- **Wave 1** = three subtasks, each paired with a *matched no-execution control* so the result
  answers "does this stage transfer?" rather than "does any agentic SFT beat a 1.8% base?":
  - **T2 Verify-and-Repair**: a candidate patch is already in the working tree; verify it, fix it if wrong.
  - **T3′ Probe-Localize**: func-localize (our best Hybrid-Gym task) **plus** write and run a probe
    that executes the located function. **Dropped as a candidate on 2026-09-28**: the probe has no
    oracle (nothing checks that what it observed is right), so it trains "run code", not "verify";
    getting an oracle means using the repo's existing tests, which turns it into SWE-smith / func-gen
    with tests. At most an optional ablation.
  - **T1′ = SWT-Bench-like training data** (since 2026-09-28): add a test to the repo's own suite
    that fails on the buggy code with an assertion and passes with the gold patch. This is not a new
    task format; the only new part is the experiment (train on it alone, measure SWE-V transfer),
    which SWE-Playground never ran. **Dropped by Gaokai on 2026-09-28** (an existing benchmark's data format,
    not a new subtask). Replacement candidates (regression repair, stall recovery, env repair) are
    in the T2 implementation-plan doc, §5.
- **T2 implementation plan (2026-09-28, for Yiqing)**: `docs/verify_repair_implementation_plan.md` (English).
  - Main pool = SWE-Gym with kimi-k2.5 as the free teacher: 330 teacher-solvable instances with
    flawed candidates (616 candidates). R2E + qwen80b has only 96.
  - SWE-Gym first needs the testbed env fix: 20–83% of existing SWE-Gym trajectories hit
    `ModuleNotFoundError`.
  - Pilot three prompt arms: P0 neutral (18/90), P1 "known to fail its tests", P2 host-side CI
    feedback on `finish`.
  - New pre-rollout step: re-grade each candidate (and the gold patch) in-container before hiding
    the tests.
- **Revised order after the 2026-09-28 re-check (§0b)**: T2 and T1′ are co-first, as small pilots.
  - T2 gets a CI stage (hidden tests report failures, never their source) and ≤ 10% correct candidates.
  - T1′ trains the one skill measurably absent from all our data: running tests in the repo's own
    framework. Test it first with zero rollouts (0g: SWE-Play's existing `swt` trajectories alone);
    build our own SWT-like data on R2E-Gym only if 0g is positive.
  - Principle: an execution task is only meaningful with an oracle to check the execution against
    (hidden tests for T2, gold patch for T1′). T3′ has none.
  - Terminal-Bench is not expected to move from ~500 SWE-style trajectories.
- **Teacher = a free NVIDIA-gateway model.** The exact model is secondary; default
  `nvidia/qwen/qwen3-next-80b-a3b-instruct`. `kimi-k2.5`, `gpt-oss-120b` and `qwen3.6-35b-a3b` are
  also free (checked 2026-09-26). Keep one teacher fixed across a task and its control, with
  trajectories short enough for the student's 60-step budget.
- **Students = Qwen2.5-Coder-7B + Qwen3-8B.** Primary metric = SWE-bench Verified (500).
  Terminal-Bench is a gated side track (TBLite, Qwen3-8B, custom Harbor agent), dropped if the floor
  probe shows no signal.
- **Trajectory evidence (§2b) first set the order T2 > T1′ > T3′** (revised above). It also exposed a
  blocker:
  - In our SWE-V eval harness, students run the base Python without the repo's dependencies, so
    execution mostly crashes.
  - The prompt's `conda activate testbed` line has an upstream typo.
  - This must be fixed (or taught) before execution training can pay off.
- **Before any rollout** (Wave 0):
  - fix the eval execution environment;
  - harden R2E-Gym rollouts (hidden tests and future git history are visible to the agent today);
  - refactor the R2E evaluator into hooks;
  - run a zero-rollout prior experiment on existing data;
  - establish the noise floor.

## 0. Progress (2026-09-27)

**Eval environment fixed** (`benchmarks/swebench/testbed_env.py`, on by default in `swebench-infer`,
opt-out `--no-testbed-env`, recorded in run metadata):
- Root cause: SWE-bench images activate `testbed` only in `/root/.bashrc`, but the agent runs as
  `openhands`, whose `python` is base conda 3.11. Also, the env's editable install points at
  `/testbed` while the agent edits a copy in `/workspace/<repo>`, so even an activated env would
  run the unedited code.
- Fix: at workspace prep, activate `testbed` in the agent's `~/.bashrc` and repoint the
  editable-install pointers to the working copy (PYTHONPATH fallback for non-editable installs).
- Verified in real containers for all 12 SWE-V repos' install styles; `import pkg` resolves to the
  agent's working copy.
- A/B on 12 easy50 instances (qwen3-next-80b):

  | | fix OFF | fix ON |
  |---|---|---|
  | agent code runs that succeed | 20/47 (43%) | 38/53 (72%) |
  | `pip install` attempts | 15 | 1 |
  | resolved | 8/12 | 9/12 (n.s.) |

- SWE-Gym keeps the old behavior until validated (meson-python pandas installs are not repointed).
- Local-docker note: the local sweb.eval images ship mode-777 files, so final patches carry
  mode-change noise. Harmless to grading, and absent from the GPU-side predictions.

**T2 verify-and-repair implemented** (`benchmarks/hybridgym_verifyrepair/`, CLI
`hybridgym-verifyrepair-infer --harness {r2egym,swebench} --candidates <jsonl>`):
- The candidate is applied as an uncommitted change through a new `prepare_repo` hook in both
  harnesses.
- R2E runs delete the hidden tests and all git history past HEAD (verified live: no
  `/r2e_tests`, 0 refs, `git log --all` == HEAD history).
- The final diff is graded by the benchmark's own grader.
- `build_candidates.py` builds candidates from graded rollouts. The R2E pool is 835 (626 flawed /
  209 correct).
- Unit tests cover all of the above.

**Feasibility results** (teacher = qwen3-next-80b, free):

| run | candidates | resolved | notes |
|---|---|---|---|
| SWE-V easy50, validation only (never train on it) | 20 flawed = real failed-but-localized 7B student patches | **11/20** (7/10, 4/10) | the same teacher from scratch solves 8/10 of these instances, so flawed candidates *anchor* the teacher; successful runs show repro-fails → edit → rerun-passes |
| SWE-V easy50, correct control | 9 resolved 7B student patches | **9/9** | the teacher does not break correct patches (5 kept unchanged) |
| R2E pilot | 20 flawed + 4 correct | **0/20** flawed, 4/4 correct | these instances are hard: from scratch qwen80b 2/20, opus45 7/20; 10/20 flawed candidates rubber-stamped |
| R2E teacher-solvable | 90 flawed on instances qwen80b solved once from scratch | **18/90 (20%)** | edited + re-ran: 16/41 resolved; left the flawed candidate unchanged: 1/33 (37% of runs); 18 usable trajectories, median 15 actions |

**Takeaways:**
- T2 yield depends on pairing a teacher-solvable instance with a weaker model's near-miss patch.
  Only 93 such pairs exist in the current R2E pool.
- Even on those, the free teacher yields 20%. The main loss is rubber-stamping the flawed
  candidate (37%); when it engages in the repro → edit → re-run loop it succeeds 39% of the time.
  500 trajectories at this yield would need about 2,500 rollouts and far more candidate pairs than
  exist.
- Levers, in order of cost:
  - a prompt that states the attempt fails the project's tests for this issue (realistic CI
    framing; removes the rubber-stamp option but also the correct-candidate branch);
  - near-miss mutations of correct patches (controlled difficulty, unlimited pairs);
  - a stronger teacher (opus45 solves 282 of the pool's instances).
- To scale, generate student-like flawed candidates on teacher-solvable instances:
  - best: our 7B students' own failed R2E patches (on-policy);
  - or: mutated near-miss versions of correct patches;
  - or: a stronger teacher (opus45 solves 282 of the pool's instances).
- Successful T2 trajectories are short (median ~15 actions) and contain exactly the behaviour the
  failure analysis says students lack. Whether that transfers is the training experiment.

## 0b. Re-check against Yiqing's brainstorming hints (2026-09-28)

The hints:
1. design from agent trajectories on SWE-bench, SWT-Bench and Commit-0 (on babel);
2. look at SWE-Play's `general` split (LLM-proposed tasks);
3. consider Terminal-Bench.

Scripts and outputs are in the session scratchpad (`behavior_compare.py`, `sweplay/`, `agentB/`).

**Hint 1: babel trajectories are not reachable from this box** (no DNS or ssh route to babel).
- Our failure taxonomy (§2b) rests only on SWE-V easy50 student trajectories. We have seen no
  SWT-Bench or Commit-0 student trajectories, and none are public on HF.
- **Ask:** copy `/data/tir/projects/tir5/users/yiqingxi/openhands/evaluation/evaluation_outputs`
  (or a sample per benchmark) to taurus, then rerun the §2b taxonomy on SWT and Commit-0.
- Those runs come from the OpenHands V0 harness, which activates `testbed`. The env bug fixed in §0
  is specific to the SDK harness (the GPU-side SWE-V sweep).
- **The SDK SWT-Bench harness has the same bug:** `benchmarks/swtbench/run_infer.py` copies
  `/testbed` to `/workspace/<repo>` and never activates `testbed`. Yet its prompt requires
  `python reproduction.py` and running the test framework. If SWT-Bench becomes a target, it needs
  the `testbed_env` step, and SDK-harness SWT numbers are suspect until then.

**Hint 2: SWE-Play = SWE-Playground** (arXiv 2512.12216; 704 Claude Sonnet 4 trajectories on 28
LLM-built projects; `general` 280 / `swe` 213 / `swt` 183 / `commit0` 28).
- Its ablation (7B, SWE-V resolved %):
  - base 1.8;
  - general only 8.6;
  - swe only 11.0;
  - general + swe 11.0;
  - all 704 (adds swt + commit0) 17.0.
- Single seed, no swt-only arm, and the last step also adds 43% data. Still, adding 280 general
  trajectories gave 0 while adding 211 swt/commit0 gave +6, so volume alone does not explain it.
- **`general` is not a new task type.** All 280 fill `NotImplementedError` stubs so that
  `tests/X.sh` passes (data classes 91, algorithms 65, parsers 26, engines 22, formatting 21, CLI 20,
  integration 13, ...). It is Hybrid-Gym func-gen plus a test oracle.
  - It adds nothing on top of `swe`, and 12% of the trajectories make no source edit (which
    plausibly explains general-only's 22.6% empty patches).
  - One workflow element is useful: the agent self-checks, then runs the provided tests. **In 65% of
    trajectories the teacher's own check passed and the provided tests then failed.** Even a strong
    teacher's self-checks miss flaws, the same mechanism as T2's 37% rubber-stamping, and the
    external test signal is what forces the repair.
- **SWE-Play data hygiene:**
  - 59% of `swe` and 51% of `swt` trajectories see a `# BUG:` comment left by the bug injection
    (localization for free);
  - `swt` trajectories almost never check that the test passes after the fix (~2/183);
  - `commit0` is unverified (3/28 finish);
  - every trajectory runs OpenHands' own Python and `pip install`s pytest. Training transferred
    anyway, but the data teaches a pip reflex.
- **Behaviour vs our data** (regex over actions):

  | data | median actions | code-run share | test-runner share | run before 1st edit | run after last edit | fail→edit→pass |
  |---|---|---|---|---|---|---|
  | SWE-Play general | 34 | 26% | 16% | 14% | 86% | 68% |
  | SWE-Play swe | 38 | 26% | 4.5% | 99% | 99% | 88% |
  | SWE-Play swt | 24 | 32% | 24% | 98% | 98% | 51% |
  | R2E qwen80b 663 (T2's control) | 15 | 19% | **0.1%** | 72% | 97% | 44% |
  | R2E opus45 1127 | 44 | 34% | 14% | 100% | 100% | 81% |
  | **T2 resolved 18** | 16 | 23% | **2.9%** | 89% | 89% | 56% |
  | func-loc claude45 / qwen80b | 19 / 7 | 1.7% / 0% | ~0% | 3% / 0% | 19% / 0% | 0% |

  - **T2 as run so far behaves almost like its own matched control** (R2E issue solving by the same
    teacher). So T2-vs-control mainly measures the start state (a near-miss patch), and a gain over
    the control is less likely than §3 assumed.
  - The one behaviour absent from all our data is **running tests in the repo's own framework**:
    0.1–3% of actions, against 16–24% in SWE-Play and 27.6% "verification" on SWE-bench.

**Hint 3: Terminal-Bench.** Re-checked numbers are in §2.
- Base Qwen3-8B scores about 2.5 on TB2 and 4–7 on TBLite.
- Every positive TB result at ≤14B used terminal or env tasks and 6k–100k trajectories. SWE-style
  sources at 10k trajectories reach only 4–6 on TB2 (OpenThoughts-Agent).
- So ~500 T1′/T2/T3′ trajectories are not expected to move TB, and format effects would dominate
  any change. TB stays a probe, recalibrated in §5. If TB becomes a goal in its own right, T4 (env
  repair) is the candidate, at thousands of trajectories.

**Refinements (applied in the TL;DR, §3 and §5):**
1. **T2 → T2-CI.**
   - After its own check, the agent may run `run_ci.sh`, which runs the hidden F2P + P2P tests as
     root and prints only test ids and assertion messages (test source stays unreadable).
   - Pilot on the same 90 teacher-solvable pairs against the 18/90 baseline.
   - Train with and without the CI turns (or use CI only to reject trajectories), because SWE-V has
     no CI.
   - Correct candidates ≤ 10%: "verify and stop" trajectories risk teaching empty patches.
   - Borrow from SWE-Review: reproduce on the original code before reading the candidate diff, and
     keep only decision-correct trajectories.
2. **T1′ → SWT-format reproduction test on R2E-Gym, co-first with T2.**
   - Output = a test patch in the repo's suite. Verifier: it fails on the buggy tree with an
     assertion (not an import/syntax error) and passes after the gold patch rebuilt from
     `parsed_commit_content`. SWE-Play's `swt` data lacks the pass-after-fix check; ours has it.
   - It trains exactly the missing test-runner behaviour, directly targets SWT-Bench (one of
     Hybrid-Gym's three eval benchmarks), and needs no candidate pool.
   - **Risk:** 76% of R2E-Gym-Lite issues contain a code block that starts with an import (a
     pasteable repro), so the §3 triviality gate (≤ 40%) fails on R2E. Accept this for the pilot
     (the skill is framework integration and a correct failure, not discovering the repro) and
     measure the added test-runner steps. SWE-Gym or SWE-rebench real issues are the fallback.
3. **New zero-rollout arm (0g):** train on SWE-Play `swt` 183 alone vs `swe` 183 (tool names mapped
   `execute_bash`→`terminal`, `str_replace_editor`→`file_editor`, our system prompt). This is
   SWE-Playground's missing swt-only ablation and costs only GPU time.
   - Confounds: Sonnet 4 teacher, synthetic repos, `# BUG:` leaks, pip reflex.
4. **`general`-style stub-and-test** (implement a stubbed function, then conform to held-out tests)
   is noted as a T3′ alternative only. SWE-Play shows ~0 marginal gain, and it teaches
   magic-number tuning to pass tests.
5. **Novelty:** the withdrawn "Atomic Skills" paper (2604.05013) trained jointly on
   localize/edit/test-gen/repro/review. Our claim must be the *isolated, individually trained*
   stage with a matched control, which nobody has published.

## 1. Success criterion (fixed before running)

Base Qwen2.5-Coder-7B is 1.8% on SWE-V, so "beats base" is not informative. A subtask **works** if,
pooled over both students and 2 seeds/resamples:

- gain ≥ **+3.0 pts** resolved over its matched control (same pool, teacher, N, recipe), with the 95%
  instance-bootstrap CI above 0 and the same sign on both students; and
- non-empty-patch rate does not drop by more than 5 pts.

It is **competitive** if also within 2 pts of the func-localize-500 reference (mean of 3 resamples).
On Terminal-Bench (if the track survives its gate): paired over tasks with k=3, gain ≥ 5 pts.

Why these numbers: near-replicate 7B students disagree on 54–70 of 500 instances, so McNemar needs
a gap of about 16 instances (3.2 pts) for p<.05, and the minimum detectable effect at 80% power is
about 4.3 pts per student. Pooling two students and two seeds brings that to about 3 pts. easy50 has
an MDE of about 20 pts, so it is used for smoke tests only.

## 2. Evidence the design rests on

| fact | source |
|---|---|
| Single Hybrid-Gym tasks @500 (7B): func-loc 12.8, issue-loc 9.6, dep-search 6.4, func-gen 7.8; SWE-Gym(491) 10.6 | paper Table 5 |
| Action shares on SWE-bench: exploration 49.1, **verification 27.6**, implementation 12.7, reasoning 6.0, **execution 4.7**; Hybrid-Gym tasks: verification 0.3–9, execution ~0.1 | paper Table 6 |
| Output must be a source edit (P1): removing str_replace from func-loc collapses transfer | paper §4.1 |
| **Teacher matters more than pool**: R2E students from qwen3-next-80b score 79/500 (7B) and 103/500 (Qwen3-8B); from opus-4.5 they score 25 and 11 | `swebench-results/*/eval_report.json` |
| func-loc students: 7B claude45-1457i 68, qwen80b-909i 60; Qwen3-8B claude45 86, qwen80b 73; Qwen3-4B qwen80b 98 | same |
| Our "verification helps" easy50 ablations do **not** replicate on the full 500 (7B no-val 70 vs 68, add-verify 49 vs 47; Qwen3-4B no-val 93 vs 80). They tested re-viewing a docstring, not execution, so the hypothesis is **open** | same |
| No paper trains a 7–8B student on one isolated *stage* of issue solving and reports SWE-V transfer. Nearest: SWE-Playground (repro only inside a mix, 7B 11.0→17.0 when repro+library are added, confounded with +43% data); SWE-Review 2607.06065 (review trajectories *added* to equal-volume resolution data raise Qwen3-8B 27.6→28.4 / 31.2→36.8 / 34.0→37.8 at 1k/2k/3k; review-only never tested); OpenThoughts-Agent 2606.24855 (Qwen3-8B on 95 single *sources*, 10k each: swe-smith 32.3 SWE-V-100 / 6.4 TB2, r2egym 28.3 / 4.1, stack-pytest-withtests 18.3 / 3.8, superuser 13.3 / **10.9**); CLI-Gym (env repair → TB, 32B only). "Atomic Skills" 2604.05013 (joint RL on localize/edit/test-gen/repro/review) was **withdrawn** for data errors: not evidence, but a novelty risk | prior-art re-check 2026-09-28 |
| Qwen2.5-Coder-7B scores 0.0 on TB2 (6.25 on TB1). Base Qwen3-8B scores **2.5** on TB2 (CLI-Universe 2606.22883, avg@4) and **4.4–6.7** on TBLite (our recount of public traces; River 2608.22631 reports 3.8). The earlier "1.1 on TB2" was a fine-tuned Qwen3-8B (Endless Terminals), not the base | TerminalTraj 2602.01244, CLI-Universe, River, `DCAgent2/eval__openthoughts-tblite__qwen3-8b__lambda-traces` |
| Every positive TB result at ≤14B used terminal/env tasks and thousands of trajectories (Qwen3-8B → 10.9 on TB2 with 6k CLI-Universe trajectories, 12 with SETA RL; Qwen2.5-Coder-7B → 10.1 with 50.7k). SWE-style sources at 10k trajectories reach only 4–6 on TB2 | same + OpenThoughts-Agent |

## 2b. Trajectory evidence: would these tasks help? (2026-09-26)

Data: 398 student trajectories on SWE-V easy50 (HF `easy50_*-student_50i` and
`swe_bench_easy50_*`; 8 students, all trained on func-localize-style data). Also the training data
of those students, and full-500 reports. Four readers hand-read 42 stratified trajectories against
the gold patch. Scripts are in the session scratchpad (`traj_features.py`, `exec_refine.py`,
`dump_traj.py`, `teacher_exec.py`).

**Where the students fail** (5 main students, 243 trajectories, 70 resolved, 173 failures; astropy-7606 excluded):

| failure | share | what the hand-read sample says | fixable by execution/verification training? |
|---|---|---|---|
| right file, wrong fix, nothing run after the last edit | 28% (49) | edits that change nothing (docstring-only, re-applied code, comment tweak), code that parses but crashes on first call (made-up attribute, missing `self.x=`, duplicated kwargs), failed edits reported as done | **yes, about 6/11 exposed by re-running the issue example**; ~3–7 become resolved. Best fit: **T2** |
| right file, wrong fix, something run after | 20% (35) | the "verification" was fake: 6/10 checks crashed on the environment and were replaced by `ast.parse`; evidence of the bug present but ignored in 7/10 | **yes, a better check was feasible in 8/10**; ~4–6/10 become resolved. **T2 ≥ T1′** |
| no source edit | 36% (63) | mostly found the code but never committed to an edit (4/11), or looped / falsely claimed done (3/11) | **mostly no** (2–3/11); needs commit-to-edit / anti-loop, not execution |
| edited the wrong file | 15% (26) | not read in depth | partly (a failing repro can point at the right code) |

- **Association (does NOT survive re-validation):** among right-file edits, trajectories that ran
  repo code after the last edit resolve 49% vs 29% (+26 pts within instance). An independent
  re-derivation (own parser, 2026-09-27) found:
  - it is driven by the claude45 student: without it, +4.6 pts (p=0.77);
  - with student + instance fixed effects: +0.10 (se 0.14);
  - counting only runs without import errors: +8 pts (p=0.49);
  - the "runs" are mostly `python -c "import pkg"` one-liners under the wrong interpreter.

  So there is **no evidence yet** that verification itself converts failures in these students;
  the case for T2 rests on the failure taxonomy and the hand-read counterfactuals below, not on
  this association.
- **Full-500 pool:** 24–50% of all 500 instances are localized but unresolved for our students
  (e.g. 7B R2E-qwen80b: 63.8% localized, 15.8% resolved). That is the pool T2 targets, 2–4× the
  current resolve rate.
- **Training data behaviour transfers almost one-to-one, but ours teaches fake verification:**
  - 51% of `func_localize_claude45_1457i` trajectories run something after the last edit, and so
    do 52% of its student's SWE-V runs.
  - Those runs are syntax/docstring checks: 53% `py_compile`/`ast`, 30% `help()`/`__doc__`.
  - Hence students treat "it parses" as verified. This also explains why deleting them (no-val)
    changed nothing on the full 500.
  - func-loc data from qwen80b, gpt5mini and gpt55 has ~0–10% execution.
  - R2E issue-solving data is full of it: qwen80b reproduces first in 72% of trajectories and runs
    after the last edit in 97%. opus45 does both in 100%, but at 43 steps and with the worst
    student, so length matters too.
- **Blocker: the eval environment breaks execution.**
  - In the current harness students run the base `python3` (3.11, no repo deps): 26–66% of
    trajectories hit `ModuleNotFoundError`, and 0–4% run `conda activate`.
  - The prompt's Phase 2.1 line `./opt/miniconda3/etc/profile.d/conda.sh ; conda activate testbed`
    has a typo (`./opt` → `. /opt`), inherited from upstream.
  - The older easy50 harness (testbed active) shows ~2% module errors.
  - The full-500 GPU sweep uses the same `swebench-infer` prompt, so it likely has the same problem.
- **Where execution was not needed:** in resolved trajectories, execution was never causally
  necessary (0/9 hand-read). The issue text often states the fix, so choose training instances
  whose issue does not.
- **Data hygiene:**
  - 8/398 trajectories used curl/wget; 2 fetched the upstream GitHub patch, and 17 more pulled
    GitHub content via python or `git clone` (network access during eval = contamination risk).
  - Many old-harness (`swe_bench_easy50_*`) transcripts are truncated at the start (18–21/50 in
    two sets), which inflates their "no edit" counts.
  - astropy-7606 is never resolved in this harness (0/77; a PASS_TO_PASS test always fails).

**Design consequences** (applied below):
1. **T2 is first.** Its candidate patches should look like the observed student errors: parse
   cleanly but behave wrong (no-op edits, missing assignments, made-up APIs, partial fixes).
   Include some correct candidates and cases where an existing test contradicts the issue.
2. **Every execution task starts with an environment step:**
   - activate the env;
   - confirm `import pkg; pkg.__file__` is the repo;
   - use the repo's own test runner;
   - never `pip install` the package under test.

   The training environment must require this the same way the eval does, or the eval must be fixed.
3. **Checks must be behavioural:**
   - the same repro file fails before and passes after, asserting expected values;
   - a crashed check is not a pass (never fall back to `ast.parse`);
   - look at `git diff` before `finish`.
4. **T1′ must re-run its repro after the edit** (or pair with T2), since detection is not repair.
   Since 2026-09-28 T1′ is detection only (SWT format, no source edit), so repair comes from T2.
   T3′ is the weakest on this evidence; keep it as the clean func-loc ablation, third in line.

## 3. Design: stage-isolated subtasks, each with a built-in control

Each subtask = an executable repo + a **start-state transform** + a **required output that includes
a source edit** (P1) + an **automatic verifier** that runs *inside the rollout container right after
the agent finishes*. Grading in-container avoids a second image pull, and the per-instance cleanup
deletes the base image afterwards anyway. Each subtask is paired with the no-execution condition it
extends, so a win isolates the execution stage.

### Wave-1 task specs

| | **T2 Verify-and-Repair** (+ CI stage) | **T3′ Probe-Localize** | **T1′ Repro-Test** (SWT format) |
|---|---|---|---|
| Pool | R2E-Gym-Lite, hardened. About 950 applying unresolved candidate patches (gpt-5-mini 555 + dv4f 399) over 627 instances; correct candidates from resolved weak-model patches | SWE-smith-py env images (131 repos, one image per repo, so about 131 pulls total). Targets are pure-Python functions sampled by AST, with LLM descriptions that give no name or path (func-loc recipe) | R2E-Gym-Lite, hardened (gold patch rebuilt from `parsed_commit_content`). Falls back to SWE-Gym or SWE-rebench `filtered` (real issues) if R2E's pasteable repros make it too easy |
| Start state | buggy repo + candidate patch as an **uncommitted** change (≥ 90% flawed / ≤ 10% correct, §0b) | clean repo, docstring of target removed | buggy repo + issue |
| Agent must | decide whether the change fixes the issue by running things; fix it if not; keep what is right. T2-CI: after its own check it may run `run_ci.sh` (hidden F2P+P2P, ids and assertion messages only) | locate the target → write its docstring → write `probe.py` that imports it from the repo and calls it on realistic inputs → run it until it runs cleanly | write and run a repro, then add a test to the repo's own suite and run it with the repo's test runner until it fails for the right reason; no source edits |
| Verifier | final diff vs the original base passes the faithful R2E reward (hidden F2P+P2P) | docstring on the correct target, source diff docstring-only (func-loc evaluator), **and** `probe.py` exits 0 under a stdlib `sys.settrace` tracer that records the target's code object executing from the repo file | diff touches test files only; the new test fails on the buggy tree with an assertion-type failure (not an Import/Syntax/Name error of the test) and passes after the gold patch (SWT-Bench's F→P criterion); optional tracer coverage of gold-modified lines |
| Matched control | full issue solving on the same instances, same teacher: existing `r2egym_qwen3next80b_1500i` resolved rows, subsampled to matched N and repo mix | **the same trajectories truncated after the docstring edit**, instruction swapped to the func-loc one, synthetic `finish` appended. Also a token-matched variant | issue-loc rollouts on the same instances, same teacher (needs no image); plus the zero-rollout SWE-Play swt-vs-swe arm (0g) |
| Anti-hacking | tests and future history hidden; audit whether "correct" candidates were rubber-stamped without any run | probe must import from the repo path (tracer checks `co_filename`); no source edits besides the docstring | test must import the package from the repo; `/r2e_tests` and future history hidden (they contain the gold test) |
| Target N | 500 successes | ~700 successes (so a token-matched subset remains) | 500 successes |
| Why it might win | closest to the downstream task; recycles failed rollouts that are otherwise discarded; trains the verify→repair loop | inherits the best single task and adds real execution, on 13× more repos than R2E | trains the one behaviour absent from all our data (test-runner use, §0b); SWE-Playground's only hint of type-specific transfer (+6 with swt+commit0); directly targets SWT-Bench; needs no candidate pool |
| Main risk | "issue solving with a hint": it hands over localization, the 7B's bottleneck; as run so far it behaves like its matched control (§0b), so the gain over control may be small without the CI stage | CWM found execution traces did not transfer (non-agentic); probe stage may be too short | 76% of R2E issues contain a pasteable repro, so the repro step is easy; SWT-format alone may not transfer to SWE-V (SWE-Playground never trained it alone) |

Order (revised 2026-09-28, §0b): **T2-CI and T1′ are co-first as 30–90-instance pilots.** T2 fits
the largest failure mode and reuses existing patches, labels and reward. T1′ shares T2's hardening,
needs only the gold-patch grader, and trains the missing test-runner behaviour. **T3′** stays third
(clean func-loc ablation; weakest on the trajectory evidence).

### Pilot gate (30 instances per task, before scaling)

- Teacher (qwen3-next-80b) success ≥ 50%.
- The execution stage adds ≥ 6 steps and ≥ 15% of actions; median total ≤ 45 steps (the student
  eval caps at 60 iterations; long teacher trajectories are the suspected cause of the opus-4.5
  R2E student's failure).
- 0 reward hacks in 20 hand-audited trajectories.
- T1′ only: report the share of issues with a pasteable repro (76% on R2E-Gym-Lite, so the old
  ≤ 40% gate is dropped). Instead gate on the new test failing with an assertion on ≥ 80% of
  successes and on ≥ 2 test-runner invocations per trajectory.
- T2 only: verdict accuracy on correct vs flawed candidates reported (it must not "fix" correct patches).
- T2-CI only: yield on the 90 teacher-solvable pairs vs the 18/90 baseline, and the share of
  successes that needed CI feedback (their CI-free replay is the transfer ablation).

## 4. Wave 0 — before any teacher rollout (~3 days, mostly parallel)

- **0-env. Fix the eval execution environment (blocker, see §2b).** Options:
  - (i) pre-activate `testbed` in the SWE-V workspace (env-setup command / PATH);
  - (ii) fix the Phase 2.1 typo;
  - (iii) leave the eval as is and make the new tasks teach env activation.

  (i)/(ii) change the eval, so re-grade 2–3 existing control students under the fixed harness to
  keep comparisons within one harness. Also decide whether to block network access during eval
  (`curl` of upstream fixes seen in 8/398).

- **0a. Harden R2E-Gym workspaces** (required by T1′/T2), done as root in `prepare_workspace`:
  - move `/r2e_tests` and `/testbed/run_tests.sh` into a mode-700 directory (plain `/root` is a+rx
    after the permission fix);
  - delete every ref except HEAD and expire the reflog.

  Keep this behind a flag that defaults to off, so the existing r2egym run is byte-identical.
  Evidence the leak matters: 12/89 kimi-k2.6 trajectories ran `git log --all`, 18/89 touched
  `run_tests.sh`. gpt-5-mini rarely probed (1/1500).
- **0b. Refactor `R2EGymEvaluation.evaluate_instance`** (`benchmarks/r2egym/run_infer.py:359-511`)
  into `_pre_agent`, `_instruction` and `_post_agent_grade` hooks with no-op defaults; add a test
  that the full task is unchanged.
  - Validate gold-patch reconstruction by feeding ~30 gold patches as a fake `output.jsonl` to
    `r2egym-eval` and expecting reward 1.0.
- **0c. Zero-rollout prior** (GPU only):
  - Split the 663 resolved qwen3-next-80b R2E trajectories into high- vs low-execution halves (by
    share of terminal actions that run python/pytest), length- and repo-matched, ~300 each.
  - Train both students. If the high-execution half does not beat the low one, lower our prior
    before spending rollouts.
- **0d. Controls and noise floor**:
  - func-loc-500 from `qwen80b-909i` ×3 resamples (7B + Qwen3-8B).
  - R2E-qwen80b-500 subsample (T2's control).
  - Base Qwen3-8B on SWE-V, if not already run.
- **0e. Opus failure diagnosis** (no GPU): compare step-count distributions of the opus45-421i and
  qwen80b-663i training data, and the students' max-iteration-hit rate. This sets the step cap for
  all new rollouts.
- **0f. Ask the GPU side to upload student SWE-V trajectories**, not just patches (needed for §5
  diagnostics). Also get Yiqing's babel SWE-bench/SWT-Bench/Commit-0 eval outputs onto taurus (§0b).
- **0g. SWE-Play swt-only arm** (GPU only, no rollouts):
  - convert SWE-Play `swt` (183) and a random 183 of `swe` to our format: `execute_bash`→`terminal`,
    `str_replace_editor`→`file_editor`, our system prompt, in-context example dropped;
  - train 7B + Qwen3-8B on each and grade on SWE-V.
  - swt ≥ swe would be the first evidence that reproduction-test writing transfers on its own.
  - Confounds: Sonnet 4 teacher, synthetic repos, `# BUG:` leaks, pip reflex.

## 5. Protocol

- **Rollouts** match the student eval exactly: SDK `e212d45` (+ chdir hotfix as a working-tree
  change), `system_prompt_old.j2`, default tools (terminal, file_editor, task_tracker, think,
  finish), condenser 240/2, max-iterations 60. Only the instruction template differs per task.
  - qwen3-next-80b @5 workers (the known-safe level); runs in tmux; disk guard on.
  - Critic retries cannot see task success, so use `--n-critic-runs 1` and filter by the
    in-container grade.
- **Training** is on the GPU side with the same recipe as existing students (lr 5e-5, 5 epochs,
  bs16). 2 seeds per treatment arm.
- **Eval**: SWE-V full 500, reporting resolved, non-empty, localized and non-loop.
  - Interim reads may use the 199 instances ever solved by any of our 7B students (about 60% cheaper).
  - Final numbers are always on the full 500.
- **Behavior diagnostics** (regex over student SWE-V trajectories; our LLM judges missed behavior
  in 40k-token transcripts):
  - D1: % that run repo code.
  - D2: % that write and run a repro before the first source edit.
  - D3: % that re-run something after the last edit.
  - D4: a script going from failing (traceback) to passing.
  - D5: P2P-regression rate among non-empty patches (free from harness reports).
  - Guardrails: max-iterations hit, loops, steps to first edit.

  A subtask that moves D1–D4 by ≥ 15 pts but not the resolve rate is a *behavioral* win and gets
  re-evaluated at 100 iterations.
- **Terminal-Bench track** (does not block SWE-V):
  1. Add a small custom Harbor agent mirroring the student eval: non-native tool calling (our SFT
     data is text-format function calls), `system_prompt_old.j2`, condenser 240/2, max-iterations
     60, SDK pinned to the training version. Harbor's stock `openhands-sdk` agent differs on all four
     and would fail our students for format reasons alone.
  2. Fix `benchmarks/terminalbench/run_infer.py` so it runs at all:
     - `--task-name` → `-i`;
     - pass `-k`/`--ak` through;
     - score each (task, attempt) pair (`eval_infer.py:90-94` currently drops repeat attempts).
  3. Probe `openthoughts-tblite@2.0` (100 tasks, all built from Dockerfiles, so watch disk) at k=1
     on base Qwen3-8B, `qwen3-8b-r2egym-qwen3next80b-663i` and `qwen3-8b-func-localize-claude45-1457i`.
     Students are served from the GPU node over ngrok; Harbor runs on this box's docker.
  4. Gate: if the best student is < 5 pts above base Qwen3-8B (≈ 4–7 on TBLite, 2.5 on TB2;
     re-measure it in our harness), drop TB and report SWE-V only. Expect this outcome for ~500
     SWE-style trajectories (§0b). A TB win would need a terminal/env task (T4) at thousands of
     trajectories.
- **Hand-off**: push `synthetic-code-training/<task>_qwen3next80b_<N>i` with the existing schema
  `{instance_id, resolved, messages}`. Every grader writes `output.report.json` with `resolved_ids`:
  if it is missing, `convert_and_push.py` silently labels every row resolved.

## 6. Implementation map (all in the worktree, repo conventions: one dir per task, CLI in pyproject)

| path | what |
|---|---|
| `benchmarks/r2egym/run_infer.py` | hook refactor + optional hardening flag (0a/0b) |
| `benchmarks/hybridgym_verifyrepair/` | T2: `build_dataset.py` (candidate pool from local gpt-5-mini/dv4f outputs + labels, per-repo cap against the 52%-numpy skew), `run_infer.py` (R2E subclass; `_pre_agent` applies the candidate uncommitted), `prompts/default.j2`; grading reuses `r2egym-eval` |
| `benchmarks/hybridgym_probelocalize/` | T3′: target sampler + description generator, SWE-smith-image `run_infer.py` with in-container grader (func-loc evaluator + settrace probe runner), truncation-control builder |
| `benchmarks/hybridgym_reprotest/` | T1′: gold-patch rebuild from `parsed_commit_content`, F→P test grader (buggy tree, then gold patch), issue-loc control run |
| `benchmarks/utils/exec_grading.py` | shared: run a command as root in the container, the settrace harness, hidden-file restore |
| `benchmarks/terminalbench/` | wrapper fixes + `harbor_agent.py` |
| `tools/action_components/` | Table-6-style component shares for new tasks and the D1–D5 diagnostics |
| `tools/swebench_grade/compare_students.py` | paired McNemar + instance bootstrap between two graded students |
| `benchmarks/*/tests/` | verifier unit tests on synthetic repos (fail→pass, hack attempts rejected) |

## 7. Schedule and resources (estimates)

- **Wave 0**: ~3 days on this box, plus GPU time for 0c/0d (about 12 students).
- **T2 and T3′ pilots**: ~2 days. Scaling each to 500–700 successes: ~3 days at 5 workers.
- **Training and eval**: 2 tasks × 2 arms × 2 students × 2 seeds, turnaround set by the GPU side.
- **T1′** pilots alongside T2-CI (revised 2026-09-28); it reuses the R2E hardening and adds one grader.
- **Constraints**:
  - Docker Hub 200 pulls/hr, shared; the pull cache is down. T2 needs ≤ 627 pulls, T3′ ~131.
  - `/` has 533 GB free; `/mnt/data*` are full.
  - Gateway worker ceiling is box-wide; other runs (e.g. `verbosity-think`) share it.
  - Never `docker system prune -a` on this shared box.

## 8. If Wave 1 fails, and what is deferred

- Behavior moved but resolve rate didn't → re-evaluate at 100 iterations and on Qwen3-8B.
- Still flat, and 0c was flat → execution is not the 7B bottleneck; pivot to T6 (localize-then-implement,
  the paper's own future-work item).
- **Deferred** (Wave 2):
  - winners rebuilt on SWE-rebench `filtered` (6.5k instances / 1.8k repos, public per-instance
    images, CC-BY-4.0; drop the 26 instances from SWE-V repos) for diversity;
  - mixing winners with func-loc (additivity);
  - T4 env repair: only if TB survives, reframed as a patch to dependency files to satisfy P1, built
    on SWE-smith images; CLI-Gym and DockSmith are the baselines to beat;
  - T7 bug injection (SSR/BugPilot-style; also generates T1/T2 data);
  - T5 failure-localization.

## 9. Environment and file hygiene

- The main checkout `/home/gaokaizhang/benchmarks` is untouched (live runs use it).
- The worktree has:
  - its own `.venv` (uv, hardlinked);
  - the SDK submodule at the same `e212d45` with the chdir-race hotfix re-applied as an
    **uncommitted** change, so `SDK_SHORT_SHA`, output-dir names and image tags match the main checkout;
  - `vendor/Hybrid-Gym` intentionally not initialized.
- LLM configs are referenced by absolute path from the main checkout's `.llm_config/` (not copied).
- Untracked tools living only in the main checkout (`tools/swebench_grade/{grade_batch.sh,
  reapply_resolved.py,fetch_predictions.py}`, `tools/trajectory_steps/`,
  `tools/critic_rubrics_judge/`) stay there; grading keeps running from the main checkout.
- Nothing is committed or pushed yet.
