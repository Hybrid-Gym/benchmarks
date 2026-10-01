# Hybrid-Gym verify-and-repair (T2)

Training tasks isolating the verification stage of issue solving. A candidate patch for the
issue (typically another model's attempt) is applied to the repository as an uncommitted change
before the agent starts. Two tasks use this setup:

- `--task repair` (default, prompt `default.j2`): the agent decides, by running code, whether the
  change resolves the issue, and fixes, completes or reverts it if not. The final diff (candidate
  plus the agent's edits) is graded by the underlying benchmark's own grader.
- `--task judge` (prompt `judge.j2`): the agent only decides whether the change resolves the issue
  and ends its final message with `VERDICT: CORRECT` or `VERDICT: INCORRECT`.
  `hybridgym-verifyrepair-judge-eval` compares the verdict with the candidate's grade
  (`candidate_resolved`); no grader run is needed.

See `docs/hybridgym_followup_plan.md` for motivation and results.

## Candidates

A JSONL file with one row per instance:

```json
{"instance_id": "...", "candidate_patch": "diff --git ...", "source": "gpt5mini", "candidate_resolved": false}
```

Only `instance_id` and `candidate_patch` are used by inference. Build candidates from graded
R2E-Gym rollouts (their `eval_snapshot.jsonl`/`output.jsonl` plus `output.report.json`):

```bash
uv run python -m benchmarks.hybridgym_verifyrepair.build_candidates \
    --source gpt5mini=<graded_run_dir> --source dv4f=<graded_run_dir> \
    --select pool.txt --correct-fraction 0.1 --max-per-repo 100 --out candidates.jsonl
```

Every instance with a flawed (applied, unresolved) patch gets one of those; resolved patches on
other instances are added up to `--correct-fraction`, so the agent also sees attempts that only
need verifying. Keep that share small (default 10%): trajectories that verify and stop without an
edit may teach the student to submit empty patches on SWE-bench, where there is no candidate.

For the judge task, use `--paired --correct-fraction 0.5`: only instances that have both a flawed
and a correct patch are used, and half of them get the correct one, so the label cannot be guessed
from the issue.

The teacher can only produce useful trajectories on instances it can solve. In our pilot,
flawed candidates on instances the teacher cannot solve from scratch were almost never repaired,
so restrict candidates to teacher-solvable instances where possible.

## Running

```bash
# Training data (R2E-Gym images; hidden tests and future git history are removed first)
uv run hybridgym-verifyrepair-infer .llm_config/teacher.json --harness r2egym \
    --candidates candidates.jsonl --workspace docker --max-iterations 60

# Grade with the benchmark's own grader
uv run r2egym-eval <output.jsonl>

# Judge task: the agent only gives a verdict, scored against candidate_resolved
uv run hybridgym-verifyrepair-infer .llm_config/teacher.json --harness r2egym --task judge \
    --candidates judge_candidates.jsonl --workspace docker --max-iterations 60
uv run hybridgym-verifyrepair-judge-eval <output.jsonl> --candidates judge_candidates.jsonl

# SWE-bench-format instances (validation only; never train on SWE-bench Verified)
uv run hybridgym-verifyrepair-infer .llm_config/teacher.json --harness swebench \
    --candidates candidates.jsonl --workspace docker --keep-base-image
```

`--harness r2egym` requires `--workspace docker` (hiding the hidden tests uses `docker exec`).
The output directory name includes the task (`verifyrepair-` / `verifyjudge-`) and the candidates
file name, so different tasks and candidate sets never share (and resume from) one directory.
`judge_eval` writes `output.report.json` next to `output.jsonl`; its `resolved_ids` are the
instances with a correct verdict, and each result also records `ran_code` and `patch_unchanged`
(the agent left the candidate as it was) for filtering trajectories.
