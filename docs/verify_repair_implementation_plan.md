Verify-and-Repair (Gaokai Zhang)
You can take the implementation of our current r2egym task (`benchmarks/r2egym/`) and `benchmarks/hybridgym_verifyrepair/` (already runs on R2E-Gym) as a reference.
Task description: given an issue and a repo where a previous attempt at fixing it is left as an uncommitted change, the agent needs to check by running code whether the change fixes the issue, and repair it if not

Variance 1 (done): neutral prompt
Task description: the prompt says the previous attempt may be correct, incomplete or wrong. About 90% of the candidates are wrong and 10% correct
Result: 18/90 resolved on R2E-Gym; in 37% of the runs the agent kept the wrong candidate unchanged

Variance 2: known-failing prompt
Task description: the prompt says the previous attempt fails the tests of this issue. All candidates are wrong

Variance 3: CI feedback
Task description: same prompt as variance 1. When the agent calls finish, we run the hidden tests and send the names and assertion messages of the failing tests back to the agent (at most 2 rounds)

Run variance 2 and 3 on the same 90 R2E-Gym instances first, then scale up the better one

Reference implementation guide:
(data-preprocessing)
We need repos that can run, so we use R2E-Gym and SWE-Gym, where we already have graded rollouts of several models (`eval_outputs/r2egym_outputs`, `eval_outputs/swegym_outputs`)
A candidate is another model's patch that applies, edits non-test source files, and fails the hidden tests
Only keep instances that the teacher can solve from scratch; on the others the teacher almost never repairs the candidate. SWE-Gym with kimi-k2.5 as the teacher gives 330 instances, R2E-Gym with qwen3-next-80b gives 96
You can run `benchmarks/hybridgym_verifyrepair/build_candidates.py` to create the candidate file. Add a loader for SWE-Gym reports and a filter for teacher-solved instances

(training environment)
For R2E-Gym this is already done in `benchmarks/r2egym/run_infer.py`: remove the hidden tests and all git history after HEAD, fix permissions, take a base snapshot commit, then apply the candidate as an uncommitted change
Add one step before removing the hidden tests: apply the candidate, run the hidden tests, and make sure it really fails in our environment
For SWE-Gym, the agent currently runs the base python without the repo's dependencies. Turn on `testbed_env` (`benchmarks/swebench/testbed_env.py`) and check that `import <pkg>` resolves to the repo for every repo (pandas needs extra handling)

(inference)
Start from `benchmarks/hybridgym_verifyrepair/run_infer.py`
Edit the prompt (`prompts/default.j2`) for variance 2
For variance 3, after the conversation finishes, grade the diff in a new container; if it fails, send the failing tests back with `conversation.send_message()` and run again. Do not put the hidden tests into the agent's container
The remaining part should be the same (same scaffold and max 60 iterations as the student eval)

(evaluation)
Grade the final diff (candidate + agent edits) with the dataset's own grader: `benchmarks/r2egym/eval_infer.py` for R2E-Gym, `benchmarks/swegym/eval_infer.py` for SWE-Gym
Then filter the resolved trajectories: the agent must have changed the candidate, run the repo code after its last edit, not modified test files, and not looked at the hidden tests or the git history
When pushing with `convert_and_push.py`, take `resolved` from the grader's report
