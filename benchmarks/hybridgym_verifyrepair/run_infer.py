"""Verify-and-repair: check a pre-applied candidate patch and fix it if needed.

A candidate patch for the issue (e.g. a weaker model's failed attempt) is applied to
the repository as an uncommitted change before the agent starts. Two tasks share
this setup:

- ``--task repair``: the agent decides, by running code, whether the candidate
  resolves the issue and repairs it if not. The final diff (candidate plus the
  agent's edits, relative to the base commit) is graded by the underlying
  benchmark's own harness.
- ``--task judge``: the agent only decides whether the candidate resolves the issue
  and ends with a ``VERDICT:`` line, which ``judge_eval`` compares with the
  candidate's known grade.

Candidates are a JSONL file with one row per instance: ``{"instance_id": ...,
"candidate_patch": ...}`` plus optional provenance fields, which are ignored here.

Usage:
    uv run hybridgym-verifyrepair-infer <llm_config> --harness r2egym \\
        --task judge --candidates candidates.jsonl --workspace docker
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import List, Protocol

from pydantic import Field

from benchmarks.hybridgym_verifyrepair.judge_eval import (
    VERDICT_REMINDER,
    final_answer,
    parse_verdict,
)
from benchmarks.r2egym.config import INFER_DEFAULTS as R2EGYM_DEFAULTS
from benchmarks.r2egym.run_infer import R2EGymEvaluation
from benchmarks.swebench.config import INFER_DEFAULTS as SWEBENCH_DEFAULTS
from benchmarks.swebench.run_infer import SWEBenchEvaluation
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import EvalInstance, EvalMetadata
from openhands.sdk import get_logger
from openhands.sdk.conversation import BaseConversation, RemoteConversation
from openhands.sdk.workspace import RemoteWorkspace


logger = get_logger(__name__)

HARNESS_DEFAULTS = {"swebench": SWEBENCH_DEFAULTS, "r2egym": R2EGYM_DEFAULTS}
# Default prompt and output-directory prefix of each task.
TASK_PROMPTS = {"repair": "default.j2", "judge": "judge.j2"}
TASK_DIR_PREFIX = {"repair": "verifyrepair", "judge": "verifyjudge"}
MAX_VERDICT_REMINDERS = 2
CANDIDATE_PATH_IN_CONTAINER = "/tmp/candidate.patch"


def load_candidates(path: str) -> dict[str, str]:
    """Read ``instance_id -> candidate_patch``; one non-empty candidate per instance."""
    candidates: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row["instance_id"]
            if instance_id in candidates:
                raise ValueError(f"Duplicate candidate for {instance_id}")
            if not row["candidate_patch"].strip():
                raise ValueError(f"Empty candidate patch for {instance_id}")
            candidates[instance_id] = row["candidate_patch"]
    return candidates


class _CommandResult(Protocol):
    exit_code: int
    stderr: str


class _Workspace(Protocol):
    def execute_command(self, command: str) -> _CommandResult: ...

    def file_upload(self, source_path: str, destination_path: str) -> object: ...


def apply_candidate(workspace: _Workspace, repo_path: str, patch: str) -> None:
    """Apply ``patch`` to the repo as an uncommitted working-tree change.

    The patch is uploaded as a file (a command-line argument is size-limited). Files
    it creates are marked intent-to-add so they show up in the agent's ``git diff``;
    ``git apply -N`` is avoided because git 2.34 corrupts the index with it.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(patch)
        local_path = f.name
    try:
        workspace.file_upload(local_path, CANDIDATE_PATH_IN_CONTAINER)
    finally:
        os.unlink(local_path)
    patch_file = CANDIDATE_PATH_IN_CONTAINER
    res = workspace.execute_command(
        f"cd {repo_path} && git apply --whitespace=nowarn {patch_file} && "
        f"git apply --summary {patch_file} | sed -n 's/^ create mode [0-7]* //p' | "
        'while IFS= read -r f; do git add -N -- "$f"; done && '
        f"rm -f {patch_file}"
    )
    if res.exit_code != 0:
        raise RuntimeError(f"Candidate patch did not apply: {res.stderr[:500]}")


def has_verdict(conversation: BaseConversation) -> bool:
    """Whether the agent's latest answer (finish or plain message) has a verdict."""
    history = [e.model_dump(mode="json") for e in conversation.state.events]
    return parse_verdict(final_answer(history)) is not None


def request_verdict(conversation: RemoteConversation) -> None:
    """Judge task: remind the agent (at most twice) if it stopped without a verdict.

    Each reminder gets a single run, so an agent that answers with a plain message
    instead of calling ``finish`` is not pushed on by fake user responses.
    """
    for _ in range(MAX_VERDICT_REMINDERS):
        if has_verdict(conversation):
            return
        logger.info("No verdict line; sending a reminder")
        conversation.send_message(VERDICT_REMINDER)
        conversation.run()


def _with_candidates(
    instances: List[EvalInstance], candidates: dict[str, str]
) -> List[EvalInstance]:
    kept = [inst for inst in instances if inst.id in candidates]
    missing = len(candidates) - len(kept)
    if missing:
        logger.warning("%d candidates have no matching dataset instance", missing)
    logger.info("Instances with a candidate patch: %d", len(kept))
    return kept


class SWEBenchVerifyRepair(SWEBenchEvaluation):
    candidates: dict[str, str] = Field(default_factory=dict)
    task: str = "repair"

    def prepare_instances(self) -> List[EvalInstance]:
        return _with_candidates(super().prepare_instances(), self.candidates)

    def prepare_repo(
        self, workspace: RemoteWorkspace, instance: EvalInstance, repo_path: str
    ) -> None:
        apply_candidate(workspace, repo_path, self.candidates[instance.id])

    def fake_user_response(self, conversation: BaseConversation) -> str:
        # A plain-message answer with a verdict ends the judge task.
        if self.task == "judge" and has_verdict(conversation):
            return "/exit"
        return super().fake_user_response(conversation)

    def after_conversation(self, conversation: RemoteConversation) -> None:
        if self.task == "judge":
            request_verdict(conversation)


class R2EGymVerifyRepair(R2EGymEvaluation):
    candidates: dict[str, str] = Field(default_factory=dict)
    task: str = "repair"
    # The hidden tests would let the agent grade the candidate directly.
    hide_eval_artifacts: bool = True

    def prepare_instances(self) -> List[EvalInstance]:
        return _with_candidates(super().prepare_instances(), self.candidates)

    def prepare_repo(
        self, workspace: RemoteWorkspace, instance: EvalInstance, repo_path: str
    ) -> None:
        apply_candidate(workspace, repo_path, self.candidates[instance.id])

    def fake_user_response(self, conversation: BaseConversation) -> str:
        # A plain-message answer with a verdict ends the judge task.
        if self.task == "judge" and has_verdict(conversation):
            return "/exit"
        return super().fake_user_response(conversation)

    def after_conversation(self, conversation: RemoteConversation) -> None:
        if self.task == "judge":
            request_verdict(conversation)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--harness", choices=sorted(HARNESS_DEFAULTS), default="r2egym")
    pre.add_argument("--task", choices=sorted(TASK_PROMPTS), default="repair")
    pre_args = pre.parse_known_args()[0]

    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.add_argument(
        "--harness",
        choices=sorted(HARNESS_DEFAULTS),
        default="r2egym",
        help="Benchmark whose images, dataset and grader the task runs on",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_PROMPTS),
        default="repair",
        help="repair: verify and fix the candidate; judge: only give a verdict",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="JSONL with one {instance_id, candidate_patch} row per instance",
    )
    parser.add_argument(
        "--keep-base-image",
        action="store_true",
        help="swebench harness, docker workspace: keep official base images",
    )
    parser.set_defaults(**HARNESS_DEFAULTS[pre_args.harness])
    prompt_dir = Path(__file__).parent / "prompts"
    parser.set_defaults(
        prompt_path=str((prompt_dir / TASK_PROMPTS[pre_args.task]).resolve())
    )
    args = parser.parse_args()

    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    candidates = load_candidates(args.candidates)

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=(
            f"{TASK_DIR_PREFIX[args.task]}-{Path(args.candidates).stem}-"
            f"{args.dataset.replace('/', '__')}-{args.split.replace('/', '__')}"
        ),
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )
    selected_instances_file = args.select
    if selected_instances_file is None:
        # Restrict dataset loading (and --n-limit) to instances with a candidate.
        Path(structured_output_dir).mkdir(parents=True, exist_ok=True)
        selected_instances_file = str(Path(structured_output_dir) / "candidate_ids.txt")
        Path(selected_instances_file).write_text("\n".join(sorted(candidates)) + "\n")

    enable_condenser = args.enable_condenser and not args.disable_condenser
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={
            "task": "verifyrepair",
            "subtask": args.task,
            "candidates": args.candidates,
        },
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=create_critic(args),
        selected_instances_file=selected_instances_file,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    if args.harness == "swebench":
        evaluator = SWEBenchVerifyRepair(
            metadata=metadata,
            num_workers=args.num_workers,
            candidates=candidates,
            task=args.task,
            keep_base_image=args.keep_base_image,
        )
    else:
        evaluator = R2EGymVerifyRepair(
            metadata=metadata,
            num_workers=args.num_workers,
            candidates=candidates,
            task=args.task,
        )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))
    logger.info("Evaluation completed!")
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
