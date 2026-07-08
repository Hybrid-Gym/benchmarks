import json
import os
import subprocess
from typing import Any, List

from jinja2 import Environment, FileSystemLoader

from benchmarks.r2egym import constants
from benchmarks.r2egym.build_images import (
    extract_custom_tag,
    get_official_docker_image,
)
from benchmarks.r2egym.config import INFER_DEFAULTS
from benchmarks.r2egym.dataset import get_dataset
from benchmarks.utils.acp import (
    add_acp_agent_metadata,
    build_acp_agent,
    get_acp_forward_env,
    is_acp_agent,
    setup_acp_workspace,
    workspace_keepalive,
)
from benchmarks.utils.args_parser import add_prompt_path_argument, get_parser
from benchmarks.utils.build_utils import ensure_local_image
from benchmarks.utils.console_logging import summarize_instance
from benchmarks.utils.constants import EVAL_AGENT_SERVER_IMAGE
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.image_utils import remote_image_exists
from benchmarks.utils.litellm_proxy import build_eval_llm
from benchmarks.utils.llm_config import load_llm_config
from benchmarks.utils.models import (
    EvalInstance,
    EvalMetadata,
    EvalOutput,
    ToolPresetType,
)
from benchmarks.utils.version import get_phased_image_tag_prefix
from openhands.sdk import Agent, Conversation, Tool, get_logger
from openhands.sdk.agent import ACPAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.workspace import RemoteWorkspace
from openhands.tools.delegate import DelegateTool
from openhands.workspace import APIRemoteWorkspace, ApptainerWorkspace, DockerWorkspace


logger = get_logger(__name__)


def _fix_r2egym_permissions(container_id: str) -> None:
    """Repair R2E-Gym's root-owned environment for the openhands agent-server.

    R2E-Gym images run as root: the repo venv python lives under /root and
    /testbed is root-owned, so the openhands user (uid 10001) cannot use the
    prepared venv (already first on PATH) nor edit the repo. Make the uv python
    readable and hand /testbed to openhands. Docker workspace only.
    """
    if not container_id:
        logger.warning("[r2egym] no container id; skipping permission fix")
        return
    fix = (
        "chmod a+rx /root /root/.local /root/.local/share 2>/dev/null || true; "
        "chmod -R a+rX /root/.local/share/uv 2>/dev/null || true; "
        "chown -R openhands:openhands /testbed 2>/dev/null || true; "
        "true"
    )
    res = subprocess.run(
        ["docker", "exec", "--user", "root", container_id, "sh", "-c", fix],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        logger.warning(
            "[r2egym] permission fix exited %s: %s",
            res.returncode,
            (res.stderr or "").strip()[:300],
        )


def get_tools_for_preset(
    preset: ToolPresetType, enable_browser: bool = False
) -> list[Tool]:
    """Get the list of tools for the given preset."""
    from openhands.tools.preset.default import get_default_tools

    return get_default_tools(enable_browser=enable_browser)


def get_instruction(
    instance: dict,
    metadata: EvalMetadata,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    # R2E-Gym stores the bare repo name (e.g. "aiohttp"), not "owner/name".
    workspace_dir_name = instance["repo_name"]
    assert metadata.details is not None

    # Set up Jinja2 environment
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "workspace_dir_name": workspace_dir_name,
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction


class R2EGymEvaluation(Evaluation):
    """
    Process-based R2E-Gym evaluation implemented as a child of the abstract
    Evaluation orchestrator.

    Mirrors the SWE-Bench evaluator, with the R2E-Gym-specific differences:
      - the base docker image is read from the dataset (``docker_image`` column)
        rather than derived from the instance id,
      - instances are keyed on ``repo_name`` (no "owner/name" split),
      - the base commit for the final diff is captured at runtime (R2E-Gym does
        not ship one), and
      - the repo inside every image lives at ``/testbed``.

    Implements:
      - prepare_instances()
      - prepare_workspace(instance)
      - evaluate_instance(instance, workspace)
    """

    def get_official_docker_image(self, instance: EvalInstance) -> str:
        return get_official_docker_image(instance.data)

    def extract_custom_tag(self, official_docker_image: str) -> str:
        return extract_custom_tag(official_docker_image)

    def get_source_repo_path(self, instance: EvalInstance) -> str:
        return constants.REPO_PATH_IN_IMAGE

    def prepare_instances(self) -> List[EvalInstance]:
        logger.info("Setting up R2E-Gym evaluation data")

        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )

        instances: List[EvalInstance] = []
        for _, row in df.iterrows():
            inst_id = str(row["instance_id"])
            instances.append(EvalInstance(id=inst_id, data=row.to_dict()))

        logger.info("Total instances to process: %d", len(instances))
        return instances

    # ---- Hook: prepare a workspace per instance ----------------------------------
    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        """Use DockerWorkspace by default; also supports apptainer / remote."""
        forward_env = get_acp_forward_env(self.metadata.agent_type, forward_env)

        official_docker_image = self.get_official_docker_image(instance)
        build_target = constants.DEFAULT_BUILD_TARGET
        custom_tag = self.extract_custom_tag(official_docker_image)
        # For non-binary targets, append target suffix
        suffix = (
            f"-{build_target}" if build_target != constants.BUILD_TARGET_BINARY else ""
        )
        agent_server_image = (
            f"{EVAL_AGENT_SERVER_IMAGE}:"
            f"{get_phased_image_tag_prefix()}-{custom_tag}{suffix}"
        )

        if self.metadata.workspace_type == "docker":
            ensure_local_image(
                agent_server_image=agent_server_image,
                base_image=official_docker_image,
                custom_tag=custom_tag,
                target=build_target,
            )

            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                # Reclaim disk: delete the built agent-server image when the
                # workspace is cleaned up. The per-instance R2E-Gym base image it
                # was built FROM is removed separately in _cleanup_workspace.
                cleanup_image=True,
            )
            # R2E-Gym images assume a root runtime; repair permissions so the
            # openhands user can use the prepared venv and edit /testbed in place.
            _fix_r2egym_permissions(getattr(workspace, "_container_id", ""))
        elif self.metadata.workspace_type == "apptainer":
            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in "
                    "container registry, make sure to build, push it, and make it "
                    "public accessible before using apptainer workspace."
                )

            logger.info(
                f"Using apptainer workspace with pre-built image {agent_server_image} "
                f"(tag prefix: {get_phased_image_tag_prefix()})"
            )

            workspace = ApptainerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
                forward_env=forward_env or [],
                cache_dir=os.getenv("APPTAINER_CACHEDIR", None),
            )
        elif self.metadata.workspace_type == "remote":
            runtime_api_key = os.getenv("RUNTIME_API_KEY")
            if not runtime_api_key:
                raise ValueError(
                    "RUNTIME_API_KEY environment variable is not set for remote "
                    "workspace"
                )

            if not remote_image_exists(agent_server_image):
                raise RuntimeError(
                    f"Agent server image {agent_server_image} does not exist in "
                    "container registry, make sure to build, push it, and make it "
                    "public accessible before using remote workspace."
                )
            logger.info(
                f"Using remote workspace with image {agent_server_image} "
                f"(tag prefix: {get_phased_image_tag_prefix()}, "
                f"resource_factor: {resource_factor})"
            )
            startup_timeout = float(
                os.getenv(
                    "REMOTE_RUNTIME_STARTUP_TIMEOUT",
                    str(constants.DEFAULT_REMOTE_RUNTIME_STARTUP_TIMEOUT),
                )
            )
            workspace = APIRemoteWorkspace(
                runtime_api_url=os.getenv(
                    "RUNTIME_API_URL", constants.DEFAULT_RUNTIME_API_URL
                ),
                runtime_api_key=runtime_api_key,
                server_image=agent_server_image,
                target_type="source" if "source" in build_target else "binary",
                forward_env=forward_env or [],
                resource_factor=resource_factor,
                init_timeout=startup_timeout,
                startup_wait_timeout=startup_timeout,
            )
        else:
            raise ValueError(
                f"Unsupported workspace_type: {self.metadata.workspace_type}"
            )

        for cmd in self.metadata.env_setup_commands or []:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

    # ---- Hook: reclaim per-instance disk after each instance ---------------------
    def _cleanup_workspace(
        self,
        workspace: RemoteWorkspace,
        instance: EvalInstance,
        *,
        capture_archive: bool = True,
    ) -> None:
        # Capture the built agent-server image ID BEFORE cleanup untags it. The
        # local phased build tags the SAME image twice (the requested tag plus a
        # content-hashed assembly tag); cleanup_image=True only removes one tag,
        # leaving the multi-GB image alive via the other. Removing by image ID
        # afterwards drops all remaining tags at once.
        image_id: str | None = None
        if self.metadata.workspace_type == "docker":
            server_image = getattr(workspace, "_image_name", None) or getattr(
                workspace, "server_image", None
            )
            if server_image:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", "-f", "{{.Id}}", server_image],
                    capture_output=True,
                    text=True,
                )
                if inspect.returncode == 0:
                    image_id = inspect.stdout.strip()

        # Base cleanup stops the container and (with cleanup_image=True) removes
        # the primary agent-server tag for this instance.
        super()._cleanup_workspace(workspace, instance, capture_archive=capture_archive)

        # Reclaim the rest of the per-instance disk (docker mode only): any
        # remaining agent-server tags (by ID) and the R2E-Gym base image the
        # agent-server image was built FROM. Each R2E-Gym instance uses a unique
        # image, so this is safe and prevents disk from filling across a run.
        # We target exact images only (by ID or exact tag), never a wildcard, so
        # unrelated images on a shared host are untouched.
        if self.metadata.workspace_type != "docker":
            return
        images_to_remove: list[str] = []
        if image_id:
            images_to_remove.append(image_id)
        try:
            images_to_remove.append(self.get_official_docker_image(instance))
        except Exception as e:
            logger.warning(
                "[cleanup] could not resolve base image for %s: %s", instance.id, e
            )
        for image_ref in images_to_remove:
            try:
                result = subprocess.run(
                    ["docker", "rmi", "-f", image_ref],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info(
                        "[cleanup] removed image %s for %s", image_ref, instance.id
                    )
                else:
                    logger.warning(
                        "[cleanup] failed to remove image %s for %s: %s",
                        image_ref,
                        instance.id,
                        result.stderr.strip(),
                    )
            except Exception as e:
                logger.warning(
                    "[cleanup] error removing image %s for %s: %s",
                    image_ref,
                    instance.id,
                    e,
                )

    # ---- Hook: evaluate one instance ---------------------------------------------
    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        """
        Create conversation, run agent, collect history and git patch.
        Do not write files here; just return EvalOutput.
        """
        if is_acp_agent(self.metadata.agent_type):
            agent = build_acp_agent(self.metadata.agent_type, self.metadata.llm.model)
        else:
            agent_llm = build_eval_llm(self.metadata.llm)
            tools = get_tools_for_preset(
                preset=self.metadata.tool_preset,
                # Disable browser tools in CLI mode
                enable_browser=False,
            )
            if self.metadata.enable_delegation:
                tools.append(Tool(name=DelegateTool.name))
            condenser = None
            if self.metadata.enable_condenser:
                condenser = LLMSummarizingCondenser(
                    llm=build_eval_llm(self.metadata.llm, usage_id="condenser"),
                    max_size=self.metadata.condenser_max_size,
                    keep_first=self.metadata.condenser_keep_first,
                )
            # Load public skills (respects EXTENSIONS_REF env var)
            _system_prompt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils",
                "prompts",
                "system_prompt_old.j2",
            )
            with open(_system_prompt_path) as _f:
                _system_prompt = _f.read()
            agent = Agent(
                llm=agent_llm,
                tools=tools,
                system_prompt=_system_prompt,
                condenser=condenser,
            )

        assert isinstance(workspace, RemoteWorkspace)

        setup_acp_workspace(self.metadata.agent_type, workspace)

        # Edit the repo in place at /testbed. R2E-Gym installs the repo editable
        # into /testbed/.venv, so working on a copy elsewhere would decouple the
        # agent's edits from the environment it runs against.
        repo_path = self.get_source_repo_path(instance)  # /testbed
        instance.data["repo_path"] = repo_path

        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
            delete_on_close=True,
        )

        logger.info("repo_path: %s", repo_path)

        # /testbed ships with uncommitted R2E setup changes and untracked harness
        # files. Snapshot them as the diff base (no git reset, which would revert
        # the intended base state) so the patch is only the agent's net changes.
        snapshot = workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global --add safe.directory {repo_path} && "
            f"git config --global user.email '{constants.GIT_USER_EMAIL}' && "
            f"git config --global user.name '{constants.GIT_USER_NAME}' && "
            f"git add -A && "
            f"git commit --no-verify --allow-empty "
            f"-m '{constants.GIT_BASE_SNAPSHOT_MESSAGE}'"
        )
        assert snapshot.exit_code == 0, f"base snapshot failed: {snapshot.stderr}"

        # R2E-Gym ships no base_commit; the snapshot HEAD is our diff base.
        rev_parse = workspace.execute_command(f"cd {repo_path} ; git rev-parse HEAD")
        assert rev_parse.exit_code == 0, f"git rev-parse failed: {rev_parse.stderr}"
        base_commit = rev_parse.stdout.strip()
        instance.data["base_commit"] = base_commit

        instruction = get_instruction(
            instance=instance.data,
            metadata=self.metadata,
            workspace_path=workspace.working_dir,
        )
        with workspace_keepalive(self.metadata.agent_type, workspace):
            conversation.send_message(instruction)
            # Run conversation with fake user responses to handle agent messages
            run_conversation_with_fake_user_response(conversation)

        # Drop test-run byproducts the agent created (pyc / caches) so they don't
        # leak into the patch on repos that don't gitignore them.
        workspace.execute_command(
            f"cd {repo_path} && "
            "find . -name '*.pyc' -delete 2>/dev/null; "
            "find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null; "
            "rm -rf .pytest_cache 2>/dev/null; true"
        )

        # git add
        workspace.execute_command(f"cd {repo_path} ; git add -A")

        # git commit
        # Use --no-verify to bypass pre-commit hooks (e.g., husky) that can fail
        workspace.execute_command(
            f"cd {repo_path} && "
            f"git config --global user.email '{constants.GIT_USER_EMAIL}' && "
            f"git config --global user.name '{constants.GIT_USER_NAME}' && "
            f"git commit --no-verify -m '{constants.GIT_COMMIT_MESSAGE}'"
        )

        # Get git patch (diff against the captured base commit)
        git_patch_result = workspace.execute_command(
            (f"cd {repo_path} ; git --no-pager diff --no-color {base_commit} HEAD")
        )
        assert git_patch_result.exit_code == 0, (
            f"git diff failed: {git_patch_result.stderr}"
        )
        git_patch = git_patch_result.stdout

        # Log instance summary
        summarize_instance(
            instance_id=instance.id,
            conversation=conversation,
            git_patch=git_patch,
            logger=logger,
        )

        # Build test_result with git patch and optional ACP agent metadata
        test_result: dict[str, Any] = {
            "git_patch": git_patch,
        }
        if isinstance(agent, ACPAgent):
            add_acp_agent_metadata(test_result, conversation)

        # EvalOutput is your model; keep fields consistent with prior JSONL
        out = EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result=test_result,
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
        )
        return out


def main() -> None:
    parser = get_parser()
    add_prompt_path_argument(parser, __file__)
    parser.set_defaults(**INFER_DEFAULTS)
    args = parser.parse_args()

    # Validate n_critic_runs
    if args.n_critic_runs < 1:
        raise ValueError(f"n_critic_runs must be >= 1, got {args.n_critic_runs}")

    llm = load_llm_config(args.llm_config_path)
    logger.info("Using LLM config: %s", llm.model_dump_json(indent=2))

    dataset_description = (
        args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
    )

    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=dataset_description,
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )

    # Create critic instance from parsed arguments
    critic = create_critic(args)
    logger.info(f"Using critic: {type(critic).__name__}")
    logger.info(f"Using tool preset: {args.tool_preset}")

    # Handle condenser configuration
    # --disable-condenser takes precedence over --enable-condenser and defaults
    enable_condenser = args.enable_condenser
    if args.disable_condenser:
        enable_condenser = False

    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=["export PIP_CACHE_DIR=~/.cache/pip"],
        n_critic_runs=args.n_critic_runs,
        critic=critic,
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        tool_preset=args.tool_preset,
        enable_delegation=args.enable_delegation,
        agent_type=args.agent_type,
        enable_condenser=enable_condenser,
        condenser_max_size=args.condenser_max_size,
        condenser_keep_first=args.condenser_keep_first,
    )

    # Run orchestrator with a simple JSONL writer
    evaluator = R2EGymEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
    )

    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))

    logger.info("Evaluation completed!")
    # Emit machine-readable path for callers
    print(json.dumps({"output_json": str(evaluator.output_path)}))


if __name__ == "__main__":
    main()
