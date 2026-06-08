# evaluate:
# (1) the (file) localization accuracy of swe-bench
# (2) the non-empty rate of generated patch
# (3) tool call distribution analysis

# input: output.jsonl files
# the generated patch can be found in data[i]["test_result"]["git_patch"]
# instance id can be found in data[i]["instance_id"]

# the golden patch can be found in the hf dataset: SWE-bench/SWE-bench_Verified (split: test; golden patch key: data[i]["patch"])
# each golden patch is like: diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py\n--- a/astropy/modeling/separable.py\n+++ b/astropy/modeling/separable.py\n@@ -242,7 +242,7 @@ def _cstack(left, right):\ncright = _coord_matrix(right, 'right', noutp)\nelse:\ncright = np.zeros((noutp, right.shape[1]))\n- cright[-right.shape[0]:, -right.shape[1]:] = 1\n+ cright[-right.shape[0]:, -right.shape[1]:] = right\n\nreturn np.hstack([cleft, cright])\n
# instance id can be found in data[i]["instance_id"]

import argparse
import json
import os
import re
import shutil
from collections import defaultdict


def parse_git_patch(patch_text):
    """
    Parse a git patch and return a list of dictionaries for each modified file.

    Args:
        patch_text (str): The git patch text

    Returns:
        list: List of dictionaries, each containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)
    """
    # Split the patch into lines
    lines = patch_text.split("\n")

    files = []
    current_file = None
    added_lines = []
    removed_lines = []

    # Pattern to match file path in diff header
    file_pattern = r"^diff --git a/(.+) b/(.+)$"

    for line in lines:
        # Extract filename from diff header
        if line.startswith("diff --git"):
            # Save previous file if exists
            if current_file is not None:
                files.append(
                    {
                        "filename": current_file,
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                    }
                )

                current_file = None
                added_lines = []
                removed_lines = []

            # Start new file
            match = re.match(file_pattern, line)
            if match:
                current_file = match.group(2)  # Use the 'b/' path (new file)

        # Extract added lines (lines starting with +)
        elif (
            line.startswith("+")
            and not line.startswith("+++")
            and current_file is not None
        ):
            content = line[1:] if line.startswith("+") else line
            content = content.strip()
            if content != "":
                added_lines.append(content)

        # Extract removed lines (lines starting with -)
        elif (
            line.startswith("-")
            and not line.startswith("---")
            and current_file is not None
        ):
            content = line[1:] if line.startswith("-") else line
            content = content.strip()
            if content != "":
                removed_lines.append(content)

    # Don't forget the last file
    if current_file is not None:
        files.append(
            {
                "filename": current_file,
                "added_lines": added_lines,
                "removed_lines": removed_lines,
            }
        )

    return files


def check_add_comments_only(patch_dict):
    """
    Check if a patch dictionary only adds comments and doesn't remove any lines.

    Args:
        patch_dict (dict): Dictionary containing:
            - filename: The modified file path
            - added_lines: List of added lines (with + prefix)
            - removed_lines: List of removed lines (with - prefix)

    Returns:
        bool: True if only comments were added, False otherwise
    """

    # if patch_dict['is_new_file']:
    #     return False

    # if patch_dict['is_within_docstring']:
    #     return False

    # Create a set of removed lines (stripped) for matching
    removed_lines_stripped = {line.strip() for line in patch_dict["removed_lines"]}

    # Track which removed lines have been matched
    matched_removed_lines = set()

    # Check if all added lines are comments or match removed lines with comments appended
    for line in patch_dict["added_lines"]:
        line = line.strip()

        # Check if it's a pure comment (starts with # or is empty/whitespace)
        is_pure_comment = line.startswith("#") or line == "" or line.isspace()

        if is_pure_comment:
            continue

        # Check if it's a removed line with an inline comment appended
        # Split on '#' to separate code from comment
        if "#" in line:
            code_part = line.split("#", 1)[0].strip()
            # Check if the code part matches any removed line
            if code_part in removed_lines_stripped:
                matched_removed_lines.add(code_part)
                continue

        # Check if it exactly matches a removed line (same code, no comment added)
        if line in removed_lines_stripped:
            matched_removed_lines.add(line)
            continue

        # If it's neither a pure comment nor a removed line (with or without comment), it's not comment-only
        return False

    # Check if all removed lines are either comments or have been matched
    for line in patch_dict["removed_lines"]:
        line_stripped = line.strip()
        # If it's a comment or empty, it's fine
        if (
            line_stripped.startswith("#")
            or line_stripped == ""
            or line_stripped.isspace()
        ):
            continue
        # Otherwise, it must have been matched by an added line
        if line_stripped not in matched_removed_lines:
            return False

    return True


def patch2file_paths(patch):
    # get all file paths from the patch
    file_paths = set()

    # Parse git diff format to extract file paths
    # Look for lines like "diff --git a/path/to/file b/path/to/file"
    diff_lines = patch.split("\n")

    for line in diff_lines:
        if line.startswith("diff --git"):
            # Extract file path from "diff --git a/path/to/file b/path/to/file"
            parts = line.split()
            if len(parts) >= 4:
                # Remove the 'a/' prefix to get the actual file path
                file_path = parts[2][2:]  # Remove 'a/' prefix
                file_paths.add(file_path)

    return file_paths


# ---------------------------------------------------------------------------
# Format-agnostic helpers (support both old OH string-action format and new
# SDK dict-action format where action is a dict with a "kind" field)
# ---------------------------------------------------------------------------


def _get_action_type(turn):
    """Return a normalised action-type string ('run', 'edit', 'read', 'finish', …)."""
    action = turn.get("action")
    if action is None:
        return None
    if isinstance(action, str):
        return action  # old format
    if isinstance(action, dict):
        kind = action.get("kind", "")
        if kind == "TerminalAction":
            return "run"
        elif kind == "FileEditorAction":
            cmd = action.get("command", "")
            return "read" if cmd == "view" else "edit"
        elif kind == "FinishAction":
            return "finish"
        elif kind == "ThinkAction":
            return "think"
        else:
            return kind.lower().replace("action", "")
    return None


def _get_action_argument_str(turn):
    """Return a serialised string of the action arguments (for deduplication)."""
    action = turn.get("action")
    if isinstance(action, str):
        # old format: arguments live in tool_call_metadata
        try:
            arg = turn["tool_call_metadata"]["model_response"]["choices"][0]["message"][
                "tool_calls"
            ][0]["function"]["arguments"]
            return arg.replace("\n", "")
        except Exception:
            return ""
    if isinstance(action, dict):
        action_copy = {k: v for k, v in action.items() if k != "kind"}
        return json.dumps(action_copy, sort_keys=True).replace("\n", "")
    return ""


def _get_terminal_command(turn):
    """Return the shell command string for a 'run' / TerminalAction turn."""
    action = turn.get("action")
    if isinstance(action, str):
        arg_str = _get_action_argument_str(turn)
        try:
            return json.loads(arg_str)["command"].strip()
        except Exception:
            return ""
    if isinstance(action, dict):
        return action.get("command", "").strip()
    return ""


def _get_edit_path(turn):
    """Return the file path targeted by an edit/read action."""
    action = turn.get("action")
    if isinstance(action, str):
        try:
            return json.loads(
                turn["tool_call_metadata"]["model_response"]["choices"][0]["message"][
                    "tool_calls"
                ][0]["function"]["arguments"]
            )["path"]
        except Exception:
            return None
    if isinstance(action, dict):
        return action.get("path")
    return None


def _get_obs_success(obs_entry):
    """Return True if the observation indicates success."""
    if "observation" in obs_entry:
        # new SDK format
        return not obs_entry["observation"].get("is_error", False)
    if "success" in obs_entry:
        return obs_entry["success"]
    if "content" in obs_entry:
        return "ERROR" not in str(obs_entry.get("content", ""))
    return True


def _has_obs(obs_entry):
    """Return True if obs_entry contains an observation we can inspect."""
    return "observation" in obs_entry or "content" in obs_entry


def _get_agent_text(turn):
    """Return a best-effort text dump of an agent turn (thought + action content)."""
    parts = []
    # message field (old format)
    msg = turn.get("message")
    if msg:
        parts.append(str(msg))
    # thought (both formats)
    thought = turn.get("thought") or []
    if isinstance(thought, list):
        for item in thought:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
    # action content (new format)
    action = turn.get("action")
    if isinstance(action, dict):
        kind = action.get("kind", "")
        if kind == "TerminalAction":
            parts.append(action.get("command", ""))
        elif kind == "FileEditorAction":
            if action.get("path"):
                parts.append(action["path"])
    return " ".join(parts)


def _get_thought_text(turn):
    """Return the first thought text from a turn."""
    thought = turn.get("thought") or []
    if isinstance(thought, list):
        for item in thought:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
    return ""


# ---------------------------------------------------------------------------
# Analysis functions (updated for both formats)
# ---------------------------------------------------------------------------


def history_has_file_paths(history, file_path):
    for i in range(len(history) - 1):
        if history[i]["source"] == "agent":
            if file_path in _get_agent_text(history[i]):
                return True
    return False


def edit_history_has_file_paths(history, file_path):
    for i in range(len(history) - 1):
        if history[i]["source"] == "agent":
            if _get_action_type(history[i]) == "edit":
                edit_path = _get_edit_path(history[i])
                if edit_path and file_path in edit_path:
                    return True
    return False


def get_tool_call_distribution(history, instance_id=0, duplicate_thresh=3):
    """
    Analyze tool call distribution from history, filtering out repeated failed tool calls.
    Returns:
    - tool_call_counts: dict with tool type -> count of all calls
    - successful_tool_call_counts: dict with tool type -> count of successful calls
    """
    tool_call_counts = defaultdict(int)
    successful_tool_call_counts = defaultdict(int)

    # Track repeated failed calls to filter them out
    action_list = []
    agent_steps = []

    for turn in history:
        if turn["source"] == "agent" and "action" in turn:
            action = turn.get("action")
            if action is None:
                continue
            if isinstance(action, str) and action == "system":
                continue
            agent_steps.append(turn)

    for i in range(len(history) - 1):
        turn = history[i]
        if (
            turn["source"] != "agent"
            or "action" not in turn
            or turn.get("action") is None
        ):
            continue

        mapped = _get_action_type(turn)
        if mapped is None or mapped in ("think", "system"):
            continue

        action_argument = _get_action_argument_str(turn)
        if not action_argument:
            continue

        action_type = "oh_" + mapped
        action_key = f"{action_type}:{action_argument}".replace("\n", "").replace(
            "\\n", ""
        )

        if _has_obs(history[i + 1]):
            is_success = _get_obs_success(history[i + 1])

            if action_key not in [x[0] for x in action_list]:
                # only record once for each action
                if is_success:
                    # Successful call - always count
                    tool_call_counts[action_type] += 1
                    successful_tool_call_counts[action_type] += 1
                else:
                    # Failed call - only count if not repeated
                    tool_call_counts[action_type] += 1

        # Special handling for 'run' action to categorize unix commands
        if action_type == "oh_run":
            cmd = _get_terminal_command(turn)
            if not cmd:
                action_list.append((action_key, i))
                continue

            READ_CMD_LIST = [
                "cat ",
                "tail ",
                "head ",
                "grep ",
                "sed ",
                "awk ",
                "vi ",
                "vim ",
                "nano ",
            ]
            EDIT_CMD_LIST = ["sed ", "awk ", "vi ", "vim ", "nano "]
            EXECUTION_CMD_LIST = ["python ", "python3 ", "bash ", "sh ", "./"]

            if any(cmd.startswith(cmd_key) for cmd_key in READ_CMD_LIST):
                unix_type = "unix_read"
            elif (
                any(cmd.startswith(cmd_key) for cmd_key in EDIT_CMD_LIST)
                or ("cat " in cmd and " > " in cmd)
                or ("echo " in cmd and " > " in cmd)
            ):
                unix_type = "unix_edit"
            elif any(cmd.startswith(cmd_key) for cmd_key in EXECUTION_CMD_LIST):
                unix_type = "unix_exec"
            else:
                unix_type = "unix_other"

            # Count unix command types
            if _has_obs(history[i + 1]):
                is_success = _get_obs_success(history[i + 1])
                if action_key not in [x[0] for x in action_list]:
                    # only record once for each action
                    if is_success:
                        tool_call_counts[unix_type] += 1
                        successful_tool_call_counts[unix_type] += 1
                    else:
                        tool_call_counts[unix_type] += 1

        action_list.append((action_key, i))

    tool_call_counts["read"] = (
        tool_call_counts["oh_read"] + tool_call_counts["unix_read"]
    )
    tool_call_counts["edit"] = (
        tool_call_counts["oh_edit"] + tool_call_counts["unix_edit"]
    )

    successful_tool_call_counts["read"] = (
        successful_tool_call_counts["oh_read"]
        + successful_tool_call_counts["unix_read"]
    )
    successful_tool_call_counts["edit"] = (
        successful_tool_call_counts["oh_edit"]
        + successful_tool_call_counts["unix_edit"]
    )

    has_consecutive_duplicates_flag, start_idx, L = has_loop(
        [x[0] for x in action_list], min_consecutive=duplicate_thresh
    )

    # single_consecutive_duplicates_flag, single_start_idx, single_L = has_loop([x[0] for x in action_list], min_consecutive=duplicate_thresh, max_loop_length=1)
    # if has_consecutive_duplicates_flag and not single_consecutive_duplicates_flag:
    #     print(instance_id)
    #     print(action_list[start_idx:start_idx+L*duplicate_thresh])
    #     print()

    return (
        agent_steps,
        tool_call_counts,
        successful_tool_call_counts,
        has_consecutive_duplicates_flag,
    )


def has_loop(action_list, min_consecutive=3, max_loop_length=1):
    n = len(action_list)
    for L in range(1, min(n // min_consecutive + 1, max_loop_length + 1)):
        for start_idx in range(n - L * min_consecutive + 1):
            subarray = action_list[start_idx : start_idx + L]
            pattern = subarray * min_consecutive
            if action_list[start_idx : start_idx + L * min_consecutive] == pattern:
                return True, start_idx, L

    return False, None, None


def never_call_tools(history):
    never_call_tools_flag = True
    no_tool_call_step_num = 0
    for i in range(1, len(history) - 1):
        turn = history[i]
        if (
            turn["source"] != "agent"
            or "action" not in turn
            or turn.get("action") is None
        ):
            continue
        mapped = _get_action_type(turn)
        if mapped in ("condensation", "system", None):
            continue
        if mapped == "think":
            no_tool_call_step_num += 1
        else:
            never_call_tools_flag = False
    return never_call_tools_flag, no_tool_call_step_num


def get_avg_word_count(history):
    word_counts = []
    for i in range(4, len(history)):
        turn = history[i]
        if (
            turn["source"] != "agent"
            or "action" not in turn
            or turn.get("action") is None
        ):
            continue
        mapped = _get_action_type(turn)
        if mapped in ("condensation", "system", None):
            continue
        text = _get_thought_text(turn)
        if text:
            word_counts.append(len(text.split()))

    return sum(word_counts) / len(word_counts) if len(word_counts) > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="/data/tir/projects/tir5/users/yiqingxi/benchmarks/evaluation_outputs/swe_bench_easy50_outputs/princeton-nlp__SWE-bench_Verified-test/openai/qwen25-coder-7b-func-localize-claude47-1467i-5e-0-00005lr-bs16-bf16_sdk_e212d45_maxiter_60/output.jsonl",
    )
    parser.add_argument("--total_num", type=int, default=50)
    parser.add_argument("--eval_limit", type=int, default=-1)
    parser.add_argument("--split", type=str, default="easy50")
    parser.add_argument("--update_file", action="store_true", default=False)
    args = parser.parse_args()

    SUBSET_IDS = []
    if args.split == "easy50":
        with open("benchmarks/swebench/easy50_instances.txt", "r") as f:
            for line in f:
                SUBSET_IDS.append(line.strip())

    instance_id2golden_patch = {}
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            try:
                instance_id2golden_patch[data["instance_id"]] = data["instance"][
                    "patch"
                ]
            except Exception:
                continue

    # Fallback: load golden patches from local cache (new format has instance=null)
    if not instance_id2golden_patch:
        golden_patch_file = os.path.join(
            os.path.dirname(__file__), "golden_patches.json"
        )
        with open(golden_patch_file, "r") as f:
            instance_id2golden_patch = json.load(f)

    # extract all file paths from the golden patch
    instance_id2file_paths = {}
    for instance_id, golden_patch in instance_id2golden_patch.items():
        file_paths = patch2file_paths(golden_patch)
        instance_id2file_paths[instance_id] = file_paths

    # load swe-bench-eval file (if any)
    swe_bench_eval_file = args.input_file.replace(".jsonl", ".swebench_eval.jsonl")
    id2resolved = {}
    if os.path.exists(swe_bench_eval_file):
        with open(swe_bench_eval_file, "r") as f:
            for line in f:
                data = json.loads(line)
                id2resolved[data["instance_id"]] = data["test_result"]["report"][
                    "resolved"
                ]

    local_eval_file = os.path.dirname(args.input_file) + "/report.json"
    if os.path.exists(local_eval_file):
        with open(local_eval_file, "r") as f:
            local_eval_data = json.load(f)
        for instance_id in local_eval_data["resolved_ids"]:
            id2resolved[instance_id] = True

    # load the file_path from generated patch
    num_lines = 0
    resolved_count = 0
    non_empty_count = 0
    correct_coarse_localization_count = 0
    correct_edit_localization_count = 0
    correct_localization_count = 0
    stuck_in_loop_count = 0
    never_call_tools_count = 0
    comment_only_count = 0
    total_count = 0
    non_empty_set = set()
    success_lozalization_set = set()
    stuck_in_loop_set = set()
    trajectory_length = []
    agent_steps_length = []
    infra_error_count = 0
    total_no_tool_call_step_num = 0
    avg_word_count_list = []

    # Tool call distribution analysis
    tool_type_list = [
        "read",
        "edit",
        "oh_run",
        "oh_read",
        "oh_edit",
        "unix_read",
        "unix_edit",
        "unix_exec",
        "unix_other",
    ]
    per_instance_success_dict = {tool_type: 0 for tool_type in tool_type_list}
    global_tool_call_counts = defaultdict(int)
    global_successful_tool_call_counts = defaultdict(int)

    lines_with_results = []
    with open(args.input_file, "r") as f:
        for line_idx, line in enumerate(f):
            if args.eval_limit > 0 and line_idx >= args.eval_limit:
                break
            data = json.loads(line)
            num_lines += 1
            try:
                instance_id = data["instance_id"]
                generated_patch = data["test_result"]["git_patch"]
                if len(SUBSET_IDS) > 0:
                    if instance_id not in SUBSET_IDS:
                        continue
            except Exception:
                # print("Error: git_patch not found in data")
                # print(data["instance_id"])
                continue

            if len(data["history"]) <= 4:
                continue

            patch_dicts = parse_git_patch(generated_patch)
            comment_only_flag = False
            for patch_dict in patch_dicts:
                if patch_dict["filename"] in instance_id2file_paths.get(
                    instance_id, set()
                ):
                    if check_add_comments_only(patch_dict):
                        comment_only_flag = True
                        break

            if comment_only_flag:
                comment_only_count += 1

            never_call_tools_flag, no_tool_call_step_num = never_call_tools(
                data["history"]
            )
            if never_call_tools_flag:
                never_call_tools_count += 1
                # from IPython import embed; embed(); exit()

            total_no_tool_call_step_num += no_tool_call_step_num

            avg_word_count = get_avg_word_count(data["history"])
            if avg_word_count > 0:
                avg_word_count_list.append(avg_word_count)

            if instance_id in id2resolved:
                resolved_count += id2resolved[instance_id]

            # Get golden file paths for this instance
            golden_file_paths = instance_id2file_paths.get(instance_id, set())

            coarse_localization_flag = False
            for file_path in golden_file_paths:
                if history_has_file_paths(data["history"], file_path):
                    coarse_localization_flag = True
                    break

            correct_coarse_localization_count += coarse_localization_flag

            edit_localization_flag = False
            for file_path in golden_file_paths:
                if edit_history_has_file_paths(data["history"], file_path):
                    edit_localization_flag = True
                    break

            correct_edit_localization_count += edit_localization_flag

            # localization accuracy
            if generated_patch:
                non_empty_count += 1
                non_empty_set.add(instance_id)

                # Get generated file paths
                generated_file_paths = patch2file_paths(generated_patch)

                # Check if any generated file path matches any golden file path
                if golden_file_paths and generated_file_paths:
                    # Check for intersection between golden and generated file paths
                    intersection = golden_file_paths.intersection(generated_file_paths)
                    if intersection:
                        correct_localization_count += 1
                        success_lozalization_set.add(instance_id)

            # Tool call distribution analysis
            (
                agent_steps,
                tool_call_counts,
                successful_tool_call_counts,
                has_consecutive_duplicates_flag,
            ) = get_tool_call_distribution(data["history"], data["instance_id"])
            if has_consecutive_duplicates_flag:
                stuck_in_loop_count += 1
                stuck_in_loop_set.add(instance_id)
            # if data["error"] is not None and "loop" in data["error"].lower():
            #     stuck_in_loop_count += 1
            #     stuck_in_loop_set.add(instance_id)

            # Aggregate global counts
            for tool_type, count in tool_call_counts.items():
                global_tool_call_counts[tool_type] += count

            for tool_type, count in successful_tool_call_counts.items():
                global_successful_tool_call_counts[tool_type] += count
                if count > 0 and tool_type in per_instance_success_dict:
                    per_instance_success_dict[tool_type] += 1

            infra_error_flag = False
            total_count += 1
            trajectory_length.append(len(data["history"]))
            agent_steps_length.append(len(agent_steps))
            if len(agent_steps) > 0:
                if _get_action_type(agent_steps[-1]) != "finish":
                    if (
                        data["error"] is not None
                        and "loop" not in data["error"].lower()
                        and "maximum iteration" not in data["error"].lower()
                    ):
                        infra_error_count += 1
                        infra_error_flag = True
            if not infra_error_flag:
                lines_with_results.append(line)

    if args.update_file:
        print(
            f"{len(lines_with_results)} lines with results after filtering out infra errors"
        )
        shutil.copy(args.input_file, args.input_file.replace(".jsonl", ".backup.jsonl"))
        with open(args.input_file, "w") as f:
            for line in lines_with_results:
                f.write(line)

    # Calculate and print results
    if total_count > 0:
        if args.total_num > 0:
            total_count = args.total_num
        accuracy = correct_localization_count / total_count

        print()
        ckpt_name = args.input_file.split("/")[-2].split("-cp32768ctx")[0]
        print(f"Checkpoint: {ckpt_name}")
        print(f"File: {args.input_file}")
        print(f"Total count: {total_count} ({num_lines} lines)")
        print()
        if len(id2resolved) == 0:
            resolved_count = "?"
        print(
            f"{resolved_count} resolved / {correct_localization_count} localized (file) / {non_empty_count} non-empty"
        )
        # print(f"{correct_edit_localization_count} edit-attempted / {correct_coarse_localization_count} appeared (file) / {stuck_in_loop_count} have loop / {never_call_tools_count} never call tools")
        print(
            f"{correct_edit_localization_count} edit-attempted / {correct_coarse_localization_count} appeared (file) / {stuck_in_loop_count} have loop / {never_call_tools_count} never call tools / {comment_only_count} localized but comment only"
        )
        print("Tool:", end=" ")
        print(
            f"run: {per_instance_success_dict['oh_run']}/{total_count} (Exec: {per_instance_success_dict['unix_exec']}/{total_count})"
        )
        print(
            f"read: {per_instance_success_dict['read']}/{total_count} (OH: {per_instance_success_dict['oh_read']}/{total_count}, Unix: {per_instance_success_dict['unix_read']}/{total_count})"
        )
        print(
            f"edit: {per_instance_success_dict['edit']}/{total_count} (OH: {per_instance_success_dict['oh_edit']}/{total_count}, Unix: {per_instance_success_dict['unix_edit']}/{total_count})"
        )
        print()

        print()
        print(
            f"Avg Trajectory Length: {sum(trajectory_length) / len(trajectory_length):.4f}"
        )
        print(
            f"Avg Word Count: {sum(avg_word_count_list) / len(avg_word_count_list):.4f}"
        )
        print(f"Total No Tool Call Step Num: {total_no_tool_call_step_num}")
        # print(f"Avg Agent Steps: {sum(agent_steps_length) / len(agent_steps_length):.4f}")
        print(f"Infra Error Count: {infra_error_count}")
        print()
