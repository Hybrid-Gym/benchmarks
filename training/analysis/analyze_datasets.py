"""
Analyze two HuggingFace trajectory datasets:
  - synthetic-code-training/func_localize_gpt5mini_1346i
  - synthetic-code-training/func_localize_gpt5mini_loc_strategy_1289i

Statistics computed per dataset:
  (1) % trajectories with >1 keyword search actions BEFORE first file-editing action
  (2) % trajectories with >1 keyword search actions overall
  (3) % trajectories where: before the first file-read/edit action, the target file
      appears in at least one prior keyword search output, AND that search has <=50
      unique files in its output
  (4) Same as (3) but threshold <=100
  (5) Same as (3) but threshold <=30
"""

import re
import json
import sys
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Parsing helpers (adapted from extract_localization_steps.py)
# ---------------------------------------------------------------------------

_XML_CALL_RE = re.compile(
    r"<function=(?P<tool>\w+)>\n?(?P<body>.*?)</function>",
    re.DOTALL,
)
_XML_PARAM_RE = re.compile(
    r"<parameter=(?P<key>\w+)>(?P<value>.*?)</parameter>",
    re.DOTALL,
)
_RESULT_RE = re.compile(
    r"^EXECUTION RESULT of \[(?P<tool>\w+)\]:\n?(?P<output>.*)",
    re.DOTALL | re.MULTILINE,
)


class ToolCall:
    def __init__(self, tool, params, result=None):
        self.tool = tool
        self.params = params
        self.result = result


def parse_xml_calls(content):
    calls = []
    for m in _XML_CALL_RE.finditer(content):
        tool = m.group("tool")
        body = m.group("body")
        params = {pm.group("key"): pm.group("value").strip()
                  for pm in _XML_PARAM_RE.finditer(body)}
        calls.append(ToolCall(tool=tool, params=params))
    return calls


def parse_xml_result(content):
    m = _RESULT_RE.match(content.strip())
    return m.group("output").strip() if m else None


def try_parse_json_calls(content):
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    calls_raw = obj.get("function_calls") or obj.get("tool_calls") or []
    if not calls_raw:
        return []
    calls = []
    for c in calls_raw:
        tool = (c.get("tool") or c.get("name") or
                c.get("function", {}).get("name", ""))
        params = c.get("parameters") or c.get("arguments") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"raw": params}
        calls.append(ToolCall(tool=tool, params=params))
    return calls


def extract_tool_calls(messages):
    """Walk messages, pair assistant tool calls with their results."""
    calls = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "assistant":
            new_calls = try_parse_json_calls(content)
            if not new_calls:
                new_calls = parse_xml_calls(content)

            if new_calls:
                result_text = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                    result_text = parse_xml_result(
                        messages[i + 1].get("content", "") or ""
                    )
                    if result_text is None:
                        result_text = (messages[i + 1].get("content") or "").strip()
                for c in new_calls:
                    c.result = result_text
                calls.extend(new_calls)
        i += 1
    return calls


# ---------------------------------------------------------------------------
# Action classification
# ---------------------------------------------------------------------------

def is_keyword_search(call):
    """
    Keyword search = terminal grep/rg command, or find -name command.
    These are actions that search for keywords/patterns across files.
    """
    if call.tool.lower() != "terminal":
        return False
    cmd = (call.params.get("command") or call.params.get("cmd") or
           call.params.get("input") or "").strip()
    if not cmd:
        return False
    cmd_lower = cmd.lower()
    if re.search(r'\b(grep|rg)\b', cmd_lower):
        return True
    if re.search(r'\bfind\b.*-name\b', cmd_lower):
        return True
    return False


def is_file_read(call):
    """File read = file_editor view, or terminal cat/head/tail/less/more."""
    tool = call.tool.lower()
    if tool == "file_editor":
        return (call.params.get("command") or "") == "view"
    if tool == "terminal":
        cmd = (call.params.get("command") or call.params.get("cmd") or
               call.params.get("input") or "").strip()
        if re.search(r'\b(cat|head|tail|less|more)\b', cmd.lower()):
            return True
    return False


def is_file_edit(call):
    """File edit = file_editor str_replace/insert/create, or terminal sed -i / git apply."""
    tool = call.tool.lower()
    if tool == "file_editor":
        return (call.params.get("command") or "") in ("str_replace", "insert", "create")
    if tool == "terminal":
        cmd = (call.params.get("command") or call.params.get("cmd") or
               call.params.get("input") or "").strip()
        cmd_lower = cmd.lower()
        if re.search(r'\bsed\s+-i\b', cmd_lower):
            return True
        if re.search(r'\bgit\s+apply\b', cmd_lower):
            return True
    return False


def get_file_path(call):
    """Extract file path from a file read or edit action."""
    tool = call.tool.lower()
    if tool == "file_editor":
        return call.params.get("path") or None
    if tool == "terminal":
        cmd = (call.params.get("command") or call.params.get("cmd") or
               call.params.get("input") or "").strip()
        m = re.search(r'\b(?:cat|head|tail|less|more)\s+([^\s|><&;]+)', cmd)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Grep output parsing
# ---------------------------------------------------------------------------

def extract_files_from_search_output(output, cmd=""):
    """
    Parse grep/rg/find output and return the set of unique file paths mentioned.

    grep/rg line formats:
      - path/to/file.py:lineno:content
      - path/to/file.py:content
      - path/to/file.py          (grep -l or find output)
      - Binary file path/to/file matches
    """
    if not output:
        return set()

    files = set()
    cmd_lower = (cmd or "").lower()
    is_find = re.search(r'\bfind\b', cmd_lower) and not re.search(r'\b(grep|rg)\b', cmd_lower)

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        # "Binary file X matches"
        m = re.match(r'Binary file (.+?) matches', line)
        if m:
            files.add(m.group(1).strip())
            continue

        # For find output: each non-empty line is a file path
        if is_find:
            # Skip lines that look like error messages
            if not line.startswith('/') and not line.startswith('./') and not line.startswith('../'):
                if '/' not in line and not re.match(r'^\S+\.\w+$', line):
                    continue
            files.add(line)
            continue

        # grep/rg output: "filepath:..." or just "filepath"
        # Try to match "path/to/file.ext:..."
        m = re.match(r'^((?:[./\w\-][\w./\-]*?)(?:\.\w+))(?::\d+)?:', line)
        if m:
            files.add(m.group(1))
            continue

        # Plain file path (grep -l output)
        if re.match(r'^[./\w][\w./\-]*\.\w+$', line):
            files.add(line)

    return files


def file_in_output(target_file, files_set, raw_output):
    """
    Check whether target_file appears in a keyword search result.
    Matches on full path, suffix match, or basename match.
    """
    if not target_file:
        return False

    target_basename = target_file.split('/')[-1]

    # Direct match
    if target_file in files_set:
        return True

    # Suffix / prefix match against parsed files
    for f in files_set:
        f_basename = f.split('/')[-1]
        if (f.endswith(target_file) or target_file.endswith(f) or
                f_basename == target_basename):
            return True

    # Fallback: check raw output text
    if target_file in (raw_output or ""):
        return True
    if target_basename and target_basename in (raw_output or ""):
        return True

    return False


# ---------------------------------------------------------------------------
# Per-trajectory analysis
# ---------------------------------------------------------------------------

def analyze_trajectory(messages):
    calls = extract_tool_calls(messages)

    # Index actions
    keyword_searches = []   # (idx, call, files_in_output)
    first_edit_idx = None
    first_read_edit_idx = None
    first_read_edit_call = None

    for idx, call in enumerate(calls):
        if first_edit_idx is None and is_file_edit(call):
            first_edit_idx = idx

        if first_read_edit_idx is None and (is_file_read(call) or is_file_edit(call)):
            first_read_edit_idx = idx
            first_read_edit_call = call

        if is_keyword_search(call):
            cmd = (call.params.get("command") or call.params.get("cmd") or
                   call.params.get("input") or "")
            files = extract_files_from_search_output(call.result or "", cmd)
            keyword_searches.append((idx, call, files))

    # Searches before first edit
    searches_before_first_edit = [
        (idx, c, f) for (idx, c, f) in keyword_searches
        if first_edit_idx is None or idx < first_edit_idx
    ]

    # Searches before first read/edit
    searches_before_first_read_edit = [
        (idx, c, f) for (idx, c, f) in keyword_searches
        if first_read_edit_idx is None or idx < first_read_edit_idx
    ]

    # Stat 1 & 2
    n_searches_before_edit = len(searches_before_first_edit)
    n_searches_total = len(keyword_searches)

    # Stats 3/4/5
    covered_thresholds = {30: False, 50: False, 100: False}
    target_file = None

    if first_read_edit_call is not None:
        target_file = get_file_path(first_read_edit_call)

        if target_file:
            for (s_idx, s_call, s_files) in searches_before_first_read_edit:
                if file_in_output(target_file, s_files, s_call.result or ""):
                    n_files = len(s_files)
                    for threshold in (30, 50, 100):
                        if n_files <= threshold:
                            covered_thresholds[threshold] = True

    return {
        "n_searches_before_edit": n_searches_before_edit,
        "n_searches_total": n_searches_total,
        "target_file": target_file,
        "covered_thresholds": covered_thresholds,
        "has_first_read_edit": first_read_edit_call is not None,
    }


# ---------------------------------------------------------------------------
# Dataset-level analysis
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name):
    print(f"\nLoading {dataset_name} ...", flush=True)
    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    n = len(ds)
    print(f"Total trajectories: {n}", flush=True)

    stat1 = 0   # >1 keyword search before first edit
    stat2 = 0   # >1 keyword search overall
    stat3 = {30: 0, 50: 0, 100: 0}  # covered within threshold (denominator = all)
    has_read_edit_with_path = 0

    for i, row in enumerate(ds):
        messages = row.get("messages", [])
        r = analyze_trajectory(messages)

        if r["n_searches_before_edit"] > 1:
            stat1 += 1
        if r["n_searches_total"] > 1:
            stat2 += 1
        if r["target_file"]:
            has_read_edit_with_path += 1
            for t in (30, 50, 100):
                if r["covered_thresholds"][t]:
                    stat3[t] += 1

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{n}", flush=True)

    print(f"\n{'='*65}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*65}")
    print(f"Total trajectories                         : {n}")
    print(f"(1) >1 kw-search before first file-edit    : {stat1:5d}  ({100*stat1/n:.1f}%)")
    print(f"(2) >1 kw-search overall                   : {stat2:5d}  ({100*stat2/n:.1f}%)")
    print(f"")
    print(f"    [Trajectories w/ identifiable first     ")
    print(f"     read/edit file path]                  : {has_read_edit_with_path:5d}  ({100*has_read_edit_with_path/n:.1f}%)")
    print(f"")
    print(f"(3) File covered, <=50 files in search out : {stat3[50]:5d}  ({100*stat3[50]/n:.1f}%) [of all] / ({100*stat3[50]/has_read_edit_with_path:.1f}%) [of identifiable]" if has_read_edit_with_path else "")
    print(f"(4) File covered, <=100 files in search out: {stat3[100]:5d}  ({100*stat3[100]/n:.1f}%) [of all] / ({100*stat3[100]/has_read_edit_with_path:.1f}%) [of identifiable]" if has_read_edit_with_path else "")
    print(f"(5) File covered, <=30 files in search out : {stat3[30]:5d}  ({100*stat3[30]/n:.1f}%) [of all] / ({100*stat3[30]/has_read_edit_with_path:.1f}%) [of identifiable]" if has_read_edit_with_path else "")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    "synthetic-code-training/func_localize_gpt5mini_1346i",
    "synthetic-code-training/func_localize_gpt5mini_loc_strategy_1289i",
    "synthetic-code-training/student-func-localize-gpt5mini-1346i",
    "synthetic-code-training/student-func-localize-gpt5mini-1346i-strategy-prompt",
    "synthetic-code-training/student-func-localize-gpt5mini-loc-strategy-1289i",
]

if __name__ == "__main__":
    for ds_name in DATASETS:
        analyze_dataset(ds_name)
