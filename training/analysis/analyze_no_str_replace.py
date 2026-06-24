"""
Analyze GPT55 training trajectories that never call str_replace.
Check whether they still produce a valid patch (via file_editor:insert, terminal, etc.)
Also breaks down the full edit-method distribution across the entire dataset.
"""

from datasets import load_dataset
import re
from collections import Counter


def get_file_editor_commands(messages):
    """Return Counter of file_editor command types used in assistant messages."""
    counts = Counter()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for block in re.findall(r"<function=file_editor>(.*?)</function>", content, re.DOTALL):
            m = re.search(r"<parameter=command>(.*?)</parameter>", block, re.DOTALL)
            if m:
                counts[m.group(1).strip()] += 1
    return counts


def get_edit_methods(messages):
    """
    Return a set of edit methods actually used:
      - 'str_replace'   : file_editor str_replace
      - 'insert'        : file_editor insert
      - 'create'        : file_editor create
      - 'terminal_edit' : sed -i / awk redirect / git apply in terminal
      - 'git_diff_seen' : a non-empty git diff appeared in an observation (edit confirmed)
    """
    methods = set()
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        role = msg.get("role", "")

        if role == "assistant":
            if "<function=file_editor>" in content:
                for cmd in re.findall(
                    r"<function=file_editor>.*?<parameter=command>(.*?)</parameter>",
                    content,
                    re.DOTALL,
                ):
                    cmd = cmd.strip()
                    if cmd in ("str_replace", "insert", "create"):
                        methods.add(cmd)

            if "<function=terminal>" in content:
                for cmd in re.findall(
                    r"<parameter=command>(.*?)</parameter>", content, re.DOTALL
                ):
                    if re.search(r"\bsed\s+-i\b|\bawk\b.*>\s*\S+|>\s*\S+\.py\b", cmd):
                        methods.add("terminal_edit")
                    if "git apply" in cmd or "patch -p" in cmd:
                        methods.add("terminal_edit")

        elif role == "user":
            if "diff --git" in content and "@@" in content:
                methods.add("git_diff_seen")

    return methods


def get_finish_message(messages):
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if "<function=finish>" in content:
            m = re.search(r"<parameter=message>(.*?)</parameter>", content, re.DOTALL)
            if m:
                return m.group(1).strip()
    return None


def main():
    print("Loading synthetic-code-training/func_localize_gpt55_1477i ...")
    ds = load_dataset("synthetic-code-training/func_localize_gpt55_1477i", split="train")
    n = len(ds)
    print(f"Total examples: {n}\n")

    # ── Full dataset edit-method breakdown ───────────────────────────────────
    edit_method_counter = Counter()
    category_counts = Counter()  # str_replace / insert_only / other_edit / no_edit

    no_str_replace_examples = []

    for ex in ds:
        msgs = ex["messages"]
        methods = get_edit_methods(msgs)

        edit_method_counter.update(methods)

        if "str_replace" in methods:
            category_counts["has_str_replace"] += 1
        elif "insert" in methods:
            category_counts["insert_only"] += 1
            no_str_replace_examples.append((ex, methods))
        elif methods - {"git_diff_seen"}:
            category_counts["other_edit_only"] += 1
            no_str_replace_examples.append((ex, methods))
        else:
            category_counts["no_edit_at_all"] += 1
            no_str_replace_examples.append((ex, methods))

    print("── Edit method presence (entire dataset) ──────────────────────────")
    for method in ("str_replace", "insert", "create", "terminal_edit", "git_diff_seen"):
        cnt = edit_method_counter[method]
        print(f"  {method:20s}: {cnt:4d} / {n}  ({100*cnt/n:.1f}%)")

    print()
    print("── Trajectory categories ───────────────────────────────────────────")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:25s}: {cnt:4d} / {n}  ({100*cnt/n:.1f}%)")

    print()
    total_no_str = len(no_str_replace_examples)
    print(f"Trajectories WITHOUT str_replace: {total_no_str} / {n} ({100*total_no_str/n:.1f}%)")

    # Break them down
    insert_only = [(ex, m) for ex, m in no_str_replace_examples if "insert" in m]
    other_edit  = [(ex, m) for ex, m in no_str_replace_examples
                   if "insert" not in m and (m - {"git_diff_seen"})]
    no_edit     = [(ex, m) for ex, m in no_str_replace_examples
                   if not (m - {"git_diff_seen"}) and "insert" not in m]

    print(f"  └─ use file_editor:insert (valid patch): {len(insert_only)}")
    print(f"  └─ other edit method (terminal etc.):    {len(other_edit)}")
    print(f"  └─ no edit whatsoever (phantom success): {len(no_edit)}")

    # ── Insert-only: show sample ──────────────────────────────────────────────
    if insert_only:
        print()
        print("── Sample insert-only trajectories (first 5) ──────────────────────")
        for ex, methods in insert_only[:5]:
            fe_cmds = get_file_editor_commands(ex["messages"])
            finish_msg = get_finish_message(ex["messages"]) or ""
            print(f"\n  {ex['instance_id']}")
            print(f"    file_editor cmds: {dict(fe_cmds)}")
            print(f"    edit methods:     {methods}")
            print(f"    finish (preview): {finish_msg[:200]}")

    # ── No-edit: show sample ──────────────────────────────────────────────────
    if no_edit:
        print()
        print("── Truly phantom-success trajectories (no edit at all) ─────────────")
        for ex, methods in no_edit[:10]:
            fe_cmds = get_file_editor_commands(ex["messages"])
            finish_msg = get_finish_message(ex["messages"]) or "(no finish call)"
            print(f"\n  {ex['instance_id']}")
            print(f"    file_editor cmds: {dict(fe_cmds)}")
            print(f"    finish (preview): {finish_msg[:300]}")


if __name__ == "__main__":
    main()
