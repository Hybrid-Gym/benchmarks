# Filter output.with_completions.jsonl.gz to instances with valid docstring patches.
#
# Applies the judge_valid_docstring_patch rule from Hybrid-Gym/evaluation/convert_data.py:
# the appended-line block must end in a triple-quote or triple-backtick fence.
#
# Writes output_success.with_completions.jsonl.gz next to the input.
import argparse
import gzip
import json
import os


def parse_git_diff(diff_text):
    if not diff_text:
        return []
    added = []
    in_hunk = False
    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
    return added


def judge_valid_docstring_patch(diff_text):
    added = parse_git_diff(diff_text)
    if not added:
        return False
    block = "\n".join(added).strip()
    return block.endswith('"""') or block.endswith("'''") or block.endswith("```")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="Path to output.with_completions.jsonl.gz")
    p.add_argument(
        "--dst",
        default=None,
        help="Output path (default: output_success.with_completions.jsonl.gz next to src)",
    )
    args = p.parse_args()

    dst = args.dst or os.path.join(
        os.path.dirname(args.src),
        "output_success.with_completions.jsonl.gz",
    )

    total = kept = 0
    with gzip.open(args.src, "rt") as f_in, gzip.open(dst, "wt") as f_out:
        for line in f_in:
            r = json.loads(line)
            total += 1
            patch = (r.get("test_result") or {}).get("git_patch", "")
            if judge_valid_docstring_patch(patch):
                r["resolved"] = True
                f_out.write(json.dumps(r) + "\n")
                kept += 1

    pct = 100 * kept / total if total else 0
    print(f"Filtered {total} → {kept} resolved ({pct:.1f}%)")
    print(f"Wrote: {dst}  ({os.path.getsize(dst) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
