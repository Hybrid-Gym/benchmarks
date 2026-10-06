"""
Analyze the text that appears before tool/action calls in the claude45 dataset.
Categorizes message types and computes average token counts.

Dataset: synthetic-code-training/func_localize_claude45_1457i
"""

import re
import json
from collections import defaultdict, Counter
from datasets import load_dataset


# Simple whitespace tokenizer (approx GPT-style token count heuristic)
def approx_tokens(text: str) -> int:
    # ~4 chars per token is a rough approximation
    return max(1, len(text) // 4)


def extract_preaction_text(content: str) -> str | None:
    """
    Given an assistant message content string, extract the text before the
    first <function= tool call. Returns None if there is no text (empty or
    tool call starts at position 0).
    """
    # Find the first tool call tag
    match = re.search(r"<function=", content)
    if match is None:
        return None  # No tool call in this message
    pre = content[: match.start()].strip()
    return pre if pre else None


# ── Message type classifier ──────────────────────────────────────────────────

CATEGORY_PATTERNS = [
    # Confidence/conclusion before writing
    ("conclusion_then_write",   re.compile(
        r"(i'm confident|i've confirmed|now i'm|now i've|i have confirmed|i have identified)",
        re.I)),
    # Result of previous action + verification/next step
    ("result_then_verify",      re.compile(
        r"(has been (added|updated|removed|written|created|modified) successfully"
        r"|successfully (added|updated|written|created|modified)"
        r"|the (syntax is valid|output shows|result is))",
        re.I)),
    # Discovery / found something + investigation
    ("discovery_then_investigate", re.compile(
        r"(i found|i('ve| have) found|found it|found the|let me (also )?check|let me (also )?look)",
        re.I)),
    # Error handling / alternative approach
    ("error_then_alternative",  re.compile(
        r"(error is unrelated|unrelated to my changes|instead:|alternative|instead,)",
        re.I)),
    # Simple action announcement ("Now I'll ...", "Let me ...")
    ("simple_action",           re.compile(
        r"^(now i('ll| will)|let me|i('ll| will) now|i('ll| will) start|i('ll| will) add"
        r"|i('ll| will) write|i('ll| will) view|i('ll| will) verify|i('ll| will) check)",
        re.I)),
]


def classify(text: str) -> str:
    for name, pattern in CATEGORY_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(num_rows: int = 200):
    print(f"Loading dataset (first {num_rows} rows)...")
    ds = load_dataset(
        "synthetic-code-training/func_localize_claude45_1457i",
        split="train",
    )
    ds = ds.select(range(min(num_rows, len(ds))))

    all_preaction_texts = []        # (text, token_count)
    category_texts = defaultdict(list)  # category -> list of (text, token_count)
    empty_count = 0
    total_tool_calls = 0

    for row in ds:
        messages = row.get("messages") or row.get("conversations") or []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role != "assistant":
                continue
            if "<function=" not in content:
                continue
            total_tool_calls += 1
            pre = extract_preaction_text(content)
            if pre is None:
                empty_count += 1
            else:
                toks = approx_tokens(pre)
                all_preaction_texts.append((pre, toks))
                cat = classify(pre)
                category_texts[cat].append((pre, toks))

    # ── Stats ────────────────────────────────────────────────────────────────
    has_text = len(all_preaction_texts)
    print(f"\n{'='*60}")
    print(f"Total assistant turns with tool calls : {total_tool_calls}")
    print(f"  - with pre-action text              : {has_text}  "
          f"({100*has_text/total_tool_calls:.1f}%)")
    print(f"  - empty (no text before tool call)  : {empty_count}  "
          f"({100*empty_count/total_tool_calls:.1f}%)")

    if all_preaction_texts:
        token_counts = [t for _, t in all_preaction_texts]
        print(f"\nAverage pre-action text tokens (approx) : {sum(token_counts)/len(token_counts):.1f}")
        print(f"Median                                  : {sorted(token_counts)[len(token_counts)//2]}")
        print(f"Min / Max                               : {min(token_counts)} / {max(token_counts)}")

    print(f"\n{'─'*60}")
    print("Breakdown by message type:")
    print(f"{'─'*60}")
    for cat, items in sorted(category_texts.items(), key=lambda x: -len(x[1])):
        toks = [t for _, t in items]
        print(f"  {cat:<35s} count={len(items):3d}  avg_tokens={sum(toks)/len(toks):5.1f}")

    print(f"\n{'─'*60}")
    print("Example pre-action texts per category:")
    print(f"{'─'*60}")
    for cat, items in sorted(category_texts.items(), key=lambda x: -len(x[1])):
        print(f"\n[{cat}]")
        for text, toks in items[:3]:
            truncated = text[:200].replace("\n", " ")
            print(f"  ({toks} tok) {truncated!r}")

    # ── Per-position analysis: is text more common on first vs later calls? ──
    print(f"\n{'─'*60}")
    print("Position within trajectory (first call vs later calls):")
    print(f"{'─'*60}")
    first_with = first_empty = later_with = later_empty = 0
    for row in ds:
        messages = row.get("messages") or row.get("conversations") or []
        tool_call_idx = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            if "<function=" not in msg.get("content", ""):
                continue
            pre = extract_preaction_text(msg["content"])
            if tool_call_idx == 0:
                if pre: first_with += 1
                else:   first_empty += 1
            else:
                if pre: later_with += 1
                else:   later_empty += 1
            tool_call_idx += 1

    def pct(a, b): return f"{100*a/(a+b):.1f}%" if (a+b) else "n/a"
    print(f"  First tool call  — with text: {first_with}  empty: {first_empty}  "
          f"({pct(first_with, first_empty)} have text)")
    print(f"  Later tool calls — with text: {later_with}  empty: {later_empty}  "
          f"({pct(later_with, later_empty)} have text)")


if __name__ == "__main__":
    analyze(num_rows=200)
