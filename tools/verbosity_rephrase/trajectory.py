"""Split non-fncall trajectories into a text part and a tool-call part per assistant turn.

A trajectory is a `messages` list of {role, content}. An assistant turn is prose followed by
one tool call rendered as text:

    Let's search for the redirect handling code in client.py:

    <function=terminal>
    <parameter=command>grep -n "redirect" client.py</parameter>
    </function>

Its result is the next user turn ("EXECUTION RESULT of [function]: ..."). Parallel calls are
flattened into consecutive assistant turns followed by as many user turns, paired positionally.

Two irregular shapes are kept verbatim as the call part: malformed calls (`<tool_call>{json}`
from qwen, `<invoke ...>` from claude, each answered by a "Please continue working..." nudge)
and the few pure-text turns that have no call at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


FUNCTION_BLOCK = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>", re.DOTALL
)
# A malformed `<tool_call>` may be garbled (stray token first, `</tool_call>` as the opener),
# so the bare JSON object is an anchor too.
MALFORMED_START = re.compile(
    r"(<tool_call>|</tool_call>|<invoke\b|<function_calls>|^\{\"name\": \")",
    re.MULTILINE,
)
DROP_TOOLS = frozenset({"think", "task_tracker"})


@dataclass
class Turn:
    msg_idx: int
    text: str  # prose before the call, stripped
    call: str  # call part verbatim ("" for a pure-text turn)
    tool: str | None  # function name, "<malformed>", or None for a pure-text turn
    result_idx: int | None  # paired user turn
    kind: str  # "function" | "malformed" | "text"


@dataclass
class Skeleton:
    """A trajectory with think/task_tracker turns removed and every kept assistant turn split."""

    messages: list[dict]
    keep: list[int]  # surviving indices into `messages`, in order
    turns: dict[int, Turn] = field(
        default_factory=dict
    )  # kept assistant turns by msg_idx
    dropped: list[int] = field(
        default_factory=list
    )  # msg_idx of the dropped assistant turns


def split_content(content: str) -> tuple[str, str, str | None, str]:
    """Return (text, call, tool, kind) for one assistant turn."""
    m = FUNCTION_BLOCK.search(content)
    if m:
        return (
            content[: m.start()].strip(),
            content[m.start() :],
            m.group(1),
            "function",
        )
    m = MALFORMED_START.search(content)
    if m:
        return (
            content[: m.start()].strip(),
            content[m.start() :],
            "<malformed>",
            "malformed",
        )
    return content.strip(), "", None, "text"


def pair_results(messages: list[dict]) -> dict[int, int | None]:
    """Map each assistant index to its result index: the k-th call of a run gets the k-th result."""
    pairs: dict[int, int | None] = {}
    i, n = 0, len(messages)
    while i < n:
        if messages[i]["role"] != "assistant":
            i += 1
            continue
        j = i
        while j < n and messages[j]["role"] == "assistant":
            j += 1
        k = j
        while k < n and messages[k]["role"] == "user":
            k += 1
        for pos, a in enumerate(range(i, j)):
            pairs[a] = j + pos if j + pos < k else None
        i = k
    return pairs


def build_skeleton(messages: list[dict]) -> Skeleton:
    pairs = pair_results(messages)
    turns: dict[int, Turn] = {}
    dropped: list[int] = []
    drop: set[int] = set()
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        text, call, tool, kind = split_content(m["content"])
        res = pairs.get(idx)
        if tool in DROP_TOOLS:
            dropped.append(idx)
            drop.update(i for i in (idx, res) if i is not None)
            continue
        turns[idx] = Turn(idx, text, call, tool, res, kind)
    keep = [i for i in range(len(messages)) if i not in drop]
    return Skeleton(messages, keep, turns, dropped)


def render(text: str, call: str) -> str:
    """Join a text part and a call part the way the datasets format them."""
    text = text.strip()
    if not call:
        return text
    if not text:
        return call
    return f"{text}\n\n{call}"


def assemble(sk: Skeleton, texts: dict[int, str]) -> list[dict]:
    """Rebuild `messages` with each kept assistant turn's text part replaced by `texts[msg_idx]`.

    Turns absent from `texts` keep their original bytes. A pure-text turn keeps its content
    when the new text is empty, since an empty assistant turn cannot be trained on.
    """
    out: list[dict] = []
    for i in sk.keep:
        m = sk.messages[i]
        t = sk.turns.get(i)
        if t is None or i not in texts:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        new_text = texts[i]
        if t.kind == "text" and not new_text.strip():
            new_text = t.text
        out.append({"role": "assistant", "content": render(new_text, t.call)})
    return out
