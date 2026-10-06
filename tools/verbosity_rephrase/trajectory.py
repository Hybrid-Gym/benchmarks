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

`think` and `task_tracker` turns are the agent's thinking/planning steps. A skeleton drops them
together with their result turns, keeps them verbatim, or (think only) splits them like any other
turn, in which case the `thought` argument of the call is rephrased as well as the prose before it.
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
THOUGHT = re.compile(r"<parameter=thought>(.*?)</parameter>", re.DOTALL)


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
    """A trajectory with its think/task_tracker turns dropped, kept verbatim or split, and every other assistant turn split."""

    messages: list[dict]
    keep: list[int]  # surviving indices into `messages`, in order
    turns: dict[int, Turn] = field(
        default_factory=dict
    )  # kept assistant turns by msg_idx
    dropped: list[int] = field(
        default_factory=list
    )  # msg_idx of the dropped think/task_tracker turns


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


THOUGHT_LOGGED = "Your thought has been logged."


def pair_results(messages: list[dict]) -> dict[int, int | None]:
    """Map each assistant index to its result index: the k-th call of a run gets the k-th result.

    A run with fewer results than calls (qwen3.5-397b: 11 runs, e.g. a call to a tool that does not exist
    next to a think call, answered only by "Your thought has been logged.") gives each such result to a
    think call first, so that dropping the think turns never leaves another call answered by a think result.
    """
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
        calls, results = list(range(i, j)), list(range(j, k))
        if len(results) < len(calls):
            thinks = [
                a for a in calls if split_content(messages[a]["content"])[2] == "think"
            ]
            logged = [r for r in results if THOUGHT_LOGGED in messages[r]["content"]]
            pairs.update(zip(thinks, logged))
            taken = set(pairs.values())
            calls = [a for a in calls if a not in pairs]
            results = [r for r in results if r not in taken]
        for pos, a in enumerate(calls):
            pairs[a] = results[pos] if pos < len(results) else None
        i = k
    return pairs


def thought_of(call: str) -> str:
    """The `thought` argument of a think call, stripped ("" if it has none)."""
    m = THOUGHT.search(call)
    return m.group(1).strip() if m else ""


def with_thought(call: str, thought: str) -> str:
    """A think call with its `thought` argument replaced, keeping the whitespace around the argument."""
    m = THOUGHT.search(call)
    if m is None:
        raise ValueError("not a think call with a thought argument")
    body = m.group(1)
    lead = body[: len(body) - len(body.lstrip())]
    trail = body[len(body.rstrip()) :]
    return call[: m.start(1)] + lead + thought + trail + call[m.end(1) :]


def build_skeleton(
    messages: list[dict], think: str = "drop", plan: str = "drop"
) -> Skeleton:
    """`think` / `plan` say what happens to think / task_tracker turns: "drop" (with their result
    turns), "keep" (verbatim), or "split" (a turn like any other; for think, whose thought is then
    rephrased as well)."""
    policy = {"think": think, "task_tracker": plan}
    pairs = pair_results(messages)
    turns: dict[int, Turn] = {}
    dropped: list[int] = []
    drop: set[int] = set()
    for idx, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        text, call, tool, kind = split_content(m["content"])
        res = pairs.get(idx)
        how = policy.get(tool or "", "split")
        if how == "drop":
            dropped.append(idx)
            drop.update(i for i in (idx, res) if i is not None)
            continue
        if how == "keep":
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


def assemble(
    sk: Skeleton, texts: dict[int, str], thoughts: dict[int, str] | None = None
) -> list[dict]:
    """Rebuild `messages` with each split assistant turn's text part replaced by `texts[msg_idx]`
    and each split think turn's thought by `thoughts[msg_idx]`.

    Turns absent from `texts` (kept think turns included) keep their original bytes. A pure-text turn keeps its content
    when the new text is empty, since an empty assistant turn cannot be trained on.
    """
    thoughts = thoughts or {}
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
        call = with_thought(t.call, thoughts[i]) if i in thoughts else t.call
        out.append({"role": "assistant", "content": render(new_text, call)})
    return out
