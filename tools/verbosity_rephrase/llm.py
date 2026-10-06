"""Prompting, reply parsing and the gateway client for the rephraser (one agent turn -> several lengths).

The rephraser sees only the current assistant turn (its prose and its tool call). All versions
of a turn come from one response, each derived from another, so they share one meaning. Lengths are measured with the student tokenizer (tokens.py); a version outside
its band is re-requested with the measured count as feedback.

Three length specs:
  fixed   t300 / t100 / t50 / t20: the same absolute targets for every turn
  scaled  x0.5 / x2 / x4 / x8: multiples of the turn's own prose length, written as a ladder
          (x0.5 shortens the prose; x2 elaborates it; x4 elaborates x2; x8 elaborates x4); a
          multiple whose target falls outside MIN_TARGET..MAX_TARGET (every multiple of an
          empty turn) is not requested
          x32: written afterwards from the longest accepted rung (x8, else x4, x2, the prose) as
          the SOURCE, in consecutive parts of at most PART_MAX tokens (each request sees the
          text so far and writes the next part), capped at MAX_TARGET_PARTS
  thought the same multiples of a think call's `thought` argument; thoughts are long (claude45
          median 304 tokens), so only x0.5 and the rungs of at most THOUGHT_LADDER_MAX tokens are
          written in one reply, every longer rung in parts from the rung below it
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import threading
import time

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tokens import count_tokens


Messages = list[ChatCompletionMessageParam]
DEFAULT_TOLERANCE = 0.25  # 20 -> 15..25, 50 -> 38..62, 100 -> 75..125, 300 -> 225..375
WORDS_PER_TOKEN = 0.75
WORD_CALIBRATION = 1.35  # models undershoot a bare word target by 15-20 %; the token band stays the criterion
PARTS_WORD_CALIBRATION = 0.9  # a part written on its own comes out 1.5x its chunk at 1.35 (smoke test, 26 parts)
LONG_ASK = 0.6  # per doubling of the target above 300 tokens: the model writes a shrinking fraction of a long ask (~68 % at 300-600, ~44 % at 600-1000, ~35 % beyond)
MIN_TARGET = 4  # tokens; halving a shorter turn is meaningless
MAX_TARGET = 2000  # tokens; above this the model stops well short however it is asked
MAX_TARGET_PARTS = 8000  # tokens; cap for versions written in parts (x32 of a 250-token turn, the same turns x8 reaches)
THOUGHT_LADDER_MAX = 600  # tokens; a thought's rungs above this are written in parts (one-reply rungs of 600-2000 tokens miss the band 5-26 % of the time)
THOUGHT_X32_CLAMP = 16000  # tokens; a thought's x32 target is at most this (full 32x for thoughts up to 500 tokens, 87 % of claude45's): longer versions degenerate into filler
PART_MAX = 1500  # tokens; longest part asked for in one request
PARTS_STOP = 0.9  # stop adding parts once the text reaches this fraction of the target (the band is wider)
MAX_SOFAR_CHARS = 8000  # the text so far is shown head + tail beyond this
SOFAR_TAIL = 6000
MAX_CALL_CHARS = 3500  # longer parts are shown head + tail
MAX_TEXT_CHARS = 14000
FORBIDDEN = (
    "<function=",
    "</function>",
    "<parameter=",
    "EXECUTION RESULT",
    "<tool_call>",
    "</tool_call>",
    "<invoke",
    "</parameter>",
)

# Phrases that betray the rewriting task instead of the agent's own words. The rewritten thoughts of
# family C narrated the task ("the thought doesn't mention testing, so I won't plan that", "as the
# summary says", "I'm supposed to keep the same conclusions", "the next part of my reasoning"), and so
# did the longest prose of family B ("the tool call I'm making is a view command"). A version or part
# containing one of these that the unit's original text does not also contain is rejected and asked
# for again, with the reason. Precision matters more than recall: a docstring's "summary line", "the
# next part of the task", a doctest "prompt", "the original wording" of a removed docstring, "the
# source text" a parser reads, "I'm rewriting the condition" or "inventory" are the agent's own words
# and must pass. Each pattern applies to the listed fields ("text" = prose, "thought" = a think call's
# thought): a thought legitimately discusses its docstring's "summary", prose has no such summary.
_ALL = ("text", "thought")
META_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"(?<![.\w`])(THOUGHT|TEXT|SOURCE|SUMMARY|WRITTEN SO FAR|TOOL CALL)\b"
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\bthe thought\b(?! (of|that|process|experiment|behind|occurs|occurred|crossed))",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\b(the|this) (rewritten thought|original thought|given thought|marked step|tool call)\b|"
            r"\b(expanded|condensed|shorter|longer|rewritten) version of (the|this|my) (thought|text|reasoning|commentary)\b|"
            r"\bparaphrase of the (thought|text)\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\b(the|this) summary (says|states|mentions|confirms|parameter|notes|indicates|describes|frames|calls|"
            r"refers|labels|itself|already|explicitly|only|gives|tells|suggests|hints|emphasizes|highlights|captures|"
            r"reads|provided|given|of (the|this) (tool call|step|call))\b|\b(as|per) the summary\b(?! line)",
            re.I,
        ),
        ("text",),
    ),
    (
        re.compile(
            r"\b(the|this) summary (parameter|says to|tells (me )?to|for this step|of this (step|tool call|call)|"
            r"only mentions|explicitly says)\b|\bas the summary (says|states|notes|puts|frames|labels)\b|"
            r"\bthe scope of this summary\b",
            re.I,
        ),
        ("thought",),
    ),
    (
        re.compile(
            r"\b(the|these|those|your) (instructions|rules|guidelines) (say|says|said|state|states|mention|mentions|"
            r"ask|asks|tell|tells|require|requires|forbid|forbids|specify|specifies|instruct|instructs|indicate|"
            r"call for|are clear|explicitly)\b|\bas instructed\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\b(next|final|first|second|last|opening|previous|this) part of (my|the|this) "
            r"(reasoning|commentary|thinking|thought|response|explanation|write-?up|narrative)\b|"
            r"\b(write|writing|continue|continuing|begin|beginning|start|starting|produce|producing) the "
            r"(next|final|opening|first|last|second) part\b|"
            r"\bin this part(,| I| of (my|the) (commentary|reasoning|text|thought))",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\bfrom the (thought|text|summary) alone\b|\breasoning from the (thought|text)\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\bI('m| am) (asked|told|instructed|supposed|expected|required) to (write|rewrite|expand|elaborate|"
            r"produce|reason|think|stay|keep|avoid|not|only|preserve|maintain|match|continue|condense|shorten|lengthen)\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\b(word|token) (count|limit|budget|target)\b|\blength (requirement|target|band|budget)\b|\b\d+ tokens\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\breach(ing)? (any )?new conclusions?\b|\bplan(ning)? (any )?different steps?\b|"
            r"\badd(ing)? (any )?new (steps|discoveries|conclusions)\b|\bno new (facts|steps|conclusions|discoveries)\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\bthe (thought|text|summary|passage|excerpt|instructions) (doesn't|does not|didn't|did not|never|only|"
            r"explicitly|already|also|itself) (mention|say|state|specify|specifies|include|cover|address|tell|talk|"
            r"discuss|note|indicate|describe|refer|mentions|says|states|includes|covers|addresses|tells|notes|indicates|"
            r"describes|refers)\b|"
            r"\bnot (mentioned|stated|specified|given|provided|covered|included|present|shown|contained) (in|by) "
            r"(the |this |my )?(thought|text|summary|passage|excerpt)\b|"
            r"\bbeyond the scope of (this|the) (thought|summary|text)\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\bthe agent('s)? (own |next |previous |original |current )?(step|steps|reasoning|voice|thought|thoughts|"
            r"commentary|words|intent|intention|task|goal|run|perspective)\b|"
            r"\bthe agent (is (working|trying|looking|supposed|expected|analyzing|reasoning|about)|was doing|"
            r"who (just|wrote|examined|made)|would|will|should|needs?|wants?|must|has to|itself)\b|"
            r"\bwhatever (task )?the agent\b|\bme, the agent\b|\bwritten by the agent\b|\bthe assistant\b|\bas an AI\b",
            re.I,
        ),
        _ALL,
    ),
    (
        re.compile(
            r"\[WRITE COMMENT\]|\bthe marked step\b|\bstep \d+ of the (run|trajectory)\b",
            re.I,
        ),
        _ALL,
    ),
)

# Residue of the reply format: a JSON wrapper, a stray `</think>` block, a version's key. None of it is
# ever the agent's words, so a text containing any is rejected outright (family B's pushed x2 / x4 prose
# ended in `"}\n\n{` for 3 % of the turns, and 5-7 % of the x32 prose had `</think>{"part": "` inside).
RESIDUE_PATTERNS = (
    re.compile(r"</?think>"),
    re.compile(r'\{\s*"part"|"part"\s*:'),
    re.compile(r'"(x0\.5|x2|x4|x8|x32|t20|t50|t100|t300)(_a|_b)?"\s*:'),
    re.compile(r"^\s*[{}]"),
    re.compile(r"[{}]\s*$"),
    re.compile(r'"\s*}\s*(,|\{|$|\s+")'),
)
_JSON_TAIL = re.compile(r'(?<=[.!?)\]`\'])\s*"\s*}\s*,?\s*\{?\s*$')


def meta_hits(text: str, orig: str = "", field: str = "text") -> list[str]:
    """The meta-language phrases in `text` (a `field` of a unit) that `orig`, the agent's own text, lacks."""
    low = orig.lower()
    out = []
    for p, fields in META_PATTERNS:
        if field not in fields:
            continue
        for m in p.finditer(text):
            if m.group(0).lower() not in low:
                out.append(m.group(0))
                break
    return out


def residue_hits(text: str) -> list[str]:
    return [m.group(0) for p in RESIDUE_PATTERNS for m in [p.search(text)] if m]


def strip_json_tail(text: str) -> str:
    """`text` without a trailing `"}` / `"}\\n\\n{` left by a reply that held two JSON objects."""
    return _JSON_TAIL.sub("", text) if _JSON_TAIL.search(text) else text


def after_think(raw: str) -> str:
    """A reply without its `<think>...</think>` block(s): the text after the last `</think>`."""
    return raw.rsplit("</think>", 1)[-1] if "</think>" in raw else raw


RULES = """- Preserve the meaning and intent of TEXT. Do not invent facts, findings, file contents, or results that TEXT and the TOOL CALL do not state or imply. Longer versions elaborate on the same intent (what I am looking for, why this step, how it relates to the task, what I expect to learn); they never add new steps or new discoveries. Shorter versions condense the same content.
- If TEXT is empty, write what the agent would say just before this tool call: what the call does and why it helps, based only on the call itself. The call's `summary` parameter states the agent's own intent - use it as such, never refer to "the summary"; ignore the `security_risk` parameter entirely. For the longer versions, walk through the call concretely: which file, range, pattern or command it targets and what each part is for, what output I expect to see, and what I will do with it next.
- Do not guess what the overall task is about (bug, feature, docstring, ...) unless TEXT says so; when elaborating, stay with what this step examines or changes and what that tells the agent, rather than inventing specifics about the task or the code.
- Write in the agent's voice: first person, present tense, addressed to no one in particular (e.g. "Let me ...", "I'll ...", "Now I need to ..."). Keep the original tone, including openers like "Great!" or "I see the issue" when TEXT has them.
- Plain prose only. No headings, no lists unless TEXT used them, no code blocks, no XML tags, no quotation of the tool call, no meta-language about "the tool call", "the message" or these instructions.
- Write as the agent, in the moment: the output is what the agent itself says at this step, nothing else. Never refer to TEXT, the TOOL CALL, a SOURCE, the parts, the target length or these rules, and never say what TEXT does or does not mention, what you must not invent, or what you are keeping the same: the agent has never seen these instructions. Call things by their names ("this view", "the grep"), not "the tool call"."""

SYSTEM_PROMPT = (
    """You rewrite the natural-language commentary that a software-engineering agent writes alongside a tool call, at controlled lengths.

You are given ONE agent turn: its TEXT (the prose the agent wrote before the tool call; may be empty) and its TOOL CALL. Produce the prose for this turn at {n} target lengths.

Rules:
"""
    + RULES
    + """
- {order}

Length targets (1 token is about {wpt} words):
{targets}

Respond with ONLY a JSON object of the form {form}."""
)

THOUGHT_RULES = """- Preserve the reasoning, conclusions and intent of THOUGHT. Do not invent facts, findings, file contents, code, line numbers or results that THOUGHT does not state or imply. Longer versions think the same reasoning through in more depth (what each observation means, why each alternative THOUGHT weighs is kept or ruled out, how the conclusion follows, what it implies for the next step); they never reach new conclusions, add new discoveries or plan different steps. Shorter versions condense the same reasoning and keep its conclusions and concrete references (files, functions, line numbers).
- Keep the voice and form of THOUGHT: the agent reasoning to itself in the first person. Use lists or code only where THOUGHT does; code or text that THOUGHT quotes is reproduced exactly, never altered or extended, and quoted at most once - a shorter version may refer to a quotation instead of repeating it.
- SUMMARY is the agent's own one-line label for this step: use it as context, never mention it.
- No headings unless THOUGHT has them, no XML tags or tool-call markup, no meta-language about "the thought", "the summary" or these instructions.
- Write as the agent, in the moment: the output is the agent's own thinking at this step, nothing else. Never refer to THOUGHT, SUMMARY, a SOURCE, the parts, the target length or these rules, and never say what THOUGHT does or does not mention, what you must not invent, what you are keeping the same, or what "the agent" does: the agent has never seen these instructions and speaks of its task, the description, the code and its docstring in its own words."""

THOUGHT_SYSTEM_PROMPT = (
    """You rewrite the private reasoning that a software-engineering agent records with its `think` tool between tool calls, at controlled lengths.

You are given ONE thinking step: its THOUGHT and the agent's SUMMARY of it. Produce the thought at {n} target lengths.

Rules:
"""
    + THOUGHT_RULES
    + """
- {order}

Length targets (1 token is about {wpt} words):
{targets}

Respond with ONLY a JSON object of the form {form}."""
)

PARTS_PROMPT = (
    """You write the natural-language commentary that a software-engineering agent writes alongside a tool call, at a controlled length.

You are given ONE agent turn: its TEXT (the prose the agent wrote before the tool call; may be empty) and its TOOL CALL{source_clause}. The commentary for this turn is to be {label}{longer} and is produced over one or more requests, each writing the next part; the parts are concatenated into one continuous text.

Rules:
"""
    + RULES
    + """
- When WRITTEN SO FAR is given, continue seamlessly from its last sentence: do not restart, recap or repeat what it already says, and do not announce or number the part. Each part adds detail on aspects not yet covered - what this step examines, why now, what each element of the call is for, what I expect to see, how I will read it, what I will do next depending on what I find - always anchored in TEXT and the TOOL CALL.

Length of this part (1 token is about {wpt} words): {struct}.

Respond with ONLY a JSON object of the form {{"part": "..."}}."""
)

THOUGHT_PARTS_PROMPT = (
    """You rewrite the private reasoning that a software-engineering agent records with its `think` tool between tool calls, at a controlled length.

You are given ONE thinking step: its THOUGHT and the agent's SUMMARY of it{source_clause}. The rewritten thought is to be {label} and is produced over one or more requests, each writing the next part; the parts are concatenated into one continuous text.

Rules:
"""
    + THOUGHT_RULES
    + """
- When WRITTEN SO FAR is given, continue seamlessly from its last sentence: do not restart, recap or repeat what it already says, and do not announce or number the part. Each part reasons further about aspects not yet covered - what each observation in THOUGHT means and why it matters, the alternatives THOUGHT weighs and why each is kept or ruled out, how the conclusion follows, what it implies for the next step - always anchored in THOUGHT.

Length of this part (1 token is about {wpt} words): {struct}.

Respond with ONLY a JSON object of the form {{"part": "..."}}."""
)

SUMMARY = re.compile(r"<parameter=summary>(.*?)</parameter>", re.DOTALL)

RETRY_INSTRUCTIONS = """Some versions were outside their length band. Rewrite ONLY the versions listed below, condensing or expanding the SOURCE text (keep its meaning; same rules as before). Each version gets two candidates of different lengths, written independently:

{items}

Respond with ONLY a JSON object containing exactly these keys: {keys}."""


def band(tokens: int, tolerance: float) -> tuple[int, int]:
    return round(tokens * (1 - tolerance)), round(tokens * (1 + tolerance))


def words_for(tokens: int, parts: bool = False) -> int:
    """The word count to ask for.

    The ladder ask (four versions in one reply) is calibrated up and, above 300 tokens, inflated
    by LONG_ASK. A part written on its own (`parts`) is neither: the model then writes more than
    the words asked (1.3-2x the chunk with the ladder ask, 1.5x with the bare one).
    """
    if parts:
        return round(tokens * WORDS_PER_TOKEN * PARTS_WORD_CALIBRATION)
    words = tokens * WORDS_PER_TOKEN * WORD_CALIBRATION
    if tokens > 300:
        words *= 1 + LONG_ASK * math.log2(tokens / 300)
    return round(words)


def shape_of(tokens: int, parts: bool = False) -> str:
    w = words_for(tokens, parts)
    if w <= 12:
        return "one short sentence"
    if w <= 30:
        return "one sentence"
    if w <= 55:
        return "two or three sentences"
    if w <= 90:
        return "four or five sentences"
    if w <= 160:
        return "one paragraph of six to nine sentences"
    k = max(2, round(w / 110))
    return f"{k} paragraphs of about {round(w / k)} words each"


def struct_of(tokens: int, parts: bool = False) -> str:
    w = words_for(tokens, parts)
    return f"{shape_of(tokens, parts)}, about {w} words in total (never fewer than {round(w * 0.85)} words)"


class FixedLengths:
    """Family A: four absolute lengths, the same for every turn."""

    name = "fixed"
    keys = ("t300", "t100", "t50", "t20")
    LADDER = keys  # all written in one reply
    PARTS: tuple[str, ...] = ()
    SYSTEM = SYSTEM_PROMPT
    MORE = "say more about what this step examines and what I expect to learn"
    TARGETS = {"t300": 300, "t100": 100, "t50": 50, "t20": 20}
    # Models follow a structure with a floor ("three paragraphs of about 110 words each, never
    # fewer than 280 words") far better than a bare word count; see the README.
    SHAPE = {
        "t300": "three paragraphs of about 110 words each",
        "t100": "one paragraph of five or six sentences",
        "t50": "three sentences",
        "t20": "one sentence",
    }
    STRUCT = {
        "t300": "three paragraphs of about 110 words each, about 330 words in total (never fewer than 280 words)",
        "t100": "one paragraph of five or six sentences, about 100 words (never fewer than 85)",
        "t50": "three sentences, about 50 words (never fewer than 42)",
        "t20": "one sentence of about 22 words (never fewer than 17)",
    }

    def targets(self, orig_tokens: int) -> dict[str, int]:
        return dict(self.TARGETS)

    def split(self, targets: dict[str, int]) -> tuple[dict[str, int], list[str]]:
        """(the keys written in one reply with their targets, the keys written in parts afterwards)."""
        return dict(targets), []

    def block(self, text: str, call: str) -> str:
        return turn_block(text, call)

    def cap(self, key: str) -> int:
        return MAX_TARGET

    def order_rule(self, keys: list[str]) -> str:
        return "Write t300 first, then condense it to t100, then t50, then t20."

    def label(self, key: str, tokens: int, orig_tokens: int) -> str:
        return f"{tokens} tokens"

    def struct(self, key: str, tokens: int) -> str:
        return self.STRUCT[key]

    def shape(self, key: str, tokens: int) -> str:
        return self.SHAPE[key]

    def source(self, text: str, versions: dict[str, dict]) -> str:
        """What a retry condenses or expands: the accepted t300, else the original prose, else the best t300 attempt."""
        v = versions.get("t300")
        if v and v["ok"]:
            return v["text"]
        return text or (
            v["text"] if v else "(no usable source - write from the TOOL CALL)"
        )


class ScaledLengths:
    """Family B: multiples of the turn's own prose length, written as a ladder from short to long."""

    name = "scaled"
    keys = ("x0.5", "x2", "x4", "x8", "x32")
    MULT = {"x0.5": 0.5, "x2": 2, "x4": 4, "x8": 8, "x32": 32}
    LADDER = ("x0.5", "x2", "x4", "x8")  # one reply
    PARTS = ("x32",)  # written afterwards, in parts, from the longest accepted rung
    CAP: dict[str, int] = {"x32": MAX_TARGET_PARTS}
    SYSTEM = SYSTEM_PROMPT
    PARTS_SYSTEM = PARTS_PROMPT
    SUBJECT = "TEXT"  # what the prompts call the text being rewritten
    NOUN = "commentary"
    LONGER = " - far longer than TEXT -"
    CLOSE = "bring the commentary to a close that leads into the tool call"
    MORE = "say more about what this step examines and what I expect to learn"

    def targets(self, orig_tokens: int) -> dict[str, int]:
        out = {}
        for k, m in self.MULT.items():
            t = round(m * orig_tokens)
            if MIN_TARGET <= t <= self.cap(k):
                out[k] = t
        return out

    def cap(self, key: str) -> int:
        return self.CAP.get(key, MAX_TARGET)

    def split(self, targets: dict[str, int]) -> tuple[dict[str, int], list[str]]:
        """(the keys written in one reply with their targets, the keys written in parts afterwards)."""
        return {k: t for k, t in targets.items() if k in self.LADDER}, [
            k for k in self.PARTS if k in targets
        ]

    def block(self, text: str, call: str) -> str:
        return turn_block(text, call)

    def order_rule(self, keys: list[str]) -> str:
        parts = []
        if "x0.5" in keys:
            parts.append(
                f"Write x0.5 by shortening {self.SUBJECT} to half its length: keep every point and concrete "
                "reference it makes, just say each more briefly."
            )
        steps = []
        prev, prev_mult = self.SUBJECT, 1.0
        for k in keys:
            if self.MULT[k] <= 1:
                continue
            factor = self.MULT[k] / prev_mult
            times = "twice" if factor == 2 else f"{factor:g} times"
            steps.append(f"{k} by elaborating {prev} to {times} its length")
            prev, prev_mult = k, self.MULT[k]
        if steps:
            parts.append(
                "Write "
                + ", then ".join(steps)
                + ": each step keeps everything already said and "
                "adds more about what this step examines, why, and what I expect to learn."
            )
        return " ".join(parts)

    def label(self, key: str, tokens: int, orig_tokens: int) -> str:
        m = self.MULT[key]
        times = "half of" if m < 1 else f"{m} times"
        return f"{times} {self.SUBJECT}'s {orig_tokens} tokens = {tokens} tokens"

    def struct(self, key: str, tokens: int) -> str:
        return struct_of(tokens)

    def shape(self, key: str, tokens: int) -> str:
        return shape_of(tokens)

    def source(self, text: str, versions: dict[str, dict]) -> str:
        """Every multiple is anchored to the original prose, so retries expand or shorten it directly."""
        return text

    def parts_source(
        self, key: str, text: str, versions: dict[str, dict]
    ) -> tuple[str | None, str]:
        """(rung key, text) of the longest accepted rung above 1x and below `key`, else (None, the prose)."""
        for k in reversed(self.keys):
            v = versions.get(k)
            if 1 < self.MULT[k] < self.MULT[key] and v and v["ok"]:
                return k, v["text"]
        return None, text


class ThoughtLengths(ScaledLengths):
    """The thought of a think call, at the scaled family's multiples of its own length.

    x0.5, and the rungs of up to THOUGHT_LADDER_MAX tokens, come from one reply with retries; every
    longer rung is written in parts from the longest accepted rung below it (x4 from x2, x8 from x4, x32
    from x8). The single-reply x0.5 keeps the MAX_TARGET cap; x2 / x4 / x8 go up to
    MAX_TARGET_PARTS. x32 is clamped rather than capped: a thought whose 32x would exceed
    THOUGHT_X32_CLAMP is written at THOUGHT_X32_CLAMP, so every thought still grows from x8 to x32.
    """

    name = "thought"
    CAP = {
        "x2": MAX_TARGET_PARTS,
        "x4": MAX_TARGET_PARTS,
        "x8": MAX_TARGET_PARTS,
        "x32": THOUGHT_X32_CLAMP,
    }
    SYSTEM = THOUGHT_SYSTEM_PROMPT
    PARTS_SYSTEM = THOUGHT_PARTS_PROMPT
    SUBJECT = "THOUGHT"
    NOUN = "thought"
    LONGER = ""
    CLOSE = "bring the reasoning to its conclusion"
    MORE = "reason through it in more depth"

    def targets(self, orig_tokens: int) -> dict[str, int]:
        out = super().targets(orig_tokens)
        if self.MULT["x32"] * orig_tokens > THOUGHT_X32_CLAMP:
            out["x32"] = THOUGHT_X32_CLAMP
        return out

    def label(self, key: str, tokens: int, orig_tokens: int) -> str:
        if round(self.MULT[key] * orig_tokens) > tokens:  # clamped
            return f"{tokens} tokens (the most allowed; {self.MULT[key]:g} times {self.SUBJECT}'s {orig_tokens} tokens would be more)"
        return super().label(key, tokens, orig_tokens)

    def split(self, targets: dict[str, int]) -> tuple[dict[str, int], list[str]]:
        ladder = {
            k: t
            for k, t in targets.items()
            if k == "x0.5" or (k in self.LADDER and t <= THOUGHT_LADDER_MAX)
        }
        return ladder, [k for k in self.keys if k in targets and k not in ladder]

    def block(self, text: str, call: str) -> str:
        """`text` is the thought, `call` the whole think call (for its summary)."""
        m = SUMMARY.search(call)
        summary = m.group(1).strip() if m else ""
        return f"THOUGHT:\n{_clip(text, MAX_TEXT_CHARS, 2000)}\n\nSUMMARY:\n{summary or '(none)'}"


SPECS = {
    "fixed": FixedLengths(),
    "scaled": ScaledLengths(),
    "thought": ThoughtLengths(),
}
Spec = FixedLengths | ScaledLengths


def system_prompt(
    spec: Spec, targets: dict[str, int], orig_tokens: int, tolerance: float
) -> str:
    lines = "\n".join(
        f"- {k}: {spec.label(k, t, orig_tokens)} = {spec.struct(k, t)}; acceptable {lo}-{hi} tokens"
        for k, t in targets.items()
        for lo, hi in [band(t, tolerance)]
    )
    form = "{" + ", ".join(f'"{k}": "..."' for k in targets) + "}"
    return spec.SYSTEM.format(
        n=("one", "two", "three", "four")[len(targets) - 1],
        order=spec.order_rule(list(targets)),
        wpt=WORDS_PER_TOKEN,
        targets=lines,
        form=form,
    )


def _clip(s: str, limit: int, tail: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - tail] + "\n[... truncated ...]\n" + s[-tail:]


def turn_block(text: str, call: str) -> str:
    text_shown = _clip(text, MAX_TEXT_CHARS, 2000) if text else "(empty)"
    call_shown = (
        _clip(call, MAX_CALL_CHARS, 700) if call else "(none - this turn is prose only)"
    )
    return f"TEXT:\n{text_shown}\n\nTOOL CALL:\n{call_shown}"


def first_round_messages(
    spec: Spec, targets: dict[str, int], text: str, call: str, tolerance: float
) -> Messages:
    return [
        {
            "role": "system",
            "content": system_prompt(spec, targets, count_tokens(text), tolerance),
        },
        {"role": "user", "content": spec.block(text, call)},
    ]


def retry_messages(
    spec: Spec,
    targets: dict[str, int],
    text: str,
    call: str,
    source: str,
    failing: dict[str, tuple[int, int]],
    tolerance: float,
    notes: dict[str, str] | None = None,
) -> Messages:
    """`failing` maps key -> (tokens, words) of the rejected attempt; (0, 0) means missing or invalid.

    The word count to ask for is derived from the attempt's own tokens-per-word ratio, so the
    feedback is calibrated to this very text (code identifiers tokenize densely). `notes` gives, per
    key, why an attempt of acceptable length was rejected (it spoke about the rewriting task).
    """
    items = []
    asked: list[str] = []
    for k, target in targets.items():
        if k not in failing:
            continue
        lo, hi = band(target, tolerance)
        n, w = failing[k]
        if n == 0:
            want = words_for(target)
            alt = round(want * 1.25)
            why = (notes or {}).get(k) or "your previous attempt was missing or invalid"
        else:
            ratio = n / max(w, 1)
            want = round(target / ratio)
            if n < lo:
                alt = round(want * 1.5)
                min_words = round(lo / ratio)
                why = (
                    f"your previous attempt had {w} words = {n} tokens, too SHORT (about {max(want - w, 1)} words missing); "
                    f"anything under {min_words} words will be rejected again, so {spec.MORE}"
                )
            else:
                # a gentle cut: models drop dense identifiers when condensing, so an aggressive cut lands under the band
                alt = round((want + w) / 2)
                why = f"your previous attempt had {w} words = {n} tokens, too LONG (about {max(w - want, 1)} words too many)"
        items.append(
            f"- {k}: {target} tokens, acceptable {lo}-{hi} tokens; {why}. "
            f'Give two candidates: "{k}_a" of about {want} words and "{k}_b" of about {alt} words ({spec.shape(k, target)}); count the words.'
        )
        asked += [f"{k}_a", f"{k}_b"]
    keys = ", ".join(f'"{k}"' for k in asked)
    user = (
        f"{spec.block(text, call)}\n\nSOURCE (the version to condense or expand):\n{source}\n\n"
        + RETRY_INSTRUCTIONS.format(items="\n".join(items), keys=keys)
    )
    return [
        {
            "role": "system",
            "content": system_prompt(spec, targets, count_tokens(text), tolerance),
        },
        {"role": "user", "content": user},
    ]


def parts_messages(
    spec: ScaledLengths,
    key: str,
    target: int,
    orig_tokens: int,
    text: str,
    call: str,
    source_key: str | None,
    source: str,
    sofar: str,
    sofar_tokens: int,
    chunk: int,
    final: bool,
    note: str = "",
) -> Messages:
    """One request of a version written in parts: the turn, the SOURCE rung (first part only), the text so far, and the ask for the next `chunk` tokens.

    `note` is a sentence about the previous reply (why it was unusable, or that it came out short).
    """
    times = f"{spec.MULT[source_key]:g}-times" if source_key else ""
    subject, noun = spec.SUBJECT, spec.NOUN
    source_clause = (
        f", plus SOURCE, an earlier {times} elaboration of {subject} that this {noun} elaborates further"
        if source_key and not sofar
        else ""
    )
    system = spec.PARTS_SYSTEM.format(
        source_clause=source_clause,
        label=spec.label(key, target, orig_tokens),
        longer=spec.LONGER,
        wpt=WORDS_PER_TOKEN,
        struct=struct_of(chunk, parts=True),
    )
    user = spec.block(text, call)
    if source_key and not sofar:
        user += f"\n\nSOURCE ({times} {subject}'s length):\n{source}"
    if sofar:
        user += (
            f"\n\nWRITTEN SO FAR ({sofar_tokens} of the {target} tokens):\n"
            f"{_clip(sofar, MAX_SOFAR_CHARS, SOFAR_TAIL)}"
        )
    shape = shape_of(chunk, parts=True)
    if not sofar and final:
        ask = f"Write the whole {noun} now: {shape}."
    elif not sofar:
        ask = f"Write the opening part now: {shape}. More parts will follow, so do not wrap up or conclude."
    elif final:
        ask = f"Write the final part now: {shape}, and {spec.CLOSE}."
    else:
        ask = f"Write the next part now: {shape}. More parts will follow, so do not wrap up or conclude."
    if note:
        ask += f" {note}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{user}\n\n{ask}"},
    ]


_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)


def _unescape(v: str) -> str:
    return (
        v.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _lenient_extract(s: str, keys: list[str]) -> dict[str, str]:
    """Pull `"key": "value"` pairs out of almost-JSON, where quotes inside values are left unescaped."""
    mark = re.compile('"(' + "|".join(map(re.escape, keys)) + r')"\s*:\s*"')
    marks = list(mark.finditer(s))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        chunk = re.sub(
            r'"\s*[,}]?\s*\{?\s*$', "", s[m.end() : end].rstrip()
        )  # closing quote (+ `,` or `}`, + the `{` of a second object the model started)
        out[m.group(1)] = _unescape(chunk)
    return out


def extract_partial(raw: str, key: str) -> str | None:
    """The value of `key` in a reply whose JSON never closes (cut off mid-string, or the model stopped without the closing brace): everything after its opening quote, minus a trailing quote/brace.

    The LAST occurrence of the key counts: the model sometimes drafts a value, emits `</think>` and
    starts the object again.
    """
    ms = list(re.finditer('"' + re.escape(key) + r'"\s*:\s*"', after_think(raw or "")))
    if not ms:
        return None
    m = ms[-1]
    v = re.sub(r'"?\s*}?\s*(```)?\s*$', "", after_think(raw)[m.end() :].rstrip())
    v = _unescape(re.sub(r"\\+$", "", v)).strip()
    if not v or any(f in v for f in FORBIDDEN):
        return None
    return v


def parse_versions(raw: str, keys: list[str]) -> dict[str, str]:
    """Extract the requested keys from the model's reply, in reply order; missing or invalid keys are absent."""
    s = after_think(raw or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    m = _JSON_SPAN.search(s)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0), strict=False)
    except ValueError:
        obj = _lenient_extract(m.group(0), keys)
    if not isinstance(obj, dict):
        return {}
    return {
        k: strip_json_tail(v.strip())
        for k, v in obj.items()
        if k in keys
        and isinstance(v, str)
        and v.strip()
        and not any(f in v for f in FORBIDDEN)
    }


class AdaptiveLimiter:
    """Cap on in-flight requests that backs off on a 429 and creeps back up on clean calls.

    The gateway's throughput swings with other tenants' load: a worker count that saturates a
    slow endpoint triggers 429 storms once it speeds up, and one that is safe when it is fast
    idles when it is slow. On a rate limit the cap drops to 3/4 (never below `lo`) and every
    request waits out a short cooldown, which is the small-scale form of the full stop that
    clears a storm; after `up_after` consecutive clean calls the cap grows by one, up to `hi`.
    """

    def __init__(
        self,
        start: int,
        lo: int,
        hi: int,
        up_after: int = 30,
        cooldown_s: float = 20.0,
        log: bool = True,
    ):
        self.cap = max(lo, min(start, hi))
        self.lo, self.hi = lo, hi
        self.up_after = up_after
        self.cooldown_s = cooldown_s
        self.log = log
        self.cv = threading.Condition()
        self.inflight = 0
        self.ok_streak = 0
        self.cooldown_until = 0.0

    def acquire(self) -> None:
        with self.cv:
            while self.inflight >= self.cap or time.time() < self.cooldown_until:
                self.cv.wait(timeout=2.0)
            self.inflight += 1

    def release(self, rate_limited: bool) -> None:
        with self.cv:
            self.inflight -= 1
            if rate_limited:
                new = max(self.lo, self.cap * 3 // 4)
                self.cooldown_until = time.time() + self.cooldown_s
                if new != self.cap and self.log:
                    print(f"  limiter: cap {self.cap} -> {new} (429)", file=sys.stderr)
                self.cap, self.ok_streak = new, 0
            else:
                self.ok_streak += 1
                if self.ok_streak >= self.up_after and self.cap < self.hi:
                    if self.log:
                        print(
                            f"  limiter: cap {self.cap} -> {self.cap + 1} (clean streak)",
                            file=sys.stderr,
                        )
                    self.cap, self.ok_streak = self.cap + 1, 0
            self.cv.notify_all()


class Rephraser:
    """OpenAI-compatible client with gateway-friendly retries and an adaptive in-flight cap."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2500,
        extra_body: dict | None = None,
        max_attempts: int = 8,
        backoff_base_s: float = 3.0,
        request_timeout_s: float = 600.0,
        limiter: AdaptiveLimiter | None = None,
    ):
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=request_timeout_s, max_retries=0
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.limiter = limiter

    def complete(
        self,
        messages: Messages,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        """Return (content, usage); raise the last error once max_attempts are exhausted."""
        for attempt in range(1, self.max_attempts + 1):
            if self.limiter:
                self.limiter.acquire()
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                    if temperature is None
                    else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    extra_body=self.extra_body,
                )
            except Exception as e:  # noqa: BLE001 - every gateway failure is retryable
                desc = f"{type(e).__name__}: {str(e)[:140]}"
                if self.limiter:
                    self.limiter.release("429" in desc or "RateLimit" in desc)
                if attempt == self.max_attempts:
                    raise
                delay = (
                    self.backoff_base_s * 2 ** (attempt - 1) * (0.5 + random.random())
                )
                if "429" in desc or "RateLimit" in desc:
                    delay = max(delay, 10.0 * attempt)
                print(
                    f"  retry {attempt}/{self.max_attempts} in {delay:.0f}s: {desc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue
            if self.limiter:
                self.limiter.release(False)
            usage = {
                "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                "completion_tokens": getattr(r.usage, "completion_tokens", None),
                "finish_reason": r.choices[0].finish_reason,
            }
            return r.choices[0].message.content or "", usage
        raise RuntimeError("max_attempts must be >= 1")
