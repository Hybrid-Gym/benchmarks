"""Prompting, reply parsing and the gateway client for the rephraser (one agent turn -> several lengths).

The rephraser sees only the current assistant turn (its prose and its tool call). All versions
of a turn come from one response, each derived from another, so they share one meaning. Lengths are measured with the student tokenizer (tokens.py); a version outside
its band is re-requested with the measured count as feedback.

Two length specs (families):
  fixed   t300 / t100 / t50 / t20: the same absolute targets for every turn
  scaled  x0.5 / x2 / x4 / x8: multiples of the turn's own prose length, written as a ladder
          (x0.5 shortens the prose; x2 elaborates it; x4 elaborates x2; x8 elaborates x4); a
          multiple whose target falls outside MIN_TARGET..MAX_TARGET (every multiple of an
          empty turn) is not requested
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tokens import count_tokens


Messages = list[ChatCompletionMessageParam]
DEFAULT_TOLERANCE = 0.25  # 20 -> 15..25, 50 -> 38..62, 100 -> 75..125, 300 -> 225..375
WORDS_PER_TOKEN = 0.75
WORD_CALIBRATION = 1.35  # models undershoot a bare word target by 15-20 %; the token band stays the criterion
LONG_ASK = 0.6  # per doubling of the target above 300 tokens: the model writes a shrinking fraction of a long ask (~68 % at 300-600, ~44 % at 600-1000, ~35 % beyond)
MIN_TARGET = 4  # tokens; halving a shorter turn is meaningless
MAX_TARGET = 2000  # tokens; above this the model stops well short however it is asked
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
)

SYSTEM_PROMPT = """You rewrite the natural-language commentary that a software-engineering agent writes alongside a tool call, at controlled lengths.

You are given ONE agent turn: its TEXT (the prose the agent wrote before the tool call; may be empty) and its TOOL CALL. Produce the prose for this turn at {n} target lengths.

Rules:
- Preserve the meaning and intent of TEXT. Do not invent facts, findings, file contents, or results that TEXT and the TOOL CALL do not state or imply. Longer versions elaborate on the same intent (what I am looking for, why this step, how it relates to the task, what I expect to learn); they never add new steps or new discoveries. Shorter versions condense the same content.
- If TEXT is empty, write what the agent would say just before this tool call: what the call does and why it helps, based only on the call itself. The call's `summary` parameter states the agent's own intent - use it as such, never refer to "the summary"; ignore the `security_risk` parameter entirely. For the longer versions, walk through the call concretely: which file, range, pattern or command it targets and what each part is for, what output I expect to see, and what I will do with it next.
- Do not guess what the overall task is about (bug, feature, docstring, ...) unless TEXT says so; when elaborating, stay with what this step examines or changes and what that tells the agent, rather than inventing specifics about the task or the code.
- Write in the agent's voice: first person, present tense, addressed to no one in particular (e.g. "Let me ...", "I'll ...", "Now I need to ..."). Keep the original tone, including openers like "Great!" or "I see the issue" when TEXT has them.
- Plain prose only. No headings, no lists unless TEXT used them, no code blocks, no XML tags, no quotation of the tool call, no meta-language about "the tool call", "the message" or these instructions.
- {order}

Length targets (1 token is about {wpt} words):
{targets}

Respond with ONLY a JSON object of the form {form}."""

RETRY_INSTRUCTIONS = """Some versions were outside their length band. Rewrite ONLY the versions listed below, condensing or expanding the SOURCE text (keep its meaning; same rules as before). Each version gets two candidates of different lengths, written independently:

{items}

Respond with ONLY a JSON object containing exactly these keys: {keys}."""


def band(tokens: int, tolerance: float) -> tuple[int, int]:
    return round(tokens * (1 - tolerance)), round(tokens * (1 + tolerance))


def words_for(tokens: int) -> int:
    words = tokens * WORDS_PER_TOKEN * WORD_CALIBRATION
    if tokens > 300:
        words *= 1 + LONG_ASK * math.log2(tokens / 300)
    return round(words)


def shape_of(tokens: int) -> str:
    w = words_for(tokens)
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


def struct_of(tokens: int) -> str:
    w = words_for(tokens)
    return f"{shape_of(tokens)}, about {w} words in total (never fewer than {round(w * 0.85)} words)"


class FixedLengths:
    """Family A: four absolute lengths, the same for every turn."""

    name = "fixed"
    keys = ("t300", "t100", "t50", "t20")
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
    keys = ("x0.5", "x2", "x4", "x8")
    MULT = {"x0.5": 0.5, "x2": 2, "x4": 4, "x8": 8}

    def targets(self, orig_tokens: int) -> dict[str, int]:
        out = {}
        for k, m in self.MULT.items():
            t = round(m * orig_tokens)
            if MIN_TARGET <= t <= MAX_TARGET:
                out[k] = t
        return out

    def order_rule(self, keys: list[str]) -> str:
        parts = []
        if "x0.5" in keys:
            parts.append(
                "Write x0.5 by shortening TEXT to half its length: keep every point and concrete "
                "reference it makes, just say each more briefly."
            )
        steps = []
        prev, prev_mult = "TEXT", 1.0
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
        return f"{times} TEXT's {orig_tokens} tokens = {tokens} tokens"

    def struct(self, key: str, tokens: int) -> str:
        return struct_of(tokens)

    def shape(self, key: str, tokens: int) -> str:
        return shape_of(tokens)

    def source(self, text: str, versions: dict[str, dict]) -> str:
        """Every multiple is anchored to the original prose, so retries expand or shorten it directly."""
        return text


SPECS = {"fixed": FixedLengths(), "scaled": ScaledLengths()}
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
    return SYSTEM_PROMPT.format(
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
        {"role": "user", "content": turn_block(text, call)},
    ]


def retry_messages(
    spec: Spec,
    targets: dict[str, int],
    text: str,
    call: str,
    source: str,
    failing: dict[str, tuple[int, int]],
    tolerance: float,
) -> Messages:
    """`failing` maps key -> (tokens, words) of the rejected attempt; (0, 0) means missing or invalid.

    The word count to ask for is derived from the attempt's own tokens-per-word ratio, so the
    feedback is calibrated to this very text (code identifiers tokenize densely).
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
            why = "your previous attempt was missing or invalid"
        else:
            ratio = n / max(w, 1)
            want = round(target / ratio)
            if n < lo:
                alt = round(want * 1.5)
                min_words = round(lo / ratio)
                why = (
                    f"your previous attempt had {w} words = {n} tokens, too SHORT (about {max(want - w, 1)} words missing); "
                    f"anything under {min_words} words will be rejected again, so say more about what this step examines and what I expect to learn"
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
        f"{turn_block(text, call)}\n\nSOURCE (the version to condense or expand):\n{source}\n\n"
        + RETRY_INSTRUCTIONS.format(items="\n".join(items), keys=keys)
    )
    return [
        {
            "role": "system",
            "content": system_prompt(spec, targets, count_tokens(text), tolerance),
        },
        {"role": "user", "content": user},
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
            r'"\s*[,}]?\s*$', "", s[m.end() : end].rstrip()
        )  # closing quote (+ `,` or `}`)
        out[m.group(1)] = _unescape(chunk)
    return out


def parse_versions(raw: str, keys: list[str]) -> dict[str, str]:
    """Extract the requested keys from the model's reply, in reply order; missing or invalid keys are absent."""
    s = (raw or "").strip()
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
        k: v.strip()
        for k, v in obj.items()
        if k in keys
        and isinstance(v, str)
        and v.strip()
        and not any(f in v for f in FORBIDDEN)
    }


class Rephraser:
    """OpenAI-compatible client with gateway-friendly retries."""

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

    def complete(
        self,
        messages: Messages,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        """Return (content, usage); raise the last error once max_attempts are exhausted."""
        for attempt in range(1, self.max_attempts + 1):
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
                if attempt == self.max_attempts:
                    raise
                desc = f"{type(e).__name__}: {str(e)[:140]}"
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
            usage = {
                "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                "completion_tokens": getattr(r.usage, "completion_tokens", None),
                "finish_reason": r.choices[0].finish_reason,
            }
            return r.choices[0].message.content or "", usage
        raise RuntimeError("max_attempts must be >= 1")
