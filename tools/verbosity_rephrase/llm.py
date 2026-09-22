"""Prompting, reply parsing and the gateway client for the rephraser (one agent turn -> four lengths).

The rephraser sees only the current assistant turn (its prose and its tool call). It writes the
~300-token version first and condenses it to ~100 / ~50 / ~20 in the same response so the four
versions share one meaning. Lengths are measured with the student tokenizer (tokens.py); a version
outside its band is re-requested with the measured count as feedback.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


Messages = list[ChatCompletionMessageParam]
TARGETS = {"t300": 300, "t100": 100, "t50": 50, "t20": 20}
KEY_ORDER = ["t300", "t100", "t50", "t20"]
DEFAULT_TOLERANCE = 0.25  # 20 -> 15..25, 50 -> 38..62, 100 -> 75..125, 300 -> 225..375
WORDS_PER_TOKEN = 0.75
WORD_CALIBRATION = 1.35  # models undershoot a bare word target by 15-20 %; the token band stays the criterion
# Models follow a structure with a floor ("three paragraphs of about 110 words each, never fewer
# than 280 words") far better than a bare word count; see the README for the prompt experiment.
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

You are given ONE agent turn: its TEXT (the prose the agent wrote before the tool call; may be empty) and its TOOL CALL. Produce the prose for this turn at four target lengths.

Rules:
- Preserve the meaning and intent of TEXT. Do not invent facts, findings, file contents, or results that TEXT and the TOOL CALL do not state or imply. Longer versions elaborate on the same intent (what I am looking for, why this step, how it relates to the task, what I expect to learn); they never add new steps or new discoveries. Shorter versions condense the same content.
- If TEXT is empty, write what the agent would say just before this tool call: what the call does and why it helps, based only on the call itself. The call's `summary` parameter states the agent's own intent - use it as such, never refer to "the summary"; ignore the `security_risk` parameter entirely. For the longer versions, walk through the call concretely: which file, range, pattern or command it targets and what each part is for, what output I expect to see, and what I will do with it next.
- Do not guess what the overall task is about (bug, feature, docstring, ...) unless TEXT says so; when elaborating, stay with what this step examines or changes and what that tells the agent, rather than inventing specifics about the task or the code.
- Write in the agent's voice: first person, present tense, addressed to no one in particular (e.g. "Let me ...", "I'll ...", "Now I need to ..."). Keep the original tone, including openers like "Great!" or "I see the issue" when TEXT has them.
- Plain prose only. No headings, no lists unless TEXT used them, no code blocks, no XML tags, no quotation of the tool call, no meta-language about "the tool call", "the message" or these instructions.
- Write t300 first, then condense it to t100, then t50, then t20.

Length targets (1 token is about {wpt} words):
{targets}

Respond with ONLY a JSON object of the form {{"t300": "...", "t100": "...", "t50": "..." , "t20": "..."}}."""

RETRY_INSTRUCTIONS = """Some versions were outside their length band. Rewrite ONLY the versions listed below, condensing or expanding the SOURCE text (keep its meaning; same rules as before). Each version gets two candidates of different lengths, written independently:

{items}

Respond with ONLY a JSON object containing exactly these keys: {keys}."""


def band(key: str, tolerance: float) -> tuple[int, int]:
    t = TARGETS[key]
    return round(t * (1 - tolerance)), round(t * (1 + tolerance))


def words_for(key: str) -> int:
    return round(TARGETS[key] * WORDS_PER_TOKEN * WORD_CALIBRATION)


def system_prompt(tolerance: float) -> str:
    targets = "\n".join(
        f"- {k}: {TARGETS[k]} tokens = {STRUCT[k]}; acceptable {lo}-{hi} tokens"
        for k in KEY_ORDER
        for lo, hi in [band(k, tolerance)]
    )
    return SYSTEM_PROMPT.format(wpt=WORDS_PER_TOKEN, targets=targets)


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


def first_round_messages(text: str, call: str, tolerance: float) -> Messages:
    return [
        {"role": "system", "content": system_prompt(tolerance)},
        {"role": "user", "content": turn_block(text, call)},
    ]


def retry_messages(
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
    for k in KEY_ORDER:
        if k not in failing:
            continue
        lo, hi = band(k, tolerance)
        n, w = failing[k]
        if n == 0:
            want = words_for(k)
            alt = round(want * 1.25)
            why = "your previous attempt was missing or invalid"
        else:
            ratio = n / max(w, 1)
            want = round(TARGETS[k] / ratio)
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
            f"- {k}: {TARGETS[k]} tokens, acceptable {lo}-{hi} tokens; {why}. "
            f'Give two candidates: "{k}_a" of about {want} words and "{k}_b" of about {alt} words ({SHAPE[k]}); count the words.'
        )
        asked += [f"{k}_a", f"{k}_b"]
    keys = ", ".join(f'"{k}"' for k in asked)
    user = (
        f"{turn_block(text, call)}\n\nSOURCE (the version to condense or expand):\n{source}\n\n"
        + RETRY_INSTRUCTIONS.format(items="\n".join(items), keys=keys)
    )
    return [
        {"role": "system", "content": system_prompt(tolerance)},
        {"role": "user", "content": user},
    ]


_JSON_SPAN = re.compile(r"\{.*\}", re.DOTALL)
_KEY_MARK = re.compile(r'"(t300|t100|t50|t20)(_[a-z])?"\s*:\s*"')


def _unescape(v: str) -> str:
    return (
        v.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _lenient_extract(s: str) -> dict[str, str]:
    """Pull `"key": "value"` pairs out of almost-JSON, where quotes inside values are left unescaped."""
    marks = list(_KEY_MARK.finditer(s))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        key = m.group(1) + (m.group(2) or "")
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        chunk = re.sub(
            r'"\s*[,}]?\s*$', "", s[m.end() : end].rstrip()
        )  # closing quote (+ `,` or `}`)
        out[key] = _unescape(chunk)
    return out


def parse_versions(raw: str, keys: list[str]) -> dict[str, str]:
    """Extract the requested keys from the model's reply; missing or invalid keys are absent."""
    s = (raw or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s)
    m = _JSON_SPAN.search(s)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0), strict=False)
    except ValueError:
        obj = _lenient_extract(m.group(0))
    if not isinstance(obj, dict):
        return {}
    out: dict[str, str] = {}
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip() and not any(f in v for f in FORBIDDEN):
            out[k] = v.strip()
    return out


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
        request_timeout_s: float = 180.0,
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
        self, messages: Messages, temperature: float | None = None
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
                    max_tokens=self.max_tokens,
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
