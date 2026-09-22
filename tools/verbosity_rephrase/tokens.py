"""Token counting with the student models' tokenizer (Qwen3-4B/8B and Qwen2.5-Coder share one vocabulary)."""

from __future__ import annotations

import functools

from tokenizers import Tokenizer


TOKENIZER_REPO = "Qwen/Qwen3-8B"


@functools.lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TOKENIZER_REPO)


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids) if text else 0
