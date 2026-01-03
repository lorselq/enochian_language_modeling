from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class NgramSlice:
    word_index: int
    start: int
    end: int
    ngram: str


def tokenize_words(text: str) -> list[str]:
    """Extract alphabetic tokens preserving order."""
    return [match.group(0) for match in _WORD_RE.finditer(text)]


def iter_char_ngrams(word: str, *, max_len: int = 7) -> Iterator[tuple[int, int, str]]:
    lowered = word.lower()
    length = len(lowered)
    for start in range(length):
        for size in range(1, max_len + 1):
            end = start + size
            if end > length:
                break
            yield start, end, lowered[start:end]


def expand_sentence_ngrams(text: str, *, max_len: int = 7) -> list[NgramSlice]:
    slices: list[NgramSlice] = []
    for word_index, word in enumerate(tokenize_words(text)):
        for start, end, ngram in iter_char_ngrams(word, max_len=max_len):
            slices.append(
                NgramSlice(
                    word_index=word_index,
                    start=start,
                    end=end,
                    ngram=ngram,
                )
            )
    return slices
