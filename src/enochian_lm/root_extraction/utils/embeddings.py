"""Shared embedding resources and helpers for crews and tools.

This module centralizes access to heavyweight embedding models so that
long-running crews only need to instantiate them once per process.  It
also collects a few lightweight helpers that were previously duplicated
across modules.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from gensim.models import FastText
from sentence_transformers import SentenceTransformer

from enochian_lm.common.config import get_config_paths

__all__ = [
    "get_fasttext_model",
    "get_sentence_transformer",
    "safe_output",
    "select_definitions",
    "stream_text",
]


# --- FastText wrapper + caching -------------------------------------------------


class FastTextWrapper:
    """
    Thin wrapper around a FastText-like model that normalizes vector access.

    It guarantees a .get_word_vector(token: str) method, regardless of whether
    the underlying model is:
      - Facebook fasttext (C++/python bindings),
      - gensim.models.FastText,
      - gensim KeyedVectors, etc.

    All other attributes/methods are proxied to the underlying model.
    """

    def __init__(self, model: FastText):
        self._model = model

    def get_word_vector(self, token: str):
        m = self._model

        # Facebook fasttext-style API
        if hasattr(m, "get_word_vector"):
            return m.get_word_vector(token)

        # gensim FastText: vectors live under .wv
        if hasattr(m, "wv") and hasattr(m.wv, "get_vector"):
            return m.wv.get_vector(token)

        # gensim KeyedVectors or similar
        if hasattr(m, "get_vector"):
            return m.get_vector(token)

        raise TypeError(
            f"Cannot obtain vector for token {token!r} from model of type {type(m)!r}"
        )

    def __getattr__(self, name: str):
        # Delegate everything else (most_similar, wv, etc.) to the real model
        return getattr(self._model, name)


_FASTTEXT_LOCK = Lock()
_FASTTEXT_MODEL: Optional[FastTextWrapper] = None
_FASTTEXT_PATH: Optional[str] = None

_SENTENCE_LOCK = Lock()
_SENTENCE_MODELS: Dict[str, SentenceTransformer] = {}


def _resolve_fasttext_path(model_path: Optional[Path | str]) -> str:
    if model_path is not None:
        return str(model_path)
    paths = get_config_paths()
    return str(paths["model_output"])


def get_fasttext_model(model_path: Optional[Path | str] = None) -> FastTextWrapper:
    """Return a process-wide FastText wrapper, loading the underlying model on first use."""

    global _FASTTEXT_MODEL, _FASTTEXT_PATH

    desired_path = _resolve_fasttext_path(model_path)

    with _FASTTEXT_LOCK:
        # Cache hit with same path: reuse wrapper
        if _FASTTEXT_MODEL is not None and _FASTTEXT_PATH == desired_path:
            return _FASTTEXT_MODEL

        # Load gensim FastText model and wrap it
        base_model = FastText.load(desired_path)
        wrapped = FastTextWrapper(base_model)

        _FASTTEXT_MODEL = wrapped
        _FASTTEXT_PATH = desired_path
        return wrapped


# --- Sentence-transformer caching ----------------------------------------------


def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Return a cached sentence-transformer instance by model name."""

    with _SENTENCE_LOCK:
        model = _SENTENCE_MODELS.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _SENTENCE_MODELS[model_name] = model
        return model


# --- Misc helpers --------------------------------------------------------------


def stream_text(text: str, *, delay: float = 0.001) -> None:
    """Emit text to stdout with a small delay between characters."""

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            break


def select_definitions(def_list, max_words: int = 75):
    """Select definitions without exceeding ``max_words``."""

    selected = []
    total_words = 0

    for definition in def_list:
        bracket_index = definition.find(" [")
        if bracket_index != -1:
            word_slice = definition[:bracket_index]
        else:
            word_slice = definition

        word_count = len(word_slice.split())
        if total_words + word_count > max_words:
            break

        selected.append(definition)
        total_words += word_count

    return selected


def safe_output(crew_output) -> dict:
    """Defensively extract the raw output dictionary from a Crew result."""

    if not crew_output:
        return {}

    try:
        return getattr(crew_output, "raw_output", {})
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[!] Failed to extract output: {exc}")
        return {}
