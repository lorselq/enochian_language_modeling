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

_FASTTEXT_LOCK = Lock()
_FASTTEXT_MODEL: Optional[FastText] = None
_FASTTEXT_PATH: Optional[str] = None

_SENTENCE_LOCK = Lock()
_SENTENCE_MODELS: Dict[str, SentenceTransformer] = {}


def _resolve_fasttext_path(model_path: Optional[Path | str]) -> str:
    if model_path is not None:
        return str(model_path)
    paths = get_config_paths()
    return str(paths["model_output"])


def get_fasttext_model(model_path: Optional[Path | str] = None) -> FastText:
    """Return a process-wide FastText instance, loading it on first use."""

    global _FASTTEXT_MODEL, _FASTTEXT_PATH

    desired_path = _resolve_fasttext_path(model_path)

    with _FASTTEXT_LOCK:
        if _FASTTEXT_MODEL is not None and _FASTTEXT_PATH == desired_path:
            return _FASTTEXT_MODEL

        _FASTTEXT_MODEL = FastText.load(desired_path)
        _FASTTEXT_PATH = desired_path
        return _FASTTEXT_MODEL


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Return a cached sentence-transformer instance by model name."""

    with _SENTENCE_LOCK:
        model = _SENTENCE_MODELS.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _SENTENCE_MODELS[model_name] = model
        return model


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
