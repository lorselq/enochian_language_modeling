"""Shared embedding resources and helpers for crews and tools.

This module centralizes access to heavyweight embedding models so that
long-running crews only need to instantiate them once per process.  It
also collects a few lightweight helpers that were previously duplicated
across modules.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import TypedDict

from gensim.models import FastText
import numpy as np
from sentence_transformers import SentenceTransformer

from enochian_lm.common.config import get_config_paths
from enochian_lm.common.types import Vector

__all__ = [
    "cluster_definition_counts",
    "cluster_definitions",
    "get_fasttext_model",
    "get_sentence_transformer",
    "get_sentence_transformer_if_available",
    "safe_output",
    "select_definitions",
    "stream_text",
]


LOGGER = logging.getLogger(__name__)


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

    def get_word_vector(self, token: str) -> Vector:
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
_FASTTEXT_MODEL: FastTextWrapper | None = None
_FASTTEXT_PATH: str | None = None
_SENTENCE_EMBED_CACHE: dict[tuple[str, str], np.ndarray] = {}

_SENTENCE_LOCK = Lock()
_SENTENCE_MODELS: dict[str, SentenceTransformer] = {}
_SENTENCE_MODEL_FAILURES: set[tuple[str, bool]] = set()


def _resolve_fasttext_path(model_path: Path | str | None) -> str:
    if model_path is not None:
        return str(model_path)
    paths = get_config_paths()
    return str(paths["model_output"])


def get_fasttext_model(model_path: Path | str | None = None) -> FastTextWrapper:
    """Return a process-wide FastText wrapper, loading the underlying model on first use."""

    global _FASTTEXT_MODEL, _FASTTEXT_PATH

    desired_path = _resolve_fasttext_path(model_path)

    with _FASTTEXT_LOCK:
        # Cache hit with same path: reuse wrapper
        if _FASTTEXT_MODEL is not None and _FASTTEXT_PATH == desired_path:
            return _FASTTEXT_MODEL

        # Load gensim FastText model and wrap it.
        #
        # Some lightweight test shims expose a FastText-like class without a
        # `.load(...)` classmethod. In that constrained case we instantiate the
        # shim directly so higher-level translation tests can still run without
        # pulling the heavy model artifact.
        loader = getattr(FastText, "load", None)
        if callable(loader):
            base_model = loader(desired_path)
        else:  # pragma: no cover - exercised only in dependency-shim test setups
            base_model = FastText()
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


def get_sentence_transformer_if_available(
    model_name: str = "all-MiniLM-L6-v2",
    *,
    local_files_only: bool = False,
) -> SentenceTransformer | None:
    """Return a cached sentence-transformer, memoizing load failures.

    Translation needs semantic helpers when they are locally available, but
    `--no-llm` runs must never sit around repeatedly probing remote model
    hosts. This wrapper remembers failed loads and immediately returns `None`
    on later calls so deterministic translation can fall back to lexical
    heuristics instead of paying the same offline penalty over and over.
    """

    cache_key = (model_name, local_files_only)
    with _SENTENCE_LOCK:
        if cache_key in _SENTENCE_MODEL_FAILURES:
            return None
        cached = _SENTENCE_MODELS.get(model_name)
        if cached is not None:
            return cached

    try:
        if local_files_only:
            model = SentenceTransformer(model_name, local_files_only=True)
        else:
            model = SentenceTransformer(model_name)
    except TypeError:
        # Older sentence-transformers releases may not support
        # `local_files_only`. In strict local-only mode we prefer an immediate
        # deterministic fallback over risking a network probe.
        if local_files_only:
            with _SENTENCE_LOCK:
                _SENTENCE_MODEL_FAILURES.add(cache_key)
            return None
        raise
    except Exception as exc:  # pragma: no cover - exercised via translation fallbacks
        LOGGER.warning(
            "Sentence-transformer %s unavailable (local_only=%s): %s",
            model_name,
            local_files_only,
            exc,
        )
        with _SENTENCE_LOCK:
            _SENTENCE_MODEL_FAILURES.add(cache_key)
        return None

    with _SENTENCE_LOCK:
        _SENTENCE_MODELS[model_name] = model
    return model


# --- Semantic clustering helpers ------------------------------------------------


class DefinitionCluster(TypedDict):
    members: list[int]
    representative: int


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def cluster_definitions(
    definitions: Sequence[str],
    model: SentenceTransformer | None = None,
    *,
    similarity_threshold: float = 0.8,
    scores: Sequence[float | None] | None = None,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> list[DefinitionCluster]:
    """Cluster definition strings by semantic similarity using sentence embeddings.

    Returns a list of clusters with the index of a representative definition
    chosen by the highest provided score (or first entry if no score).
    """
    if not definitions:
        return []

    if scores is None:
        scores = [0.0] * len(definitions)
    elif len(scores) != len(definitions):
        raise ValueError("Scores must match the number of definitions.")

    embedder = model or get_sentence_transformer(model_name)
    embeddings: list[np.ndarray | None] = [None] * len(definitions)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for index, definition in enumerate(definitions):
        if not definition:
            continue
        key = (model_name, definition)
        cached = _SENTENCE_EMBED_CACHE.get(key)
        if cached is not None:
            embeddings[index] = cached
            continue
        uncached_indices.append(index)
        uncached_texts.append(definition)

    if uncached_texts:
        encoded_batch = embedder.encode(
            uncached_texts,
            normalize_embeddings=True,
        )

        encoded_rows: list[object] | None = None
        if len(uncached_texts) == 1:
            encoded_rows = [encoded_batch]
        elif isinstance(encoded_batch, np.ndarray):
            if encoded_batch.ndim == 2 and encoded_batch.shape[0] == len(uncached_texts):
                encoded_rows = [encoded_batch[row_index] for row_index in range(encoded_batch.shape[0])]
        elif isinstance(encoded_batch, Sequence) and not isinstance(encoded_batch, (str, bytes)):
            encoded_sequence = list(encoded_batch)
            if (
                len(encoded_sequence) == len(uncached_texts)
                and encoded_sequence
                and isinstance(encoded_sequence[0], (Sequence, np.ndarray))
                and not isinstance(encoded_sequence[0], (str, bytes))
            ):
                encoded_rows = encoded_sequence

        # Some lightweight test doubles still return one vector for list input.
        # Fall back to per-string encode in that case while keeping production
        # models on the batched path above.
        if encoded_rows is None:
            encoded_rows = [
                embedder.encode(text, normalize_embeddings=True)
                for text in uncached_texts
            ]

        for index, encoded in zip(uncached_indices, encoded_rows, strict=False):
            embedding = np.array(encoded, dtype=float)
            key = (model_name, definitions[index])
            _SENTENCE_EMBED_CACHE[key] = embedding
            embeddings[index] = embedding

    clusters: list[dict[str, object]] = []
    for idx, embedding in enumerate(embeddings):
        if embedding is None:
            clusters.append(
                {"sum": None, "count": 0, "members": [idx]}
            )
            continue

        best_idx = None
        best_sim = 0.0
        for c_idx, cluster in enumerate(clusters):
            centroid = cluster.get("sum")
            if centroid is None:
                continue
            centroid_vec = centroid / max(1, int(cluster.get("count", 1)))
            similarity = float(np.dot(embedding, centroid_vec))
            if similarity >= similarity_threshold and similarity > best_sim:
                best_sim = similarity
                best_idx = c_idx

        if best_idx is None:
            clusters.append(
                {"sum": embedding.copy(), "count": 1, "members": [idx]}
            )
        else:
            cluster = clusters[best_idx]
            cluster["sum"] = cluster["sum"] + embedding
            cluster["count"] = int(cluster.get("count", 0)) + 1
            cluster["members"].append(idx)

    output: list[DefinitionCluster] = []
    for cluster in clusters:
        members: list[int] = list(cluster["members"])
        best_member = members[0]
        best_score = scores[best_member] or 0.0
        for member in members[1:]:
            score = scores[member] or 0.0
            if score > best_score:
                best_score = score
                best_member = member
        output.append({"members": members, "representative": best_member})
    return output


def cluster_definition_counts(
    definition_glosses: Mapping[str, Sequence[tuple[str, float | None]]],
    model: SentenceTransformer | None = None,
    *,
    similarity_threshold: float = 0.8,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> dict[str, int]:
    """Return clustered definition counts per morph for semantic deduplication."""
    counts: dict[str, int] = {}
    for morph, glosses in definition_glosses.items():
        if not glosses:
            continue
        texts = [gloss for gloss, _score in glosses if gloss]
        scores = [score for _gloss, score in glosses if _gloss]
        if not texts:
            continue
        clusters = cluster_definitions(
            texts,
            model,
            similarity_threshold=similarity_threshold,
            scores=scores,
            model_name=model_name,
        )
        counts[morph.upper()] = len(clusters)
    return counts


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
