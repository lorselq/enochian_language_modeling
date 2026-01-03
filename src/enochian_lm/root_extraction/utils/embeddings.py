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
from typing import Dict, Mapping, Optional, Sequence, TypedDict

from gensim.models import FastText
import numpy as np
from sentence_transformers import SentenceTransformer

from enochian_lm.common.config import get_config_paths

__all__ = [
    "cluster_definition_counts",
    "cluster_definitions",
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
_SENTENCE_EMBED_CACHE: Dict[tuple[str, str], np.ndarray] = {}

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
    model: Optional[SentenceTransformer] = None,
    *,
    similarity_threshold: float = 0.8,
    scores: Optional[Sequence[Optional[float]]] = None,
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
    embeddings: list[Optional[np.ndarray]] = []
    for definition in definitions:
        key = (model_name, definition)
        cached = _SENTENCE_EMBED_CACHE.get(key)
        if cached is not None:
            embeddings.append(cached)
            continue
        if not definition:
            embeddings.append(None)
            continue
        encoded = embedder.encode(definition, normalize_embeddings=True)
        embedding = np.array(encoded, dtype=float)
        _SENTENCE_EMBED_CACHE[key] = embedding
        embeddings.append(embedding)

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
    definition_glosses: Mapping[str, Sequence[tuple[str, Optional[float]]]],
    model: Optional[SentenceTransformer] = None,
    *,
    similarity_threshold: float = 0.8,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> Dict[str, int]:
    """Return clustered definition counts per morph for semantic deduplication."""
    counts: Dict[str, int] = {}
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
