from __future__ import annotations

"""Strategy-based reranking utilities (task 3.1).

Implements a lightweight, interpretable reranking layer on top of Phase 2 scores.

Strategies
----------
prefer-fewer
    bonus = -0.5 * len(morphs)                 (favor fewer morphs)
prefer-known
    bonus = 0.3 * (count of morphs with uses > 5)  (favor well-attested morphs)
prefer-balance
    bonus = -0.2 * variance(lengths)           (favor similar morph lengths)

Returns the list sorted by (base_score + bonus), descending.
"""

from typing import Any, Dict, Iterable

from .decomposition import Decomposition
from .repository import (
    ClusterRecord,
    MorphHypothesisRecord,
    ResidualSemanticRecord,
    WordEvidence,
)


def apply_strategy(
    decompositions: list[tuple[Decomposition, float]],
    strategy: str,
    evidence: WordEvidence,
) -> list[tuple[Decomposition, float]]:
    """Apply the requested reranking strategy and return decompositions sorted by final score."""
    if not decompositions:
        return []

    strategy_key = (strategy or "").lower().strip()

    support_counts: Dict[str, int] | None = None
    if strategy_key == "prefer-known":
        support_counts = _compile_support_counts(evidence)

    reranked: list[tuple[Decomposition, float]] = []
    for decomp, base_score in decompositions:
        base = _safe_number(base_score, default=0.0)
        bonus = _strategy_bonus(
            decomp=decomp,
            strategy=strategy_key,
            support_counts=support_counts,
        )
        reranked.append((decomp, base + bonus))

    reranked.sort(key=lambda pair: pair[1], reverse=True)
    return reranked


def _strategy_bonus(
    *,
    decomp: Decomposition,
    strategy: str,
    support_counts: Dict[str, int] | None,
) -> float:
    if strategy == "prefer-fewer":
        # Spec: bonus = -0.5 * len(morphs)
        return -0.5 * len(decomp.morphs)

    if strategy == "prefer-known":
        if not support_counts:
            return 0.0
        attested = sum(
            1 for morph in decomp.morphs if support_counts.get((morph or "").upper(), 0) > 5
        )
        return 0.3 * attested

    if strategy == "prefer-balance":
        variance = _length_variance(decomp.morphs)
        return -0.2 * variance

    return 0.0


def _compile_support_counts(evidence: WordEvidence) -> Dict[str, int]:
    """Aggregate per-morph usage counts across evidence types (case-insensitive keys)."""
    counts: Dict[str, int] = {}

    def bump(key: str) -> None:
        norm = (key or "").upper()
        if not norm:
            return
        counts[norm] = counts.get(norm, 0) + 1

    for cluster in _safe_iter(evidence.direct_clusters):
        # ClusterRecord: uses .ngram (per your earlier code)
        bump(getattr(cluster, "ngram", ""))

    for residual in _safe_iter(evidence.residual_semantics):
        # ResidualSemanticRecord: uses .residual
        bump(getattr(residual, "residual", ""))

    for hypothesis in _safe_iter(evidence.morph_hypotheses):
        # MorphHypothesisRecord: uses .morph
        bump(getattr(hypothesis, "morph", ""))

    return counts


def _length_variance(morphs: Iterable[str]) -> float:
    lengths = [len(m or "") for m in morphs]
    if len(lengths) <= 1:
        return 0.0
    mean = sum(lengths) / float(len(lengths))
    return sum((length - mean) ** 2 for length in lengths) / float(len(lengths))


def _safe_iter(value: Iterable[object] | None) -> Iterable[object]:  # pragma: no cover
    return value or ()


def _safe_number(value: Any, *, default: float) -> float:  # pragma: no cover
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
