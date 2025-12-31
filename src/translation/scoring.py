from __future__ import annotations

"""Soft scoring utilities for decomposition candidates (task 2.3)."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List

from .decomposition import Decomposition
from .repository import (
    ClusterRecord,
    MorphHypothesisRecord,
    ResidualSemanticRecord,
    WordEvidence,
)


@dataclass
class ScoringWeights:
    """Configurable weights for the composite score.

    If custom weights do not sum to 1.0 they are normalized 
    automatically so the final score remains comparable 
    across configurations.
    """

    beam_prior: float = 0.3
    avg_cluster_quality: float = 0.25
    residual_coverage: float = 0.25
    acceptance_bonus: float = 0.2

    def normalized(self) -> "ScoringWeights":
        total = (
            self.beam_prior
            + self.avg_cluster_quality
            + self.residual_coverage
            + self.acceptance_bonus
        )
        if total <= 0:
            # Fall back to equal weights to avoid division by zero and keep the
            # scoring function usable.
            return ScoringWeights(0.25, 0.25, 0.25, 0.25)

        scale = 1.0 / total
        return ScoringWeights(
            beam_prior=self.beam_prior * scale,
            avg_cluster_quality=self.avg_cluster_quality * scale,
            residual_coverage=self.residual_coverage * scale,
            acceptance_bonus=self.acceptance_bonus * scale,
        )


def score_decomposition(
    decomp: Decomposition,
    evidence: WordEvidence,
    weights: ScoringWeights | None = None,
) -> float:
    """Return a composite soft score for ``decomp``.

    Components:

    - beam_prior: raw beam / TF–IDF score.
    - avg_cluster_quality: per-morph quality based on cluster cohesion /
      semantic_coverage.
    - residual_coverage: 1 - residual_ratio from the decomposition breakdown.
    - acceptance_bonus: counts of clusters, residuals, and hypotheses aligned
      with the morphs (weighted 1.0 / 0.5 / 0.3).
    """

    active = (weights or ScoringWeights()).normalized()

    beam_raw = (
        decomp.beam_score_normalized
        if decomp.beam_score_normalized is not None
        else decomp.beam_score
    )
    beam_prior = _safe_number(beam_raw, default=0.0)
    avg_cluster_quality = _average_cluster_quality(
        decomp.morphs,
        evidence.direct_clusters,
    )

    residual_ratio = _safe_number(decomp.breakdown.get("residual_ratio"), default=1.0)
    residual_coverage = max(0.0, min(1.0, 1.0 - residual_ratio))

    acceptance = _acceptance_bonus(
        decomp.morphs,
        evidence.direct_clusters,
        evidence.residual_semantics,
        evidence.morph_hypotheses,
    )

    return (
        active.beam_prior * beam_prior
        + active.avg_cluster_quality * avg_cluster_quality
        + active.residual_coverage * residual_coverage
        + active.acceptance_bonus * acceptance
    )


def score_decomposition_unweighted(
    decomp: Decomposition,
    evidence: WordEvidence,
) -> float:
    """Return an unweighted composite score for ``decomp``.

    This sums the raw component scores without applying the normalized
    ScoringWeights multipliers.
    """

    beam_raw = (
        decomp.beam_score_normalized
        if decomp.beam_score_normalized is not None
        else decomp.beam_score
    )
    beam_prior = _safe_number(beam_raw, default=0.0)
    avg_cluster_quality = _average_cluster_quality(
        decomp.morphs,
        evidence.direct_clusters,
    )

    residual_ratio = _safe_number(decomp.breakdown.get("residual_ratio"), default=1.0)
    residual_coverage = max(0.0, min(1.0, 1.0 - residual_ratio))

    acceptance = _acceptance_bonus(
        decomp.morphs,
        evidence.direct_clusters,
        evidence.residual_semantics,
        evidence.morph_hypotheses,
    )

    return beam_prior + avg_cluster_quality + residual_coverage + acceptance


def _safe_number(value: Any, default: float = 0.0) -> float:
    """Best-effort cast to float, used at the edges of the pipeline."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average_cluster_quality(
    morphs: Iterable[str],
    clusters: List[ClusterRecord],
) -> float:
    """Compute average cluster quality across morphs using cohesion / coverage.

    For each morph, the best available cluster quality is used. If no cluster
    exists for a morph, its contribution is 0.0. The average is weighted by
    morph length to avoid overvaluing many tiny morphs.
    """

    morph_list = [m.upper() for m in morphs]
    if not morph_list:
        return 0.0

    total_len = sum(len(morph) for morph in morph_list)
    if total_len <= 0:
        return 0.0

    morph_set = set(morph_list)

    # Best quality per morph.
    best_scores: Dict[str, float] = {}
    for cluster in clusters:
        morph = cluster.ngram.upper()
        if morph not in morph_set:
            # Ignore clusters that do not correspond to the current decomposition.
            continue

        metrics = [
            metric
            for metric in (cluster.cohesion, cluster.semantic_coverage)
            if metric is not None
        ]
        if not metrics:
            continue

        # If both cohesion and coverage are present, this is (cohesion + coverage) / 2.
        quality = sum(metrics) / float(len(metrics))
        if morph not in best_scores or quality > best_scores[morph]:
            best_scores[morph] = quality

    total_quality = 0.0
    for morph in morph_list:
        total_quality += len(morph) * best_scores.get(morph, 0.0)

    return total_quality / float(total_len)


def _acceptance_bonus(
    morphs: Iterable[str],
    clusters: List[ClusterRecord],
    residuals: List[ResidualSemanticRecord],
    hypotheses: List[MorphHypothesisRecord],
) -> float:
    """Calculate acceptance bonus for a set of morphs.

    Evidence is compressed, length-weighted, and normalized to reduce the
    incentive to split a supported morph into many sub-morphs.
    """

    cluster_counts: Dict[str, int] = {}
    for cluster in clusters:
        key = cluster.ngram.upper()
        cluster_counts[key] = cluster_counts.get(key, 0) + 1

    residual_counts: Dict[str, int] = {}
    for residual in residuals:
        key = residual.residual.upper()
        residual_counts[key] = residual_counts.get(key, 0) + 1

    hypothesis_counts: Dict[str, int] = {}
    for hypothesis in hypotheses:
        key = hypothesis.morph.upper()
        hypothesis_counts[key] = hypothesis_counts.get(key, 0) + 1

    morph_list = [m.upper() for m in morphs]
    total_len = sum(len(morph) for morph in morph_list)
    if total_len <= 0:
        return 0.0

    bonus = 0.0
    for morph in morph_list:
        key = morph.upper()
        raw = (
            cluster_counts.get(key, 0)
            + 0.5 * residual_counts.get(key, 0)
            + 0.3 * hypothesis_counts.get(key, 0)
        )
        if raw <= 0:
            continue
        compressed = math.log1p(raw)
        bonus += len(morph) * compressed

    normalized = bonus / float(total_len)
    return min(1.5, normalized)
