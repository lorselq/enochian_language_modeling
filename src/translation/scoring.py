from __future__ import annotations

"""Soft scoring utilities for decomposition candidates (task 2.3)."""

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from enochian_lm.common.types import FastTextModel, MaybeNumber, Vector
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

    beam_prior: float = 0.25
    avg_cluster_quality: float = 0.20
    residual_coverage: float = 0.20
    acceptance_bonus: float = 0.15
    specificity_bonus: float = 0.20  # Reward morphs with fewer definitions

    def normalized(self) -> "ScoringWeights":
        total = (
            self.beam_prior
            + self.avg_cluster_quality
            + self.residual_coverage
            + self.acceptance_bonus
            + self.specificity_bonus
        )
        if total <= 0:
            # Fall back to equal weights to avoid division by zero and keep the
            # scoring function usable.
            return ScoringWeights(0.2, 0.2, 0.2, 0.2, 0.2)

        scale = 1.0 / total
        return ScoringWeights(
            beam_prior=self.beam_prior * scale,
            avg_cluster_quality=self.avg_cluster_quality * scale,
            residual_coverage=self.residual_coverage * scale,
            acceptance_bonus=self.acceptance_bonus * scale,
            specificity_bonus=self.specificity_bonus * scale,
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
    - specificity_bonus: rewards morphs with fewer definitions (more specific).
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

    specificity = _specificity_bonus(
        decomp.morphs,
        evidence.direct_clusters,
    )

    return (
        active.beam_prior * beam_prior
        + active.avg_cluster_quality * avg_cluster_quality
        + active.residual_coverage * residual_coverage
        + active.acceptance_bonus * acceptance
        + active.specificity_bonus * specificity
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

    specificity = _specificity_bonus(
        decomp.morphs,
        evidence.direct_clusters,
    )

    return beam_prior + avg_cluster_quality + residual_coverage + acceptance + specificity


def _safe_number(value: MaybeNumber, default: float = 0.0) -> float:
    """Best-effort cast to float, used at the edges of the pipeline."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _average_cluster_quality(
    morphs: Iterable[str],
    clusters: list[ClusterRecord],
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
    best_scores: dict[str, float] = {}
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
    clusters: list[ClusterRecord],
    residuals: list[ResidualSemanticRecord],
    hypotheses: list[MorphHypothesisRecord],
) -> float:
    """Calculate acceptance bonus for a set of morphs.

    Evidence is compressed, length-weighted, and normalized to reduce the
    incentive to split a supported morph into many sub-morphs.
    """

    cluster_counts: dict[str, int] = {}
    for cluster in clusters:
        key = cluster.ngram.upper()
        cluster_counts[key] = cluster_counts.get(key, 0) + 1

    residual_counts: dict[str, int] = {}
    for residual in residuals:
        key = residual.residual.upper()
        residual_counts[key] = residual_counts.get(key, 0) + 1

    hypothesis_counts: dict[str, int] = {}
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


def _specificity_bonus(
    morphs: Iterable[str],
    clusters: list[ClusterRecord],
) -> float:
    """Calculate specificity bonus rewarding morphs with fewer definitions.

    Morphs with fewer cluster definitions are more specific/less ambiguous.
    For example, NAZ with 1 definition is preferred over NA with 5 definitions.

    Returns a score in [0, 1] where higher means more specific.
    """
    # Count distinct definitions per morph
    morph_def_counts: dict[str, int] = {}
    for cluster in clusters:
        key = cluster.ngram.upper()
        morph_def_counts[key] = morph_def_counts.get(key, 0) + 1

    morph_list = [m.upper() for m in morphs]
    if not morph_list:
        return 0.0

    total_len = sum(len(m) for m in morph_list)
    if total_len <= 0:
        return 0.0

    # Calculate length-weighted specificity
    # specificity = 1 / (1 + log(num_definitions))
    # This gives: 1 def -> 1.0, 2 defs -> 0.59, 5 defs -> 0.38, 10 defs -> 0.30
    weighted_specificity = 0.0
    for morph in morph_list:
        num_defs = morph_def_counts.get(morph, 0)
        if num_defs == 0:
            # No definitions = unknown, neutral specificity
            specificity = 0.5
        else:
            specificity = 1.0 / (1.0 + math.log1p(num_defs - 1))
        # Weight by morph length (longer morphs matter more)
        weighted_specificity += len(morph) * specificity

    return weighted_specificity / float(total_len)


# ---------------------------------------------------------------------------
# Semantic coherence scoring for singleton validation
# ---------------------------------------------------------------------------


@dataclass
class CoherenceResult:
    """Result of semantic coherence analysis for a decomposition.

    Attributes
    ----------
    score:
        Overall coherence score in [0, 1]. Higher is better.
    singleton_cohesion:
        Average pairwise similarity among singletons. Higher means
        singletons are semantically related (good).
    large_morph_diversity:
        Average pairwise dissimilarity among larger morphs. Higher means
        large morphs are semantically distinct (good).
    singleton_count:
        Number of singleton morphs in the decomposition.
    large_morph_count:
        Number of non-singleton morphs in the decomposition.
    """

    score: float
    singleton_cohesion: float
    large_morph_diversity: float
    singleton_count: int
    large_morph_count: int


def compute_semantic_coherence(
    morphs: list[str],
    fasttext_model: FastTextModel | None,
    *,
    cohesion_threshold: float = 0.3,
    diversity_threshold: float = 0.7,
) -> CoherenceResult:
    """Compute semantic coherence for a decomposition.

    The coherence model is based on the observation that:
    - Singletons (single-letter morphs) act as grammatical particles and should
      be semantically similar to each other (they cohere).
    - Larger morphs carry distinct semantic content and should be dissimilar
      from each other (they diversify meaning).

    For a decomposition like NAZ + P + SA + D:
    - P and D (singletons) should have high similarity to each other.
    - NAZ and SA (large morphs) should have low similarity to each other.

    Parameters
    ----------
    morphs:
        List of morph strings from a decomposition.
    fasttext_model:
        A FastText model with `wv` attribute supporting `get_word_vector`.
    cohesion_threshold:
        Minimum expected similarity among singletons. Pairs below this
        are considered incoherent.
    diversity_threshold:
        Maximum expected similarity among large morphs. Pairs above this
        are considered too similar (redundant).

    Returns
    -------
    CoherenceResult:
        Contains the overall coherence score and component metrics.
    """
    if not morphs or fasttext_model is None:
        return CoherenceResult(
            score=0.5,  # Neutral score when we can't compute
            singleton_cohesion=0.0,
            large_morph_diversity=0.0,
            singleton_count=0,
            large_morph_count=0,
        )

    # Separate morphs by length
    singletons = [m.upper() for m in morphs if len(m) == 1]
    large_morphs = [m.upper() for m in morphs if len(m) > 1]

    # Get vectors for all morphs
    vectors: dict[str, Vector] = {}
    wv = getattr(fasttext_model, "wv", fasttext_model)
    for morph in set(singletons + large_morphs):
        try:
            vec = wv.get_word_vector(morph.lower())
            if vec is not None:
                vectors[morph] = np.asarray(vec, dtype=float)
        except (KeyError, AttributeError):
            pass

    # Compute singleton cohesion (should be HIGH for good decompositions)
    singleton_cohesion = _compute_avg_pairwise_similarity(
        singletons, vectors
    )

    # Compute large morph diversity (should be LOW similarity = HIGH diversity)
    large_morph_similarity = _compute_avg_pairwise_similarity(
        large_morphs, vectors
    )
    large_morph_diversity = 1.0 - large_morph_similarity

    # Compute overall coherence score
    # - Reward high singleton cohesion (above threshold)
    # - Reward high large morph diversity (similarity below threshold)
    cohesion_score = 0.0
    if len(singletons) >= 2:
        if singleton_cohesion >= cohesion_threshold:
            cohesion_score = min(1.0, singleton_cohesion / cohesion_threshold)
        else:
            # Penalize low cohesion among singletons
            cohesion_score = singleton_cohesion / cohesion_threshold * 0.5

    diversity_score = 0.0
    if len(large_morphs) >= 2:
        if large_morph_similarity <= diversity_threshold:
            diversity_score = 1.0 - (large_morph_similarity / diversity_threshold)
        else:
            # Penalize high similarity (redundancy) among large morphs
            diversity_score = 0.0

    # Combine scores - weight cohesion higher for singleton-heavy decompositions
    singleton_weight = len(singletons) / max(1, len(morphs))
    large_weight = len(large_morphs) / max(1, len(morphs))

    if len(singletons) >= 2 and len(large_morphs) >= 2:
        # Both groups present - blend scores
        score = singleton_weight * cohesion_score + large_weight * diversity_score
    elif len(singletons) >= 2:
        # Only singletons to compare - use cohesion only
        score = cohesion_score
    elif len(large_morphs) >= 2:
        # Only large morphs to compare - use diversity only
        score = diversity_score
    else:
        # Can't compare pairs - neutral score
        score = 0.5

    return CoherenceResult(
        score=score,
        singleton_cohesion=singleton_cohesion,
        large_morph_diversity=large_morph_diversity,
        singleton_count=len(singletons),
        large_morph_count=len(large_morphs),
    )


def _compute_avg_pairwise_similarity(
    morphs: list[str],
    vectors: dict[str, Vector],
) -> float:
    """Compute average pairwise cosine similarity among morphs.

    Returns 0.0 if fewer than 2 morphs have vectors.
    """
    morph_vecs = [(m, vectors[m]) for m in morphs if m in vectors]
    if len(morph_vecs) < 2:
        return 0.0

    similarities: list[float] = []
    for i in range(len(morph_vecs)):
        for j in range(i + 1, len(morph_vecs)):
            _, vec_i = morph_vecs[i]
            _, vec_j = morph_vecs[j]
            sim = _cosine_similarity(vec_i, vec_j)
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


def _cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def score_decomposition_with_coherence(
    decomp: Decomposition,
    evidence: WordEvidence,
    fasttext_model: FastTextModel | None,
    *,
    weights: ScoringWeights | None = None,
    coherence_weight: float = 0.15,
) -> tuple[float, CoherenceResult]:
    """Score a decomposition including semantic coherence.

    This extends the standard scoring with a coherence component that
    validates singleton usage. The coherence score is blended with
    the base score using ``coherence_weight``.

    Parameters
    ----------
    decomp:
        The decomposition to score.
    evidence:
        Word evidence for the decomposition.
    fasttext_model:
        FastText model for computing semantic similarity.
    weights:
        Optional scoring weights for the base score.
    coherence_weight:
        Weight for the coherence component (default 0.15).
        The base score weight is (1 - coherence_weight).

    Returns
    -------
    tuple[float, CoherenceResult]:
        The combined score and the coherence analysis result.
    """
    base_score = score_decomposition(decomp, evidence, weights=weights)
    coherence = compute_semantic_coherence(decomp.morphs, fasttext_model)

    # Blend base score with coherence
    combined = (1.0 - coherence_weight) * base_score + coherence_weight * coherence.score

    return combined, coherence
