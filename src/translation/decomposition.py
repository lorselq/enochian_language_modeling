from __future__ import annotations

"""Segmentation and decomposition utilities for task 2.1.

This module wraps MorphemeCandidateFinder to produce Decomposition objects
that can be further filtered and scored in later tasks (2.2 / 2.3).
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from .repository import WordEvidence

LOGGER = logging.getLogger(__name__)


@dataclass
class MorphSupportStats:
    has_cluster: bool = False
    has_residual: bool = False
    hypothesis_max: Optional[float] = None  # max delta_cosine across hypotheses
    uses: int = 0  # total appearances across evidence


@dataclass
class Decomposition:
    """Represent a single segmentation hypothesis for a word.

    Parameters
    ----------
    morphs:
        Ordered list of morph strings representing the segmentation path.
        All morphs are normalized to uppercase.
    beam_score:
        Aggregate TF–IDF / beam-search score returned by ``segment_target``.
    breakdown:
        Coverage metadata describing how much of the surface form is explained.
        At minimum this includes:
            - ``segments``: list of covered spans or segment records
            - ``uncovered``: list of residual spans, if any
            - ``coverage_ratio``: fraction of characters covered by morphs
            - ``residual_ratio``: 1.0 - coverage_ratio
    morph_support:
        Mapping of morph → provenance label:
            - ``"cluster"``
            - ``"residual"``
            - ``"hypothesis"``
            - ``"unknown"``
        based on the supplied :class:`WordEvidence`.
    """

    morphs: List[str]
    beam_score: float
    breakdown: Dict[str, object]
    morph_support: Dict[str, str] = field(default_factory=dict)


class DecompositionEngine:
    """Generate candidate decompositions using ``MorphemeCandidateFinder``.

    This is a thin wrapper around :meth:`MorphemeCandidateFinder.segment_target`
    that normalizes inputs, attaches evidence-aware support labels, and
    constructs :class:`Decomposition` instances for downstream filtering and
    scoring.
    """

    def __init__(self, candidate_finder: MorphemeCandidateFinder) -> None:
        self.candidate_finder = candidate_finder

    def generate_decompositions(
        self, word: str, evidence: WordEvidence
    ) -> List[Decomposition]:
        """Return all plausible decompositions for ``word``.

        The beam-search is delegated to ``segment_target``. Each returned path
        is wrapped as a :class:`Decomposition` and enriched with:

        - normalized morph strings (uppercase)
        - the raw beam score
        - a coverage breakdown
        - per-morph support labels derived from ``evidence``

        Notes
        -----
        * This method deliberately does **not** deduplicate decompositions that
          share the same morph sequence. Later scoring / filtering stages are
          responsible for letting competing analyses "fight it out".
        """

        if not word:
            return []

        normalized = word.upper()
        parses = self.candidate_finder.segment_target(normalized)

        decompositions: List[Decomposition] = []

        support_lookup = _build_support_lookup(evidence)

        for path, score, _ngram_scores, coverage in parses:
            # Normalize morphs so that lookups and downstream comparisons are
            # case-insensitive and consistent with the evidence index.
            morphs = [str(m).upper() for m in path]

            breakdown = _build_breakdown(self.candidate_finder, normalized, coverage)

            morph_support = {
                morph: _classify_support(morph, support_lookup) for morph in morphs
            }

            decompositions.append(
                Decomposition(
                    morphs=morphs,
                    beam_score=float(score),
                    breakdown=breakdown,
                    morph_support=morph_support,
                )
            )

        return decompositions


def _build_breakdown(
    candidate_finder: MorphemeCandidateFinder,
    word: str,
    coverage: list[dict[str, float | int | str]],
) -> Dict[str, object]:
    """Build a coverage breakdown dictionary.

    Preference order:

    1. If ``candidate_finder`` exposes a private ``_build_breakdown`` helper,
       delegate to it for a canonical representation.
    2. Otherwise, fall back to a minimal but well-formed coverage summary that
       exposes the keys expected by later scoring steps.
    """

    coverage_list: list[dict[str, float | int | str]] = list(coverage)

    # Prefer the candidate-finder's own helper if it exists.
    if hasattr(candidate_finder, "_build_breakdown"):
        build_method: Callable[
            [str, list[dict[str, float | int | str]]], dict[str, object]
        ] = getattr(candidate_finder, "_build_breakdown")
        return build_method(word, coverage_list)

    # Fallback: compute simple coverage / residual spans from ``start`` / ``end``
    # fields if available. This is intentionally conservative and is only used
    # when the helper is missing.
    word_len = len(word)
    if word_len == 0:
        return {
            "segments": coverage_list,
            "uncovered": [],
            "coverage_ratio": 0.0,
            "residual_ratio": 1.0,
        }

    # Normalize segments into [start, end) ranges within [0, word_len]
    ranges: List[Tuple[int, int]] = []
    for segment in coverage_list:
        start_raw = segment.get("start", 0)
        end_raw = segment.get("end", start_raw)
        try:
            start = int(start_raw)
            end = int(end_raw)
        except (TypeError, ValueError):
            continue

        start = max(0, min(word_len, start))
        end = max(start, min(word_len, end))
        if start < end:
            ranges.append((start, end))

    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    covered = sum(end - start for start, end in merged)
    coverage_ratio = covered / word_len if word_len else 0.0

    # Build uncovered spans between merged covered ranges.
    uncovered: List[Dict[str, List[int]]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            uncovered.append({"span": [cursor, start]})
        cursor = end
    if cursor < word_len:
        uncovered.append({"span": [cursor, word_len]})

    return {
        "segments": coverage_list,
        "uncovered": uncovered,
        "coverage_ratio": coverage_ratio,
        "residual_ratio": 1.0 - coverage_ratio,
    }


def _build_support_lookup(evidence: WordEvidence) -> Dict[str, str]:
    """Build a case-insensitive lookup table of morph → support label.

    Priority is:

    1. direct clusters (``"cluster"``)
    2. residual semantics (``"residual"``)
    3. morph hypotheses (``"hypothesis"``)

    Later sources never overwrite earlier ones for the same morph key.
    """

    # Normalize everything to uppercase for consistent lookups.
    support: Dict[str, str] = {}

    for cluster in evidence.direct_clusters:
        key = cluster.ngram.upper()
        support.setdefault(key, "cluster")

    for residual in evidence.residual_semantics:
        key = residual.residual.upper()
        support.setdefault(key, "residual")

    for hypothesis in evidence.morph_hypotheses:
        key = hypothesis.morph.upper()
        support.setdefault(key, "hypothesis")

    return support


def _classify_support(morph: str, support_lookup: Dict[str, str]) -> str:
    """Return the provenance label for ``morph`` given a support lookup."""
    return support_lookup.get(morph.upper(), "unknown")


def _residual_ratio(decomp: Decomposition) -> float:
    """Return the residual ratio from a decomposition breakdown.

    Defaults to 1.0 when missing or malformed to keep filtering conservative.
    """
    value = decomp.breakdown.get("residual_ratio", 1.0)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 1.0
    return 1.0


def _compile_support_stats(evidence: WordEvidence) -> Dict[str, MorphSupportStats]:
    """Aggregate per-morph support details for filtering decisions."""

    stats: Dict[str, MorphSupportStats] = {}

    def ensure(key: str) -> MorphSupportStats:
        key = key.upper()
        if key not in stats:
            stats[key] = MorphSupportStats()
        return stats[key]

    # Clusters
    for cluster in evidence.direct_clusters:
        entry = ensure(cluster.ngram)
        entry.has_cluster = True
        entry.uses += 1

    # Residual semantics
    for residual in evidence.residual_semantics:
        entry = ensure(residual.residual)
        entry.has_residual = True
        entry.uses += 1

    # Hypotheses
    for hypothesis in evidence.morph_hypotheses:
        entry = ensure(hypothesis.morph)
        entry.uses += 1
        delta = hypothesis.delta_cosine
        if delta is None:
            continue
        if entry.hypothesis_max is None or delta > entry.hypothesis_max:
            entry.hypothesis_max = float(delta)

    return stats


def apply_hard_filters(
    decompositions: List[Decomposition],
    evidence: WordEvidence,
    min_support_threshold: float = 0.2,
) -> List[Decomposition]:
    """Apply the hard-filtering rules described in task 2.2.

    Filters are applied in order:

    1. Every morph must be supported by clusters, residuals, or sufficiently
       strong hypotheses (delta_cosine >= min_support_threshold).
    2. Decompositions with high residual ratios (>0.5) are discarded when a
       better-covered alternative exists.
    3. When possible, decompositions that use well-attested morphs (>3 uses)
       are preferred over those composed entirely of singletons.
    """

    if not decompositions:
        return []

    support_stats = _compile_support_stats(evidence)

    def has_support(morph: str) -> bool:
        stats = support_stats.get(morph.upper())
        if stats is None:
            return False
        if stats.has_cluster or stats.has_residual:
            return True
        if stats.hypothesis_max is None:
            return False
        return stats.hypothesis_max >= min_support_threshold

    # ------------------------------------------------------------------
    # Filter 1: every morph must have support.
    # ------------------------------------------------------------------
    supported: List[Decomposition] = []
    for decomp in decompositions:
        missing = [m for m in decomp.morphs if not has_support(m)]
        if missing:
            LOGGER.debug(
                "Discarding %s due to unsupported morphs: %s",
                decomp.morphs,
                missing,
            )
            continue
        supported.append(decomp)

    if not supported:
        return []

    # ------------------------------------------------------------------
    # Filter 2: discard high-residual decomps when a better exists.
    # ------------------------------------------------------------------
    min_residual = min(_residual_ratio(d) for d in supported)
    coverage_filtered: List[Decomposition] = []
    for decomp in supported:
        ratio = _residual_ratio(decomp)
        if ratio > 0.5 and min_residual <= 0.5:
            LOGGER.debug(
                "Discarding %s due to high residual_ratio %.3f (best=%.3f)",
                decomp.morphs,
                ratio,
                min_residual,
            )
            continue
        coverage_filtered.append(decomp)

    if not coverage_filtered:
        return []

    # ------------------------------------------------------------------
    # Filter 3: prefer well-attested morphs (>3 uses) over pure singletons.
    # ------------------------------------------------------------------
    def attestation_score(decomp: Decomposition) -> int:
        score = 0
        for morph in decomp.morphs:
            stats = support_stats.get(morph.upper())
            if stats is not None and stats.uses > 3:
                score += 1
        return score

    max_attestation = max(attestation_score(d) for d in coverage_filtered)
    if max_attestation == 0:
        # Nobody uses well-attested morphs; let soft scoring handle it.
        return coverage_filtered

    attested: List[Decomposition] = []
    for decomp in coverage_filtered:
        score = attestation_score(decomp)
        if score == 0:
            LOGGER.debug(
                "Discarding %s due to singleton-only morph usage (best score=%d)",
                decomp.morphs,
                max_attestation,
            )
            continue
        attested.append(decomp)

    return attested if attested else coverage_filtered
