from __future__ import annotations

"""Segmentation and decomposition utilities for task 2.1.

This module wraps MorphemeCandidateFinder to produce Decomposition objects
that can be further filtered and scored in later tasks (2.2 / 2.3).
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, cast, Dict, Iterable, List, Tuple, Optional, TypedDict
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from .repository import WordEvidence

LOGGER = logging.getLogger(__name__)


class CoverageSegment(TypedDict):
    """Coverage segment from beam search segmentation."""

    start: int
    end: int
    ngram: str
    canonical: str
    tfidf: float


@dataclass
class MorphSupportStats:
    has_cluster: bool = False
    has_residual: bool = False
    has_attested: bool = False
    has_dictionary: bool = False
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
    canonicals:
        Ordered list of canonical dictionary entries corresponding to each morph
        slice. These are auxiliary and may be shorter/longer than ``morphs``
        when coverage data is incomplete.
    beam_score:
        Aggregate TF–IDF / beam-search score returned by ``segment_target``.
    beam_score_normalized:
        Beam score normalized to [0, 1] within the current candidate set.
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
    canonicals: List[str]
    beam_score: float
    breakdown: Dict[str, object] = field(default_factory=dict)
    beam_score_normalized: Optional[float] = None
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
        self,
        word: str,
        evidence: WordEvidence,
        *,
        force_dictionary: bool = False,
        allow_whole_word: bool = True,
        n_best: int | None = None,
        definition_counts: Dict[str, int] | None = None,
    ) -> tuple[List[Decomposition], Dict[str, object]]:
        """Return all plausible decompositions for ``word`` plus diagnostics.

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

        diagnostics: Dict[str, object] = {
            "fallback_used": False,
            "fallback_morphs": [],
            "parse_count": 0,
            "decomposition_count": 0,
            "extra_ngram_keys": 0,
            "extra_ngram_entries": 0,
            "dictionary_ngram_keys": 0,
            "dictionary_ngram_entries": 0,
            "dictionary_forced": force_dictionary,
        }

        if not word:
            return [], diagnostics

        normalized = word.upper()
        extra_ngrams = _build_evidence_ngrams(
            normalized,
            evidence,
            candidate_finder=self.candidate_finder,
            definition_counts=definition_counts,
        )
        diagnostics["extra_ngram_keys"] = len(extra_ngrams)
        diagnostics["extra_ngram_entries"] = sum(
            len(entries) for entries in extra_ngrams.values()
        )
        dictionary_ngrams: Dict[str, List[Tuple[str, int, int]]] = {}
        if force_dictionary:
            dictionary_ngrams = _build_dictionary_ngrams(
                normalized, candidate_finder=self.candidate_finder
            )
            diagnostics["dictionary_ngram_keys"] = len(dictionary_ngrams)
            diagnostics["dictionary_ngram_entries"] = sum(
                len(entries) for entries in dictionary_ngrams.values()
            )

        merged = (
            _merge_ngrams(extra_ngrams, dictionary_ngrams)
            if dictionary_ngrams
            else extra_ngrams
        )
        parses = self.candidate_finder.segment_target(
            normalized,
            extra_ngrams=merged,
            n_best=n_best,
            definition_counts=definition_counts,
        )
        diagnostics["parse_count"] = len(parses)

        if not parses and not force_dictionary:
            dictionary_ngrams = _build_dictionary_ngrams(
                normalized, candidate_finder=self.candidate_finder
            )
            diagnostics["dictionary_ngram_keys"] = len(dictionary_ngrams)
            diagnostics["dictionary_ngram_entries"] = sum(
                len(entries) for entries in dictionary_ngrams.values()
            )
            if dictionary_ngrams:
                merged = _merge_ngrams(extra_ngrams, dictionary_ngrams)
                parses = self.candidate_finder.segment_target(
                    normalized,
                    extra_ngrams=merged,
                    n_best=n_best,
                    definition_counts=definition_counts,
                )
                diagnostics["parse_count"] = len(parses)
                if parses:
                    diagnostics["fallback_used"] = True
                    diagnostics["fallback_morphs"] = sorted(
                        {
                            canon
                            for entries in dictionary_ngrams.values()
                            for canon, _, _ in entries
                        }
                    )

        decompositions: List[Decomposition] = []

        support_lookup = _build_support_lookup(evidence)

        for path, score, _ngram_scores, coverage in parses:
            segments = cast(list[CoverageSegment], coverage)
            morphs, canonicals = _segment_tokens(
                normalized, segments, path
            )
            if not allow_whole_word and len(normalized) > 1:
                if len(morphs) == 1 and morphs[0] == normalized:
                    continue
            breakdown = _build_breakdown(
                self.candidate_finder, normalized, segments
            )

            morph_support = {
                morph: _classify_support(morph, support_lookup) for morph in morphs
            }

            decompositions.append(
                Decomposition(
                    morphs=morphs,
                    canonicals=canonicals,
                    beam_score=float(score),
                    breakdown=breakdown,
                    morph_support=morph_support,
                )
            )

        _normalize_beam_scores(decompositions)
        diagnostics["decomposition_count"] = len(decompositions)
        return decompositions, diagnostics


def _build_breakdown(
    candidate_finder: MorphemeCandidateFinder,
    word: str,
    coverage: Iterable[CoverageSegment],
) -> Dict[str, object]:
    """Build a coverage breakdown dictionary.

    Preference order:

    1. If ``candidate_finder`` exposes a private ``_build_breakdown`` helper,
       delegate to it for a canonical representation.
    2. Otherwise, fall back to a minimal but well-formed coverage summary that
       exposes the keys expected by later scoring steps.
    """
    # Prefer the candidate-finder's own helper if it exists.
    if hasattr(candidate_finder, "_build_breakdown"):
        build_method: Callable[[str, list[CoverageSegment]], dict[str, object]] = (
            getattr(candidate_finder, "_build_breakdown")
        )
        return build_method(word, list(coverage))

    # Fallback: compute simple coverage / residual spans from ``start`` / ``end``
    # fields if available. This is intentionally conservative and is only used
    # when the helper is missing.
    coverage_list = list(coverage)
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


def _segment_tokens(
    word: str,
    coverage: List[CoverageSegment],
    canonicals: Iterable[str],
) -> tuple[List[str], List[str]]:
    """Return morph slices (ngrams) and canonicals for a segmentation."""
    ngrams: List[str] = []
    canon_list: List[str] = [str(c).upper() for c in canonicals]

    if coverage:
        for segment in coverage:
            ngram = segment.get("ngram")
            if isinstance(ngram, str) and ngram:
                ngrams.append(ngram.upper())

    if not ngrams:
        ngrams = [str(word).upper()] if word else []

    return ngrams, canon_list


def _normalize_beam_scores(decompositions: List[Decomposition]) -> None:
    if not decompositions:
        return

    adjusted_scores: List[float] = []
    for decomp in decompositions:
        morph_count = max(1, len(decomp.morphs))
        adjusted_scores.append(float(decomp.beam_score) / morph_count)

    min_score = min(adjusted_scores)
    max_score = max(adjusted_scores)

    if max_score == min_score:
        for decomp in decompositions:
            decomp.beam_score_normalized = 1.0
        return

    span = max_score - min_score
    for decomp, adjusted in zip(decompositions, adjusted_scores):
        decomp.beam_score_normalized = (adjusted - min_score) / span


def _build_support_lookup(evidence: WordEvidence) -> Dict[str, str]:
    """Build a case-insensitive lookup table of morph → support label.

    Priority is:

    1. direct clusters (``"cluster"``)
    2. residual semantics (``"residual"``)
    3. morph hypotheses (``"hypothesis"``)
    4. attested definitions (``"attested"``)
    5. dictionary matches (``"dictionary"``)

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

    for attested in evidence.attested_definitions:
        key = attested.source_word.upper()
        support.setdefault(key, "attested")
        root_key = attested.root_ngram.upper()
        support.setdefault(root_key, "attested")

    for morph in evidence.dictionary_morphs:
        support.setdefault(morph.upper(), "dictionary")

    return support


def _build_evidence_ngrams(
    word: str,
    evidence: WordEvidence,
    *,
    candidate_finder: MorphemeCandidateFinder,
    definition_counts: Dict[str, int] | None = None,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Build an extra ngram index based on evidence-backed morphs.

    Key change vs. the old behavior:
    - Treat evidence-backed morphs as *boost signals*, not just "missing ngrams".
    - Use a small DF so the boost meaningfully affects IDF.
    - Avoid boosting 1-char morphs and the full word (both encourage degenerate parses).
    """

    morphs: set[str] = set()
    morphs.update(cluster.ngram for cluster in evidence.direct_clusters)
    morphs.update(residual.residual for residual in evidence.residual_semantics)
    morphs.update(hypothesis.morph for hypothesis in evidence.morph_hypotheses)
    morphs.update(attested.source_word for attested in evidence.attested_definitions)
    morphs.update(attested.root_ngram for attested in evidence.attested_definitions)
    morphs.update(evidence.dictionary_morphs.keys())

    if not morphs or not word:
        return {}

    word = word.upper()

    # We only boost *multi-char* submorphs (otherwise the beam search will happily
    # prefer "N+A+Z+P+S+A+D" forever).
    min_len = max(candidate_finder.min_n, 2)
    max_len = candidate_finder.max_n

    total_docs = max(1, int(candidate_finder.total_docs))

    # Aggressive IDF boosts:
    df_edge = 1  # strongest boost for prefix/suffix candidates (NAZ, PSAD)
    df_inner = max(1, int(total_docs * 0.02))  # still strong, but not insane

    support_stats = _compile_support_stats(evidence)

    candidates: List[Tuple[int, int, int, str]] = []
    # rank tuple = (edge_bonus, length, uses, morph)
    for morph in morphs:
        if not morph:
            continue
        normalized = morph.strip().upper()
        if not normalized:
            continue
        if normalized not in word:
            continue
        if not (min_len <= len(normalized) <= max_len):
            continue
        if len(normalized) >= len(word):
            # Don't boost the full word; we're trying to get good *decompositions*.
            continue

        stats = support_stats.get(normalized)
        uses = stats.uses if stats is not None else 0
        edge_bonus = (
            1 if (word.startswith(normalized) or word.endswith(normalized)) else 0
        )
        candidates.append((edge_bonus, len(normalized), uses, normalized))

    if not candidates:
        return {}

    # Prefer edge morphs, then longer morphs, then higher-evidence morphs.
    candidates.sort(reverse=True)

    cap = 50  # keep this bounded per word
    extra: Dict[str, List[Tuple[str, int, int]]] = {}
    seen: set[str] = set()

    for edge_bonus, _ln, _uses, normalized in candidates:
        if normalized in seen:
            continue
        seen.add(normalized)

        key = normalized.lower()
        base_df = df_edge if edge_bonus else df_inner

        # Use definition counts to adjust DF for specificity:
        # - Fewer definitions = more specific = lower DF = higher IDF = better score
        # - More definitions = more ambiguous = higher DF = lower IDF = worse score
        if definition_counts:
            def_count = definition_counts.get(normalized, 0)
            if def_count > 0:
                # Scale DF by definition count: more defs = higher DF = lower score
                # Use log scale to prevent extreme values
                import math
                specificity_factor = math.log1p(def_count)  # 1 def -> 0.69, 28 defs -> 3.37
                df = max(1, int(base_df * (1 + specificity_factor)))
            else:
                df = base_df
        else:
            df = base_df

        # IMPORTANT: do NOT skip keys already in ngram_index.
        # We want evidence to be able to override / bias existing weights.
        extra.setdefault(key, []).append((normalized, 1, df))

        if len(seen) >= cap:
            break

    return extra


def _build_dictionary_ngrams(
    word: str,
    *,
    candidate_finder: MorphemeCandidateFinder,
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Build extra ngrams for dictionary-backed substrings in ``word``.

    This fallback helps when the ngram index is missing substrings that are
    present in the dictionary, enabling decomposition suggestions even when
    direct evidence is absent.
    """
    if not word:
        return {}

    total_docs = max(1, candidate_finder.total_docs)
    fallback_df = max(1, int(total_docs * 0.5))
    min_n = candidate_finder.min_n
    max_n = candidate_finder.max_n
    word_upper = word.upper()
    known_words = {w.upper() for w in candidate_finder.known_words}

    extra: Dict[str, List[Tuple[str, int, int]]] = {}
    for start in range(len(word_upper)):
        for end in range(start + min_n, min(len(word_upper), start + max_n) + 1):
            slice_text = word_upper[start:end]
            if slice_text not in known_words:
                continue
            key = slice_text.lower()
            if key in candidate_finder.ngram_index:
                continue
            extra.setdefault(key, []).append((slice_text, 1, fallback_df))

    return extra


def _merge_ngrams(
    left: Dict[str, List[Tuple[str, int, int]]],
    right: Dict[str, List[Tuple[str, int, int]]],
) -> Dict[str, List[Tuple[str, int, int]]]:
    merged: Dict[str, List[Tuple[str, int, int]]] = {**left}
    for key, entries in right.items():
        merged.setdefault(key, []).extend(entries)
    return merged


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

    # Attested definitions
    for attested in evidence.attested_definitions:
        entry = ensure(attested.source_word)
        entry.has_attested = True
        entry.uses += 1
        root_entry = ensure(attested.root_ngram)
        root_entry.has_attested = True
        root_entry.uses += 1

    # Dictionary matches
    for morph in evidence.dictionary_morphs:
        entry = ensure(morph)
        entry.has_dictionary = True
        entry.uses += 1

    return stats


def apply_hard_filters(
    decompositions: List[Decomposition],
    evidence: WordEvidence,
    min_support_threshold: float = 0.2,
) -> tuple[List[Decomposition], Dict[str, object]]:
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
        return [], {
            "stage1_dropped": 0,
            "stage2_dropped": 0,
            "stage3_dropped": 0,
            "unsupported_morphs": [],
            "dictionary_supported_morphs": [],
            "attested_supported_morphs": [],
            "dictionary_supported_count": 0,
            "attested_supported_count": 0,
            "min_residual_ratio": None,
            "max_attestation_score": None,
            "min_support_threshold": min_support_threshold,
            "filter_traces": [],
        }

    support_stats = _compile_support_stats(evidence)
    diagnostics: Dict[str, object] = {
        "stage1_dropped": 0,
        "stage2_dropped": 0,
        "stage3_dropped": 0,
        "unsupported_morphs": [],
        "dictionary_supported_morphs": [],
        "attested_supported_morphs": [],
        "dictionary_supported_count": 0,
        "attested_supported_count": 0,
        "min_residual_ratio": None,
        "max_attestation_score": None,
        "min_support_threshold": min_support_threshold,
        "filter_traces": [],
    }
    unsupported_counts: Dict[str, int] = {}
    dictionary_support_counts: Dict[str, int] = {}
    attested_support_counts: Dict[str, int] = {}

    def has_support(morph: str) -> bool:
        stats = support_stats.get(morph.upper())
        if stats is None:
            return False
        if stats.has_cluster or stats.has_residual:
            return True
        if stats.has_attested or stats.has_dictionary:
            return True
        if stats.hypothesis_max is None:
            return False
        return stats.hypothesis_max >= min_support_threshold

    def is_dictionary_supported_only(morph: str) -> bool:
        stats = support_stats.get(morph.upper())
        if stats is None:
            return False
        has_primary = stats.has_cluster or stats.has_residual
        if stats.hypothesis_max is not None:
            has_primary = has_primary or stats.hypothesis_max >= min_support_threshold
        return not has_primary and stats.has_dictionary

    def is_attested_supported_only(morph: str) -> bool:
        stats = support_stats.get(morph.upper())
        if stats is None:
            return False
        has_primary = stats.has_cluster or stats.has_residual
        if stats.hypothesis_max is not None:
            has_primary = has_primary or stats.hypothesis_max >= min_support_threshold
        return not has_primary and stats.has_attested

    # ------------------------------------------------------------------
    # Filter 1: every morph must have support.
    # ------------------------------------------------------------------
    supported: List[Decomposition] = []
    filter_traces: List[Dict[str, object]] = []
    for decomp in decompositions:
        missing = [m for m in decomp.morphs if not has_support(m)]
        trace: Dict[str, object] = {
            "morphs": list(decomp.morphs),
            "missing_morphs": missing,
            "morph_support": dict(decomp.morph_support),
            "residual_ratio": _residual_ratio(decomp),
            "segments": _summarize_segments(decomp.breakdown),
        }
        filter_traces.append(trace)
        if missing:
            diagnostics["stage1_dropped"] = cast(int, diagnostics["stage1_dropped"]) + 1
            for morph in missing:
                unsupported_counts[morph] = unsupported_counts.get(morph, 0) + 1
            LOGGER.debug(
                "Discarding %s due to unsupported morphs: %s",
                decomp.morphs,
                missing,
            )
            continue
        for morph in decomp.morphs:
            if is_dictionary_supported_only(morph):
                dictionary_support_counts[morph] = (
                    dictionary_support_counts.get(morph, 0) + 1
                )
            if is_attested_supported_only(morph):
                attested_support_counts[morph] = (
                    attested_support_counts.get(morph, 0) + 1
                )
        supported.append(decomp)

    diagnostics["filter_traces"] = filter_traces

    if unsupported_counts:
        top_n = 5
        ranked = sorted(
            unsupported_counts.items(), key=lambda item: (-item[1], item[0])
        )
        diagnostics["unsupported_morphs"] = [
            {"morph": morph, "count": count} for morph, count in ranked[:top_n]
        ]

    if dictionary_support_counts:
        ranked = sorted(
            dictionary_support_counts.items(), key=lambda item: (-item[1], item[0])
        )
        diagnostics["dictionary_supported_morphs"] = [
            {"morph": morph, "count": count} for morph, count in ranked
        ]
        diagnostics["dictionary_supported_count"] = sum(
            dictionary_support_counts.values()
        )

    if attested_support_counts:
        ranked = sorted(
            attested_support_counts.items(), key=lambda item: (-item[1], item[0])
        )
        diagnostics["attested_supported_morphs"] = [
            {"morph": morph, "count": count} for morph, count in ranked
        ]
        diagnostics["attested_supported_count"] = sum(attested_support_counts.values())

    if not supported:
        return [], diagnostics

    # ------------------------------------------------------------------
    # Filter 2: discard high-residual decomps when a better exists.
    # ------------------------------------------------------------------
    min_residual = min(_residual_ratio(d) for d in supported)
    diagnostics["min_residual_ratio"] = min_residual
    coverage_filtered: List[Decomposition] = []
    for decomp in supported:
        ratio = _residual_ratio(decomp)
        if ratio > 0.5 and min_residual <= 0.5:
            diagnostics["stage2_dropped"] = cast(int, diagnostics["stage2_dropped"]) + 1
            LOGGER.debug(
                "Discarding %s due to high residual_ratio %.3f (best=%.3f)",
                decomp.morphs,
                ratio,
                min_residual,
            )
            continue
        coverage_filtered.append(decomp)

    if not coverage_filtered:
        return [], diagnostics

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

    attestation_scores = {
        tuple(d.morphs): attestation_score(d) for d in coverage_filtered
    }
    for decomp in coverage_filtered:
        _update_attestation_trace(
            filter_traces, decomp, attestation_scores[tuple(decomp.morphs)]
        )
    max_attestation = max(attestation_scores.values())
    diagnostics["max_attestation_score"] = max_attestation
    if max_attestation == 0:
        # Nobody uses well-attested morphs; let soft scoring handle it.
        return coverage_filtered, diagnostics

    attested: List[Decomposition] = []
    for decomp in coverage_filtered:
        score = attestation_scores[tuple(decomp.morphs)]
        if score == 0:
            diagnostics["stage3_dropped"] = cast(int, diagnostics["stage3_dropped"]) + 1
            LOGGER.debug(
                "Discarding %s due to singleton-only morph usage (best score=%d)",
                decomp.morphs,
                max_attestation,
            )
            continue
        attested.append(decomp)

    return (attested if attested else coverage_filtered), diagnostics


def _summarize_segments(breakdown: Dict[str, object]) -> List[Dict[str, str]]:
    raw_segments = breakdown.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    summary: List[Dict[str, str]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        ngram = segment.get("ngram")
        canonical = segment.get("canonical")
        if isinstance(ngram, str) or isinstance(canonical, str):
            summary.append(
                {
                    "ngram": ngram if isinstance(ngram, str) else "",
                    "canonical": canonical if isinstance(canonical, str) else "",
                }
            )
    return summary


def _update_attestation_trace(
    traces: List[Dict[str, object]],
    target: Decomposition,
    score: int,
) -> None:
    if not traces:
        return
    target_key = tuple(target.morphs)
    updated = False
    for trace in traces:
        morphs = trace.get("morphs")
        if isinstance(morphs, list) and tuple(morphs) == target_key:
            trace["attestation_score"] = score
            updated = True
    if updated:
        return
