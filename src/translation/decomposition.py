from __future__ import annotations

"""Segmentation and decomposition utilities for task 2.1.

This module wraps MorphemeCandidateFinder to produce Decomposition objects
that can be further filtered and scored in later tasks (2.2 / 2.3).
"""

import json
import logging
import re
from typing import Final, Literal, cast
from collections.abc import Iterable
from dataclasses import dataclass, field
from collections.abc import Callable

from enochian_lm.root_extraction.utils.candidate_finder import (
    CoverageSegment,
    MorphemeCandidateFinder,
)
from .repository import WordEvidence

LOGGER = logging.getLogger(__name__)


@dataclass
class MorphSupportStats:
    has_cluster: bool = False
    has_residual: bool = False
    has_attested: bool = False
    has_dictionary: bool = False
    hypothesis_max: float | None = None  # max delta_cosine across hypotheses
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

    morphs: list[str]
    canonicals: list[str]
    beam_score: float
    breakdown: dict[str, object] = field(default_factory=dict)
    beam_score_normalized: float | None = None
    morph_support: dict[str, str] = field(default_factory=dict)


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
        definition_counts: dict[str, int] | None = None,
        definition_glosses: dict[str, list[tuple[str, float | None]]] | None = None,
        evidence_mode: str | None = "all",
    ) -> tuple[list[Decomposition], dict[str, object]]:
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

        diagnostics: dict[str, object] = {
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
            evidence_mode=evidence_mode,
        )
        diagnostics["extra_ngram_keys"] = len(extra_ngrams)
        diagnostics["extra_ngram_entries"] = sum(
            len(entries) for entries in extra_ngrams.values()
        )
        dictionary_ngrams: dict[str, list[tuple[str, int, int]]] = {}
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
            definition_glosses=definition_glosses,
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
                    definition_glosses=definition_glosses,
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

        decompositions: list[Decomposition] = []

        if not parses:
            _normalize_beam_scores(decompositions)
            diagnostics["decomposition_count"] = len(decompositions)
            return decompositions, diagnostics

        support_lookup = _build_support_lookup(evidence, evidence_mode=evidence_mode)

        for path, score, _ngram_scores, coverage in parses:
            if not coverage:
                continue
            morphs, canonicals = _segment_tokens(normalized, coverage, path)
            if not allow_whole_word and len(normalized) > 1:
                if len(morphs) == 1 and morphs[0] == normalized:
                    continue
            breakdown = _build_breakdown(self.candidate_finder, normalized, coverage)

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
) -> dict[str, object]:
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
    ranges: list[tuple[int, int]] = []
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
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    covered = sum(end - start for start, end in merged)
    coverage_ratio = covered / word_len if word_len else 0.0

    # Build uncovered spans between merged covered ranges.
    uncovered: list[dict[str, list[int]]] = []
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
    coverage: list[CoverageSegment],
    canonicals: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return morph slices (ngrams) and canonicals for a segmentation."""
    ngrams: list[str] = []
    canon_list: list[str] = [str(c).upper() for c in canonicals]

    if coverage:
        for segment in coverage:
            ngram = segment.get("ngram")
            if isinstance(ngram, str) and ngram:
                ngrams.append(ngram.upper())

    if not ngrams:
        ngrams = [str(word).upper()] if word else []

    return ngrams, canon_list


def _normalize_beam_scores(decompositions: list[Decomposition]) -> None:
    if not decompositions:
        return

    adjusted_scores: list[float] = []
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


def _cluster_has_definition(cluster: object) -> bool:
    glossator_def = getattr(cluster, "glossator_def", None)
    raw_definitions = getattr(cluster, "raw_definitions", None)
    residual_headline = getattr(cluster, "residual_headline", None)

    return (
        _first_non_empty(
            _extract_glossator_definition(glossator_def),
            _first_cluster_raw_definition(raw_definitions),
            residual_headline,
        )
        is not None
    )


EvidenceMode = Literal["all", "clusters-only", "residuals-only"]
_EVIDENCE_MODES: Final[set[str]] = {"all", "clusters-only", "residuals-only"}


def _normalize_evidence_mode(evidence_mode: str | None) -> EvidenceMode:
    normalized = (evidence_mode or "all").strip().lower()
    if normalized in _EVIDENCE_MODES:
        return cast(EvidenceMode, normalized)
    return "all"


def _evidence_flags(evidence_mode: EvidenceMode) -> tuple[bool, bool, bool]:
    return (
        evidence_mode != "residuals-only",
        evidence_mode != "clusters-only",
        evidence_mode == "all",
    )


def _first_cluster_raw_definition(raw_definitions: object) -> str | None:
    if not isinstance(raw_definitions, list):
        return None
    for raw in raw_definitions:
        enhanced_def = getattr(raw, "enhanced_def", None)
        definition = getattr(raw, "definition", None)
        text = _first_non_empty(enhanced_def, definition)
        if text is not None:
            return text
    return None


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
            continue
        try:
            if value:
                text = str(value).strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def _extract_glossator_definition(payload: object) -> str | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return _definition_from_glossator_json(payload)
    if not isinstance(payload, str):
        return None

    text = payload.strip()
    if not text:
        return None

    parsed = _parse_glossator_json(text)
    if isinstance(parsed, dict):
        return _definition_from_glossator_json(parsed)
    return text if text else None


def _parse_glossator_json(text: str) -> dict | None:
    for attempt in (_load_json, _load_json_from_code_fence, _load_nested_raw_text):
        result = attempt(text)
        if isinstance(result, dict):
            return result
    return None


def _load_json(text: str) -> dict | None:
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _load_json_from_code_fence(text: str) -> dict | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    return _load_json(match.group(1).strip())


def _load_nested_raw_text(text: str) -> dict | None:
    data = _load_json(text)
    if not isinstance(data, dict):
        return None
    raw_text = data.get("RAW_TEXT")
    if not isinstance(raw_text, str):
        return None
    raw_text = raw_text.strip()
    if not raw_text:
        return None
    return _load_json_from_code_fence(raw_text) or _load_json(raw_text)


def _definition_from_glossator_json(payload: dict) -> str | None:
    for key in ("DEFINITION", "Definition", "definition", "gloss", "GLOSS"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _build_support_lookup(
    evidence: WordEvidence,
    *,
    evidence_mode: str | None = "all",
) -> dict[str, str]:
    """Build a case-insensitive lookup table of morph → support label.

    Priority is:

    1. direct clusters (``"cluster"``)
    2. residual semantics (``"residual"``)
    3. morph hypotheses (``"hypothesis"``)
    4. attested definitions (``"attested"``)
    5. dictionary matches (``"dictionary"``)

    Later sources never overwrite earlier ones for the same morph key.
    """

    evidence_mode = _normalize_evidence_mode(evidence_mode)
    clusters_enabled, residuals_enabled, hypotheses_enabled = _evidence_flags(
        evidence_mode
    )
    # Normalize everything to uppercase for consistent lookups.
    support: dict[str, str] = {}

    if clusters_enabled:
        for cluster in evidence.direct_clusters:
            if not _cluster_has_definition(cluster):
                continue
            key = cluster.ngram.upper()
            support.setdefault(key, "cluster")

    if residuals_enabled:
        for residual in evidence.residual_semantics:
            key = residual.residual.upper()
            support.setdefault(key, "residual")

    if hypotheses_enabled:
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
    definition_counts: dict[str, int] | None = None,
    evidence_mode: str | None = "all",
) -> dict[str, list[tuple[str, int, int]]]:
    """Build an extra ngram index based on evidence-backed morphs.

    Key change vs. the old behavior:
    - Treat evidence-backed morphs as *boost signals*, not just "missing ngrams".
    - Use a small DF so the boost meaningfully affects IDF.
    - Avoid boosting 1-char morphs and the full word (both encourage degenerate parses),
      except when the 1-char morph is explicitly dictionary-backed.
    """

    evidence_mode = _normalize_evidence_mode(evidence_mode)
    clusters_enabled, residuals_enabled, hypotheses_enabled = _evidence_flags(
        evidence_mode
    )
    morphs: set[str] = set()
    if clusters_enabled:
        morphs.update(
            cluster.ngram
            for cluster in evidence.direct_clusters
            if _cluster_has_definition(cluster)
        )
    if residuals_enabled:
        morphs.update(residual.residual for residual in evidence.residual_semantics)
    if hypotheses_enabled:
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

    support_stats = _compile_support_stats(evidence, evidence_mode=evidence_mode)

    candidates: list[tuple[int, int, int, str]] = []
    # rank tuple = (edge_bonus, length, uses, morph)
    for morph in morphs:
        if not morph:
            continue
        normalized = morph.strip().upper()
        if not normalized:
            continue
        if normalized not in word:
            continue
        if len(normalized) < min_len:
            if len(normalized) != 1 or normalized not in evidence.dictionary_morphs:
                continue
        if len(normalized) > max_len:
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
    extra: dict[str, list[tuple[str, int, int]]] = {}
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
) -> dict[str, list[tuple[str, int, int]]]:
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

    extra: dict[str, list[tuple[str, int, int]]] = {}
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
    left: dict[str, list[tuple[str, int, int]]],
    right: dict[str, list[tuple[str, int, int]]],
) -> dict[str, list[tuple[str, int, int]]]:
    merged: dict[str, list[tuple[str, int, int]]] = {**left}
    for key, entries in right.items():
        merged.setdefault(key, []).extend(entries)
    return merged


def _classify_support(morph: str, support_lookup: dict[str, str]) -> str:
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


def _compile_support_stats(
    evidence: WordEvidence,
    *,
    evidence_mode: str | None = "all",
) -> dict[str, MorphSupportStats]:
    """Aggregate per-morph support details for filtering decisions."""

    evidence_mode = _normalize_evidence_mode(evidence_mode)
    clusters_enabled, residuals_enabled, hypotheses_enabled = _evidence_flags(
        evidence_mode
    )
    stats: dict[str, MorphSupportStats] = {}

    def ensure(key: str) -> MorphSupportStats:
        key = key.upper()
        if key not in stats:
            stats[key] = MorphSupportStats()
        return stats[key]

    # Clusters
    if clusters_enabled:
        for cluster in evidence.direct_clusters:
            if not _cluster_has_definition(cluster):
                continue
            entry = ensure(cluster.ngram)
            entry.has_cluster = True
            entry.uses += 1

    # Residual semantics
    if residuals_enabled:
        for residual in evidence.residual_semantics:
            entry = ensure(residual.residual)
            entry.has_residual = True
            entry.uses += 1

    # Hypotheses
    if hypotheses_enabled:
        for hypothesis in evidence.morph_hypotheses:
            entry = ensure(hypothesis.morph)
            entry.uses += 1
            delta = hypothesis.delta_cosine
            if delta is None:
                continue
            if entry.hypothesis_max is None or delta > entry.hypothesis_max:
                entry.hypothesis_max = float(delta)

    # Attested definitions
    if hypotheses_enabled:
        for attested in evidence.attested_definitions:
            entry = ensure(attested.source_word)
            entry.has_attested = True
            entry.uses += 1
            root_entry = ensure(attested.root_ngram)
            root_entry.has_attested = True
            root_entry.uses += 1

    # Dictionary matches
    if hypotheses_enabled:
        for morph in evidence.dictionary_morphs:
            entry = ensure(morph)
            entry.has_dictionary = True
            entry.uses += 1

    return stats


def apply_hard_filters(
    decompositions: list[Decomposition],
    evidence: WordEvidence,
    min_support_threshold: float = 0.2,
    *,
    evidence_mode: str | None = "all",
) -> tuple[list[Decomposition], dict[str, object]]:
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

    support_stats = _compile_support_stats(evidence, evidence_mode=evidence_mode)
    # Use local counters to avoid cast() when incrementing
    stage1_dropped = 0
    stage2_dropped = 0
    stage3_dropped = 0

    unsupported_counts: dict[str, int] = {}
    dictionary_support_counts: dict[str, int] = {}
    attested_support_counts: dict[str, int] = {}

    # Build diagnostics dict - counters will be updated before return
    diagnostics: dict[str, object] = {
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

    def _finalize_diagnostics() -> None:
        """Update diagnostics dict with final counter values."""
        diagnostics["stage1_dropped"] = stage1_dropped
        diagnostics["stage2_dropped"] = stage2_dropped
        diagnostics["stage3_dropped"] = stage3_dropped

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
    supported: list[Decomposition] = []
    filter_traces: list[dict[str, object]] = []
    for decomp in decompositions:
        missing = [m for m in decomp.morphs if not has_support(m)]
        trace: dict[str, object] = {
            "morphs": list(decomp.morphs),
            "missing_morphs": missing,
            "morph_support": dict(decomp.morph_support),
            "residual_ratio": _residual_ratio(decomp),
            "segments": _summarize_segments(decomp.breakdown),
        }
        filter_traces.append(trace)
        if missing:
            stage1_dropped += 1
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
        _finalize_diagnostics()
        return [], diagnostics

    # ------------------------------------------------------------------
    # Filter 2: discard high-residual decomps when a better exists.
    # ------------------------------------------------------------------
    min_residual = min(_residual_ratio(d) for d in supported)
    diagnostics["min_residual_ratio"] = min_residual
    coverage_filtered: list[Decomposition] = []
    for decomp in supported:
        ratio = _residual_ratio(decomp)
        if ratio > 0.5 and min_residual <= 0.5:
            stage2_dropped += 1
            LOGGER.debug(
                "Discarding %s due to high residual_ratio %.3f (best=%.3f)",
                decomp.morphs,
                ratio,
                min_residual,
            )
            continue
        coverage_filtered.append(decomp)

    if not coverage_filtered:
        _finalize_diagnostics()
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
        _finalize_diagnostics()
        return coverage_filtered, diagnostics

    attested: list[Decomposition] = []
    for decomp in coverage_filtered:
        score = attestation_scores[tuple(decomp.morphs)]
        if score == 0:
            stage3_dropped += 1
            LOGGER.debug(
                "Discarding %s due to singleton-only morph usage (best score=%d)",
                decomp.morphs,
                max_attestation,
            )
            continue
        attested.append(decomp)

    _finalize_diagnostics()
    return (attested if attested else coverage_filtered), diagnostics


def _summarize_segments(breakdown: dict[str, object]) -> list[dict[str, str]]:
    raw_segments = breakdown.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    summary: list[dict[str, str]] = []
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
    traces: list[dict[str, object]],
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
