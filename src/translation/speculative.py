"""Speculative decomposition candidates for agentic translation.

The normal translation scorer intentionally prefers stronger, chunkier, more
prevalent analyses. This module adds a bounded read-only exploration lane for
low-prevalence but evidence-backed decompositions, especially singleton-backed
splits that should be visible to agentic and phrase-level context.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
import math
import re

from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder

from .decomposition import (
    DEFAULT_MAX_FULL_SEGMENTATIONS,
    DEFAULT_MAX_PARTIAL_PER_INDEX,
    Decomposition,
    _collect_attested_pieces,
    build_decompositions_from_segmentations,
    enumerate_attested_segmentations_with_diagnostics,
)
from .repository import WordEvidence
from .profiles import (
    TranslationProfile,
    apply_profile_preferred_meanings,
    score_decomposition_profile,
)
from .strategies import extract_definition_candidates, select_top_k


@dataclass(frozen=True)
class SpeculativeProfile:
    """Configuration for bounded speculative branch generation."""

    name: str = "default"
    max_branches: int = 5
    max_root_senses: int | None = 0
    require_singleton_or_dense_split: bool = True
    singleton_penalty: float = 0.08
    min_score: float = 0.70


DEFAULT_SPECULATIVE_PROFILE = SpeculativeProfile()

_MOTIF_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "separation",
        "separate",
        "separated",
        "separating",
        "division",
        "divide",
        "dividing",
        "divided",
        "distinction",
        "distinct",
        "distinguish",
        "differentiation",
        "difference",
        "apart",
        "part",
        "parts",
        "another",
        "otherness",
        "equal",
    ),
    (
        "disturbance",
        "disturb",
        "vex",
        "vexing",
        "affliction",
        "afflict",
        "force",
        "pressure",
    ),
    (
        "pillar",
        "pillars",
        "column",
        "support",
        "rectangular",
        "geometry",
        "form",
    ),
)


def build_speculative_candidates(
    word: str,
    *,
    evidence: WordEvidence,
    candidate_finder: MorphemeCandidateFinder,
    existing_candidates: Sequence[Mapping[str, object]],
    allow_dictionary: bool,
    evidence_mode: str,
    profile: SpeculativeProfile = DEFAULT_SPECULATIVE_PROFILE,
    translation_profile: TranslationProfile | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return bounded speculative candidates plus diagnostics.

    The output candidates use the same public shape as regular compositional
    candidates so phrase translation can consume them without special casing.
    """

    normalized = (word or "").strip().upper()
    if not normalized:
        return [], _empty_diagnostics(profile)

    attested_pieces = _collect_attested_pieces(evidence, evidence_mode=evidence_mode)
    if not attested_pieces:
        return [], _empty_diagnostics(profile)

    segmentations, enum_diag = enumerate_attested_segmentations_with_diagnostics(
        normalized,
        attested_pieces,
        max_partial_per_index=DEFAULT_MAX_PARTIAL_PER_INDEX,
        max_full_segmentations=DEFAULT_MAX_FULL_SEGMENTATIONS,
        min_piece_len=1,
    )
    segmentations = [
        segmentation
        for segmentation in segmentations
        if not (len(segmentation) == 1 and str(segmentation[0]).upper() == normalized)
    ]
    if not segmentations:
        return [], {
            **_empty_diagnostics(profile),
            "enumerator": enum_diag,
            "attested_piece_count": len(attested_pieces),
        }

    decompositions = build_decompositions_from_segmentations(
        normalized,
        segmentations,
        candidate_finder=candidate_finder,
        evidence=evidence,
        evidence_mode=evidence_mode,
    )
    existing_paths = {
        tuple(str(morph).upper() for morph in candidate.get("morphs", []))
        for candidate in existing_candidates
        if isinstance(candidate, Mapping)
    }
    scored: list[tuple[Decomposition, float, dict[str, object]]] = []
    definition_candidates = extract_definition_candidates(
        {
            morph
            for decomp in decompositions
            for morph in decomp.morphs
            if morph
        },
        evidence,
        max_per_morph=profile.max_root_senses or 0,
        allow_dictionary=allow_dictionary,
    )

    for decomp in decompositions:
        path = tuple(decomp.morphs)
        if path in existing_paths:
            continue
        score, scoring = _score_decomposition(
            decomp,
            definition_candidates=definition_candidates,
            profile=profile,
            translation_profile=translation_profile,
        )
        if score < profile.min_score:
            continue
        decomp.score_breakdown = {
            **dict(decomp.score_breakdown or {}),
            "speculative": scoring,
            "total": score,
        }
        scored.append((decomp, score, scoring))

    scored.sort(key=lambda item: item[1], reverse=True)
    scored = scored[: max(1, profile.max_branches)]
    if not scored:
        return [], {
            **_empty_diagnostics(profile),
            "enumerator": enum_diag,
            "attested_piece_count": len(attested_pieces),
            "considered_decomposition_count": len(decompositions),
        }

    candidates = select_top_k(
        [(decomp, score) for decomp, score, _scoring in scored],
        k=max(1, profile.max_branches),
        evidence=evidence,
        allow_dictionary=allow_dictionary,
        translation_profile=translation_profile,
    )
    scoring_by_path = {tuple(decomp.morphs): scoring for decomp, _score, scoring in scored}
    for candidate in candidates:
        path = tuple(str(morph).upper() for morph in candidate.get("morphs", []))
        scoring = scoring_by_path.get(path, {})
        if translation_profile is not None:
            apply_profile_preferred_meanings(
                candidate,
                definition_candidates,
                profile=translation_profile,
            )
        else:
            _apply_preferred_motif_meanings(candidate, definition_candidates)
        candidate["analysis_type"] = "speculative_compositional"
        candidate["speculative"] = True
        candidate["speculative_profile"] = profile.name
        candidate["decision_trace"] = {
            "selection_reason": _selection_reason(path, scoring),
            "speculative_score": scoring.get("score"),
            "motif_groups": scoring.get("motif_groups", []),
            "head_modifier_analysis": candidate.get("head_modifier_analysis")
            or scoring.get("head_modifier_analysis"),
            "singleton_count": scoring.get("singleton_count", 0),
            "support_ratio": scoring.get("support_ratio", 0.0),
        }
        warnings = [
            str(warning)
            for warning in candidate.get("warnings", [])
            if isinstance(warning, str)
        ]
        warnings.append(
            "Speculative branch: low-prevalence decomposition retained for context review."
        )
        candidate["warnings"] = warnings

    return candidates, {
        "enabled": True,
        "profile": profile.__dict__,
        "attested_piece_count": len(attested_pieces),
        "enumerator": enum_diag,
        "considered_decomposition_count": len(decompositions),
        "returned_count": len(candidates),
        "paths": [" + ".join(candidate.get("morphs", [])) for candidate in candidates],
    }


def _apply_preferred_motif_meanings(
    candidate: dict[str, object],
    definition_candidates: Mapping[str, list[dict[str, object]]],
) -> None:
    """Prefer the root senses that explain why a speculative branch survived."""

    meanings_raw = candidate.get("meanings")
    meanings = meanings_raw if isinstance(meanings_raw, list) else []
    by_morph = {
        str(meaning.get("morph") or meaning.get("canonical") or "").upper(): meaning
        for meaning in meanings
        if isinstance(meaning, dict)
    }
    for morph in [str(m).upper() for m in candidate.get("morphs", []) if isinstance(m, str)]:
        preferred = _preferred_motif_candidate(definition_candidates.get(morph, []))
        if preferred is None:
            continue
        meaning = by_morph.get(morph)
        if meaning is None:
            meaning = {"morph": morph, "canonical": morph}
            meanings.append(meaning)
            by_morph[morph] = meaning
        definition = _display_definition(preferred)
        raw_definition = _clean_display_definition(
            preferred.get("raw_definition") or definition
        )
        semantic_core = _as_string_list(preferred.get("semantic_core_terms"))
        negative = _as_string_list(preferred.get("negative_contrast"))
        source = str(preferred.get("source") or "derived")
        meaning.update(
            {
                "morph": morph,
                "canonical": morph,
                "definition": definition,
                "raw_definition": raw_definition,
                "surface_gloss": definition,
                "surface_gloss_strategy": preferred.get("surface_gloss_strategy")
                or "speculative_motif",
                "semantic_core": semantic_core,
                "semantic_core_terms": semantic_core,
                "negative_contrast": negative,
                "provenance": source,
                "cluster_id": preferred.get("cluster_id"),
                "source_cluster_id": preferred.get("source_cluster_id")
                or preferred.get("cluster_id"),
                "source_variant": preferred.get("source_variant"),
                "definition_trace": {
                    "selected_definition": definition,
                    "raw_selected_definition": raw_definition,
                    "selected_source": source,
                    "selected_quality": preferred.get("quality"),
                    "selected_semantic_core": semantic_core,
                    "selected_negative_contrast": negative,
                    "surface_gloss": definition,
                    "surface_gloss_strategy": "speculative_motif",
                    "runner_ups": [],
                    "suppressed": [],
                    "blind_dictionary_fallback": False,
                    "negative_contrast_penalties": [],
                    "meta_linguistic_rejections": [],
                    "selected_source_detail": {
                        key: preferred.get(key)
                        for key in (
                            "cluster_id",
                            "source_cluster_id",
                            "source_variant",
                            "source_run_id",
                            "source_cluster_index",
                        )
                        if preferred.get(key) is not None
                    },
                },
            }
        )
    candidate["meanings"] = meanings


def _preferred_motif_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if not candidates:
        return None
    scored: list[tuple[float, Mapping[str, object]]] = []
    for item in candidates:
        text = _definition_text([item])
        motif_hits = 0
        for keywords in _MOTIF_GROUPS:
            if any(_keyword_present(text, keyword) for keyword in keywords):
                motif_hits += 1
        quality = _safe_float(item.get("quality"), default=0.0)
        scored.append((motif_hits * 10.0 + quality, item))
    scored.sort(key=lambda row: row[0], reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][1]


def _display_definition(candidate: Mapping[str, object]) -> str:
    raw_definition = _clean_display_definition(candidate.get("raw_definition"))
    definition = _clean_display_definition(candidate.get("definition"))
    if raw_definition and (
        not definition
        or len(raw_definition.split()) > len(definition.split())
        or len(raw_definition) > len(definition) + 12
    ):
        return raw_definition
    return definition or raw_definition


def _clean_display_definition(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .;:")


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _score_decomposition(
    decomp: Decomposition,
    *,
    definition_candidates: Mapping[str, list[dict[str, object]]],
    profile: SpeculativeProfile,
    translation_profile: TranslationProfile | None,
) -> tuple[float, dict[str, object]]:
    morphs = [morph.upper() for morph in decomp.morphs if morph]
    if not morphs:
        return 0.0, {"score": 0.0, "reason": "empty"}
    if profile.require_singleton_or_dense_split:
        has_singleton = any(len(morph) == 1 for morph in morphs)
        if not has_singleton and len(morphs) < 3:
            return 0.0, {"score": 0.0, "reason": "not_speculative_shape"}

    coverage = _safe_float(decomp.breakdown.get("coverage_ratio"), default=0.0)
    support_ratio = _support_ratio(decomp)
    singleton_count = sum(1 for morph in morphs if len(morph) == 1)
    dense_split_bonus = min(0.20, max(0, len(morphs) - 2) * 0.05)
    motif_score, motif_groups = _motif_score(morphs, definition_candidates)
    head_modifier_analysis = (
        score_decomposition_profile(
            morphs,
            definition_candidates,
            profile=translation_profile,
        )
        if translation_profile is not None
        else {}
    )
    profile_score = _safe_float(head_modifier_analysis.get("score"), default=0.0)
    motif_score = max(motif_score, profile_score)
    scarcity_score = _scarcity_score(morphs, definition_candidates)
    score = (
        0.38 * coverage
        + 0.28 * support_ratio
        + 0.42 * motif_score
        + 0.12 * scarcity_score
        + dense_split_bonus
        - profile.singleton_penalty * singleton_count
    )
    score = max(0.0, score)
    return score, {
        "score": score,
        "coverage": coverage,
        "support_ratio": support_ratio,
        "singleton_count": singleton_count,
        "motif_score": motif_score,
        "motif_groups": motif_groups,
        "translation_profile_score": profile_score,
        "head_modifier_analysis": head_modifier_analysis,
        "scarcity_score": scarcity_score,
        "dense_split_bonus": dense_split_bonus,
    }


def _support_ratio(decomp: Decomposition) -> float:
    support = dict(decomp.morph_support or {})
    if not decomp.morphs:
        return 0.0
    supported = sum(1 for morph in decomp.morphs if support.get(morph) != "unknown")
    return supported / float(len(decomp.morphs))


def _motif_score(
    morphs: Sequence[str],
    definition_candidates: Mapping[str, list[dict[str, object]]],
) -> tuple[float, list[dict[str, object]]]:
    groups: list[dict[str, object]] = []
    for keywords in _MOTIF_GROUPS:
        matched: list[str] = []
        for morph in morphs:
            text = _definition_text(definition_candidates.get(morph, []))
            if any(_keyword_present(text, keyword) for keyword in keywords):
                matched.append(morph)
        if matched:
            groups.append(
                {
                    "keywords": list(keywords[:5]),
                    "matched_morphs": matched,
                    "matched_count": len(matched),
                }
            )
    if not groups:
        return 0.0, []
    best = max(len(group["matched_morphs"]) for group in groups)
    shared_bonus = 0.20 if best >= 2 else 0.0
    coverage = min(1.0, best / float(max(1, len(morphs))))
    return min(1.0, coverage + shared_bonus), groups


def _scarcity_score(
    morphs: Sequence[str],
    definition_candidates: Mapping[str, list[dict[str, object]]],
) -> float:
    values: list[float] = []
    for morph in morphs:
        count = len(definition_candidates.get(morph, []))
        if count <= 0:
            values.append(0.0)
        else:
            values.append(1.0 / (1.0 + math.log1p(float(count))))
    return sum(values) / float(len(values)) if values else 0.0


def _definition_text(candidates: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for candidate in candidates:
        for key in ("definition", "semantic_core", "negative_contrast"):
            value = candidate.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                parts.extend(str(item) for item in value)
    return " ".join(parts).lower()


def _keyword_present(text: str, keyword: str) -> bool:
    if not text or not keyword:
        return False
    return re.search(rf"\b{re.escape(keyword.lower())}\b", text) is not None


def _selection_reason(path: Sequence[str], scoring: Mapping[str, object]) -> str:
    motifs = scoring.get("motif_groups")
    motif_count = len(motifs) if isinstance(motifs, list) else 0
    return (
        "Speculative full-cover split retained because "
        f"{' + '.join(path)} has attested support and {motif_count} shared motif "
        "group(s), while using low-prevalence singleton-capable roots."
    )


def _empty_diagnostics(profile: SpeculativeProfile) -> dict[str, object]:
    return {
        "enabled": True,
        "profile": profile.__dict__,
        "attested_piece_count": 0,
        "considered_decomposition_count": 0,
        "returned_count": 0,
        "paths": [],
    }


def _safe_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
