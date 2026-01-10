from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import math
import re

import numpy as np

from enochian_lm.common.types import MaybeNumber
from enochian_lm.root_extraction.utils.embeddings import (
    cluster_definitions,
    get_sentence_transformer,
)
from .decomposition import Decomposition
from .repository import ClusterRecord, DictionaryMorph, WordEvidence


# ---------------------------
# Task 3.1: reranking strategy
# ---------------------------

def apply_strategy(
    decompositions: list[tuple[Decomposition, float]],
    strategy: str,
    evidence: WordEvidence,
) -> list[tuple[Decomposition, float]]:
    if not decompositions:
        return []

    strategy_key = (strategy or "").lower().strip()

    support_counts: dict[str, int] | None = None
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
    support_counts: dict[str, int] | None,
) -> float:
    if strategy == "prefer-fewer":
        return -0.5 * len(decomp.morphs)

    if strategy == "prefer-known":
        if not support_counts:
            return 0.0
        attested = sum(
            1 for morph in decomp.morphs
            if support_counts.get((morph or "").upper(), 0) > 5
        )
        return 0.3 * attested

    if strategy == "prefer-balance":
        lengths = [len(morph or "") for morph in decomp.morphs]
        total_len = sum(lengths)
        chunkiness = 0.0
        if total_len > 0:
            chunkiness = sum(length ** 2 for length in lengths) / float(total_len ** 2)
        return 0.5 * chunkiness - 0.2 * _length_variance(decomp.morphs)

    return 0.0


def _compile_support_counts(evidence: WordEvidence) -> dict[str, int]:
    """Aggregate per-morph usage counts across evidence types (case-insensitive keys)."""
    counts: dict[str, int] = {}

    def bump(key: str) -> None:
        norm = (key or "").upper()
        if not norm:
            return
        counts[norm] = counts.get(norm, 0) + 1

    # These lists are always lists (default_factory=list), not Optional.
    for cluster in evidence.direct_clusters:
        bump(cluster.ngram)

    for residual in evidence.residual_semantics:
        bump(residual.residual)

    for hypothesis in evidence.morph_hypotheses:
        bump(hypothesis.morph)

    for attested in evidence.attested_definitions:
        bump(attested.source_word)
        bump(attested.root_ngram)

    for morph in evidence.dictionary_morphs:
        bump(morph)

    return counts


# ---------------------------
# Task 3.2: top-k selection
# ---------------------------

def select_top_k(
    ranked: list[tuple[Decomposition, float]],
    k: int = 3,
    *,
    evidence: WordEvidence | None = None,
) -> list[dict[str, object]]:
    if not ranked:
        return []

    try:
        top_k = int(k)
    except (TypeError, ValueError):
        top_k = 0
    if top_k <= 0:
        return []

    ordered: list[tuple[Decomposition, float]] = sorted(
        ((d, _safe_number(s, default=0.0)) for d, s in ranked),
        key=lambda pair: pair[1],
        reverse=True,
    )

    seen: set[tuple[str, ...]] = set()
    deduped: list[tuple[Decomposition, float]] = []
    for decomp, score in ordered:
        key = tuple(decomp.morphs)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((decomp, score))

    tie_warning: str | None = None
    if len(deduped) >= 2:
        delta = abs(deduped[0][1] - deduped[1][1])
        if delta < 0.05:
            tie_warning = "alternate decomposition exists"

    results: list[dict[str, object]] = []
    for idx, (decomp, score) in enumerate(deduped[:top_k], start=1):
        warnings: list[str] = []
        if tie_warning and idx <= 2:
            warnings.append(tie_warning)

        canonicals = list(decomp.canonicals) if decomp.canonicals else []
        results.append(
            {
                "rank": idx,
                "morphs": list(decomp.morphs),
                "canonicals": canonicals,
                "score": score,
                "breakdown": decomp.breakdown,
                "score_breakdown": dict(decomp.score_breakdown)
                if decomp.score_breakdown
                else None,
                "meanings": _extract_meanings(decomp=decomp, evidence=evidence),
                "warnings": warnings,
            }
        )

    return results


def _extract_meanings(
    *,
    decomp: Decomposition,
    evidence: WordEvidence | None,
) -> list[dict[str, object]]:
    support_lookup: dict[str, str] = {
        k.upper(): v for k, v in (decomp.morph_support or {}).items()
    }

    if evidence is None:
        return [
            {
                "morph": (morph or "").upper(),
                "canonical": None,
                "definition": None,
                "definitions": [],
                "provenance": "unknown",
            }
            for morph in decomp.morphs
        ]

    candidates = extract_definition_candidates(
        decomp.morphs,
        evidence,
        max_per_morph=4,
    )
    for idx, morph in enumerate(decomp.morphs):
        key = (morph or "").upper()
        canonical = (
            decomp.canonicals[idx].upper()
            if idx < len(decomp.canonicals)
            else None
        )
        extra_defs = _meaning_from_evidence(
            morph=key,
            evidence=evidence,
            canonical=canonical,
        )
        candidates.setdefault(key, [])
        existing = {str(item.get("definition", "")).strip().lower() for item in candidates[key]}
        for entry in extra_defs:
            definition = entry.get("definition")
            if not isinstance(definition, str):
                continue
            normalized = definition.strip()
            if not normalized:
                continue
            key_norm = normalized.lower()
            if key_norm in existing:
                continue
            existing.add(key_norm)
            candidates[key].append(
                {
                    "definition": normalized,
                    "source": entry.get("provenance"),
                    "quality": 0.0,
                }
            )

    selections, _beam_results = _select_definition_combination(
        decomp.morphs,
        candidates,
    )

    meanings: list[dict[str, object]] = []
    for idx, morph in enumerate(decomp.morphs):
        key = (morph or "").upper()
        canonical = (
            decomp.canonicals[idx].upper()
            if idx < len(decomp.canonicals)
            else None
        )
        selected = selections.get(key)
        definition = selected.get("definition") if selected else None
        provenance = selected.get("source") if selected else None

        if not provenance or provenance == "unknown":
            provenance = support_lookup.get(key, "unknown")

        meanings.append(
            {
                "morph": key,
                "canonical": canonical,
                "definition": definition,
                "definitions": [
                    entry.get("definition")
                    for entry in candidates.get(key, [])
                    if isinstance(entry.get("definition"), str)
                ],
                "provenance": provenance,
                "anchor_strength": compute_anchor_strength(
                    key,
                    candidates.get(key, []),
                ),
            }
        )

    return meanings


def _meaning_from_evidence(
    *,
    morph: str,
    evidence: WordEvidence | None,
    canonical: str | None,
) -> list[dict[str, object]]:
    if evidence is None:
        return []

    results: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(definition: str | None, provenance: str) -> None:
        if not isinstance(definition, str):
            return
        normalized = definition.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        results.append({"definition": normalized, "provenance": provenance})

    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(cluster.glossator_def),
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        add(definition, "cluster")

    for residual in evidence.residual_semantics:
        if residual.residual.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(residual.glossator_def),
            residual.residual_headline,
        )
        add(definition, "residual")

    for hypothesis in evidence.morph_hypotheses:
        if hypothesis.morph.upper() != morph:
            continue
        seed_glosses = ", ".join(g for g in hypothesis.seed_glosses if g and g.strip()) or None
        definition = _first_non_empty(hypothesis.proposed_gloss, seed_glosses, hypothesis.anchor)
        add(definition, "hypothesis")

    for attested in evidence.attested_definitions:
        if attested.source_word.upper() != morph:
            continue
        definition = _first_non_empty(attested.definition)
        add(definition, "attested")

    entry = evidence.dictionary_morphs.get(morph)
    if entry is not None:
        definition = _first_non_empty(entry.definition, ", ".join(entry.senses))
        add(definition, "dictionary")

    if canonical and canonical != morph:
        for entry in _meaning_from_evidence(
            morph=canonical,
            evidence=evidence,
            canonical=None,
        ):
            provenance = entry.get("provenance")
            definition = entry.get("definition")
            if isinstance(provenance, str) and provenance:
                add(definition, f"canonical_{provenance}")
            else:
                add(definition, "canonical_unknown")

    return results


def _first_cluster_raw_definition(cluster: ClusterRecord) -> str | None:
    # raw_definitions is a list[RawDefinition] (dataclass), so use attributes.
    for raw in cluster.raw_definitions:
        definition = _first_non_empty(raw.enhanced_def, raw.definition)
        if definition is not None:
            return definition
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
    return None


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


def _length_variance(morphs: Iterable[str]) -> float:
    lengths = [len(m or "") for m in morphs]
    if len(lengths) <= 1:
        return 0.0
    mean = sum(lengths) / float(len(lengths))
    return sum((length - mean) ** 2 for length in lengths) / float(len(lengths))


def _safe_number(value: MaybeNumber, *, default: float) -> float:
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


# ---------------------------------------------
# Beam-scoring helpers for definition candidates
# ---------------------------------------------

def extract_definition_candidates(
    morphs: Iterable[str],
    evidence: WordEvidence,
    *,
    max_per_morph: int = 3,
) -> dict[str, list[dict[str, object]]]:
    """Return candidate definitions per morph with quality metadata."""
    try:
        limit = max(0, int(max_per_morph))
    except (TypeError, ValueError):
        limit = 0

    results: dict[str, list[dict[str, object]]] = {}
    for raw_morph in morphs:
        morph = (raw_morph or "").upper()
        if not morph:
            continue

        candidates = _collect_definition_candidates(morph, evidence)
        if candidates:
            candidates.sort(
                key=lambda item: (_candidate_quality(item), len(item.get("definition", ""))),
                reverse=True,
            )

        if limit:
            candidates = candidates[:limit]

        results[morph] = candidates

    return results


def compute_anchor_strength(
    morph: str,
    candidates: Iterable[dict[str, object]],
) -> float:
    """Compute anchor strength using morph length, candidate count, and quality."""
    morph_len = len((morph or "").strip())
    candidate_list = list(candidates)
    length_score = min(1.0, math.log1p(morph_len) / math.log1p(6))
    scarcity_score = 1.0 / float(len(candidate_list)) if candidate_list else 0.0

    quality_values: list[float] = []
    for candidate in candidate_list:
        for key in ("semantic_coverage", "cohesion", "semantic_cohesion"):
            quality_values.append(_safe_number(candidate.get(key), default=0.0))
    quality_score = (
        sum(quality_values) / float(len(quality_values)) if quality_values else 0.0
    )

    blended = 0.45 * length_score + 0.35 * scarcity_score + 0.20 * quality_score
    return max(0.0, min(1.0, blended))


def compute_complementarity_band_similarity(
    definitions: Iterable[str],
    *,
    target: float = 0.45,
    radius: float = 0.20,
) -> float:
    """Compute complementarity-band similarity using sentence-transformer embeddings."""
    cleaned = [text.strip() for text in definitions if isinstance(text, str) and text.strip()]
    if len(cleaned) < 2:
        return 0.0

    embedder = get_sentence_transformer("paraphrase-MiniLM-L6-v2")
    vectors = [
        np.array(embedder.encode(text, normalize_embeddings=True), dtype=float)
        for text in cleaned
    ]

    sims: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarity = float(np.dot(vectors[i], vectors[j]))
            value = 1.0 - abs(similarity - target) / radius
            sims.append(max(0.0, min(1.0, value)))

    return sum(sims) / float(len(sims)) if sims else 0.0


CONTRADICTION_KEYWORD_PAIRS: list[tuple[str, str]] = [
    ("unite", "divide"),
    ("whole", "part"),
    ("light", "dark"),
    ("still", "move"),
    ("good", "wicked"),
    ("righteous", "sinful"),
    ("light", "dark"),
    ("life", "death"),
    ("love", "hate"),
    ("order", "chaos"),
    ("create", "destroy"),
]


def compute_contradiction_penalty(
    definitions_by_morph: Mapping[str, Iterable[str]],
    *,
    keyword_pairs: Iterable[tuple[str, str]] | None = None,
    penalty_per_pair: float = 0.20,
    max_penalty: float = 0.50,
) -> float:
    """Penalize contradictory keyword pairs across morph definitions."""
    pairs = list(keyword_pairs) if keyword_pairs is not None else CONTRADICTION_KEYWORD_PAIRS
    if not pairs:
        return 0.0

    corpus = " ".join(
        definition.lower()
        for definitions in definitions_by_morph.values()
        for definition in definitions
        if isinstance(definition, str) and definition.strip()
    )
    if not corpus:
        return 0.0

    detected = 0
    for left, right in pairs:
        if _keyword_present(corpus, left) and _keyword_present(corpus, right):
            detected += 1

    penalty = penalty_per_pair * detected
    return min(max_penalty, max(0.0, penalty))


def _keyword_present(corpus: str, keyword: str) -> bool:
    if not keyword:
        return False
    pattern = rf"\\b{re.escape(keyword.lower())}\\b"
    return re.search(pattern, corpus) is not None


@dataclass(frozen=True)
class DefinitionBeamResult:
    selections: dict[str, dict[str, object]]
    score: float
    components: dict[str, float]


def _select_definition_combination(
    morphs: Iterable[str],
    candidates: Mapping[str, list[dict[str, object]]],
    *,
    beam_width: int = 6,
    max_defs_per_morph: int = 4,
) -> tuple[dict[str, dict[str, object]], list[DefinitionBeamResult]]:
    morph_list = [m.upper() for m in morphs if m]
    if not morph_list:
        return {}, []

    anchor_strengths = {
        morph: compute_anchor_strength(morph, candidates.get(morph, []))
        for morph in morph_list
    }
    prepared = _prepare_candidates(morph_list, candidates, max_defs_per_morph)

    beam_results = _beam_search_definitions(
        morph_list,
        prepared,
        anchor_strengths,
        beam_width=beam_width,
    )
    if not beam_results:
        return {}, []

    locked = _lock_definition_anchors(
        beam_results,
        morph_list,
        prepared,
    )
    if locked:
        beam_results = _beam_search_definitions(
            morph_list,
            locked,
            anchor_strengths,
            beam_width=beam_width,
        )

    best = beam_results[0]
    return best.selections, beam_results


def _prepare_candidates(
    morphs: list[str],
    candidates: Mapping[str, list[dict[str, object]]],
    max_defs_per_morph: int,
) -> dict[str, list[dict[str, object]]]:
    prepared: dict[str, list[dict[str, object]]] = {}
    for morph in morphs:
        entries = list(candidates.get(morph, []))
        if entries:
            entries.sort(
                key=lambda entry: (
                    _safe_number(entry.get("quality"), default=0.0),
                    len(str(entry.get("definition", ""))),
                ),
                reverse=True,
            )
        if max_defs_per_morph > 0:
            entries = entries[:max_defs_per_morph]
        if not entries:
            entries = [
                {
                    "definition": None,
                    "source": "unknown",
                    "quality": 0.0,
                }
            ]
        prepared[morph] = entries
    return prepared


def _beam_search_definitions(
    morphs: list[str],
    candidates: Mapping[str, list[dict[str, object]]],
    anchor_strengths: Mapping[str, float],
    *,
    beam_width: int,
) -> list[DefinitionBeamResult]:
    beams: list[DefinitionBeamResult] = [
        DefinitionBeamResult(selections={}, score=0.0, components={})
    ]
    for morph in morphs:
        new_beams: list[DefinitionBeamResult] = []
        for beam in beams:
            for candidate in candidates.get(morph, []):
                selections = {**beam.selections, morph: candidate}
                score, components = _score_definition_combo(
                    selections,
                    anchor_strengths,
                )
                new_beams.append(
                    DefinitionBeamResult(
                        selections=selections,
                        score=score,
                        components=components,
                    )
                )
        new_beams.sort(key=lambda item: item.score, reverse=True)
        beams = new_beams[: max(1, beam_width)]
    return beams


def _score_definition_combo(
    selections: Mapping[str, dict[str, object]],
    anchor_strengths: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    definitions_by_morph: dict[str, list[str]] = {}
    qualities: list[float] = []
    anchors: list[float] = []

    for morph, candidate in selections.items():
        definition = candidate.get("definition")
        if isinstance(definition, str) and definition.strip():
            definitions_by_morph[morph] = [definition]
        else:
            definitions_by_morph[morph] = []
        qualities.append(_safe_number(candidate.get("quality"), default=0.0))
        anchors.append(_safe_number(anchor_strengths.get(morph), default=0.0))

    complementarity = compute_complementarity_band_similarity(
        [definition for defs in definitions_by_morph.values() for definition in defs]
    )
    anchor_alignment = sum(anchors) / float(len(anchors)) if anchors else 0.0
    definition_quality = sum(qualities) / float(len(qualities)) if qualities else 0.0
    contradiction = compute_contradiction_penalty(definitions_by_morph)

    score = (
        0.45 * complementarity
        + 0.30 * anchor_alignment
        + 0.15 * definition_quality
        - 0.10 * contradiction
    )
    components = {
        "complementarity": complementarity,
        "anchor_alignment": anchor_alignment,
        "definition_quality": definition_quality,
        "contradiction_penalty": contradiction,
    }
    return score, components


def _lock_definition_anchors(
    beam_results: list[DefinitionBeamResult],
    morphs: list[str],
    candidates: Mapping[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    if not beam_results:
        return {}
    locked: dict[str, list[dict[str, object]]] = {}
    for morph in morphs:
        selected_defs = [
            result.selections.get(morph, {}).get("definition")
            for result in beam_results
        ]
        unique_defs = [d for d in selected_defs if isinstance(d, str) and d.strip()]
        if not unique_defs:
            continue
        candidate_defs = [
            candidate.get("definition")
            for candidate in candidates.get(morph, [])
            if isinstance(candidate.get("definition"), str)
        ]
        cluster_map = _definition_cluster_map(candidate_defs)
        clusters = {
            cluster_map[definition]
            for definition in unique_defs
            if definition in cluster_map
        }
        if len(clusters) == 1:
            target_cluster = next(iter(clusters))
            locked[morph] = [
                candidate
                for candidate in candidates.get(morph, [])
                if isinstance(candidate.get("definition"), str)
                and cluster_map.get(candidate.get("definition")) == target_cluster
            ]

    if not locked:
        return {}

    merged = {**candidates}
    for morph, entries in locked.items():
        if entries:
            merged[morph] = entries
    return merged


def _definition_cluster_map(definitions: list[str]) -> dict[str, int]:
    cleaned = [definition.strip() for definition in definitions if isinstance(definition, str)]
    if not cleaned:
        return {}
    if len(cleaned) == 1:
        return {cleaned[0]: 0}

    embedder = get_sentence_transformer("paraphrase-MiniLM-L6-v2")
    clusters = cluster_definitions(
        cleaned,
        model=embedder,
        similarity_threshold=0.8,
    )
    mapping: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for member_idx in cluster.get("members", []):
            if isinstance(member_idx, int) and 0 <= member_idx < len(cleaned):
                mapping[cleaned[member_idx]] = idx
    return mapping


def _collect_definition_candidates(
    morph: str,
    evidence: WordEvidence,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_candidate(payload: dict[str, object]) -> None:
        definition = payload.get("definition")
        if not isinstance(definition, str):
            return
        normalized = definition.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        payload["definition"] = normalized
        payload["quality"] = _candidate_quality(payload)
        candidates.append(payload)

    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(cluster.glossator_def),
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        if definition:
            add_candidate(
                {
                    "definition": definition,
                    "source": "cluster",
                    "cluster_id": cluster.cluster_id,
                    "semantic_coverage": cluster.semantic_coverage,
                    "cohesion": cluster.cohesion,
                    "semantic_cohesion": cluster.semantic_cohesion,
                    "residual_ratio": cluster.residual_ratio,
                    "residual_explained": cluster.residual_explained,
                }
            )

    for residual in evidence.residual_semantics:
        if residual.residual.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(residual.glossator_def),
            residual.residual_headline,
        )
        if definition:
            add_candidate(
                {
                    "definition": definition,
                    "source": "residual",
                    "semantic_coverage": residual.semantic_coverage,
                    "cohesion": residual.cohesion,
                    "semantic_cohesion": residual.semantic_cohesion,
                    "residual_ratio": residual.residual_ratio,
                    "residual_explained": residual.residual_explained,
                    "derivational_validity": residual.derivational_validity,
                }
            )

    for hypothesis in evidence.morph_hypotheses:
        if hypothesis.morph.upper() != morph:
            continue
        seed_glosses = ", ".join(g for g in hypothesis.seed_glosses if g and g.strip())
        definition = _first_non_empty(hypothesis.proposed_gloss, seed_glosses, hypothesis.anchor)
        if definition:
            add_candidate(
                {
                    "definition": definition,
                    "source": "hypothesis",
                    "anchor": hypothesis.anchor,
                    "delta_cosine": hypothesis.delta_cosine,
                    "residual_before": hypothesis.residual_before,
                    "residual_after": hypothesis.residual_after,
                }
            )

    for attested in evidence.attested_definitions:
        if attested.source_word.upper() != morph:
            continue
        definition = _first_non_empty(attested.definition)
        if definition:
            add_candidate(
                {
                    "definition": definition,
                    "source": "attested",
                    "cluster_id": attested.cluster_id,
                }
            )

    entry: DictionaryMorph | None = evidence.dictionary_morphs.get(morph)
    if entry is not None:
        definition = _first_non_empty(entry.definition, ", ".join(entry.senses))
        if definition:
            add_candidate(
                {
                    "definition": definition,
                    "source": "dictionary",
                }
            )

    return candidates


def _candidate_quality(candidate: dict[str, object]) -> float:
    metrics = [
        candidate.get("semantic_coverage"),
        candidate.get("cohesion"),
        candidate.get("semantic_cohesion"),
        candidate.get("delta_cosine"),
    ]
    values = [_safe_number(metric, default=0.0) for metric in metrics if metric is not None]
    return sum(values) / float(len(values)) if values else 0.0
