from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re

import numpy as np

from enochian_lm.common.types import MaybeNumber
from enochian_lm.root_extraction.utils.embeddings import (
    cluster_definitions,
    get_sentence_transformer_if_available,
)
from .decomposition import Decomposition
from .placeholder_glosses import (
    clean_lexical_gloss,
    gloss_overlaps_negative_contrast,
    is_meta_linguistic_gloss,
    normalize_semantic_terms,
    semantic_core_gloss,
    surface_gloss_from_sources,
)
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
    allow_dictionary: bool = True,
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

    prepared_results: list[dict[str, object]] = []
    for decomp, score in deduped:
        warnings: list[str] = []
        canonicals = list(decomp.canonicals) if decomp.canonicals else []
        meanings = _extract_meanings(
            decomp=decomp,
            evidence=evidence,
            allow_dictionary=allow_dictionary,
        )
        bundle = compose_semantic_bundle(meanings)
        adjusted_score = float(score) + _bundle_score_bonus(bundle)
        prepared_results.append(
            {
                "morphs": list(decomp.morphs),
                "canonicals": canonicals,
                "score": adjusted_score,
                "breakdown": decomp.breakdown,
                "score_breakdown": dict(decomp.score_breakdown)
                if decomp.score_breakdown
                else None,
                "meanings": meanings,
                "warnings": warnings,
                **bundle,
            }
        )

    prepared_results.sort(
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )

    results: list[dict[str, object]] = []
    for idx, prepared in enumerate(prepared_results[:top_k], start=1):
        warnings = list(prepared.get("warnings") or [])
        if tie_warning and idx <= 2:
            warnings.append(tie_warning)
        results.append(
            {
                "rank": idx,
                "morphs": list(prepared.get("morphs") or []),
                "canonicals": list(prepared.get("canonicals") or []),
                "score": float(prepared.get("score") or 0.0),
                "breakdown": prepared.get("breakdown"),
                "score_breakdown": prepared.get("score_breakdown"),
                "meanings": list(prepared.get("meanings") or []),
                "semantic_bundle": list(prepared.get("semantic_bundle") or []),
                "bundle_surface_gloss": prepared.get("bundle_surface_gloss"),
                "bundle_head_gloss": prepared.get("bundle_head_gloss"),
                "bundle_function_profile": prepared.get("bundle_function_profile"),
                "bundle_coherence_score": prepared.get("bundle_coherence_score"),
                "warnings": warnings,
            }
        )

    return results


def _extract_meanings(
    *,
    decomp: Decomposition,
    evidence: WordEvidence | None,
    allow_dictionary: bool = True,
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
        allow_dictionary=allow_dictionary,
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
            allow_dictionary=allow_dictionary,
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
                    "raw_definition": entry.get("raw_definition"),
                    "source": entry.get("provenance"),
                    "semantic_core_terms": list(entry.get("semantic_core_terms") or []),
                    "negative_contrast": list(entry.get("negative_contrast") or []),
                    "surface_gloss_strategy": entry.get("surface_gloss_strategy") or "cleaned_definition",
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
        semantic_core_terms = list(selected.get("semantic_core_terms") or []) if selected else []
        negative_contrast = list(selected.get("negative_contrast") or []) if selected else []
        surface_gloss = selected.get("definition") if selected else None
        surface_gloss_strategy = (
            str(selected.get("surface_gloss_strategy") or "unresolved")
            if selected
            else "unresolved"
        )

        if not provenance or provenance == "unknown":
            provenance = support_lookup.get(key, "unknown")
        trace = _definition_trace(
            morph=key,
            selected=selected,
            candidates=candidates.get(key, []),
            evidence=evidence,
            allow_dictionary=allow_dictionary,
        )

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
                "raw_definition": (
                    selected.get("raw_definition")
                    if isinstance(selected, Mapping)
                    else None
                ),
                "semantic_core": semantic_core_terms,
                "semantic_core_terms": semantic_core_terms,
                "negative_contrast": negative_contrast,
                "surface_gloss": surface_gloss,
                "surface_gloss_strategy": surface_gloss_strategy,
                "provenance": provenance,
                "anchor_strength": compute_anchor_strength(
                    key,
                    candidates.get(key, []),
                ),
                "definition_trace": trace,
            }
        )

    return meanings


def _meaning_from_evidence(
    *,
    morph: str,
    evidence: WordEvidence | None,
    canonical: str | None,
    allow_dictionary: bool = True,
) -> list[dict[str, object]]:
    if evidence is None:
        return []

    results: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(
        definition: str | None,
        provenance: str,
        *,
        semantic_core_terms: Sequence[str] | None = None,
        negative_contrast: Sequence[str] | None = None,
    ) -> None:
        surface_gloss, strategy = surface_gloss_from_sources(
            semantic_core=list(semantic_core_terms or []),
            definition=definition,
            negative_contrast=list(negative_contrast or []),
            token=morph,
        )
        if not isinstance(surface_gloss, str):
            return
        normalized = surface_gloss.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        results.append(
            {
                "definition": normalized,
                "raw_definition": definition.strip() if isinstance(definition, str) else None,
                "provenance": provenance,
                "semantic_core_terms": list(semantic_core_terms or []),
                "negative_contrast": list(negative_contrast or []),
                "surface_gloss_strategy": strategy,
            }
        )

    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition, semantic_core_terms, negative_contrast = _glossator_candidate_metadata(
            cluster.glossator_def
        )
        definition = _first_non_empty(
            definition,
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        add(
            definition,
            "cluster",
            semantic_core_terms=semantic_core_terms or cluster.semantic_core,
            negative_contrast=negative_contrast or cluster.negative_contrast,
        )

    for residual in evidence.residual_semantics:
        if residual.residual.upper() != morph:
            continue
        definition, semantic_core_terms, negative_contrast = _glossator_candidate_metadata(
            residual.glossator_def
        )
        definition = _first_non_empty(
            definition,
            residual.residual_headline,
        )
        add(
            definition,
            "residual",
            semantic_core_terms=semantic_core_terms or residual.semantic_core,
            negative_contrast=negative_contrast or residual.negative_contrast,
        )

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
    if allow_dictionary and entry is not None:
        definition = _first_non_empty(entry.definition, ", ".join(entry.senses))
        add(definition, "dictionary")

    if canonical and canonical != morph:
        for entry in _meaning_from_evidence(
            morph=canonical,
            evidence=evidence,
            canonical=None,
            allow_dictionary=allow_dictionary,
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


def _semantic_core_from_glossator_json(payload: Mapping[str, object]) -> list[str]:
    return normalize_semantic_terms(
        payload.get("SEMANTIC_CORE", payload.get("semantic_core"))
    )


def _negative_contrast_from_glossator_json(payload: Mapping[str, object]) -> list[str]:
    return normalize_semantic_terms(
        payload.get("NEGATIVE_CONTRAST", payload.get("negative_contrast"))
    )


def _glossator_candidate_metadata(payload: object) -> tuple[str | None, list[str], list[str]]:
    if payload is None:
        return None, [], []
    if isinstance(payload, Mapping):
        return (
            _definition_from_glossator_json(payload),
            _semantic_core_from_glossator_json(payload),
            _negative_contrast_from_glossator_json(payload),
        )
    if not isinstance(payload, str):
        return None, [], []

    text = payload.strip()
    if not text:
        return None, [], []
    parsed = _parse_glossator_json(text)
    if not isinstance(parsed, Mapping):
        return None, [], []
    return (
        _definition_from_glossator_json(parsed),
        _semantic_core_from_glossator_json(parsed),
        _negative_contrast_from_glossator_json(parsed),
    )


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


_FUNCTION_CANONICAL_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "imperative_existential",
        "let there be",
        ("let there be", "bring into being", "establish create"),
    ),
    (
        "within_self",
        "within itself",
        ("within itself", "in itself", "within her"),
    ),
    (
        "conjunction",
        "and",
        ("conjunction", "additive", "links", "linkage", "parallel elements"),
    ),
    (
        "relative",
        "that",
        ("which", "that", "relativizer", "relative clause", "subordination"),
    ),
    (
        "feminine_locative_possessive",
        "her",
        ("her", "she", "feminine", "possessive", "locative relation"),
    ),
    (
        "locative",
        "in",
        ("locative", "inside", "within", "in"),
    ),
)

_FUNCTION_ABSTRACT_HINTS: dict[str, set[str]] = {
    "imperative_existential": {"existence", "state", "copula", "being"},
    "within_self": {"within", "inside", "self", "locative", "relation"},
    "conjunction": {"conjunction", "additive", "parallel", "elements", "link", "linkage"},
    "relative": {"relation", "subordination", "relative", "clause", "entity"},
    "feminine_locative_possessive": {"possession", "possessive", "locative", "relation", "feminine", "referent"},
    "locative": {"locative", "inside", "within", "position", "relation"},
}
_BUNDLE_STOPWORD_CLASS_GLOSSES = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "her",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "through",
    "upon",
    "which",
    "with",
}


def compose_semantic_bundle(meanings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Collapse ordered morph meanings into one word-level semantic bundle.

    Single-word translation already knows the morph-by-morph reading for a
    decomposition. Phrase translation needs that richer structure preserved so
    blind-mode ranking can prefer coherent decompositions instead of collapsing
    immediately to one gloss string. This helper keeps the ordered per-morph
    evidence and derives a human-facing bundle head plus a lightweight
    coherence score for downstream ranking and phrase assembly.
    """

    bundle_entries: list[dict[str, object]] = []
    surface_candidates: list[str] = []
    content_entries: list[dict[str, object]] = []
    function_entries: list[dict[str, object]] = []

    for meaning in meanings:
        if not isinstance(meaning, Mapping):
            continue
        entry = _compose_bundle_entry(meaning)
        bundle_entries.append(entry)
        head_gloss = str(entry.get("head_gloss") or "").strip()
        if head_gloss and head_gloss not in surface_candidates:
            surface_candidates.append(head_gloss)
        if str(entry.get("kind") or "") == "function":
            function_entries.append(entry)
        else:
            content_entries.append(entry)

    bundle_head = _bundle_head_gloss(content_entries, function_entries)
    bundle_surface = bundle_head
    bundle_function_profile = _bundle_function_profile(content_entries, function_entries)
    bundle_coherence = _bundle_coherence_score(
        bundle_entries,
        bundle_head_gloss=bundle_head,
    )

    return {
        "semantic_bundle": bundle_entries,
        "bundle_surface_gloss": bundle_surface,
        "bundle_head_gloss": bundle_head,
        "bundle_function_profile": bundle_function_profile,
        "bundle_coherence_score": bundle_coherence,
        "bundle_surface_candidates": surface_candidates,
        "bundle_selection_reason": _bundle_selection_reason(
            bundle_entries,
            bundle_head_gloss=bundle_head,
            bundle_function_profile=bundle_function_profile,
        ),
    }


def _compose_bundle_entry(meaning: Mapping[str, object]) -> dict[str, object]:
    """Normalize one morph meaning into a bundle entry for phrase composition.

    The phrase layer needs a stable per-morph record that already knows whether
    it behaves more like a function word or a content word. Building that here
    keeps all decomposition candidates on the same lexical footing before the
    phrase parser ever sees them.
    """

    morph = str(meaning.get("morph") or "").upper()
    raw_definition = (
        str(meaning.get("raw_definition") or "").strip() or None
    )
    surface_gloss = (
        str(meaning.get("surface_gloss") or "").strip() or None
    )
    if not surface_gloss:
        definition = meaning.get("definition")
        if isinstance(definition, str) and definition.strip():
            surface_gloss = definition.strip()
    semantic_core_terms = normalize_semantic_terms(
        meaning.get("semantic_core_terms") or meaning.get("semantic_core")
    )
    negative_contrast = normalize_semantic_terms(
        meaning.get("negative_contrast")
    )
    provenance = str(meaning.get("provenance") or "unknown")
    trace = meaning.get("definition_trace")
    quality = 0.0
    if isinstance(trace, Mapping):
        quality = _safe_number(trace.get("selected_quality"), default=0.0)
    quality = max(quality, _safe_number(meaning.get("anchor_strength"), default=0.0))

    function_profile, canonical_gloss = _infer_function_profile(
        surface_gloss=surface_gloss,
        semantic_core_terms=semantic_core_terms,
        raw_definition=raw_definition,
    )
    chosen_gloss = _choose_bundle_entry_gloss(
        surface_gloss=surface_gloss,
        canonical_gloss=canonical_gloss,
        function_profile=function_profile,
        semantic_core_terms=semantic_core_terms,
        negative_contrast=negative_contrast,
    )

    return {
        "morph": morph,
        "surface_gloss": surface_gloss,
        "semantic_core_terms": semantic_core_terms,
        "negative_contrast": negative_contrast,
        "provenance": provenance,
        "function_profile": function_profile,
        "kind": "function" if function_profile else "content",
        "head_gloss": chosen_gloss,
        "quality": quality,
        "raw_definition": raw_definition,
    }


def _infer_function_profile(
    *,
    surface_gloss: str | None,
    semantic_core_terms: Sequence[str],
    raw_definition: str | None,
) -> tuple[str | None, str | None]:
    """Infer whether a morph behaves like a normalized function word.

    Blind-mode phrase rendering needs compact English for conjunctions,
    relative markers, feminine possessives, and locatives. This helper detects
    those cases from the surviving semantic payload without flattening content
    words into grammar labels.
    """

    normalized_gloss = (surface_gloss or "").strip().lower()
    normalized_terms = [term.strip().lower() for term in semantic_core_terms if term]
    raw_lower = (raw_definition or "").strip().lower()

    for profile, canonical, hints in _FUNCTION_CANONICAL_RULES:
        for hint in hints:
            hint_lower = hint.lower()
            if normalized_gloss == hint_lower:
                return profile, canonical
            if hint_lower in normalized_terms:
                return profile, canonical
            if hint_lower and _raw_definition_contains_hint(raw_lower, hint_lower):
                if profile == "feminine_locative_possessive" and "glory" in raw_lower:
                    continue
                return profile, canonical

    if (
        "feminine" in raw_lower
        and "possessive" in raw_lower
        and "locative" in raw_lower
        and "glory" not in raw_lower
    ):
        return "feminine_locative_possessive", "her"
    return None, None


def _choose_bundle_entry_gloss(
    *,
    surface_gloss: str | None,
    canonical_gloss: str | None,
    function_profile: str | None,
    semantic_core_terms: Sequence[str],
    negative_contrast: Sequence[str],
) -> str | None:
    """Pick the lexical head for one bundle entry with soft function preference.

    The user wants concise function-word normalization, but only as a medium
    preference. This helper therefore keeps richer lexical phrasing when it is
    materially better, while still canonicalizing abstract grammar prose such
    as `parallel elements` or `possession`.
    """

    if canonical_gloss is None:
        return surface_gloss

    current = surface_gloss
    if current is None:
        return canonical_gloss
    if is_meta_linguistic_gloss(current) or len(current.split()) > 2:
        return canonical_gloss

    generic_hints = _FUNCTION_ABSTRACT_HINTS.get(function_profile or "", set())
    current_tokens = _bundle_gloss_tokens(current)
    if current_tokens and current_tokens <= generic_hints:
        return canonical_gloss

    current_score = _bundle_similarity_score(
        current,
        semantic_terms=semantic_core_terms,
        negative_contrast=negative_contrast,
    )
    canonical_score = _bundle_similarity_score(
        canonical_gloss,
        semantic_terms=semantic_core_terms,
        negative_contrast=negative_contrast,
    )
    if canonical_score + 0.20 >= current_score:
        return canonical_gloss
    return current


def _bundle_head_gloss(
    content_entries: Sequence[Mapping[str, object]],
    function_entries: Sequence[Mapping[str, object]],
) -> str | None:
    """Choose the word-level lexical head from ordered bundle entries.

    Phrase assembly needs one compact lexical anchor per token even when the
    underlying candidate preserves several morph meanings. This chooser favors
    the strongest surviving content gloss and only falls back to a function
    gloss when the bundle is purely grammatical.
    """

    if content_entries:
        ranked = sorted(
            content_entries,
            key=lambda entry: (
                not _is_stopword_class_gloss(str(entry.get("head_gloss") or "")),
                _safe_number(entry.get("quality"), default=0.0),
                -len(str(entry.get("head_gloss") or "").split()),
            ),
            reverse=True,
        )
        for entry in ranked:
            head_gloss = entry.get("head_gloss")
            if isinstance(head_gloss, str) and head_gloss.strip():
                return head_gloss.strip()
    for entry in function_entries:
        head_gloss = entry.get("head_gloss")
        if isinstance(head_gloss, str) and head_gloss.strip():
            return head_gloss.strip()
    return None


def _bundle_function_profile(
    content_entries: Sequence[Mapping[str, object]],
    function_entries: Sequence[Mapping[str, object]],
) -> str:
    """Summarize whether a bundle behaves like a function or content word."""

    if content_entries:
        return "content"
    for entry in function_entries:
        profile = entry.get("function_profile")
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
    return "unknown"


def _bundle_coherence_score(
    bundle_entries: Sequence[Mapping[str, object]],
    *,
    bundle_head_gloss: str | None,
) -> float:
    """Estimate whether an ordered morph bundle yields a usable word reading.

    Blind-mode ranking needs a cheap proxy for “does this decomposition look
    phrase-usable?” This score rewards lexical heads, non-meta wording, and
    per-entry quality while demoting bundles that still read like analysis
    prose instead of translation.
    """

    if not bundle_entries:
        return 0.0

    quality_values: list[float] = []
    lexical_entries = 0
    function_entries = 0
    for entry in bundle_entries:
        quality_values.append(_safe_number(entry.get("quality"), default=0.0))
        head_gloss = str(entry.get("head_gloss") or "").strip()
        if head_gloss and not is_meta_linguistic_gloss(head_gloss):
            lexical_entries += 1
        if str(entry.get("kind") or "") == "function":
            function_entries += 1

    average_quality = sum(quality_values) / float(len(quality_values)) if quality_values else 0.0
    lexical_ratio = lexical_entries / float(len(bundle_entries))
    score = 0.35 * average_quality + 0.45 * lexical_ratio
    if bundle_head_gloss:
        score += 0.20
    if function_entries and lexical_entries:
        score += 0.05
    return max(0.0, min(1.0, score))


def _bundle_selection_reason(
    bundle_entries: Sequence[Mapping[str, object]],
    *,
    bundle_head_gloss: str | None,
    bundle_function_profile: str,
) -> str:
    """Summarize how ordered morph evidence collapsed into one word reading."""

    if not bundle_entries:
        return "No bundle evidence survived."
    if bundle_head_gloss is None:
        return "No lexical bundle head survived after cleanup."
    entry_count = len(bundle_entries)
    if bundle_function_profile != "content":
        return (
            f"Ordered bundle normalized {entry_count} morph reading(s) into the "
            f'function gloss "{bundle_head_gloss}".'
        )
    return (
        f"Ordered bundle preferred the lexical head \"{bundle_head_gloss}\" from "
        f"{entry_count} surviving morph reading(s)."
    )


def _bundle_score_bonus(bundle: Mapping[str, object]) -> float:
    """Add a small bundle-viability bonus to compositional candidate scores.

    Decomposition ranking should still be dominated by the underlying evidence
    model, but phrase translation benefits when candidates that already produce
    a usable lexical head rise slightly above equally scored glossary-like
    alternatives.
    """

    coherence = _safe_number(bundle.get("bundle_coherence_score"), default=0.0)
    head_gloss = str(bundle.get("bundle_head_gloss") or "").strip()
    if not head_gloss:
        return -0.25
    if is_meta_linguistic_gloss(head_gloss):
        return -0.35
    return 0.75 * coherence


def _bundle_gloss_tokens(text: str) -> set[str]:
    """Tokenize a bundle gloss without depending on placeholder helpers.

    Bundle scoring needs a tiny lexical overlap check, but the phrase/placeholder
    layer keeps its tokenizer private. This local copy keeps the strategies
    layer self-contained while still using the same stopword policy.
    """

    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", text)
        if token.lower() not in {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "that", "the", "to", "through", "upon", "which", "with"}
    }


def _is_stopword_class_gloss(text: str) -> bool:
    """Return whether a gloss is mostly function-word scaffolding.

    Bundle head selection should avoid collapsing onto purely grammatical
    stopwords when richer lexical candidates survive in the same bundle.
    """

    tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
    if not tokens:
        return True
    return all(token in _BUNDLE_STOPWORD_CLASS_GLOSSES for token in tokens)


def _raw_definition_contains_hint(raw_definition: str, hint: str) -> bool:
    """Match canonical function hints only at token/phrase boundaries.

    Raw-definition scans are useful for function-word normalization, but raw
    substring checks can misfire on words like ``inherent`` because they
    contain ``her``. Boundary-aware matching keeps those false positives from
    collapsing lexical glosses into function shorthand.
    """

    normalized_hint = " ".join(hint.split()).strip()
    if not normalized_hint:
        return False
    pattern = re.compile(
        rf"(?<![a-zA-Z]){re.escape(normalized_hint)}(?![a-zA-Z])"
    )
    return bool(pattern.search(raw_definition))


def _bundle_similarity_score(
    candidate: str,
    *,
    semantic_terms: Sequence[str],
    negative_contrast: Sequence[str],
) -> float:
    """Score one bundle gloss candidate against semantic-core hints.

    Bundle gloss choice only needs a lightweight lexical alignment check. This
    helper intentionally mirrors the semantic-core-first policy without pulling
    in the broader placeholder-gloss selection machinery.
    """

    candidate_tokens = _bundle_gloss_tokens(candidate)
    if not candidate_tokens:
        return -1.0

    best_score = -1.0
    for term in semantic_terms:
        term_tokens = _bundle_gloss_tokens(term)
        if not term_tokens:
            continue
        overlap = len(candidate_tokens & term_tokens) / float(len(term_tokens))
        score = overlap
        if candidate.strip().lower() == term.strip().lower():
            score += 0.05
        blocked = gloss_overlaps_negative_contrast(candidate, negative_contrast)
        score -= 0.25 * float(len(blocked))
        best_score = max(best_score, score)
    return best_score


# ---------------------------------------------
# Beam-scoring helpers for definition candidates
# ---------------------------------------------

def extract_definition_candidates(
    morphs: Iterable[str],
    evidence: WordEvidence,
    *,
    max_per_morph: int = 3,
    allow_dictionary: bool = True,
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

        candidates = _collect_definition_candidates(
            morph,
            evidence,
            allow_dictionary=allow_dictionary,
        )
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

    embedder = get_sentence_transformer_if_available(
        "paraphrase-MiniLM-L6-v2",
        local_files_only=True,
    )
    if embedder is None:
        return 0.0

    try:
        vectors = [
            np.array(embedder.encode(text, normalize_embeddings=True), dtype=float)
            for text in cleaned
        ]
    except Exception:
        return 0.0

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


def compute_contradiction_penalty_for_candidates(
    morphs: Iterable[str],
    candidates: Mapping[str, list[dict[str, object]]],
    *,
    beam_width: int = 6,
    max_defs_per_morph: int = 4,
    keyword_pairs: Iterable[tuple[str, str]] | None = None,
    penalty_per_pair: float = 0.20,
    max_penalty: float = 0.50,
) -> float:
    selections, _beam_results = _select_definition_combination(
        morphs,
        candidates,
        beam_width=beam_width,
        max_defs_per_morph=max_defs_per_morph,
    )
    if not selections:
        return 0.0

    definitions_by_morph: dict[str, list[str]] = {}
    for morph, selection in selections.items():
        definition = selection.get("definition")
        if isinstance(definition, str) and definition.strip():
            definitions_by_morph[morph] = [definition]
        else:
            definitions_by_morph[morph] = []

    return compute_contradiction_penalty(
        definitions_by_morph,
        keyword_pairs=keyword_pairs,
        penalty_per_pair=penalty_per_pair,
        max_penalty=max_penalty,
    )


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

    embedder = get_sentence_transformer_if_available(
        "paraphrase-MiniLM-L6-v2",
        local_files_only=True,
    )
    if embedder is None:
        return {definition: idx for idx, definition in enumerate(cleaned)}

    try:
        clusters = cluster_definitions(
            cleaned,
            model=embedder,
            similarity_threshold=0.8,
        )
    except Exception:
        return {definition: idx for idx, definition in enumerate(cleaned)}
    mapping: dict[str, int] = {}
    for idx, cluster in enumerate(clusters):
        for member_idx in cluster.get("members", []):
            if isinstance(member_idx, int) and 0 <= member_idx < len(cleaned):
                mapping[cleaned[member_idx]] = idx
    return mapping


def _collect_definition_candidates(
    morph: str,
    evidence: WordEvidence,
    *,
    allow_dictionary: bool = True,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_candidate(payload: dict[str, object]) -> None:
        raw_definition = payload.get("raw_definition")
        if not isinstance(raw_definition, str):
            raw_definition = payload.get("definition")
        semantic_core_terms = normalize_semantic_terms(payload.get("semantic_core_terms"))
        negative_contrast = normalize_semantic_terms(payload.get("negative_contrast"))
        definition, strategy = surface_gloss_from_sources(
            semantic_core=semantic_core_terms,
            definition=raw_definition,
            negative_contrast=negative_contrast,
            token=morph,
        )
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
        payload["raw_definition"] = raw_definition.strip() if isinstance(raw_definition, str) else None
        payload["semantic_core_terms"] = semantic_core_terms
        payload["negative_contrast"] = negative_contrast
        payload["surface_gloss_strategy"] = strategy
        payload["meta_linguistic_rejection"] = (
            isinstance(raw_definition, str) and is_meta_linguistic_gloss(raw_definition)
        )
        payload["quality"] = _candidate_quality(payload)
        candidates.append(payload)

    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition, semantic_core_terms, negative_contrast = _glossator_candidate_metadata(
            cluster.glossator_def
        )
        definition = _first_non_empty(
            definition,
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        if definition:
            add_candidate(
                {
                    "raw_definition": definition,
                    "semantic_core_terms": semantic_core_terms or cluster.semantic_core,
                    "negative_contrast": negative_contrast or cluster.negative_contrast,
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
        definition, semantic_core_terms, negative_contrast = _glossator_candidate_metadata(
            residual.glossator_def
        )
        definition = _first_non_empty(
            definition,
            residual.residual_headline,
        )
        if definition:
            add_candidate(
                {
                    "raw_definition": definition,
                    "semantic_core_terms": semantic_core_terms or residual.semantic_core,
                    "negative_contrast": negative_contrast or residual.negative_contrast,
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
                    "raw_definition": definition,
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
                    "raw_definition": definition,
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
                    "raw_definition": definition,
                    "source": "dictionary",
                }
            )

    if allow_dictionary:
        return candidates

    non_dictionary = [
        candidate for candidate in candidates if str(candidate.get("source") or "") != "dictionary"
    ]
    if non_dictionary:
        return non_dictionary

    dictionary_only: list[dict[str, object]] = []
    for candidate in candidates:
        if str(candidate.get("source") or "") != "dictionary":
            continue
        fallback = dict(candidate)
        fallback["blind_dictionary_fallback"] = True
        dictionary_only.append(fallback)
    return dictionary_only


def _definition_trace(
    *,
    morph: str,
    selected: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]],
    evidence: WordEvidence | None,
    allow_dictionary: bool,
) -> dict[str, object]:
    """Capture the provenance trail behind a selected morph definition.

    Phrase-level reporting needs more than the winning gloss string. This trace
    preserves the selected source, nearby runner-ups, and any blind-mode
    suppressions so downstream footnotes can explain why a reading survived.
    """

    selected_definition = (
        selected.get("definition") if isinstance(selected, Mapping) else None
    )
    selected_raw_definition = (
        selected.get("raw_definition") if isinstance(selected, Mapping) else None
    )
    selected_semantic_core = normalize_semantic_terms(
        selected.get("semantic_core_terms") if isinstance(selected, Mapping) else []
    )
    selected_negative_contrast = normalize_semantic_terms(
        selected.get("negative_contrast") if isinstance(selected, Mapping) else []
    )
    runner_ups: list[dict[str, object]] = []
    for candidate in candidates:
        definition = candidate.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            continue
        if (
            isinstance(selected_definition, str)
            and definition.strip() == selected_definition.strip()
        ):
            continue
        runner_ups.append(
            {
                "definition": definition.strip(),
                "raw_definition": candidate.get("raw_definition"),
                "semantic_core": list(candidate.get("semantic_core_terms") or []),
                "negative_contrast": list(candidate.get("negative_contrast") or []),
                "surface_gloss_strategy": candidate.get("surface_gloss_strategy"),
                "source": str(candidate.get("source") or "unknown"),
                "quality": _safe_number(candidate.get("quality"), default=0.0),
            }
        )

    blind_dictionary_fallback = bool(
        isinstance(selected, Mapping) and selected.get("blind_dictionary_fallback")
    )
    return {
        "selected_definition": selected_definition.strip()
        if isinstance(selected_definition, str) and selected_definition.strip()
        else None,
        "raw_selected_definition": (
            selected_raw_definition.strip()
            if isinstance(selected_raw_definition, str) and selected_raw_definition.strip()
            else None
        ),
        "selected_semantic_core": selected_semantic_core,
        "selected_negative_contrast": selected_negative_contrast,
        "surface_gloss": selected_definition.strip()
        if isinstance(selected_definition, str) and selected_definition.strip()
        else None,
        "surface_gloss_strategy": (
            str(selected.get("surface_gloss_strategy") or "unresolved")
            if isinstance(selected, Mapping)
            else "unresolved"
        ),
        "selected_source": (
            str(selected.get("source") or "unknown")
            if isinstance(selected, Mapping)
            else "unknown"
        ),
        "selected_quality": (
            _safe_number(selected.get("quality"), default=0.0)
            if isinstance(selected, Mapping)
            else 0.0
        ),
        "runner_ups": runner_ups[:3],
        "suppressed": _definition_suppression_notes(
            morph=morph,
            evidence=evidence,
            allow_dictionary=allow_dictionary,
            blind_dictionary_fallback=blind_dictionary_fallback,
        ),
        "blind_dictionary_fallback": blind_dictionary_fallback,
        "negative_contrast_penalties": list(
            selected.get("negative_contrast_penalties") or []
        )
        if isinstance(selected, Mapping)
        else [],
        "meta_linguistic_rejections": list(
            selected.get("meta_linguistic_rejections") or []
        )
        if isinstance(selected, Mapping)
        else [],
    }


def _definition_suppression_notes(
    *,
    morph: str,
    evidence: WordEvidence | None,
    allow_dictionary: bool,
    blind_dictionary_fallback: bool,
) -> list[str]:
    """Describe blind-mode dictionary filtering for one morph trace."""

    if allow_dictionary or evidence is None or evidence.dictionary_morphs.get(morph) is None:
        return []
    if blind_dictionary_fallback:
        return [
            f"Blind mode reintroduced dictionary evidence for {morph} because no non-dictionary definition survived."
        ]
    return [f"Blind mode suppressed dictionary-backed definition candidates for {morph}."]


def _candidate_quality(candidate: dict[str, object]) -> float:
    metrics = [
        candidate.get("semantic_coverage"),
        candidate.get("cohesion"),
        candidate.get("semantic_cohesion"),
        candidate.get("delta_cosine"),
    ]
    values = [_safe_number(metric, default=0.0) for metric in metrics if metric is not None]
    base_quality = sum(values) / float(len(values)) if values else 0.0

    semantic_core_terms = normalize_semantic_terms(candidate.get("semantic_core_terms"))
    surface_gloss = candidate.get("definition")
    lexical_bonus = 0.12 if semantic_core_terms else 0.0
    if isinstance(surface_gloss, str) and surface_gloss.strip():
        if not is_meta_linguistic_gloss(surface_gloss):
            lexical_bonus += 0.08

    negative_contrast_penalties = gloss_overlaps_negative_contrast(
        surface_gloss,
        candidate.get("negative_contrast"),
    )
    candidate["negative_contrast_penalties"] = negative_contrast_penalties
    meta_linguistic_rejections = []
    raw_definition = candidate.get("raw_definition")
    if isinstance(raw_definition, str) and is_meta_linguistic_gloss(raw_definition):
        meta_linguistic_rejections.append(raw_definition.strip())
    candidate["meta_linguistic_rejections"] = meta_linguistic_rejections

    penalty = 0.18 * len(negative_contrast_penalties)
    if candidate.get("meta_linguistic_rejection") and not semantic_core_terms:
        penalty += 0.12

    return max(0.0, base_quality + lexical_bonus - penalty)
