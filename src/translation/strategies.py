from __future__ import annotations

from collections.abc import Iterable
import json
import re

from enochian_lm.common.types import MaybeNumber
from .decomposition import Decomposition
from .repository import ClusterRecord, WordEvidence


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

    meanings: list[dict[str, object]] = []
    for idx, morph in enumerate(decomp.morphs):
        key = (morph or "").upper()
        canonical = (
            decomp.canonicals[idx].upper()
            if idx < len(decomp.canonicals)
            else None
        )
        definition, provenance = _meaning_from_evidence(
            morph=key,
            evidence=evidence,
            canonical=canonical,
        )

        if provenance == "unknown":
            provenance = support_lookup.get(key, "unknown")

        meanings.append(
            {
                "morph": key,
                "canonical": canonical,
                "definition": definition,
                "provenance": provenance,
            }
        )

    return meanings


def _meaning_from_evidence(
    *,
    morph: str,
    evidence: WordEvidence | None,
    canonical: str | None,
) -> tuple[str | None, str]:
    if evidence is None:
        return None, "unknown"

    # 1) clusters
    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(cluster.glossator_def),
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        if definition is not None:
            return definition, "cluster"

    # 2) residuals
    for residual in evidence.residual_semantics:
        if residual.residual.upper() != morph:
            continue
        definition = _first_non_empty(
            _extract_glossator_definition(residual.glossator_def),
            residual.residual_headline,
        )
        if definition is not None:
            return definition, "residual"

    # 3) hypotheses
    for hypothesis in evidence.morph_hypotheses:
        if hypothesis.morph.upper() != morph:
            continue
        seed_glosses = ", ".join(g for g in hypothesis.seed_glosses if g and g.strip()) or None
        definition = _first_non_empty(hypothesis.proposed_gloss, seed_glosses, hypothesis.anchor)
        if definition is not None:
            return definition, "hypothesis"

    # 4) attested definitions
    for attested in evidence.attested_definitions:
        if attested.source_word.upper() != morph:
            continue
        definition = _first_non_empty(attested.definition)
        if definition is not None:
            return definition, "attested"

    # 5) dictionary entries
    entry = evidence.dictionary_morphs.get(morph)
    if entry is not None:
        definition = _first_non_empty(entry.definition, ", ".join(entry.senses))
        if definition is not None:
            return definition, "dictionary"

    if canonical and canonical != morph:
        canonical_def, canonical_prov = _meaning_from_evidence(
            morph=canonical,
            evidence=evidence,
            canonical=None,
        )
        if canonical_def is not None:
            return canonical_def, f"canonical_{canonical_prov}"

    return None, "unknown"


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
