from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

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
        return -0.2 * _length_variance(decomp.morphs)

    return 0.0


def _compile_support_counts(evidence: WordEvidence) -> Dict[str, int]:
    """Aggregate per-morph usage counts across evidence types (case-insensitive keys)."""
    counts: Dict[str, int] = {}

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

    return counts


# ---------------------------
# Task 3.2: top-k selection
# ---------------------------

def select_top_k(
    ranked: List[Tuple[Decomposition, float]],
    k: int = 3,
    *,
    evidence: WordEvidence | None = None,
) -> List[dict[str, object]]:
    if not ranked:
        return []

    try:
        top_k = int(k)
    except (TypeError, ValueError):
        top_k = 0
    if top_k <= 0:
        return []

    ordered: List[Tuple[Decomposition, float]] = sorted(
        ((d, _safe_number(s, default=0.0)) for d, s in ranked),
        key=lambda pair: pair[1],
        reverse=True,
    )

    tie_warning: str | None = None
    if len(ordered) >= 2:
        delta = abs(ordered[0][1] - ordered[1][1])
        if delta < 0.05:
            tie_warning = "alternate decomposition exists"

    results: List[dict[str, object]] = []
    for idx, (decomp, score) in enumerate(ordered[:top_k], start=1):
        warnings: List[str] = []
        if tie_warning and idx <= 2:
            warnings.append(tie_warning)

        results.append(
            {
                "rank": idx,
                "morphs": list(decomp.morphs),
                "score": score,
                "breakdown": decomp.breakdown,
                "meanings": _extract_meanings(decomp=decomp, evidence=evidence),
                "warnings": warnings,
            }
        )

    return results


def _extract_meanings(
    *,
    decomp: Decomposition,
    evidence: WordEvidence | None,
) -> List[dict[str, object]]:
    support_lookup: Dict[str, str] = {
        k.upper(): v for k, v in (decomp.morph_support or {}).items()
    }

    meanings: List[dict[str, object]] = []
    for morph in decomp.morphs:
        key = (morph or "").upper()
        definition, provenance = _meaning_from_evidence(morph=key, evidence=evidence)

        if provenance == "unknown":
            provenance = support_lookup.get(key, "unknown")

        meanings.append(
            {
                "morph": key,
                "definition": definition,
                "provenance": provenance,
            }
        )

    return meanings


def _meaning_from_evidence(
    *,
    morph: str,
    evidence: WordEvidence | None,
) -> tuple[str | None, str]:
    if evidence is None:
        return None, "unknown"

    # 1) clusters
    for cluster in evidence.direct_clusters:
        if cluster.ngram.upper() != morph:
            continue
        definition = _first_non_empty(
            cluster.glossator_def,
            _first_cluster_raw_definition(cluster),
            cluster.residual_headline,
        )
        if definition is not None:
            return definition, "cluster"

    # 2) residuals
    for residual in evidence.residual_semantics:
        if residual.residual.upper() != morph:
            continue
        definition = _first_non_empty(residual.glossator_def, residual.residual_headline)
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

    return None, "unknown"


def _first_cluster_raw_definition(cluster: ClusterRecord) -> str | None:
    # raw_definitions is a List[RawDefinition] (dataclass), so use attributes.
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


def _length_variance(morphs: Iterable[str]) -> float:
    lengths = [len(m or "") for m in morphs]
    if len(lengths) <= 1:
        return 0.0
    mean = sum(lengths) / float(len(lengths))
    return sum((length - mean) ** 2 for length in lengths) / float(len(lengths))


def _safe_number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
