from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import numbers
from pathlib import Path
from types import TracebackType
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder

from .decomposition import Decomposition, DecompositionEngine, apply_hard_filters
from .llm_synthesis import SynthesisResult, synthesize_definition
from .repository import (
    ClusterRecord,
    InsightsRepository,
    ResidualDetail,
    WordEvidence,
    FasttextNeighbor,
)
from .scoring import ScoringWeights, score_decomposition
from .strategies import apply_strategy, select_top_k
from .tokenization import expand_sentence_ngrams, tokenize_words


class InterpretationService:
    """Coordinate DB lookups and residual analysis for unseen text."""

    def __init__(
        self,
        *,
        candidate_finder: MorphemeCandidateFinder,
        repository: InsightsRepository,
        active_variants: Optional[Iterable[str]] = None,
        max_ngram_len: int = 7,
    ) -> None:
        self.candidate_finder = candidate_finder
        self.repository = repository
        self.active_variants = list(active_variants) if active_variants else None
        self.max_ngram_len = max_ngram_len

    @classmethod
    def from_config(
        cls,
        *,
        variants: Optional[Iterable[str]] = None,
        max_ngram_len: int = 7,
    ) -> "InterpretationService":
        paths = get_config_paths()
        repository = InsightsRepository(
            solo_path=Path(paths["solo"]),
            debate_path=Path(paths["debate"]),
        )
        if variants:
            repository.require_variants(variants)
        elif not repository.variants:
            raise FileNotFoundError(
                "No insights databases found. Run enochian-analysis first."
            )

        entries = load_dictionary(str(paths["dictionary"]))
        candidate_finder = MorphemeCandidateFinder(
            ngram_db_path=paths["ngram_index"],
            fasttext_model_path=paths["model_output"],
            dictionary_entries=entries,
            min_n=1,
            max_n=max_ngram_len,
        )
        return cls(
            candidate_finder=candidate_finder,
            repository=repository,
            active_variants=list(variants) if variants else None,
            max_ngram_len=max_ngram_len,
        )

    def close(self) -> None:
        self.candidate_finder.close()
        self.repository.close()

    def __enter__(self) -> "InterpretationService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def interpret_text(
        self,
        text: str,
        *,
        max_ngram_len: Optional[int] = None,
        variants: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        max_len = max_ngram_len or self.max_ngram_len
        active_variants = list(variants) if variants else self.active_variants
        if active_variants:
            self.repository.require_variants(active_variants)

        tokens = tokenize_words(text)
        ngram_slices = expand_sentence_ngrams(text, max_len=max_len)
        analyses: List[Dict[str, object]] = []
        for slice_info in ngram_slices:
            ngram_text = slice_info.ngram
            cluster_records = self.repository.fetch_clusters(
                ngram_text, variants=active_variants
            )
            candidate_breakdowns = self.candidate_finder.find_candidates(ngram_text)

            candidate_lookup: Dict[str, Dict[str, object]] = {}
            unique_candidates: List[Dict[str, object]] = []
            seen_signatures: Set[tuple] = set()

            for candidate in candidate_breakdowns:
                cleaned = _cleanup_candidate(candidate)
                signature = (
                    cleaned.get("normalized"),
                    cleaned.get("composite"),
                    cleaned.get("cos_sim"),
                    cleaned.get("target_length"),
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                unique_candidates.append(cleaned)
                for key in _candidate_lookup_keys(candidate):
                    if key not in candidate_lookup:
                        candidate_lookup[key] = cleaned

            analyses.append(
                {
                    "word_index": slice_info.word_index,
                    "span": [slice_info.start, slice_info.end],
                    "ngram": ngram_text.upper(),
                    "candidates": unique_candidates,
                    "clusters": [
                        self._serialize_cluster(
                            record, candidate_lookup, unique_candidates
                        )
                        for record in cluster_records
                    ],
                }
            )

        return {
            "text": text,
            "tokens": tokens,
            "max_ngram_len": max_len,
            "variants": active_variants or self.repository.variants,
            "analyses": analyses,
        }

    def _serialize_cluster(
        self,
        record: ClusterRecord,
        candidate_lookup: Dict[str, Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> Dict[str, object]:
        serialized: Dict[str, object] = {
            "variant": record.variant,
            "cluster_id": record.cluster_id,
            "run_id": record.run_id,
            "ngram": record.ngram,
            "cluster_index": record.cluster_index,
            "glossator_def": record.glossator_def,
            "residual_summary": {
                "explained_ratio": record.residual_explained,
                "residual_ratio": record.residual_ratio,
                "headline": record.residual_headline,
                "focus_prompt": record.residual_focus_prompt,
            },
            "metrics": {
                "semantic_coverage": record.semantic_coverage,
                "cohesion": record.cohesion,
                "semantic_cohesion": record.semantic_cohesion,
                "best_config": record.best_config,
            },
            "residual_details": self._reconcile_residual_details(
                record.residual_details, candidate_lookup, candidates
            ),
        }

        enriched_defs: List[Dict[str, object]] = []
        for raw_def in record.raw_definitions:
            normalized_options = {
                _safe_lower(raw_def.source_word),
                _safe_lower(raw_def.variant),
            }
            normalized_options.discard(None)

            matched_candidate = None
            matched_key = None
            for key in normalized_options:
                if key and key in candidate_lookup:
                    matched_candidate = candidate_lookup[key]
                    matched_key = key
                    break

            enriched = asdict(raw_def)
            enriched["normalized"] = matched_key
            enriched["candidate"] = matched_candidate
            enriched_defs.append(enriched)

        serialized["definitions"] = enriched_defs
        return serialized

    def _reconcile_residual_details(
        self,
        details: List[ResidualDetail],
        candidate_lookup: Dict[str, Dict[str, object]],
        candidates: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        reconciled: List[Dict[str, object]] = []
        matched_candidates: Set[int] = set()

        for detail in details:
            candidate = None
            normalized_key = _safe_lower(detail.normalized)
            if normalized_key and normalized_key in candidate_lookup:
                candidate = candidate_lookup[normalized_key]
                matched_candidates.add(id(candidate))

            reconciled.append(self._merge_residual_detail(detail, candidate))

        for candidate in candidates:
            if id(candidate) in matched_candidates:
                continue
            reconciled.append(_residual_detail_from_candidate(candidate))

        return reconciled

    def _merge_residual_detail(
        self, detail: ResidualDetail, candidate: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "normalized": detail.normalized,
            "definition": detail.definition,
            "coverage_ratio": detail.coverage_ratio,
            "residual_ratio": detail.residual_ratio,
            "avg_confidence": detail.avg_confidence,
            "uncovered": list(detail.uncovered),
            "low_confidence": list(detail.low_confidence),
            "source": "database",
            "candidate": candidate,
        }

        if candidate:
            raw_breakdown = candidate.get("breakdown")
            breakdown = raw_breakdown if isinstance(raw_breakdown, dict) else {}
            payload["candidate_breakdown"] = breakdown
            payload["candidate_coverage_ratio"] = breakdown.get("coverage_ratio")
            payload["candidate_residual_ratio"] = breakdown.get("residual_ratio")
            payload["candidate_uncovered"] = _extract_uncovered_texts(breakdown)
            payload["candidate_low_confidence"] = _extract_low_confidence_segments(
                breakdown
            )
        else:
            payload["candidate_breakdown"] = None

        return payload


class SingleWordTranslationService:
    """Single-word translation pipeline with optional LLM synthesis (Tasks 4.1/4.2)."""

    def __init__(
        self,
        *,
        candidate_finder: MorphemeCandidateFinder,
        repository: InsightsRepository,
        scoring_weights: ScoringWeights | None = None,
        active_variants: Optional[Iterable[str]] = None,
        max_ngram_len: int = 7,
        llm_enabled: bool = False,
        llm_use_remote: bool = False,
        llm_adapter: Callable[[List[str], List[str], Dict[str, object]], SynthesisResult] = synthesize_definition,
    ) -> None:
        self.candidate_finder = candidate_finder
        self.repository = repository
        self.scoring_weights = scoring_weights or ScoringWeights()
        self.active_variants = list(active_variants) if active_variants else None
        self.max_ngram_len = max_ngram_len
        self.llm_enabled = llm_enabled
        self.llm_use_remote = llm_use_remote
        self.llm_adapter = llm_adapter
        self._decomposition_engine = DecompositionEngine(candidate_finder)

    @classmethod
    def from_config(
        cls,
        *,
        variants: Optional[Iterable[str]] = None,
        max_ngram_len: int = 7,
        scoring_weights: ScoringWeights | None = None,
        llm_enabled: bool = False,
        llm_use_remote: bool = False,
    ) -> "SingleWordTranslationService":
        paths = get_config_paths()
        repository = InsightsRepository(
            solo_path=Path(paths["solo"]),
            debate_path=Path(paths["debate"]),
            fasttext_model_path=Path(paths["model_output"]),
        )
        if variants:
            repository.require_variants(variants)
        elif not repository.variants:
            raise FileNotFoundError(
                "No insights databases found. Run enochian-analysis first."
            )

        entries = load_dictionary(str(paths["dictionary"]))
        candidate_finder = MorphemeCandidateFinder(
            ngram_db_path=paths["ngram_index"],
            fasttext_model_path=paths["model_output"],
            dictionary_entries=entries,
            min_n=1,
            max_n=max_ngram_len,
        )

        return cls(
            candidate_finder=candidate_finder,
            repository=repository,
            scoring_weights=scoring_weights,
            active_variants=list(variants) if variants else None,
            max_ngram_len=max_ngram_len,
            llm_enabled=llm_enabled,
            llm_use_remote=llm_use_remote,
        )

    def close(self) -> None:
        self.candidate_finder.close()
        self.repository.close()

    def __enter__(self) -> "SingleWordTranslationService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def translate_word(
        self,
        word: str,
        *,
        variants: Optional[Iterable[str]] = None,
        strategy: str = "prefer-balance",
        top_k: int = 3,
        llm: Optional[bool] = None,
    ) -> Dict[str, object]:
        """Run the full single-word pipeline with optional LLM synthesis.

        Steps (Task 4.2):
        1. Fetch evidence for the target word.
        2. Generate and hard-filter decompositions.
        3. Soft-score, apply the chosen reranking strategy, and select top-k.
        4. Optionally call :func:`synthesize_definition` on the top candidate
           when ``llm`` (or ``llm_enabled``) is ``True``.
        """
        normalized = (word or "").strip().upper()
        if not normalized:
            raise ValueError("Word must be a non-empty string.")

        active_variants = list(variants) if variants else self.active_variants
        if active_variants:
            self.repository.require_variants(active_variants)

        evidence = self.repository.fetch_word_evidence(
            normalized, variants=active_variants
        )

        decompositions = self._decomposition_engine.generate_decompositions(
            normalized, evidence
        )
        filtered = apply_hard_filters(decompositions, evidence)

        ranked: List[tuple[Decomposition, float]] = []
        for decomp in filtered:
            score = score_decomposition(
                decomp, evidence, weights=self.scoring_weights
            )
            ranked.append((decomp, score))

        reranked = apply_strategy(ranked, strategy=strategy, evidence=evidence)
        selected = select_top_k(
            reranked,
            k=top_k,
            evidence=evidence,
        )

        llm_enabled = self.llm_enabled if llm is None else bool(llm)
        enriched = self._enrich_candidates(
            selected,
            evidence=evidence,
            strategy=strategy,
            llm_enabled=llm_enabled,
        )

        return {
            "word": normalized,
            "variants_queried": evidence.variants_queried,
            "strategy": strategy,
            "llm_enabled": llm_enabled,
            "llm_mode": (
                "remote"
                if llm_enabled and self.llm_use_remote
                else "local"
                if llm_enabled
                else None
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidates": enriched,
            "evidence": self._summarize_evidence(evidence),
        }

    def _enrich_candidates(
        self,
        candidates: List[dict[str, object]],
        *,
        evidence: WordEvidence,
        strategy: str,
        llm_enabled: bool,
    ) -> List[dict[str, object]]:
        enriched: List[dict[str, object]] = []
        for idx, candidate in enumerate(candidates):
            base = dict(candidate)
            meanings_raw = base.get("meanings")
            meanings: List[dict[str, object]] = (
                meanings_raw if isinstance(meanings_raw, list) else []
            )
            warnings_raw = base.get("warnings")
            warnings: List[str] = (
                [str(w) for w in warnings_raw]
                if isinstance(warnings_raw, (list, tuple))
                else []
            )

            coverage_ratio = _safe_ratio(base.get("breakdown"), "coverage_ratio")
            residual_ratio = _safe_ratio(base.get("breakdown"), "residual_ratio")

            base["concatenated_meanings"] = self._concatenate_meanings(meanings)

            if llm_enabled and idx == 0:
                morph_meanings = [
                    self._meaning_or_morph(entry) for entry in meanings if isinstance(entry, dict)
                ]
                morphs_raw = base.get("morphs")
                morphs: List[str] = (
                    [str(m) for m in morphs_raw]
                    if isinstance(morphs_raw, (list, tuple))
                    else []
                )
                context = {
                    "word": evidence.word,
                    "strategy": strategy,
                    "coverage_ratio": coverage_ratio,
                    "residual_ratio": residual_ratio,
                    "score": base.get("score"),
                    "provenance": meanings,
                    "variants": evidence.variants_queried,
                    "use_remote": self.llm_use_remote,
                }
                synthesis = self.llm_adapter(
                    morphs,
                    morph_meanings,
                    context,
                )
                base["synthesized_definition"] = synthesis.synthesized_definition
                base["concatenated_meanings"] = synthesis.concatenated_meanings
                base["confidence"] = synthesis.confidence
                base["reasoning"] = synthesis.reasoning
                if synthesis.raw_response is not None:
                    base["raw_llm_response"] = synthesis.raw_response
                if synthesis.warnings:
                    warnings.extend(synthesis.warnings)
            else:
                base["synthesized_definition"] = None
                base["confidence"] = self._coverage_confidence(
                    coverage_ratio=coverage_ratio, residual_ratio=residual_ratio
                )
                base["reasoning"] = (
                    "LLM disabled; using concatenated morph meanings."
                    if llm_enabled is False
                    else "LLM not applied to this rank."
                )

            if warnings:
                base["warnings"] = warnings

            enriched.append(base)

        return enriched

    @staticmethod
    def _concatenate_meanings(meanings: Sequence[dict[str, object]]) -> str:
        parts: List[str] = []
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            definition = meaning.get("definition")
            morph = meaning.get("morph")
            if isinstance(definition, str) and definition.strip():
                parts.append(definition.strip())
            elif isinstance(morph, str) and morph.strip():
                parts.append(morph.strip())
        return " + ".join(parts)

    @staticmethod
    def _meaning_or_morph(meaning: dict[str, object]) -> str:
        definition = meaning.get("definition")
        if isinstance(definition, str) and definition.strip():
            return definition.strip()
        morph = meaning.get("morph")
        return str(morph) if morph is not None else ""

    @staticmethod
    def _coverage_confidence(*, coverage_ratio: float, residual_ratio: float) -> float:
        return max(0.0, min(1.0, 0.35 + 0.5 * coverage_ratio - 0.2 * residual_ratio))

    @staticmethod
    def _summarize_evidence(evidence: WordEvidence) -> dict[str, object]:
        return {
            "variants_queried": list(evidence.variants_queried),
            "direct_clusters": len(evidence.direct_clusters),
            "residual_semantics": len(evidence.residual_semantics),
            "morph_hypotheses": len(evidence.morph_hypotheses),
            "fasttext_neighbors": [asdict(neighbor) for neighbor in evidence.fasttext_neighbors],
        }


def _cleanup_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key in ("normalized", "composite", "cos_sim", "confidence", "tfidf"):
        if key in candidate:
            value = candidate[key]
            cleaned[key] = float(value) if isinstance(value, numbers.Real) else value
    cleaned["breakdown"] = candidate.get("breakdown")
    cleaned["target"] = candidate.get("target")
    cleaned["target_length"] = candidate.get("target_length")
    return cleaned


def _candidate_lookup_keys(candidate: Dict[str, object]) -> Set[str]:
    keys: Set[str] = set()
    for raw in (candidate.get("normalized"), candidate.get("word")):
        lowered = _safe_lower(raw if isinstance(raw, str) else None)
        if lowered:
            keys.add(lowered)

    raw_breakdown = candidate.get("breakdown")
    breakdown = raw_breakdown if isinstance(raw_breakdown, dict) else {}
    raw_segments = breakdown.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    for segment in segments:
        lowered = _safe_lower(
            segment.get("canonical") if isinstance(segment, dict) else None
        )
        if lowered:
            keys.add(lowered)

    return keys


def _residual_detail_from_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    raw_breakdown = candidate.get("breakdown")
    breakdown_dict = raw_breakdown if isinstance(raw_breakdown, dict) else {}
    uncovered = _extract_uncovered_texts(breakdown_dict)
    low_conf = _extract_low_confidence_segments(breakdown_dict)
    return {
        "normalized": candidate.get("normalized"),
        "definition": None,
        "coverage_ratio": breakdown_dict.get("coverage_ratio"),
        "residual_ratio": breakdown_dict.get("residual_ratio"),
        "avg_confidence": None,
        "uncovered": uncovered,
        "low_confidence": low_conf,
        "source": "candidate_finder",
        "candidate": candidate,
        "candidate_breakdown": breakdown_dict,
        "candidate_coverage_ratio": breakdown_dict.get("coverage_ratio"),
        "candidate_residual_ratio": breakdown_dict.get("residual_ratio"),
        "candidate_uncovered": uncovered,
        "candidate_low_confidence": low_conf,
    }


def _extract_uncovered_texts(breakdown: Dict[str, object]) -> List[str]:
    raw_uncovered = breakdown.get("uncovered")
    uncovered = raw_uncovered if isinstance(raw_uncovered, list) else []
    texts: List[str] = []
    for entry in uncovered:
        text = entry.get("text") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _extract_low_confidence_segments(breakdown: Dict[str, object]) -> List[str]:
    raw_segments = breakdown.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    low_conf: List[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = segment.get("text")
        confidence = segment.get("semantic_confidence")
        if (
            isinstance(text, str)
            and text.strip()
            and isinstance(confidence, numbers.Real)
            and float(confidence) < 0.5
        ):
            low_conf.append(f"{text.strip()}@{float(confidence):.2f}")
    return low_conf


def _safe_lower(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered or None
    return None


def _safe_ratio(breakdown: object, key: str) -> float:
    defaults = {
        "coverage_ratio": 0.0,
        "residual_ratio": 1.0,
    }
    if not isinstance(breakdown, dict):
        return defaults.get(key, 0.0)

    value = breakdown.get(key, defaults.get(key, 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return defaults.get(key, 0.0)
