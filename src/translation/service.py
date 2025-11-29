from __future__ import annotations

import numbers
from pathlib import Path
from types import TracebackType
from typing import Dict, Iterable, List, Optional, Set

from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder

from .repository import ClusterRecord, InsightsRepository, ResidualDetail
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
            raise FileNotFoundError("No insights databases found. Run enochian-analysis first.")

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
            cluster_records = self.repository.fetch_clusters(ngram_text, variants=active_variants)
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
        serialized = {
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

        enriched_defs = []
        for raw_def in record.raw_definitions:
            normalized_options = {
                _safe_lower(raw_def.get("source_word")),
                _safe_lower(raw_def.get("variant")),
            }
            normalized_options.discard(None)
            matched_candidate = None
            matched_key = None
            for key in normalized_options:
                if key and key in candidate_lookup:
                    matched_candidate = candidate_lookup[key]
                    matched_key = key
                    break
            enriched = dict(raw_def)
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

            reconciled.append(
                self._merge_residual_detail(detail, candidate)
            )

        for candidate in candidates:
            if id(candidate) in matched_candidates:
                continue
            reconciled.append(_residual_detail_from_candidate(candidate))

        return reconciled

    def _merge_residual_detail(
        self, detail: ResidualDetail, candidate: Optional[Dict[str, object]]
    ) -> Dict[str, object]:
        payload = {
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
            breakdown = candidate.get("breakdown") or {}
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


def _cleanup_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key in ("normalized", "composite", "cos_sim", "confidence", "tfidf"):
        if key in candidate:
            value = candidate[key]
            cleaned[key] = (
                float(value) if isinstance(value, numbers.Real) else value
            )
    cleaned["breakdown"] = candidate.get("breakdown")
    cleaned["target"] = candidate.get("target")
    cleaned["target_length"] = candidate.get("target_length")
    return cleaned


def _candidate_lookup_keys(candidate: Dict[str, object]) -> Set[str]:
    keys: Set[str] = set()
    for raw in (candidate.get("normalized"), candidate.get("word")):
        lowered = _safe_lower(raw)
        if lowered:
            keys.add(lowered)
    breakdown = candidate.get("breakdown") or {}
    segments = breakdown.get("segments") or []
    for segment in segments:
        lowered = _safe_lower(segment.get("canonical"))
        if lowered:
            keys.add(lowered)
    return keys


def _residual_detail_from_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    breakdown = candidate.get("breakdown") or {}
    uncovered = _extract_uncovered_texts(breakdown)
    low_conf = _extract_low_confidence_segments(breakdown)
    return {
        "normalized": candidate.get("normalized"),
        "definition": None,
        "coverage_ratio": breakdown.get("coverage_ratio"),
        "residual_ratio": breakdown.get("residual_ratio"),
        "avg_confidence": None,
        "uncovered": uncovered,
        "low_confidence": low_conf,
        "source": "candidate_finder",
        "candidate": candidate,
        "candidate_breakdown": breakdown,
        "candidate_coverage_ratio": breakdown.get("coverage_ratio"),
        "candidate_residual_ratio": breakdown.get("residual_ratio"),
        "candidate_uncovered": uncovered,
        "candidate_low_confidence": low_conf,
    }


def _extract_uncovered_texts(breakdown: Dict[str, object]) -> List[str]:
    uncovered = breakdown.get("uncovered") or []
    texts = []
    for entry in uncovered:
        text = entry.get("text") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _extract_low_confidence_segments(breakdown: Dict[str, object]) -> List[str]:
    segments = breakdown.get("segments") or []
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
