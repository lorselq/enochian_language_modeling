from __future__ import annotations

import numbers
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.dictionary_loader import load_dictionary
from enochian_translation_team.utils.candidate_finder import MorphemeCandidateFinder

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

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
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
            candidate_index: Dict[str, Dict[str, object]] = {}
            for candidate in candidate_breakdowns:
                normalized = candidate.get("normalized")
                if not normalized:
                    continue
                key = str(normalized)
                candidate_index[key] = _cleanup_candidate(candidate)
            analyses.append(
                {
                    "word_index": slice_info.word_index,
                    "span": [slice_info.start, slice_info.end],
                    "ngram": ngram_text.upper(),
                    "candidates": list(candidate_index.values()),
                    "clusters": [
                        self._serialize_cluster(record, candidate_index)
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
        self, record: ClusterRecord, candidate_index: Dict[str, Dict[str, object]]
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
            "residual_details": [self._serialize_residual_detail(detail) for detail in record.residual_details],
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
                if key and key in candidate_index:
                    matched_candidate = candidate_index[key]
                    matched_key = key
                    break
            enriched = dict(raw_def)
            enriched["normalized"] = matched_key
            enriched["candidate"] = matched_candidate
            enriched_defs.append(enriched)
        serialized["definitions"] = enriched_defs
        return serialized

    def _serialize_residual_detail(self, detail: ResidualDetail) -> Dict[str, object]:
        return {
            "normalized": detail.normalized,
            "definition": detail.definition,
            "coverage_ratio": detail.coverage_ratio,
            "residual_ratio": detail.residual_ratio,
            "avg_confidence": detail.avg_confidence,
            "uncovered": list(detail.uncovered),
            "low_confidence": list(detail.low_confidence),
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


def _safe_lower(value: Optional[str]) -> Optional[str]:
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered or None
    return None
