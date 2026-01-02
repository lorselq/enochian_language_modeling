from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import numbers
from pathlib import Path
from types import TracebackType
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, TypeVar

import numpy as np

from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord

from .decomposition import Decomposition, DecompositionEngine, apply_hard_filters
from .llm_synthesis import SynthesisResult, synthesize_definition
from .repository import (
    ClusterRecord,
    MorphHypothesisRecord,
    InsightsRepository,
    ResidualSemanticRecord,
    ResidualDetail,
    WordEvidence,
    FasttextNeighbor,
)
from .scoring import ScoringWeights, score_decomposition, score_decomposition_unweighted
from .strategies import apply_strategy, select_top_k
from .tokenization import expand_sentence_ngrams, tokenize_words

T = TypeVar("T")


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
            min_n=2,
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
            if (
                self.candidate_finder.min_n > 1
                and not _has_segment_coverage(candidate_breakdowns)
            ):
                candidate_breakdowns = self._with_min_n(
                    1, self.candidate_finder.find_candidates, ngram_text
                )

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

    def _with_min_n(
        self,
        min_n: int,
        func: Callable[..., T],
        *args: object,
        **kwargs: object,
    ) -> T:
        previous = self.candidate_finder.min_n
        self.candidate_finder.min_n = min_n
        try:
            return func(*args, **kwargs)
        finally:
            self.candidate_finder.min_n = previous


class SingleWordTranslationService:
    """Single-word translation pipeline with optional LLM synthesis (Tasks 4.1/4.2)."""

    class EvidenceMode(str, Enum):
        ALL = "all"
        CLUSTERS_ONLY = "clusters-only"
        RESIDUALS_ONLY = "residuals-only"

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
            min_n=2,
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
        fallback_top_n: int = 5,
        evidence_mode: EvidenceMode = EvidenceMode.ALL,
        weight_enabled: bool = True,
        allow_whole_word: bool = True,
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
            normalized,
            variants=active_variants,
            dictionary_entries=self.candidate_finder.dictionary,
            min_n=self.candidate_finder.min_n,
            max_n=self.candidate_finder.max_n,
        )
        substring_support: List[str] = []
        substrings = self._substring_candidates(normalized)
        if substrings:
            (
                support_clusters,
                support_residuals,
                support_hypotheses,
            ) = self.repository.fetch_morph_support(
                substrings, variants=active_variants
            )
            if support_clusters or support_residuals or support_hypotheses:
                substring_support = sorted(
                    {
                        item.ngram.upper() for item in support_clusters
                    }
                    | {item.residual.upper() for item in support_residuals}
                    | {item.morph.upper() for item in support_hypotheses}
                )
                self._merge_support_evidence(
                    evidence,
                    support_clusters,
                    support_residuals,
                    support_hypotheses,
                )
        self._apply_evidence_mode(evidence, mode=evidence_mode)

        n_best = max(top_k * 5, self.candidate_finder.beam_width)
        decompositions, diagnostics = self._decomposition_engine.generate_decompositions(
            normalized,
            evidence,
            allow_whole_word=allow_whole_word,
            n_best=n_best,
        )
        if not decompositions and self.candidate_finder.min_n > 1:
            fallback_decomps, fallback_diag = self._with_min_n(
                1,
                self._decomposition_engine.generate_decompositions,
                normalized,
                evidence,
                allow_whole_word=allow_whole_word,
                n_best=n_best,
            )
            diagnostics["min_n_fallback"] = {
                "used": bool(fallback_decomps),
                "min_n": 1,
            }
            if fallback_decomps:
                decompositions = fallback_decomps
        if decompositions:
            morphs = {morph for decomp in decompositions for morph in decomp.morphs}
            if morphs:
                (
                    support_clusters,
                    support_residuals,
                    support_hypotheses,
                ) = self.repository.fetch_morph_support(
                    morphs, variants=active_variants
                )
                self._merge_support_evidence(
                    evidence,
                    support_clusters,
                    support_residuals,
                    support_hypotheses,
                )
                self._apply_evidence_mode(evidence, mode=evidence_mode)
        filtered, filter_diagnostics = apply_hard_filters(decompositions, evidence)
        fallback_decompositions: List[Decomposition] = []
        fallback_used = False
        fallback_mode: str | None = None
        fallback_min_coverage: float | None = None

        if not filtered:
            fallback_decompositions, fallback_diag = (
                self._decomposition_engine.generate_decompositions(
                    normalized,
                    evidence,
                    force_dictionary=True,
                    allow_whole_word=allow_whole_word,
                    n_best=n_best,
                )
            )
            diagnostics["dictionary_fallback"] = fallback_diag
            if fallback_decompositions:
                fallback_used = True
                fallback_mode = "dictionary"
                decomp_morphs = {
                    morph for decomp in fallback_decompositions for morph in decomp.morphs
                }
                if decomp_morphs:
                    (
                        support_clusters,
                        support_residuals,
                        support_hypotheses,
                    ) = self.repository.fetch_morph_support(
                        decomp_morphs, variants=active_variants
                    )
                    self._merge_support_evidence(
                        evidence,
                        support_clusters,
                        support_residuals,
                        support_hypotheses,
                    )
                    self._apply_evidence_mode(evidence, mode=evidence_mode)
                filtered, filter_diagnostics = apply_hard_filters(
                    fallback_decompositions, evidence
                )

        if not filtered:
            relaxed_source = (
                fallback_decompositions if fallback_decompositions else decompositions
            )
            relaxed, relaxed_min_coverage = _relaxed_fallback(
                relaxed_source,
                top_n=fallback_top_n,
            )
            if relaxed:
                fallback_used = True
                if fallback_mode is None:
                    fallback_mode = "relaxed"
                else:
                    fallback_mode = f"{fallback_mode}+relaxed"
                fallback_min_coverage = relaxed_min_coverage
                filtered = relaxed
                filter_diagnostics["relaxed_fallback"] = {
                    "top_n": fallback_top_n,
                    "min_coverage_ratio": relaxed_min_coverage,
                }

        ranked: List[tuple[Decomposition, float]] = []
        for decomp in filtered:
            if weight_enabled:
                score = score_decomposition(
                    decomp, evidence, weights=self.scoring_weights
                )
            else:
                score = score_decomposition_unweighted(decomp, evidence)
            ranked.append((decomp, score))

        reranked = apply_strategy(ranked, strategy=strategy, evidence=evidence)
        selected = select_top_k(
            reranked,
            k=top_k,
            evidence=evidence,
        )

        fallback_morphs: List[dict[str, object]] = []
        if not selected:
            fallback_morphs = self._fallback_morph_hints(normalized)

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
            "evidence_mode": evidence_mode.value,
            "weighting_enabled": weight_enabled,
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
            "fallback_morphs": fallback_morphs,
            "diagnostics": {
                **diagnostics,
                "substring_support": substring_support,
                "fasttext": self.repository.fasttext_diagnostics(),
                "repository": self.repository.path_diagnostics(),
                "word_lookup": self.repository.word_lookup_diagnostics(
                    normalized, variants=active_variants
                ),
                "scoring_weights": (
                    asdict(self.scoring_weights.normalized()) if weight_enabled else None
                ),
                "weighting_enabled": weight_enabled,
                "hard_filters": filter_diagnostics,
                "decomposition": {
                    "generated": len(decompositions),
                    "filtered": len(filtered),
                    "selected": len(selected),
                    "fallback_generated": len(fallback_decompositions),
                    "fallback_used": fallback_used,
                    "fallback_mode": fallback_mode,
                    "fallback_min_coverage_ratio": fallback_min_coverage,
                },
            },
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

            if llm_enabled:
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
                )

            if warnings:
                base["warnings"] = warnings

            enriched.append(base)

        return enriched

    def _with_min_n(
        self,
        min_n: int,
        func: Callable[..., T],
        *args: object,
        **kwargs: object,
    ) -> T:
        previous = self.candidate_finder.min_n
        self.candidate_finder.min_n = min_n
        try:
            return func(*args, **kwargs)
        finally:
            self.candidate_finder.min_n = previous

    def _fallback_morph_hints(self, word: str) -> List[dict[str, object]]:
        if not word:
            return []
        word_upper = word.upper()
        min_n = self.candidate_finder.min_n
        max_n = self.candidate_finder.max_n
        dictionary = self.candidate_finder.dictionary
        if not dictionary:
            return []

        hints: List[dict[str, object]] = []
        seen: set[str] = set()
        for start in range(len(word_upper)):
            for end in range(start + min_n, min(len(word_upper), start + max_n) + 1):
                slice_text = word_upper[start:end]
                key = slice_text.lower()
                if key in seen:
                    continue
                entry = dictionary.get(key)
                if not entry:
                    continue
                seen.add(key)
                definition = (
                    entry.get("enhanced_definition")
                    or _first_sense_definition(entry)
                    or None
                )
                coverage_ratio = len(slice_text) / len(word_upper)
                similarity = self._fasttext_similarity(word_upper, slice_text)
                hints.append(
                    {
                        "morph": slice_text,
                        "definition": definition,
                        "coverage_ratio": round(coverage_ratio, 3),
                        "fasttext_similarity": similarity,
                        "source": "dictionary_substring",
                    }
                )

        hints.sort(
            key=lambda item: (
                item.get("coverage_ratio") or 0.0,
                item.get("fasttext_similarity") or 0.0,
            ),
            reverse=True,
        )
        return hints

    def _fasttext_similarity(self, word: str, morph: str) -> Optional[float]:
        model = self.candidate_finder.fasttext_model
        if not model:
            return None
        try:
            word_vec = model.get_word_vector(word)
            morph_vec = model.get_word_vector(morph)
        except Exception:
            return None
        word_vec = np.asarray(word_vec, dtype=float)
        morph_vec = np.asarray(morph_vec, dtype=float)
        denom = np.linalg.norm(word_vec) * np.linalg.norm(morph_vec)
        if denom == 0:
            return None
        return float(np.dot(word_vec, morph_vec) / denom)

    def _substring_candidates(
        self, word: str, *, include_singletons: bool = True
    ) -> List[str]:
        """Generate substring candidates for evidence lookup.

        Returns substrings sorted by length descending (longest first), then
        alphabetically. This prioritizes longer, chunkier morphs over singletons.

        When ``include_singletons`` is True (default), single-letter substrings
        are included so we can discover which singletons have evidence support.
        This enables the singleton fallback path without exploding the search
        space with unsupported single letters.
        """
        if not word:
            return []
        # Include singletons for evidence discovery, but segmentation uses min_n
        min_len = 1 if include_singletons else self.candidate_finder.min_n
        max_n = self.candidate_finder.max_n
        word_upper = word.upper()
        candidates: set[str] = set()
        for start in range(len(word_upper)):
            for end in range(start + min_len, min(len(word_upper), start + max_n) + 1):
                candidates.add(word_upper[start:end])
        # Sort by length descending (prefer longer morphs), then alphabetically
        return sorted(candidates, key=lambda s: (-len(s), s))

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
    def _merge_support_evidence(
        evidence: WordEvidence,
        clusters: List[ClusterRecord],
        residuals: List[ResidualSemanticRecord],
        hypotheses: List[MorphHypothesisRecord],
    ) -> None:
        """Merge morph-level support evidence into the primary WordEvidence."""
        cluster_keys = {
            (item.variant, item.cluster_id) for item in evidence.direct_clusters
        }
        for item in clusters:
            key = (item.variant, item.cluster_id)
            if key in cluster_keys:
                continue
            evidence.direct_clusters.append(item)
            cluster_keys.add(key)

        residual_keys = {
            (item.variant, item.residual, item.parent_word, item.group_index)
            for item in evidence.residual_semantics
        }
        for item in residuals:
            key = (item.variant, item.residual, item.parent_word, item.group_index)
            if key in residual_keys:
                continue
            evidence.residual_semantics.append(item)
            residual_keys.add(key)

        hypothesis_keys = {
            (item.variant, item.hyp_id) for item in evidence.morph_hypotheses
        }
        for item in hypotheses:
            key = (item.variant, item.hyp_id)
            if key in hypothesis_keys:
                continue
            evidence.morph_hypotheses.append(item)
            hypothesis_keys.add(key)

        if clusters or residuals or hypotheses:
            evidence.fasttext_neighbors = []

    @staticmethod
    def _apply_evidence_mode(
        evidence: WordEvidence,
        *,
        mode: "SingleWordTranslationService.EvidenceMode",
    ) -> None:
        if mode == SingleWordTranslationService.EvidenceMode.ALL:
            return
        if mode == SingleWordTranslationService.EvidenceMode.CLUSTERS_ONLY:
            evidence.residual_semantics = []
            evidence.morph_hypotheses = []
            evidence.attested_definitions = []
            evidence.dictionary_morphs = {}
            evidence.fasttext_neighbors = []
            return
        if mode == SingleWordTranslationService.EvidenceMode.RESIDUALS_ONLY:
            evidence.direct_clusters = []
            evidence.morph_hypotheses = []
            evidence.attested_definitions = []
            evidence.dictionary_morphs = {}
            evidence.fasttext_neighbors = []
            return

    @staticmethod
    def _summarize_evidence(evidence: WordEvidence) -> dict[str, object]:
        return {
            "variants_queried": list(evidence.variants_queried),
            "direct_clusters": len(evidence.direct_clusters),
            "residual_semantics": len(evidence.residual_semantics),
            "morph_hypotheses": len(evidence.morph_hypotheses),
            "attested_definitions": len(evidence.attested_definitions),
            "dictionary_morphs": len(evidence.dictionary_morphs),
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


def _first_sense_definition(entry: EntryRecord) -> Optional[str]:
    senses = entry.get("senses")
    if not isinstance(senses, list) or not senses:
        return None
    for sense in senses:
        if not isinstance(sense, dict):
            continue
        definition = sense.get("definition")
        if isinstance(definition, str) and definition.strip():
            return definition.strip()
    return None


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


def _has_segment_coverage(candidates: Sequence[Dict[str, object]]) -> bool:
    for candidate in candidates:
        breakdown = candidate.get("breakdown")
        if not isinstance(breakdown, dict):
            continue
        segments = breakdown.get("segments")
        if isinstance(segments, list) and segments:
            return True
    return False


def _relaxed_fallback(
    decompositions: Sequence[Decomposition],
    *,
    top_n: int,
) -> tuple[List[Decomposition], float | None]:
    if not decompositions:
        return [], None

    try:
        limit = int(top_n)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        limit = len(decompositions)

    scored: List[tuple[Decomposition, float]] = [
        (decomp, _safe_ratio(decomp.breakdown, "coverage_ratio"))
        for decomp in decompositions
    ]
    max_coverage = max(score for _, score in scored)
    coverage_filtered = [decomp for decomp, score in scored if score >= max_coverage]
    coverage_filtered.sort(key=lambda decomp: float(decomp.beam_score), reverse=True)
    return coverage_filtered[:limit], max_coverage
