from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import numbers
from pathlib import Path
from types import TracebackType
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

import numpy as np

from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.embeddings import (
    cluster_definition_counts,
    get_sentence_transformer,
)
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord

from .decomposition import (
    DEFAULT_MAX_FULL_SEGMENTATIONS,
    DEFAULT_MAX_PARTIAL_PER_INDEX,
    Decomposition,
    DecompositionEngine,
    apply_hard_filters,
    _collect_attested_pieces,
    _build_evidence_ngrams,
    build_decompositions_from_segmentations,
    enumerate_attested_segmentations_with_diagnostics,
    _normalize_beam_scores,
)
from .llm_synthesis import (
    ConsensusSynthesisResult,
    SynthesisResult,
    synthesize_consensus,
    synthesize_definition,
)
from .repository import (
    ClusterRecord,
    DictionaryMorph,
    MorphHypothesisRecord,
    InsightsRepository,
    ResidualSemanticRecord,
    ResidualDetail,
    WordEvidence,
    FasttextNeighbor,
)
from .scoring import (
    CoherenceResult,
    ScoringWeights,
    compute_semantic_coherence,
    score_decomposition_breakdown,
    score_decomposition_with_coherence_breakdown,
)
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
        active_variants: Iterable[str] | None = None,
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
        variants: Iterable[str] | None = None,
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
        max_ngram_len: int | None = None,
        variants: Iterable[str] | None = None,
    ) -> dict[str, object]:
        max_len = max_ngram_len or self.max_ngram_len
        active_variants = list(variants) if variants else self.active_variants
        if active_variants:
            self.repository.require_variants(active_variants)

        tokens = tokenize_words(text)
        ngram_slices = expand_sentence_ngrams(text, max_len=max_len)
        analyses: list[dict[str, object]] = []
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

            candidate_lookup: dict[str, dict[str, object]] = {}
            unique_candidates: list[dict[str, object]] = []
            seen_signatures: set[tuple] = set()

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
        candidate_lookup: dict[str, dict[str, object]],
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        serialized: dict[str, object] = {
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

        enriched_defs: list[dict[str, object]] = []
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
        details: list[ResidualDetail],
        candidate_lookup: dict[str, dict[str, object]],
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        reconciled: list[dict[str, object]] = []
        matched_candidates: set[int] = set()

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
        self, detail: ResidualDetail, candidate: dict[str, object] | None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
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
        active_variants: Iterable[str] | None = None,
        max_ngram_len: int = 7,
        llm_enabled: bool = False,
        llm_use_remote: bool = False,
        llm_adapter: Callable[[list[str], list[str], dict[str, object]], SynthesisResult] = synthesize_definition,
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
        variants: Iterable[str] | None = None,
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
        variants: Iterable[str] | None = None,
        strategy: str = "prefer-balance",
        top_k: int = 3,
        llm: bool | None = None,
        fallback_top_n: int = 5,
        evidence_mode: EvidenceMode = EvidenceMode.CLUSTERS_ONLY,
        weight_enabled: bool = True,
        allow_whole_word: bool = True,
        use_beam_search: bool = False,
    ) -> dict[str, object]:
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

        dictionary_snapshot = self.candidate_finder.dictionary
        self.candidate_finder.dictionary = {}
        try:
            evidence = self.repository.fetch_word_evidence(
                normalized,
                variants=active_variants,
                dictionary_entries=None,
                min_n=self.candidate_finder.min_n,
                max_n=self.candidate_finder.max_n,
            )
            if not evidence or not getattr(evidence, "word", None):
                llm_enabled = self.llm_enabled if llm is None else bool(llm)
                return {
                    "word": normalized,
                    "variants_queried": active_variants or [],
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
                    "candidates": [],
                    "evidence": {},
                    "fallback_morphs": [],
                    "diagnostics": {
                        "evidence_missing": True,
                        "substring_support": [],
                        "fasttext": self.repository.fasttext_diagnostics(),
                        "repository": self.repository.path_diagnostics(),
                        "word_lookup": self.repository.word_lookup_diagnostics(
                            normalized, variants=active_variants
                        ),
                        "scoring_weights": (
                            asdict(self.scoring_weights.normalized())
                            if weight_enabled
                            else None
                        ),
                        "weighting_enabled": weight_enabled,
                        "hard_filters": {},
                        "decomposition": {
                            "generated": 0,
                            "filtered": 0,
                            "selected": 0,
                            "fallback_generated": 0,
                            "fallback_used": False,
                            "fallback_mode": None,
                            "fallback_min_coverage_ratio": None,
                        },
                    },
                }
            substring_support: list[str] = []
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

            # Apply evidence mode EARLY so that decomposition generation, morph_support
            # labeling, and hard filtering all respect the mode.
            # In clusters-only mode, only cluster evidence is used.
            self._apply_evidence_mode(evidence, mode=evidence_mode)

            # Fetch accepted definition counts for all possible substrings
            # This is used to penalize ambiguous morphs (many definitions) during beam search
            all_substrings = self._substring_candidates(
                normalized, include_singletons=True
            )
            definition_counts: dict[str, int] = {}
            if evidence_mode != self.EvidenceMode.RESIDUALS_ONLY:
                definition_counts = self.repository.fetch_accepted_definition_counts(
                    all_substrings, variants=active_variants
                )
            definition_glosses = self.repository.fetch_accepted_definition_glosses(
                all_substrings,
                variants=active_variants,
                include_clusters=evidence_mode != self.EvidenceMode.RESIDUALS_ONLY,
                include_residuals=evidence_mode != self.EvidenceMode.CLUSTERS_ONLY,
            )
            clustered_counts = cluster_definition_counts(
                definition_glosses,
                get_sentence_transformer("paraphrase-MiniLM-L6-v2"),
            )
            if clustered_counts:
                definition_counts = {**definition_counts, **clustered_counts}

            evidence.definition_counts = definition_counts
            evidence.definition_glosses = definition_glosses

            attested_pieces = _collect_attested_pieces(
                evidence, evidence_mode=evidence_mode.value
            )
            attested_pieces_sample = sorted(
                attested_pieces, key=lambda piece: (-len(piece), piece)
            )[:25]

            diagnostics: dict[str, object] = {
                "fallback_used": False,
                "fallback_morphs": [],
                "parse_count": 0,
                "decomposition_count": 0,
                "extra_ngram_keys": 0,
                "extra_ngram_entries": 0,
                "dictionary_ngram_keys": 0,
                "dictionary_ngram_entries": 0,
                "dictionary_forced": False,
            }
            fallback_used = False
            fallback_mode: str | None = None
            enumerator_enabled = not use_beam_search
            enumerator_params: dict[str, int] | None = None
            enumerated_full_count = 0
            enumerated_full_returned = 0
            enumerated_partial_pruned_count = 0
            fallback_used_enum = "none"
            fallback_reason: str | None = None
            attested_only_guarantee = False
            unsupported_morphs_in_generated: list[str] = []
            beam_scoring_applied = False
            beam_scoring_parse_count = 0

            if use_beam_search:
                beam_width = getattr(self.candidate_finder, "beam_width", 10)
                if beam_width is None:
                    beam_width = 10
                n_best = max(top_k * 5, beam_width)
                decompositions, diagnostics = (
                    self._decomposition_engine.generate_decompositions(
                        normalized,
                        evidence,
                        allow_whole_word=allow_whole_word,
                        n_best=n_best,
                        definition_counts=definition_counts,
                        definition_glosses=definition_glosses,
                        evidence_mode=evidence_mode.value,
                    )
                )
            else:
                max_partial = DEFAULT_MAX_PARTIAL_PER_INDEX
                max_full = DEFAULT_MAX_FULL_SEGMENTATIONS
                segmentations: list[list[str]] = []
                min_piece_len = 1
                attested_only_guarantee = True
                segmentations, enumerator_diag = (
                    enumerate_attested_segmentations_with_diagnostics(
                        normalized,
                        attested_pieces,
                        max_partial_per_index=max_partial,
                        max_full_segmentations=max_full,
                        min_piece_len=min_piece_len,
                    )
                )
                enumerated_full_count = enumerator_diag["enumerated_full_count"]
                enumerated_partial_pruned_count = enumerator_diag[
                    "enumerated_partial_pruned_count"
                ]
                enumerator_params = {
                    "max_partial_per_index": max_partial,
                    "max_full_segmentations": max_full,
                    "min_piece_len": min_piece_len,
                }
                if not allow_whole_word and len(normalized) > 1:
                    segmentations = [
                        segmentation
                        for segmentation in segmentations
                        if not (
                            len(segmentation) == 1
                            and segmentation[0] == normalized
                        )
                    ]
                enumerated_full_returned = len(segmentations)
                if not segmentations:
                    fallback_used = True
                    fallback_used_enum = "no_full_cover"
                    fallback_reason = (
                        f"no_full_cover_with_min_piece_len={min_piece_len}"
                    )
                    fallback_mode = "no_full_cover"

                unsupported_morphs_in_generated = sorted(
                    {
                        morph
                        for segmentation in segmentations
                        for morph in segmentation
                        if morph not in attested_pieces
                    }
                )
                if attested_only_guarantee and unsupported_morphs_in_generated:
                    raise RuntimeError(
                        "Attested-only enumeration produced unsupported morphs: "
                        f"{unsupported_morphs_in_generated}"
                    )

                decompositions = build_decompositions_from_segmentations(
                    normalized,
                    segmentations,
                    candidate_finder=self.candidate_finder,
                    evidence=evidence,
                    evidence_mode=evidence_mode.value,
                )
                if decompositions:
                    extra_ngrams = _build_evidence_ngrams(
                        normalized,
                        evidence,
                        candidate_finder=self.candidate_finder,
                        definition_counts=definition_counts,
                        definition_glosses=definition_glosses,
                        evidence_mode=evidence_mode.value,
                    )
                    parses = self.candidate_finder.segment_target(
                        normalized,
                        extra_ngrams=extra_ngrams,
                        definition_counts=definition_counts,
                        definition_glosses=definition_glosses,
                        restrict_to_attested=True,
                    )
                    beam_scoring_applied = True
                    beam_scoring_parse_count = len(parses)
                    beam_scores: dict[tuple[str, ...], float] = {}
                    for _path, score, _ngram_scores, coverage in parses:
                        if not coverage:
                            continue
                        morphs: list[str] = []
                        for segment in coverage:
                            ngram = segment.get("ngram")
                            if isinstance(ngram, str) and ngram:
                                morphs.append(ngram.upper())
                        if not morphs:
                            continue
                        key = tuple(morphs)
                        existing = beam_scores.get(key)
                        if existing is None or score > existing:
                            beam_scores[key] = float(score)
                    for decomp in decompositions:
                        decomp.beam_score = beam_scores.get(tuple(decomp.morphs), 0.0)
                    _normalize_beam_scores(decompositions)
                diagnostics["decomposition_count"] = len(decompositions)
                diagnostics["fallback_used"] = fallback_used
                diagnostics["enumerator"] = {
                    "attested_piece_count": len(attested_pieces),
                    "attested_pieces_sample": attested_pieces_sample,
                    "enumerator_enabled": enumerator_enabled,
                    "enumerator_params": enumerator_params,
                    "enumerated_full_count": enumerated_full_count,
                    "enumerated_full_returned": enumerated_full_returned,
                    "enumerated_partial_pruned_count": enumerated_partial_pruned_count,
                    "fallback_used": fallback_used_enum,
                    "fallback_reason": fallback_reason,
                    "attested_only_guarantee": attested_only_guarantee,
                    "unsupported_morphs_in_generated": unsupported_morphs_in_generated,
                    "beam_scoring_applied": beam_scoring_applied,
                    "beam_scoring_parse_count": beam_scoring_parse_count,
                }
            if "enumerator" not in diagnostics:
                diagnostics["enumerator"] = {
                    "attested_piece_count": len(attested_pieces),
                    "attested_pieces_sample": attested_pieces_sample,
                    "enumerator_enabled": enumerator_enabled,
                    "enumerator_params": enumerator_params,
                    "enumerated_full_count": enumerated_full_count,
                    "enumerated_full_returned": enumerated_full_returned,
                    "enumerated_partial_pruned_count": enumerated_partial_pruned_count,
                    "fallback_used": fallback_used_enum,
                    "fallback_reason": fallback_reason,
                    "attested_only_guarantee": attested_only_guarantee,
                    "unsupported_morphs_in_generated": unsupported_morphs_in_generated,
                    "beam_scoring_applied": beam_scoring_applied,
                    "beam_scoring_parse_count": beam_scoring_parse_count,
                }
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
                    # Filter support data based on evidence mode before merging
                    if evidence_mode == self.EvidenceMode.CLUSTERS_ONLY:
                        support_residuals = []
                        support_hypotheses = []
                    elif evidence_mode == self.EvidenceMode.RESIDUALS_ONLY:
                        support_clusters = []
                        support_hypotheses = []
                    self._merge_support_evidence(
                        evidence,
                        support_clusters,
                        support_residuals,
                        support_hypotheses,
                    )
            # Evidence mode already applied at the start - hard filter respects it
            filtered, filter_diagnostics = apply_hard_filters(
                decompositions,
                evidence,
                evidence_mode=evidence_mode.value,
            )
            stage1_drops_total = int(
                filter_diagnostics.get(
                    "stage1_drops_total",
                    filter_diagnostics.get("stage1_dropped", 0),
                )
            )
            stage1_drops_missing_support = int(
                filter_diagnostics.get(
                    "stage1_drops_missing_support", stage1_drops_total
                )
            )
            stage1_drops_other_reasons = int(
                filter_diagnostics.get(
                    "stage1_drops_other_reasons",
                    max(0, stage1_drops_total - stage1_drops_missing_support),
                )
            )
            diagnostics["hard_filter_stage1_drops_total"] = stage1_drops_total
            diagnostics["hard_filter_stage1_drops_missing_support"] = (
                stage1_drops_missing_support
            )
            diagnostics["hard_filter_stage1_drops_other_reasons"] = (
                stage1_drops_other_reasons
            )
            if attested_only_guarantee and stage1_drops_missing_support:
                raise RuntimeError(
                    "Attested-only enumeration dropped candidates for missing morph "
                    f"support: {stage1_drops_missing_support}"
                )
            fallback_decompositions: list[Decomposition] = []
            fallback_min_coverage: float | None = None

            ranked: list[tuple[Decomposition, float]] = []
            coherence_results: dict[tuple[str, ...], CoherenceResult] = {}
            fasttext_model = getattr(self.candidate_finder, "fasttext_model", None)

            for decomp in filtered:
                # Check if decomposition has singletons and compute coherence
                has_singletons = any(len(m) == 1 for m in decomp.morphs)

                if has_singletons and fasttext_model is not None:
                    # Use coherence-aware scoring for singleton decompositions
                    if weight_enabled:
                        score, coherence, breakdown = (
                            score_decomposition_with_coherence_breakdown(
                                decomp,
                                evidence,
                                fasttext_model,
                                weights=self.scoring_weights,
                                coherence_weight=0.05,
                                weighted=True,
                            )
                        )
                        decomp.score_breakdown = breakdown
                    else:
                        base_score, breakdown = score_decomposition_breakdown(
                            decomp,
                            evidence,
                            weighted=False,
                        )
                        coherence = compute_semantic_coherence(
                            decomp.morphs, fasttext_model
                        )
                        score = 0.95 * base_score + 0.05 * coherence.score
                        breakdown["base_score"] = base_score
                        breakdown["coherence"] = {
                            "raw": coherence.score,
                            "weight": 0.05,
                            "weighted": 0.05 * coherence.score,
                            "singleton_cohesion": coherence.singleton_cohesion,
                            "large_morph_diversity": coherence.large_morph_diversity,
                            "singleton_count": coherence.singleton_count,
                            "large_morph_count": coherence.large_morph_count,
                        }
                        breakdown["total"] = score
                        decomp.score_breakdown = breakdown
                    coherence_results[tuple(decomp.morphs)] = coherence
                else:
                    # Standard scoring for non-singleton decompositions
                    if weight_enabled:
                        score, breakdown = score_decomposition_breakdown(
                            decomp,
                            evidence,
                            weights=self.scoring_weights,
                            weighted=True,
                        )
                        decomp.score_breakdown = breakdown
                    else:
                        score, breakdown = score_decomposition_breakdown(
                            decomp,
                            evidence,
                            weighted=False,
                        )
                        decomp.score_breakdown = breakdown

                ranked.append((decomp, score))

            if coherence_results:
                diagnostics["coherence_scores"] = {
                    " + ".join(morphs): {
                        "score": result.score,
                        "singleton_cohesion": result.singleton_cohesion,
                        "large_morph_diversity": result.large_morph_diversity,
                        "singleton_count": result.singleton_count,
                        "large_morph_count": result.large_morph_count,
                    }
                    for morphs, result in coherence_results.items()
                }

            reranked = apply_strategy(ranked, strategy=strategy, evidence=evidence)
            selected = select_top_k(
                reranked,
                k=top_k,
                evidence=evidence,
            )

            fallback_morphs: list[dict[str, object]] = []

            llm_enabled = self.llm_enabled if llm is None else bool(llm)
            enriched = self._enrich_candidates(
                selected,
                evidence=evidence,
                strategy=strategy,
                llm_enabled=llm_enabled,
            )
            consensus_payload: dict[str, object] | None = None
            if llm_enabled and len(enriched) > 1:
                consensus_payload = self._build_consensus_synthesis(
                    enriched,
                    strategy=strategy,
                    llm_use_remote=self.llm_use_remote,
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
                "consensus_synthesis": consensus_payload,
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
                        asdict(self.scoring_weights.normalized())
                        if weight_enabled
                        else None
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
        finally:
            self.candidate_finder.dictionary = dictionary_snapshot

    def _enrich_candidates(
        self,
        candidates: list[dict[str, object]],
        *,
        evidence: WordEvidence,
        strategy: str,
        llm_enabled: bool,
    ) -> list[dict[str, object]]:
        enriched: list[dict[str, object]] = []
        for idx, candidate in enumerate(candidates):
            base = dict(candidate)
            meanings_raw = base.get("meanings")
            meanings: list[dict[str, object]] = (
                meanings_raw if isinstance(meanings_raw, list) else []
            )
            warnings_raw = base.get("warnings")
            warnings: list[str] = (
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
                morphs: list[str] = (
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
                base["best_estimations"] = synthesis.best_estimations
            else:
                base["synthesized_definition"] = None
                base["confidence"] = self._coverage_confidence(
                    coverage_ratio=coverage_ratio, residual_ratio=residual_ratio
                )
                base["reasoning"] = (
                    "LLM disabled; using concatenated morph meanings."
                )
                base["best_estimations"] = []

            if warnings:
                base["warnings"] = warnings

            enriched.append(base)

        return enriched

    def _build_consensus_synthesis(
        self,
        candidates: list[dict[str, object]],
        *,
        strategy: str,
        llm_use_remote: bool,
    ) -> dict[str, object]:
        coverage_values: list[float] = []
        residual_values: list[float] = []
        for candidate in candidates:
            breakdown_raw = candidate.get("breakdown")
            breakdown = breakdown_raw if isinstance(breakdown_raw, dict) else {}
            coverage = breakdown.get("coverage_ratio")
            residual = breakdown.get("residual_ratio")
            if isinstance(coverage, (int, float)):
                coverage_values.append(float(coverage))
            if isinstance(residual, (int, float)):
                residual_values.append(float(residual))

        context = {
            "strategy": strategy,
            "coverage_ratio": sum(coverage_values) / len(coverage_values)
            if coverage_values
            else 0.0,
            "residual_ratio": sum(residual_values) / len(residual_values)
            if residual_values
            else 1.0,
            "use_remote": llm_use_remote,
        }
        synthesis = synthesize_consensus(candidates, context)
        return synthesis.as_dict()

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

    def _fallback_morph_hints(self, word: str) -> list[dict[str, object]]:
        if not word:
            return []
        word_upper = word.upper()
        min_n = self.candidate_finder.min_n
        max_n = self.candidate_finder.max_n
        dictionary = self.candidate_finder.dictionary
        if not dictionary:
            return []

        hints: list[dict[str, object]] = []
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

    def _fasttext_similarity(self, word: str, morph: str) -> float | None:
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
    ) -> list[str]:
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
    def _merge_decompositions(
        primary: list[Decomposition],
        secondary: list[Decomposition],
    ) -> list[Decomposition]:
        """Merge two sets of decompositions, deduplicating by morph sequence.

        When the same morph sequence appears in both sets, the decomposition
        with the higher beam_score is kept. This allows singleton-based
        decompositions to compete fairly with chunky decompositions.
        """
        # Index by morph tuple for deduplication
        by_morphs: dict[tuple[str, ...], Decomposition] = {}

        for decomp in primary:
            key = tuple(decomp.morphs)
            existing = by_morphs.get(key)
            if existing is None or decomp.beam_score > existing.beam_score:
                by_morphs[key] = decomp

        for decomp in secondary:
            key = tuple(decomp.morphs)
            existing = by_morphs.get(key)
            if existing is None or decomp.beam_score > existing.beam_score:
                by_morphs[key] = decomp

        # Return sorted by beam_score descending for consistent ordering
        merged = list(by_morphs.values())
        merged.sort(key=lambda d: d.beam_score, reverse=True)
        return merged

    @staticmethod
    def _concatenate_meanings(meanings: Sequence[dict[str, object]]) -> str:
        parts: list[str] = []
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
        clusters: list[ClusterRecord],
        residuals: list[ResidualSemanticRecord],
        hypotheses: list[MorphHypothesisRecord],
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
            empty_morphs: dict[str, DictionaryMorph] = {}
            evidence.dictionary_morphs = empty_morphs
            evidence.fasttext_neighbors = []
            return
        if mode == SingleWordTranslationService.EvidenceMode.RESIDUALS_ONLY:
            evidence.direct_clusters = []
            evidence.morph_hypotheses = []
            evidence.attested_definitions = []
            empty_morphs: dict[str, DictionaryMorph] = {}
            evidence.dictionary_morphs = empty_morphs
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


def _cleanup_candidate(candidate: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key in ("normalized", "composite", "cos_sim", "confidence", "tfidf"):
        if key in candidate:
            value = candidate[key]
            cleaned[key] = float(value) if isinstance(value, numbers.Real) else value
    cleaned["breakdown"] = candidate.get("breakdown")
    cleaned["target"] = candidate.get("target")
    cleaned["target_length"] = candidate.get("target_length")
    return cleaned


def _candidate_lookup_keys(candidate: dict[str, object]) -> set[str]:
    keys: set[str] = set()
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


def _residual_detail_from_candidate(candidate: dict[str, object]) -> dict[str, object]:
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


def _extract_uncovered_texts(breakdown: dict[str, object]) -> list[str]:
    raw_uncovered = breakdown.get("uncovered")
    uncovered = raw_uncovered if isinstance(raw_uncovered, list) else []
    texts: list[str] = []
    for entry in uncovered:
        text = entry.get("text") if isinstance(entry, dict) else None
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _extract_low_confidence_segments(breakdown: dict[str, object]) -> list[str]:
    raw_segments = breakdown.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    low_conf: list[str] = []
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


def _first_sense_definition(entry: EntryRecord) -> str | None:
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


def _safe_lower(value: str | None) -> str | None:
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


def _has_segment_coverage(candidates: Sequence[dict[str, object]]) -> bool:
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
) -> tuple[list[Decomposition], float | None]:
    if not decompositions:
        return [], None

    try:
        limit = int(top_n)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        limit = len(decompositions)

    scored: list[tuple[Decomposition, float]] = [
        (decomp, _safe_ratio(decomp.breakdown, "coverage_ratio"))
        for decomp in decompositions
    ]
    max_coverage = max(score for _, score in scored)
    coverage_filtered = [decomp for decomp, score in scored if score >= max_coverage]
    coverage_filtered.sort(key=lambda decomp: float(decomp.beam_score), reverse=True)
    return coverage_filtered[:limit], max_coverage
