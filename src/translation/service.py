from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import math
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
    get_sentence_transformer_if_available,
)
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord

from .decomposition import (
    DEFAULT_MAX_FULL_SEGMENTATIONS,
    DEFAULT_MAX_PARTIAL_PER_INDEX,
    Decomposition,
    DecompositionEngine,
    apply_hard_filters,
    _collect_attested_cluster_ngrams,
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
    DEFAULT_LLM_CONTEXT,
    LLMRequestProgress,
    TranslationProgressReporter,
)
from .placeholder_glosses import (
    candidate_has_placeholder_gloss,
    candidate_is_placeholder_anchor,
    candidate_is_residual_placeholder_anchor,
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
from .strategies import (
    apply_strategy,
    compose_semantic_bundle,
    compute_contradiction_penalty_for_candidates,
    extract_definition_candidates,
    select_top_k,
)
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

    BLIND_RETRANSLATION_SHORT_ROOT_MAX_LEN = 4
    DECISION_SEMANTIC_WEIGHT = 0.22
    DECISION_ATTESTATION_WEIGHT = 0.14
    DECISION_SINGLETON_BURDEN_WEIGHT = 0.08
    DECISION_FEWER_MORPHS_TIE_MARGIN = 0.10
    DECISION_SINGLETON_CLEAR_WIN_MARGIN = 0.18

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
        self._clustered_definition_count_cache: dict[
            tuple[str, tuple[tuple[str, float | None], ...]],
            int,
        ] = {}

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

    def _cluster_definition_counts_cached(
        self,
        definition_glosses: dict[str, list[tuple[str, float | None]]],
        *,
        embedder,
    ) -> dict[str, int]:
        """Reuse semantic cluster counts across repeated translation calls.

        Phrase translation revisits the same substring gloss sets across many
        tokens. Caching the clustered count by morph-plus-gloss signature keeps
        the semantic-deduplication signal intact without paying the embedding
        clustering cost every time a token asks for the same morph family.
        """

        if embedder is None or not definition_glosses:
            return {}

        cached_counts: dict[str, int] = {}
        missing: dict[str, list[tuple[str, float | None]]] = {}
        for morph, glosses in definition_glosses.items():
            normalized_glosses = tuple(
                sorted(
                    (
                        str(gloss).strip(),
                        None if score is None else float(score),
                    )
                    for gloss, score in glosses
                    if isinstance(gloss, str) and gloss.strip()
                )
            )
            if not normalized_glosses:
                continue
            cache_key = (morph.upper(), normalized_glosses)
            cached = self._clustered_definition_count_cache.get(cache_key)
            if cached is not None:
                cached_counts[morph.upper()] = cached
                continue
            missing[morph] = list(normalized_glosses)

        if missing:
            clustered = cluster_definition_counts(
                missing,
                embedder,
            )
            for morph, glosses in missing.items():
                cache_key = (morph.upper(), tuple(glosses))
                count = int(clustered.get(morph.upper(), 0))
                self._clustered_definition_count_cache[cache_key] = count
                cached_counts[morph.upper()] = count

        return cached_counts

    def _prewarm_clustered_definition_counts(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
        evidence_mode: "SingleWordTranslationService.EvidenceMode" = EvidenceMode.ALL,
    ) -> None:
        """Precompute semantic cluster counts for phrase-wide substring pools.

        The phrase layer already knows when it is about to analyze many tokens.
        Warming the clustered definition counts here lets later `translate_word`
        calls reuse those semantic-deduplication results instead of rebuilding
        them token by token.
        """

        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        definition_glosses = self.repository.fetch_accepted_definition_glosses(
            unique,
            variants=variants,
            include_clusters=evidence_mode != self.EvidenceMode.RESIDUALS_ONLY,
            include_residuals=evidence_mode != self.EvidenceMode.CLUSTERS_ONLY,
        )
        if not definition_glosses:
            return

        embedder = get_sentence_transformer_if_available(
            "paraphrase-MiniLM-L6-v2",
            local_files_only=True,
        )
        if embedder is None:
            return
        self._cluster_definition_counts_cached(
            definition_glosses,
            embedder=embedder,
        )

    def translate_word(
        self,
        word: str,
        *,
        variants: Iterable[str] | None = None,
        strategy: str = "prefer-balance",
        top_k: int = 3,
        llm: bool | None = None,
        llm_context: str | None = None,
        fallback_top_n: int = 5,
        evidence_mode: EvidenceMode = EvidenceMode.CLUSTERS_ONLY,
        weight_enabled: bool = True,
        allow_whole_word: bool = True,
        use_beam_search: bool = False,
        progress_reporter: TranslationProgressReporter | None = None,
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
        try:
            evidence = self.repository.fetch_word_evidence(
                normalized,
                variants=active_variants,
                dictionary_entries=dictionary_snapshot,
                min_n=self.candidate_finder.min_n,
                max_n=self.candidate_finder.max_n,
            )
            if not evidence or not getattr(evidence, "word", None):
                llm_enabled = self.llm_enabled if llm is None else bool(llm)
                llm_progress = self._build_llm_progress_plan(
                    candidate_count=0,
                    include_consensus=False,
                    include_validation_repairs=llm_enabled,
                )
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
                    "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                    "llm_request_plan": self._serialize_llm_progress_plan(llm_progress),
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
                evidence.rejected_morphs.update(
                    self.repository.fetch_rejected_morphs(
                        substrings, variants=active_variants
                    )
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
            embedder = get_sentence_transformer_if_available(
                "paraphrase-MiniLM-L6-v2",
                local_files_only=True,
            )
            clustered_counts = self._cluster_definition_counts_cached(
                definition_glosses,
                embedder=embedder,
            )
            if clustered_counts:
                definition_counts = {**definition_counts, **clustered_counts}

            evidence.definition_counts = definition_counts
            evidence.definition_glosses = definition_glosses
            exact_candidates, blind_suppression_notes = self._collect_exact_word_candidates(
                normalized,
                evidence,
                allow_whole_word=allow_whole_word,
            )
            for candidate in exact_candidates:
                self._attach_candidate_bundle_metadata(candidate)

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
                "blind_retranslation": {
                    "enabled": not allow_whole_word,
                    "short_root_max_len": self.BLIND_RETRANSLATION_SHORT_ROOT_MAX_LEN,
                    "suppressed": blind_suppression_notes,
                },
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
                        attested_cluster_ngrams=_collect_attested_cluster_ngrams(
                            evidence,
                            evidence_mode=evidence_mode.value,
                        ),
                    )
                    beam_scoring_applied = True
                    beam_scoring_parse_count = len(parses)
                    beam_scores: dict[tuple[str, ...], float] = {}
                    parsed_paths: list[tuple[tuple[str, ...], float]] = []
                    all_morphs: set[str] = set()
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
                        parsed_paths.append((key, float(score)))
                        all_morphs.update(morphs)

                    definition_candidates = extract_definition_candidates(
                        all_morphs,
                        evidence,
                        max_per_morph=3,
                        allow_dictionary=allow_whole_word,
                    )
                    for key, score in parsed_paths:
                        penalty = compute_contradiction_penalty_for_candidates(
                            key,
                            definition_candidates,
                        )
                        adjusted = score - penalty
                        existing = beam_scores.get(key)
                        if existing is None or adjusted > existing:
                            beam_scores[key] = adjusted
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
            selected = self._select_compositional_candidates(
                filtered,
                evidence=evidence,
                diagnostics=diagnostics,
                strategy=strategy,
                top_k=top_k,
                weight_enabled=weight_enabled,
                allow_whole_word=allow_whole_word,
            )
            blind_full_cover_candidates = [
                candidate
                for candidate in selected
                if self._is_grounded_full_cover_compositional(candidate)
            ]
            if use_beam_search and not selected and not exact_candidates:
                guaranteed_decompositions, guaranteed_diag = (
                    self._enumerate_attested_full_cover_decompositions(
                        normalized,
                        evidence=evidence,
                        attested_pieces=attested_pieces,
                        allow_whole_word=allow_whole_word,
                        definition_counts=definition_counts,
                        definition_glosses=definition_glosses,
                        evidence_mode=evidence_mode.value,
                    )
                )
                diagnostics["beam_empty_fallback"] = guaranteed_diag
                if guaranteed_decompositions:
                    merged_decompositions = self._merge_decompositions(
                        decompositions,
                        guaranteed_decompositions,
                    )
                    decompositions = merged_decompositions
                    filtered, filter_diagnostics = apply_hard_filters(
                        merged_decompositions,
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
                            "stage1_drops_missing_support",
                            stage1_drops_total,
                        )
                    )
                    stage1_drops_other_reasons = int(
                        filter_diagnostics.get(
                            "stage1_drops_other_reasons",
                            max(0, stage1_drops_total - stage1_drops_missing_support),
                        )
                    )
                    diagnostics["decomposition_count"] = len(merged_decompositions)
                    diagnostics["hard_filter_stage1_drops_total"] = stage1_drops_total
                    diagnostics["hard_filter_stage1_drops_missing_support"] = (
                        stage1_drops_missing_support
                    )
                    diagnostics["hard_filter_stage1_drops_other_reasons"] = (
                        stage1_drops_other_reasons
                    )
                    selected = self._select_compositional_candidates(
                        filtered,
                        evidence=evidence,
                        diagnostics=diagnostics,
                        strategy=strategy,
                        top_k=top_k,
                        weight_enabled=weight_enabled,
                        allow_whole_word=allow_whole_word,
                    )
                    blind_full_cover_candidates = [
                        candidate
                        for candidate in selected
                        if self._is_grounded_full_cover_compositional(candidate)
                    ]
            if (
                not allow_whole_word
                and use_beam_search
                and not blind_full_cover_candidates
            ):
                guaranteed_decompositions, guaranteed_diag = (
                    self._enumerate_attested_full_cover_decompositions(
                        normalized,
                        evidence=evidence,
                        attested_pieces=attested_pieces,
                        allow_whole_word=allow_whole_word,
                        definition_counts=definition_counts,
                        definition_glosses=definition_glosses,
                        evidence_mode=evidence_mode.value,
                    )
                )
                diagnostics["blind_full_cover_fallback"] = guaranteed_diag
                if guaranteed_decompositions:
                    merged_decompositions = self._merge_decompositions(
                        decompositions,
                        guaranteed_decompositions,
                    )
                    decompositions = merged_decompositions
                    filtered, filter_diagnostics = apply_hard_filters(
                        merged_decompositions,
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
                            "stage1_drops_missing_support",
                            stage1_drops_total,
                        )
                    )
                    stage1_drops_other_reasons = int(
                        filter_diagnostics.get(
                            "stage1_drops_other_reasons",
                            max(0, stage1_drops_total - stage1_drops_missing_support),
                        )
                    )
                    diagnostics["decomposition_count"] = len(merged_decompositions)
                    diagnostics["hard_filter_stage1_drops_total"] = stage1_drops_total
                    diagnostics["hard_filter_stage1_drops_missing_support"] = (
                        stage1_drops_missing_support
                    )
                    diagnostics["hard_filter_stage1_drops_other_reasons"] = (
                        stage1_drops_other_reasons
                    )
                    selected = self._select_compositional_candidates(
                        filtered,
                        evidence=evidence,
                        diagnostics=diagnostics,
                        strategy=strategy,
                        top_k=top_k,
                        weight_enabled=weight_enabled,
                        allow_whole_word=allow_whole_word,
                    )
                    blind_full_cover_candidates = [
                        candidate
                        for candidate in selected
                        if self._is_grounded_full_cover_compositional(candidate)
                    ]

            provisional_candidates, fallback_morphs, provisional_suppression_notes = (
                self._build_provisional_candidates(
                    normalized,
                    evidence=evidence,
                    top_n=fallback_top_n,
                    existing_candidates=exact_candidates + selected,
                    allow_whole_word=allow_whole_word,
                )
            )
            for candidate in provisional_candidates:
                self._attach_candidate_bundle_metadata(candidate)
            exact_candidates, provisional_candidates, decomposition_priority_notes = (
                self._apply_blind_decomposition_priority(
                    word=normalized,
                    allow_whole_word=allow_whole_word,
                    exact_candidates=exact_candidates,
                    compositional_candidates=selected,
                    provisional_candidates=provisional_candidates,
                    had_decomposition_attempts=bool(decompositions),
                    has_full_cover_compositional=bool(blind_full_cover_candidates),
                )
            )
            blind_suppression_notes.extend(provisional_suppression_notes)
            blind_suppression_notes.extend(decomposition_priority_notes)
            if (
                not allow_whole_word
                and bool(decompositions)
                and not blind_full_cover_candidates
            ):
                blind_suppression_notes.append(
                    "Blind retranslation BUG: no grounded full-cover compositional "
                    f"candidate survived for {normalized} after attested decomposition fallback."
                )
            diagnostics["blind_retranslation"] = {
                "enabled": not allow_whole_word,
                "short_root_max_len": self.BLIND_RETRANSLATION_SHORT_ROOT_MAX_LEN,
                "suppressed": blind_suppression_notes,
            }
            if provisional_candidates and not selected:
                fallback_used = True
                fallback_mode = fallback_mode or "provisional_candidates"
            candidate_pool = self._merge_candidate_pool(
                exact_candidates,
                selected,
                provisional_candidates,
                top_k=top_k,
            )

            llm_enabled = self.llm_enabled if llm is None else bool(llm)
            llm_progress = self._build_llm_progress_plan(
                candidate_count=len(candidate_pool),
                include_consensus=llm_enabled and len(candidate_pool) > 1,
                include_validation_repairs=llm_enabled,
            )
            if llm_enabled and llm_progress is not None and progress_reporter is not None:
                progress_reporter.prepare(llm_progress)
            try:
                enriched = self._enrich_candidates(
                    candidate_pool,
                    evidence=evidence,
                    strategy=strategy,
                    llm_enabled=llm_enabled,
                    llm_context=llm_context,
                    llm_progress=llm_progress,
                    progress_reporter=progress_reporter,
                )
                consensus_payload: dict[str, object] | None = None
                if llm_enabled and len(enriched) > 1:
                    consensus_payload = self._build_consensus_synthesis(
                        enriched,
                        strategy=strategy,
                        llm_use_remote=self.llm_use_remote,
                        llm_context=llm_context,
                        llm_progress=llm_progress,
                        progress_reporter=progress_reporter,
                    )
            finally:
                if llm_enabled and progress_reporter is not None:
                    progress_reporter.done()

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
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "llm_request_plan": self._serialize_llm_progress_plan(llm_progress),
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
                        "selected": len(candidate_pool),
                        "fallback_generated": len(provisional_candidates),
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
        llm_context: str | None,
        llm_progress: LLMRequestProgress | None = None,
        progress_reporter: TranslationProgressReporter | None = None,
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

            bundle_surface = str(
                base.get("bundle_surface_gloss") or base.get("bundle_head_gloss") or ""
            ).strip()
            base["concatenated_meanings"] = (
                bundle_surface or self._concatenate_meanings(meanings)
            )

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
                    "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                    "llm_progress": llm_progress,
                    "progress_reporter": progress_reporter,
                    "candidate_rank": idx + 1,
                    "progress_stage": "candidate",
                    "progress_label": f"refining rank {idx + 1} best estimations...",
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
                base_confidence = self._coverage_confidence(
                    coverage_ratio=coverage_ratio, residual_ratio=residual_ratio
                )
                analysis_type = str(base.get("analysis_type") or "")
                if analysis_type == "dictionary_exact":
                    base["confidence"] = max(base_confidence, 0.98)
                elif analysis_type == "whole_word_anchor":
                    base["confidence"] = max(base_confidence, 0.92)
                elif analysis_type == "provisional":
                    base["confidence"] = min(base_confidence, 0.55)
                else:
                    base["confidence"] = base_confidence
                base["reasoning"] = (
                    "LLM disabled; using concatenated morph meanings."
                )
                base["best_estimations"] = []

            if candidate_is_residual_placeholder_anchor(base):
                base["confidence"] = min(float(base.get("confidence") or 0.0), 0.2)
                warnings.append(
                    "Residual-only whole-word placeholder anchor has reduced confidence."
                )
            elif candidate_is_placeholder_anchor(base):
                base["confidence"] = min(float(base.get("confidence") or 0.0), 0.25)
                warnings.append(
                    "Opaque whole-word placeholder anchor has reduced confidence."
                )
            elif candidate_has_placeholder_gloss(base):
                base["confidence"] = min(float(base.get("confidence") or 0.0), 0.45)
                warnings.append(
                    "Placeholder residual gloss lowered confidence for this candidate."
                )

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
        llm_context: str | None,
        llm_progress: LLMRequestProgress | None = None,
        progress_reporter: TranslationProgressReporter | None = None,
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
            "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
            "llm_progress": llm_progress,
            "progress_reporter": progress_reporter,
            "progress_stage": "consensus",
            "progress_label": "refining consensus best estimations...",
        }
        synthesis = synthesize_consensus(candidates, context)
        return synthesis.as_dict()

    def _build_llm_progress_plan(
        self,
        *,
        candidate_count: int,
        include_consensus: bool,
        include_validation_repairs: bool,
    ) -> LLMRequestProgress | None:
        primary_requests = candidate_count + (1 if include_consensus else 0)
        if primary_requests <= 0:
            return None
        repair_budget = primary_requests if include_validation_repairs else 0
        return LLMRequestProgress(
            total_primary_requests=primary_requests,
            validation_repair_budget=repair_budget,
        )

    @staticmethod
    def _serialize_llm_progress_plan(
        progress: LLMRequestProgress | None,
    ) -> dict[str, int] | None:
        if progress is None:
            return None
        return {
            "planned_primary_requests": progress.total_primary_requests,
            "possible_validation_repairs": progress.validation_repair_budget,
            "possible_total_requests": (
                progress.total_primary_requests + progress.validation_repair_budget
            ),
        }

    def _collect_exact_word_candidates(
        self,
        word: str,
        evidence: WordEvidence,
        *,
        allow_whole_word: bool,
    ) -> tuple[list[dict[str, object]], list[str]]:
        """Build exact-word anchor candidates before decomposition ranking.

        Single-word translation needs exact lexical anchors and whole-word
        accepted roots to compete directly with decomposed analyses. This keeps
        canonical dictionary items definitive while still allowing a strong
        whole-word database anchor to beat an over-eager split. When blind
        retranslation mode is active (`allow_whole_word=False`), this helper
        suppresses exact dictionary matches and long whole-word DB anchors so
        retranslations are forced back through productive decomposition.
        """
        candidates: list[dict[str, object]] = []
        suppression_notes: list[str] = []

        dictionary_exact = self._dictionary_exact_entry(word, evidence)
        if dictionary_exact and (dictionary_exact.definition or dictionary_exact.senses):
            definitions = [
                text
                for text in [dictionary_exact.definition, *dictionary_exact.senses]
                if isinstance(text, str) and text.strip()
            ]
            if definitions:
                if allow_whole_word:
                    candidates.append(
                        self._build_single_morph_candidate(
                            word,
                            definition=definitions[0],
                            definitions=definitions,
                            provenance="dictionary",
                            score=200.0,
                            analysis_type="dictionary_exact",
                            warnings=[],
                        )
                    )
                else:
                    suppression_notes.append(
                        f"Blind retranslation suppressed exact dictionary match for {word}."
                    )

        exact_definition_candidates = extract_definition_candidates(
            [word],
            evidence,
            max_per_morph=4,
        ).get(word, [])
        anchor_candidates = [
            entry
            for entry in exact_definition_candidates
            if entry.get("source") in {"cluster", "residual"}
        ]
        if anchor_candidates:
            selected = anchor_candidates[0]
            selected_definition = selected.get("definition")
            if isinstance(selected_definition, str) and selected_definition.strip():
                if (
                    not allow_whole_word
                    and len(word) > self.BLIND_RETRANSLATION_SHORT_ROOT_MAX_LEN
                ):
                    suppression_notes.append(
                        "Blind retranslation suppressed long whole-word DB anchor "
                        f"for {word}."
                    )
                else:
                    candidates.append(
                        self._build_single_morph_candidate(
                            word,
                            definition=selected_definition,
                            definitions=[
                                str(entry.get("definition")).strip()
                                for entry in anchor_candidates
                                if isinstance(entry.get("definition"), str)
                                and str(entry.get("definition")).strip()
                            ],
                            provenance=str(selected.get("source") or "cluster"),
                            score=150.0 + float(selected.get("quality") or 0.0),
                            analysis_type="whole_word_anchor",
                            warnings=[],
                        )
                    )

        return candidates, suppression_notes

    def _build_provisional_candidates(
        self,
        word: str,
        *,
        evidence: WordEvidence,
        top_n: int,
        existing_candidates: Sequence[dict[str, object]],
        allow_whole_word: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        """Return provisional whole-word candidates plus human-readable hints.

        The current pipeline should not fail silently when support is weak but
        there is still corpus or dictionary context worth surfacing. Provisional
        candidates make that uncertainty explicit while preserving useful
        building blocks for later phrase-level or corpus-level disambiguation.
        In blind retranslation mode, provisional exact-word fallback is only
        allowed when the token does not already have an exact dictionary entry.
        """
        fallback_morphs = self._fallback_morph_hints(word)[:top_n]
        suppression_notes: list[str] = []
        existing_types = {
            (
                tuple(str(morph) for morph in candidate.get("morphs", [])),
                str(candidate.get("analysis_type") or ""),
            )
            for candidate in existing_candidates
            if isinstance(candidate, dict)
        }
        has_exact_word_candidate = any(
            tuple(str(morph) for morph in candidate.get("morphs", [])) == (word,)
            and str(candidate.get("analysis_type") or "") != "provisional"
            for candidate in existing_candidates
            if isinstance(candidate, dict)
        )
        has_dictionary_exact = self._dictionary_exact_entry(word, evidence) is not None

        provisional: list[dict[str, object]] = []
        attested_definitions = sorted(
            {
                att.definition.strip()
                for att in evidence.attested_definitions
                if att.source_word.upper() == word
                and isinstance(att.definition, str)
                and att.definition.strip()
            }
        )
        if (
            attested_definitions
            and not allow_whole_word
            and has_dictionary_exact
            and not has_exact_word_candidate
        ):
            suppression_notes.append(
                "Blind retranslation suppressed provisional full-word fallback "
                f"for {word} because an exact dictionary entry exists."
            )
        if (
            attested_definitions
            and (allow_whole_word or not has_dictionary_exact)
            and not has_exact_word_candidate
            and ((word,), "provisional") not in existing_types
        ):
            provisional.append(
                self._build_single_morph_candidate(
                    word,
                    definition=attested_definitions[0],
                    definitions=attested_definitions,
                    provenance="attested",
                    score=40.0 + min(len(attested_definitions), 5),
                    analysis_type="provisional",
                    warnings=[
                        "Provisional whole-word reading from raw attestation only.",
                    ],
                )
            )

        return provisional, fallback_morphs, suppression_notes

    @staticmethod
    def _dictionary_exact_entry(
        word: str,
        evidence: WordEvidence,
    ) -> DictionaryMorph | None:
        """Return the exact dictionary entry for ``word`` when one is usable.

        Blind retranslation decisions depend on whether a token has an exact
        dictionary-backed whole-word reading, not just whether dictionary data
        exists for substrings. Centralizing that lookup keeps the exact-anchor
        and provisional-fallback rules aligned.
        """

        dictionary_exact = evidence.dictionary_morphs.get(word)
        if dictionary_exact is None:
            return None
        if dictionary_exact.definition or dictionary_exact.senses:
            return dictionary_exact
        return None

    def _build_single_morph_candidate(
        self,
        word: str,
        *,
        definition: str,
        definitions: Sequence[str],
        provenance: str,
        score: float,
        analysis_type: str,
        warnings: Sequence[str],
    ) -> dict[str, object]:
        """Create a normalized single-morph candidate payload.

        Exact-word anchors, whole-word database matches, and provisional raw
        attestations all share the same result schema as compositional
        candidates. Building them through one helper keeps output stable across
        `translate-word` and the upcoming phrase-level parser.
        """
        unique_definitions = [
            text
            for idx, text in enumerate(definitions)
            if text and text not in definitions[:idx]
        ]
        selected_definition = unique_definitions[0] if unique_definitions else definition
        return {
            "rank": 1,
            "morphs": [word],
            "canonicals": [word],
            "score": float(score),
            "breakdown": self._full_word_breakdown(word),
            "score_breakdown": None,
            "meanings": [
                {
                    "morph": word,
                    "canonical": word,
                    "definition": definition,
                    "definitions": unique_definitions,
                    "raw_definition": definition,
                    "semantic_core": [],
                    "semantic_core_terms": [],
                    "negative_contrast": [],
                    "surface_gloss": definition,
                    "surface_gloss_strategy": "whole_word",
                    "provenance": provenance,
                    "anchor_strength": 1.0,
                    "definition_trace": {
                        "selected_definition": selected_definition,
                        "raw_selected_definition": selected_definition,
                        "selected_semantic_core": [],
                        "selected_negative_contrast": [],
                        "surface_gloss": selected_definition,
                        "surface_gloss_strategy": "whole_word",
                        "selected_source": provenance,
                        "selected_quality": 1.0,
                        "runner_ups": [
                            {
                                "definition": alternate,
                                "raw_definition": alternate,
                                "semantic_core": [],
                                "negative_contrast": [],
                                "surface_gloss_strategy": "whole_word",
                                "source": provenance,
                                "quality": 1.0,
                            }
                            for alternate in unique_definitions
                            if isinstance(alternate, str)
                            and alternate.strip()
                            and alternate != selected_definition
                        ][:3],
                        "suppressed": [],
                        "blind_dictionary_fallback": False,
                        "negative_contrast_penalties": [],
                        "meta_linguistic_rejections": [],
                    },
                }
            ],
            "warnings": list(warnings),
            "analysis_type": analysis_type,
            "blind_mode_whole_word_rescue": False,
        }

    def _annotate_candidate(
        self,
        candidate: dict[str, object],
        *,
        evidence: WordEvidence,
        analysis_type: str,
    ) -> None:
        """Apply analysis metadata and explicit warnings to a candidate."""
        candidate["analysis_type"] = analysis_type
        warnings = list(candidate.get("warnings", []))
        rejected = sorted(
            {
                morph
                for morph in candidate.get("morphs", [])
                if isinstance(morph, str) and morph.upper() in evidence.rejected_morphs
            }
        )
        if rejected:
            warnings.append(
                "Contains explicitly rejected morph evidence: " + ", ".join(rejected)
            )
        if warnings:
            candidate["warnings"] = warnings

    def _attach_candidate_bundle_metadata(self, candidate: dict[str, object]) -> None:
        """Attach ordered semantic-bundle fields to one candidate payload.

        Word translation previously collapsed each decomposition down to a flat
        gloss list. Phrase translation now needs the ordered per-morph
        structure preserved so blind-mode scoring, phrase composition, and
        verbose traces can all see how the decomposition itself is carrying the
        meaning.
        """

        meanings_raw = candidate.get("meanings")
        meanings = meanings_raw if isinstance(meanings_raw, list) else []
        bundle = compose_semantic_bundle(meanings)
        candidate.update(bundle)
        candidate.setdefault("blind_mode_whole_word_rescue", False)

        bundle_semantic_cores = [
            list(entry.get("semantic_core_terms") or [])
            for entry in bundle.get("semantic_bundle", [])
            if isinstance(entry, dict)
        ]
        bundle_negative_contrast = [
            list(entry.get("negative_contrast") or [])
            for entry in bundle.get("semantic_bundle", [])
            if isinstance(entry, dict)
        ]
        bundle_surface_candidates = list(bundle.get("bundle_surface_candidates") or [])
        bundle_selection_reason = str(bundle.get("bundle_selection_reason") or "").strip()
        blind_mode_rescue_note = str(
            candidate.get("blind_mode_rescue_note") or ""
        ).strip()

        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            trace_raw = meaning.get("definition_trace")
            trace = dict(trace_raw) if isinstance(trace_raw, dict) else {}
            trace["bundle_semantic_cores"] = bundle_semantic_cores
            trace["bundle_negative_contrast"] = bundle_negative_contrast
            trace["bundle_surface_candidates"] = bundle_surface_candidates
            trace["bundle_selection_reason"] = bundle_selection_reason
            if blind_mode_rescue_note:
                trace["blind_mode_rescue_note"] = blind_mode_rescue_note
            meaning["definition_trace"] = trace

    def _select_compositional_candidates(
        self,
        filtered: Sequence[Decomposition],
        *,
        evidence: WordEvidence,
        diagnostics: dict[str, object],
        strategy: str,
        top_k: int,
        weight_enabled: bool,
        allow_whole_word: bool,
    ) -> list[dict[str, object]]:
        """Score decompositions via an explicit context-weighted decision function.

        Why this exists:
        decomposition quality is now tuned case-by-case, so ranking must be
        transparent and stable instead of relying on an opaque blend of legacy
        scores. This method keeps one explicit decision equation for all
        compositional reads and records the winning-vs-runner-up traces needed
        for manual debugging cycles.

        Architecture responsibility:
        this is the only place where decomposition candidates are turned into a
        final ordered shortlist. Downstream phrase assembly and CLI reporting
        consume the decision traces produced here.
        """

        ranked: list[tuple[Decomposition, float]] = []
        coherence_results: dict[tuple[str, ...], CoherenceResult] = {}
        fasttext_model = getattr(self.candidate_finder, "fasttext_model", None)

        for decomp in filtered:
            has_singletons = any(len(morph) == 1 for morph in decomp.morphs)
            if has_singletons and fasttext_model is not None:
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
                        decomp.morphs,
                        fasttext_model,
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

            rejected_morphs = [
                morph
                for morph in decomp.morphs
                if morph.upper() in evidence.rejected_morphs
            ]
            if rejected_morphs:
                score -= 2.5 * len(rejected_morphs)
                decomp.score_breakdown = dict(decomp.score_breakdown)
                decomp.score_breakdown["rejected_morphs"] = rejected_morphs
                decomp.score_breakdown["rejected_morph_penalty"] = (
                    2.5 * len(rejected_morphs)
                )

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

        strategy_ranked = apply_strategy(ranked, strategy=strategy, evidence=evidence)
        strategy_scores_by_morphs = {
            tuple(decomp.morphs): float(score)
            for decomp, score in strategy_ranked
        }

        decision_rows: list[dict[str, object]] = []
        for decomp, legacy_score in ranked:
            morph_key = tuple(decomp.morphs)
            base_component = float(
                strategy_scores_by_morphs.get(morph_key, legacy_score)
            )
            semantic_raw = self._decision_semantic_signal(
                decomp=decomp,
                coherence_results=coherence_results,
            )
            semantic_component = self.DECISION_SEMANTIC_WEIGHT * semantic_raw
            attestation_raw = self._decision_attestation_signal(
                decomp=decomp,
                evidence=evidence,
            )
            attestation_component = self.DECISION_ATTESTATION_WEIGHT * attestation_raw
            singleton_burden_raw = self._decision_singleton_burden_signal(decomp)
            singleton_burden_component = (
                self.DECISION_SINGLETON_BURDEN_WEIGHT * singleton_burden_raw
            )
            decision_score = (
                base_component
                + semantic_component
                + attestation_component
                - singleton_burden_component
            )
            decision_rows.append(
                {
                    "decomposition": decomp,
                    "morphs": list(decomp.morphs),
                    "analysis_type": "compositional",
                    "base_component": base_component,
                    "semantic_raw": semantic_raw,
                    "semantic_component": semantic_component,
                    "attestation_raw": attestation_raw,
                    "attestation_component": attestation_component,
                    "singleton_burden_raw": singleton_burden_raw,
                    "singleton_burden_component": singleton_burden_component,
                    "decision_score": decision_score,
                    "morph_count": len(decomp.morphs),
                    "singleton_count": sum(
                        1 for morph in decomp.morphs if len(morph) == 1
                    ),
                    "tie_break_applied": False,
                    "tie_break_reason": None,
                    "selection_reason": "",
                    "rejection_reason": "",
                }
            )

        decision_rows.sort(
            key=lambda row: float(row.get("decision_score") or 0.0),
            reverse=True,
        )
        decision_rows = self._apply_fewer_morph_tie_breaks(decision_rows)
        self._finalize_decision_reasons(decision_rows)

        diagnostics["decision_function"] = {
            "semantic_weight": self.DECISION_SEMANTIC_WEIGHT,
            "attestation_weight": self.DECISION_ATTESTATION_WEIGHT,
            "singleton_burden_weight": self.DECISION_SINGLETON_BURDEN_WEIGHT,
            "fewer_morphs_tie_margin": self.DECISION_FEWER_MORPHS_TIE_MARGIN,
            "singleton_clear_win_margin": self.DECISION_SINGLETON_CLEAR_WIN_MARGIN,
            "strategy": strategy,
        }
        diagnostics["decision_rows"] = [
            self._serialize_decision_row(row)
            for row in decision_rows[: max(3, top_k * 2)]
        ]

        selected = select_top_k(
            [
                (
                    row["decomposition"],
                    float(row.get("decision_score") or 0.0),
                )
                for row in decision_rows
            ],
            k=top_k,
            evidence=evidence,
            allow_dictionary=allow_whole_word,
        )
        self._attach_decision_traces_to_candidates(selected, decision_rows)
        for candidate in selected:
            self._annotate_candidate(
                candidate,
                evidence=evidence,
                analysis_type="compositional",
            )
            self._attach_candidate_bundle_metadata(candidate)
        return selected

    def _decision_semantic_signal(
        self,
        *,
        decomp: Decomposition,
        coherence_results: dict[tuple[str, ...], CoherenceResult],
    ) -> float:
        """Return the semantic compatibility signal for decision ranking.

        Why this exists:
        manual decomposition tuning needs a scalar that answers "does this split
        hang together semantically?" independently from attestation and
        singleton penalties.

        Architecture responsibility:
        converts either fasttext-based coherence or score-breakdown fallback
        values into a normalized [0, 1] signal consumed by the decision
        function.
        """

        key = tuple(decomp.morphs)
        coherence = coherence_results.get(key)
        if coherence is not None:
            return max(0.0, min(1.0, float(coherence.score)))
        breakdown = decomp.score_breakdown if isinstance(decomp.score_breakdown, dict) else {}
        components = breakdown.get("components")
        if isinstance(components, dict):
            definition_coherence = components.get("definition_coherence")
            if isinstance(definition_coherence, dict):
                raw = definition_coherence.get("raw")
                if isinstance(raw, numbers.Real):
                    # Definition coherence can be negative in edge cases.
                    return max(0.0, min(1.0, (float(raw) + 1.0) / 2.0))
        return 0.0

    def _decision_attestation_signal(
        self,
        *,
        decomp: Decomposition,
        evidence: WordEvidence,
    ) -> float:
        """Return a normalized attestation-strength signal for one decomposition.

        Why this exists:
        decomposition-first behavior still needs to favor morphs grounded in
        real evidence instead of arbitrary character splits.

        Architecture responsibility:
        computes a length-weighted attestation/specificity value that can be
        blended into the explicit decision equation.
        """

        cluster_morphs = {
            cluster.ngram.upper()
            for cluster in evidence.direct_clusters
            if isinstance(cluster.ngram, str) and cluster.ngram
        }
        residual_morphs = {
            residual.residual.upper()
            for residual in evidence.residual_semantics
            if isinstance(residual.residual, str) and residual.residual
        }
        hypothesis_morphs = {
            hypothesis.morph.upper()
            for hypothesis in evidence.morph_hypotheses
            if isinstance(hypothesis.morph, str) and hypothesis.morph
        }
        definition_counts = evidence.definition_counts or {}

        weighted_total = 0.0
        total_length = 0.0
        for morph in decomp.morphs:
            normalized = str(morph or "").upper()
            if not normalized:
                continue
            length = float(max(1, len(normalized)))
            total_length += length

            if normalized in cluster_morphs:
                source_strength = 1.0
            elif normalized in residual_morphs:
                source_strength = 0.80
            elif normalized in hypothesis_morphs:
                source_strength = 0.60
            else:
                source_strength = 0.0

            def_count_raw = definition_counts.get(normalized)
            if isinstance(def_count_raw, int) and def_count_raw > 0:
                specificity = 1.0 / (1.0 + math.log1p(float(def_count_raw)))
            elif source_strength > 0.0:
                specificity = 0.55
            else:
                specificity = 0.0

            weighted_total += length * (0.70 * source_strength + 0.30 * specificity)

        if total_length <= 0:
            return 0.0
        return max(0.0, min(1.0, weighted_total / total_length))

    @staticmethod
    def _decision_singleton_burden_signal(decomp: Decomposition) -> float:
        """Return how heavily singleton usage should be penalized.

        Why this exists:
        singletons are sometimes necessary, but singleton-heavy parses should
        only win when they carry a clear semantic/attestation advantage.

        Architecture responsibility:
        emits a bounded burden scalar used by the explicit decision function.
        """

        morphs = [str(morph or "") for morph in decomp.morphs if str(morph or "")]
        if not morphs:
            return 0.0
        singleton_count = sum(1 for morph in morphs if len(morph) == 1)
        if singleton_count == 0:
            return 0.0
        share = singleton_count / float(len(morphs))
        chain = max(0, singleton_count - 1) * 0.20
        dominance = 0.25 if singleton_count >= 3 and len(morphs) >= 4 else 0.0
        return max(0.0, min(2.0, share + chain + dominance))

    def _apply_fewer_morph_tie_breaks(
        self,
        rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Apply the fewer-morph tie-break while preserving clear singleton wins.

        Why this exists:
        decomposition-first ranking should not drift toward confetti splits when
        scores are nearly tied, but still must allow singleton-heavy winners
        when they genuinely outperform alternatives.

        Architecture responsibility:
        deterministic post-sort pass that documents when tie-break swaps occur.
        """

        ordered = [dict(row) for row in rows]
        changed = True
        while changed:
            changed = False
            for index in range(len(ordered) - 1):
                higher = ordered[index]
                lower = ordered[index + 1]
                higher_score = float(higher.get("decision_score") or 0.0)
                lower_score = float(lower.get("decision_score") or 0.0)
                delta = higher_score - lower_score
                higher_morph_count = int(higher.get("morph_count") or 0)
                lower_morph_count = int(lower.get("morph_count") or 0)
                if higher_morph_count <= lower_morph_count:
                    continue
                if delta > self.DECISION_FEWER_MORPHS_TIE_MARGIN:
                    continue
                higher_singleton_count = int(higher.get("singleton_count") or 0)
                if (
                    higher_singleton_count >= 2
                    and higher_singleton_count * 2 >= max(1, higher_morph_count)
                    and delta >= self.DECISION_SINGLETON_CLEAR_WIN_MARGIN
                ):
                    continue

                ordered[index], ordered[index + 1] = lower, higher
                ordered[index]["tie_break_applied"] = True
                ordered[index]["tie_break_reason"] = (
                    "Tie-break preferred fewer morphs "
                    f"({lower_morph_count} vs {higher_morph_count}) with score delta "
                    f"{delta:.3f}."
                )
                changed = True
        return ordered

    def _finalize_decision_reasons(self, rows: list[dict[str, object]]) -> None:
        """Attach winner/runner-up rationale to decision rows.

        Why this exists:
        manual case-by-case debugging needs deterministic "why winner beat
        runner-up" text without re-deriving ranking math from raw numbers.

        Architecture responsibility:
        enriches already-ranked rows with human-readable selection/rejection
        reasons consumed by JSON traces and verbose CLI output.
        """

        if not rows:
            return
        winner = rows[0]
        winner_score = float(winner.get("decision_score") or 0.0)
        runner_up = rows[1] if len(rows) > 1 else None
        for index, row in enumerate(rows, start=1):
            row["decision_rank"] = index
        if runner_up is not None:
            delta = winner_score - float(runner_up.get("decision_score") or 0.0)
            winner["runner_up_morphs"] = list(runner_up.get("morphs") or [])
            winner["runner_up_score_delta"] = delta
            selection_reason = (
                "Top decision score "
                f"{winner_score:.3f} beat runner-up by {delta:.3f}."
            )
            tie_break_reason = winner.get("tie_break_reason")
            if isinstance(tie_break_reason, str) and tie_break_reason.strip():
                selection_reason += f" {tie_break_reason.strip()}"
            winner["selection_reason"] = selection_reason
        else:
            winner["selection_reason"] = "Only compositional candidate available."

        for row in rows[1:]:
            delta = winner_score - float(row.get("decision_score") or 0.0)
            reasons = [f"Lower decision score by {delta:.3f}."]
            tie_break_reason = winner.get("tie_break_reason")
            if (
                isinstance(tie_break_reason, str)
                and tie_break_reason.strip()
                and int(row.get("morph_count") or 0) > int(winner.get("morph_count") or 0)
                and delta <= self.DECISION_FEWER_MORPHS_TIE_MARGIN
            ):
                reasons.append("Winner retained by fewer-morph tie-break.")
            row["rejection_reason"] = " ".join(reasons)

    @staticmethod
    def _serialize_decision_row(row: dict[str, object]) -> dict[str, object]:
        """Serialize one decision row for debug-safe JSON output."""

        return {
            "morphs": list(row.get("morphs") or []),
            "decision_rank": int(row.get("decision_rank") or 0),
            "decision_score": float(row.get("decision_score") or 0.0),
            "base_component": float(row.get("base_component") or 0.0),
            "semantic_raw": float(row.get("semantic_raw") or 0.0),
            "semantic_component": float(row.get("semantic_component") or 0.0),
            "attestation_raw": float(row.get("attestation_raw") or 0.0),
            "attestation_component": float(row.get("attestation_component") or 0.0),
            "singleton_burden_raw": float(row.get("singleton_burden_raw") or 0.0),
            "singleton_burden_component": float(
                row.get("singleton_burden_component") or 0.0
            ),
            "morph_count": int(row.get("morph_count") or 0),
            "singleton_count": int(row.get("singleton_count") or 0),
            "tie_break_applied": bool(row.get("tie_break_applied")),
            "tie_break_reason": row.get("tie_break_reason"),
            "selection_reason": row.get("selection_reason"),
            "rejection_reason": row.get("rejection_reason"),
            "runner_up_morphs": list(row.get("runner_up_morphs") or []),
            "runner_up_score_delta": (
                float(row.get("runner_up_score_delta"))
                if isinstance(row.get("runner_up_score_delta"), numbers.Real)
                else None
            ),
        }

    def _attach_decision_traces_to_candidates(
        self,
        candidates: list[dict[str, object]],
        decision_rows: Sequence[dict[str, object]],
    ) -> None:
        """Attach decision traces to selected candidates for debug consumers.

        Why this exists:
        phrase translation and verbose CLI rendering need direct access to the
        decision math for the winning and runner-up decompositions.

        Architecture responsibility:
        maps ranked decision rows onto the selected compositional payloads.
        """

        decision_map: dict[tuple[str, ...], dict[str, object]] = {}
        for row in decision_rows:
            key = tuple(str(morph) for morph in row.get("morphs") or [])
            if key not in decision_map:
                decision_map[key] = row

        for candidate in candidates:
            morphs = candidate.get("morphs")
            key = (
                tuple(str(morph) for morph in morphs)
                if isinstance(morphs, Sequence) and not isinstance(morphs, (str, bytes))
                else ()
            )
            row = decision_map.get(key)
            if row is None:
                continue
            candidate["decision_trace"] = self._serialize_decision_row(row)

    def _enumerate_attested_full_cover_decompositions(
        self,
        word: str,
        *,
        evidence: WordEvidence,
        attested_pieces: set[str],
        allow_whole_word: bool,
        definition_counts: dict[str, int],
        definition_glosses: dict[str, list[tuple[str, float | None]]],
        evidence_mode: str,
    ) -> tuple[list[Decomposition], dict[str, object]]:
        """Guarantee a singleton-capable blind decomposition pass when needed.

        Beam search is still the preferred path because it tends to surface
        chunkier, easier-to-read decompositions. Blind phrase translation,
        however, should not stop there. When the first pass fails to yield a
        grounded full-cover compositional read, this helper enumerates every
        attested full-cover segmentation, including singleton-backed paths, so
        user-facing blind mode can stay decomposition-led instead of falling
        back to whole-word rescue.
        """

        max_partial = DEFAULT_MAX_PARTIAL_PER_INDEX
        max_full = DEFAULT_MAX_FULL_SEGMENTATIONS
        segmentations, enumerator_diag = (
            enumerate_attested_segmentations_with_diagnostics(
                word,
                attested_pieces,
                max_partial_per_index=max_partial,
                max_full_segmentations=max_full,
                min_piece_len=1,
            )
        )
        if not allow_whole_word and len(word) > 1:
            segmentations = [
                segmentation
                for segmentation in segmentations
                if not (len(segmentation) == 1 and segmentation[0] == word)
            ]

        decompositions = build_decompositions_from_segmentations(
            word,
            segmentations,
            candidate_finder=self.candidate_finder,
            evidence=evidence,
            evidence_mode=evidence_mode,
        )
        self._assign_beam_scores_to_decompositions(
            word,
            decompositions,
            evidence=evidence,
            allow_whole_word=allow_whole_word,
            definition_counts=definition_counts,
            definition_glosses=definition_glosses,
            evidence_mode=evidence_mode,
        )
        return (
            decompositions,
            {
                "enumerator_enabled": True,
                "enumerator_params": {
                    "max_partial_per_index": max_partial,
                    "max_full_segmentations": max_full,
                    "min_piece_len": 1,
                },
                "enumerated_full_count": enumerator_diag.get("enumerated_full_count", 0),
                "enumerated_full_returned": len(segmentations),
                "enumerated_partial_pruned_count": enumerator_diag.get(
                    "enumerated_partial_pruned_count",
                    0,
                ),
                "decomposition_count": len(decompositions),
            },
        )

    def _assign_beam_scores_to_decompositions(
        self,
        word: str,
        decompositions: Sequence[Decomposition],
        *,
        evidence: WordEvidence,
        allow_whole_word: bool,
        definition_counts: dict[str, int],
        definition_glosses: dict[str, list[tuple[str, float | None]]],
        evidence_mode: str,
    ) -> None:
        """Backfill beam-style priors onto enumerated blind decompositions.

        The singleton-capable fallback uses explicit attested segmentation
        enumeration, which does not carry the beam scores that the primary
        search path emits. Recomputing lightweight beam priors here lets those
        fallback decompositions compete fairly with chunkier parses.
        """

        if not decompositions:
            return

        segment_target = getattr(self.candidate_finder, "segment_target", None)
        if not callable(segment_target):
            return

        extra_ngrams = _build_evidence_ngrams(
            word,
            evidence,
            candidate_finder=self.candidate_finder,
            definition_counts=definition_counts,
            definition_glosses=definition_glosses,
            evidence_mode=evidence_mode,
        )
        try:
            parses = segment_target(
                word,
                extra_ngrams=extra_ngrams,
                definition_counts=definition_counts,
                definition_glosses=definition_glosses,
                restrict_to_attested=True,
                attested_cluster_ngrams=_collect_attested_cluster_ngrams(
                    evidence,
                    evidence_mode=evidence_mode,
                ),
            )
        except Exception:
            return

        beam_scores: dict[tuple[str, ...], float] = {}
        parsed_paths: list[tuple[tuple[str, ...], float]] = []
        all_morphs: set[str] = set()
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
            parsed_paths.append((key, float(score)))
            all_morphs.update(morphs)

        definition_candidates = extract_definition_candidates(
            all_morphs,
            evidence,
            max_per_morph=3,
            allow_dictionary=allow_whole_word,
        )
        for key, score in parsed_paths:
            penalty = compute_contradiction_penalty_for_candidates(
                key,
                definition_candidates,
            )
            adjusted = score - penalty
            existing = beam_scores.get(key)
            if existing is None or adjusted > existing:
                beam_scores[key] = adjusted

        for decomp in decompositions:
            decomp.beam_score = beam_scores.get(tuple(decomp.morphs), 0.0)
        _normalize_beam_scores(list(decompositions))

    def _apply_blind_decomposition_priority(
        self,
        *,
        word: str,
        allow_whole_word: bool,
        exact_candidates: Sequence[dict[str, object]],
        compositional_candidates: Sequence[dict[str, object]],
        provisional_candidates: Sequence[dict[str, object]],
        had_decomposition_attempts: bool,
        has_full_cover_compositional: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
        """Keep blind mode decomposition-first and reserve whole-word rescue.

        `--no-whole-word(s)` is intended to stress-test decomposition quality,
        not let provisional or whole-word reads silently retake the lead.
        Whole-word candidates therefore stay out of the main winner path when a
        usable compositional bundle already exists, but remain available as
        clearly labeled rescue evidence when decomposition still collapses into
        unresolved or meta-linguistic output.
        """

        if allow_whole_word:
            return list(exact_candidates), list(provisional_candidates), []
        if not compositional_candidates and not had_decomposition_attempts:
            return list(exact_candidates), list(provisional_candidates), []
        if has_full_cover_compositional:
            return (
                [],
                [],
                [
                    "Blind retranslation reserved whole-word readings for "
                    f"{word} because decomposition already produced a grounded full-cover candidate."
                ],
            )
        if had_decomposition_attempts:
            return (
                [],
                [],
                [
                    "Blind retranslation kept whole-word readings suppressed for "
                    f"{word} because no grounded full-cover compositional candidate survived."
                ],
            )
        return list(exact_candidates), list(provisional_candidates), []

    @staticmethod
    def _candidate_has_usable_bundle(candidate: dict[str, object]) -> bool:
        """Return whether a candidate yields a phrase-usable bundle head.

        Blind-mode rescue should only reintroduce whole-word evidence when the
        decomposition still has no human-facing lexical head. This helper keeps
        that threshold explicit and shared across word and phrase ranking.
        """

        if str(candidate.get("analysis_type") or "") != "compositional":
            return False
        if candidate_has_placeholder_gloss(candidate):
            return False
        bundle_head = str(candidate.get("bundle_head_gloss") or "").strip()
        if not bundle_head:
            return False
        if candidate_is_placeholder_anchor(candidate):
            return False
        return float(candidate.get("bundle_coherence_score") or 0.0) >= 0.25

    def _merge_candidate_pool(
        self,
        *candidate_groups: Sequence[dict[str, object]],
        top_k: int,
    ) -> list[dict[str, object]]:
        """Merge and normalize candidate groups before final ranking.

        Exact anchors, compositional parses, and provisional fallbacks all need
        to coexist in one ranked pool. This method now also demotes residual
        placeholder anchors so raw evidence diagnostics do not outrank grounded
        full-cover decompositions that are better suited for human-facing
        translation.
        """
        merged: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()

        for group in candidate_groups:
            for candidate in group:
                if not isinstance(candidate, dict):
                    continue
                meanings = candidate.get("meanings", [])
                meaning_key = tuple(
                    (
                        entry.get("morph"),
                        entry.get("definition"),
                        entry.get("provenance"),
                    )
                    for entry in meanings
                    if isinstance(entry, dict)
                )
                key = (
                    candidate.get("analysis_type"),
                    tuple(candidate.get("morphs", [])),
                    meaning_key,
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(dict(candidate))

        self._demote_placeholder_candidates(merged)
        merged.sort(
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        if top_k > 0:
            merged = merged[:top_k]
        for idx, candidate in enumerate(merged, start=1):
            candidate["rank"] = idx
        return merged

    @staticmethod
    def _demote_placeholder_candidates(candidates: list[dict[str, object]]) -> None:
        """Reduce the ranking strength of placeholder-only residual candidates.

        Whole-word anchors intentionally get large base scores so they can beat
        flimsy decompositions. That heuristic breaks down when the anchor gloss
        is only a raw residual headline, so this pass lowers both those anchors
        and any other placeholder-bearing candidate relative to grounded
        compositional full-cover reads.
        """

        grounded_full_cover = [
            candidate
            for candidate in candidates
            if SingleWordTranslationService._is_grounded_full_cover_compositional(
                candidate
            )
        ]
        if grounded_full_cover:
            floor = min(float(candidate.get("score") or 0.0) for candidate in grounded_full_cover)
            for candidate in candidates:
                if not candidate_is_placeholder_anchor(candidate):
                    continue
                candidate["score"] = min(float(candidate.get("score") or 0.0), floor - 0.01)
                warnings = list(candidate.get("warnings", []))
                if candidate_is_residual_placeholder_anchor(candidate):
                    warnings.append(
                        "Residual-only whole-word placeholder anchor demoted below grounded compositional readings."
                    )
                else:
                    warnings.append(
                        "Opaque whole-word placeholder anchor demoted below grounded compositional readings."
                    )
                candidate["warnings"] = warnings

        for candidate in candidates:
            if not candidate_has_placeholder_gloss(candidate):
                continue
            if candidate_is_placeholder_anchor(candidate):
                continue
            candidate["score"] = float(candidate.get("score") or 0.0) - 0.75
            warnings = list(candidate.get("warnings", []))
            warnings.append(
                "Placeholder residual gloss penalized relative to grounded evidence."
            )
            candidate["warnings"] = warnings

    @staticmethod
    def _is_grounded_full_cover_compositional(candidate: dict[str, object]) -> bool:
        """Return whether a candidate is a usable full-cover compositional read.

        Placeholder demotion should only trigger when a real alternative exists.
        This helper defines that threshold narrowly: a compositional candidate
        that fully covers the word and whose visible glosses are not just raw
        residual diagnostics.
        """

        if str(candidate.get("analysis_type") or "") != "compositional":
            return False
        breakdown = candidate.get("breakdown")
        if not isinstance(breakdown, dict):
            return False
        coverage = breakdown.get("coverage_ratio")
        if not isinstance(coverage, numbers.Real) or float(coverage) < 0.999:
            return False
        return not candidate_has_placeholder_gloss(candidate)

    @staticmethod
    def _full_word_breakdown(word: str) -> dict[str, object]:
        """Return a normalized full-coverage breakdown for whole-word anchors."""
        return {
            "segments": [
                {
                    "start": 0,
                    "end": len(word),
                    "ngram": word,
                    "canonical": word,
                }
            ],
            "uncovered": [],
            "coverage_ratio": 1.0,
            "residual_ratio": 0.0,
        }


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
            evidence.fasttext_neighbors = []
            return
        if mode == SingleWordTranslationService.EvidenceMode.RESIDUALS_ONLY:
            evidence.direct_clusters = []
            evidence.morph_hypotheses = []
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
            "rejected_morphs": sorted(evidence.rejected_morphs),
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
