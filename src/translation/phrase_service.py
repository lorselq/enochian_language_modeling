from __future__ import annotations

"""Phrase-level translation built on top of the single-word translator.

This module upgrades translation from isolated word glossing to a global,
phrase-level search. Each token first receives one or more algorithmic
single-word analyses; the phrase layer then scores combinations of those
analyses jointly so anchors, unknown words, and local structure can constrain
each other before any optional LLM rendering happens.
"""

import copy
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from enochian_lm.common.config import get_config_paths

from .llm_synthesis import (
    DEFAULT_LLM_CONTEXT,
    PhraseRenderBundleResult,
    PhraseRenderResult,
    _build_phrase_footnote_fallback,
    render_phrase_bundle,
    render_phrase_lay_translation,
    render_phrase_translation,
)
from .memory import TranslationMemoryRepository
from .placeholder_glosses import (
    clean_lexical_gloss,
    is_meta_linguistic_gloss,
    normalize_semantic_terms,
    sanitize_human_gloss,
    specific_gloss_from_definition_and_semantic_core,
)
from .service import SingleWordTranslationService
from .strategies import compose_semantic_bundle


_BLIND_DICTIONARY_RESCUE_NOTE = (
    "Blind mode dictionary rescue used the exact dictionary entry because the "
    "surviving decomposition gloss stayed too weak for human-facing output."
)

_FUNCTION_PROFILE_CANONICAL_GLOSSES: dict[str, str] = {
    "conjunction": "and",
    "relative": "that",
    "within_self": "within itself",
    "locative": "in",
    "imperative_existential": "let there be",
    "feminine_locative_possessive": "her",
}

_GLOSS_ALIGNMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "to",
    "upon",
    "with",
}

_WEAK_PRIMARY_RESCUE_GLOSSES = {
    "and",
    "being",
    "brightness",
    "entity",
    "existence",
    "her",
    "identity",
    "in",
    "light",
    "position",
    "power",
    "source",
    "state",
    "that",
    "thing",
    "transformation",
}


class PhraseProgressReporter(Protocol):
    """Describe the phrase-stage status interface used by the CLI renderer.

    Phrase translation performs several deterministic steps around the optional
    LLM calls. Keeping those stage updates behind a small protocol lets the
    service report progress without depending on the CLI implementation.
    """

    def stage(self, message: str, **metadata: object) -> None: ...

    def llm_status(self, event: dict[str, object]) -> None: ...

    def done(self) -> None: ...
from .tokenization import tokenize_words


@dataclass(slots=True)
class PhraseTokenCandidate:
    """Represent one phrase-level lexical reading for a token."""

    token: str
    rank: int
    analysis_type: str
    definition: str | None
    raw_definition: str | None
    alternates: list[str]
    confidence: float
    score: float
    role_hint: str
    selected_source: str = "unknown"
    definition_trace: dict[str, object] = field(default_factory=dict)
    chosen_in_parse: bool = False
    morphs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    semantic_bundle: list[dict[str, object]] = field(default_factory=list)
    bundle_surface_gloss: str | None = None
    bundle_head_gloss: str | None = None
    bundle_function_profile: str = "unknown"
    bundle_coherence_score: float = 0.0
    blind_mode_whole_word_rescue: bool = False
    blind_mode_rescue_note: str | None = None
    dictionary_rescue_gloss: str | None = None
    dictionary_rescue_note: str | None = None


@dataclass(slots=True)
class PhraseRelation:
    """Describe a lightweight structural relation between adjacent tokens."""

    left_index: int
    right_index: int
    relation: str
    direction: str
    score: float


@dataclass(slots=True)
class PhraseParseCandidate:
    """Carry one globally scored candidate parse for an input phrase."""

    rank: int
    score: float
    token_choices: list[PhraseTokenCandidate] = field(default_factory=list)
    relations: list[PhraseRelation] = field(default_factory=list)
    translation_skeleton: str = ""


class PhraseTranslationService:
    """Coordinate token analysis, global phrase search, and memory updates."""

    def __init__(
        self,
        *,
        word_service: SingleWordTranslationService,
        memory_repository: TranslationMemoryRepository,
        llm_renderer: Callable[[dict[str, object], dict[str, object]], PhraseRenderResult] = render_phrase_translation,
        lay_renderer: Callable[[dict[str, object], dict[str, object]], PhraseRenderResult] = render_phrase_lay_translation,
        bundle_renderer: Callable[[dict[str, object], dict[str, object]], PhraseRenderBundleResult] | None = None,
    ) -> None:
        self.word_service = word_service
        self.memory_repository = memory_repository
        self.llm_renderer = llm_renderer
        self.lay_renderer = lay_renderer
        if bundle_renderer is not None:
            self.bundle_renderer = bundle_renderer
        elif (
            llm_renderer is render_phrase_translation
            and lay_renderer is render_phrase_lay_translation
        ):
            self.bundle_renderer = render_phrase_bundle
        else:
            self.bundle_renderer = None

    @classmethod
    def from_config(
        cls,
        *,
        variants: Iterable[str] | None = None,
        llm_enabled: bool = False,
        llm_use_remote: bool = False,
        memory_db: Path | None = None,
    ) -> "PhraseTranslationService":
        """Build a phrase translation service from repo-configured resources."""
        paths = get_config_paths()
        return cls(
            word_service=SingleWordTranslationService.from_config(
                variants=variants,
                llm_enabled=llm_enabled,
                llm_use_remote=llm_use_remote,
            ),
            memory_repository=TranslationMemoryRepository(
                memory_db or Path(paths["translation_memory"])
            ),
        )

    def close(self) -> None:
        self.word_service.close()
        self.memory_repository.close()

    def __enter__(self) -> "PhraseTranslationService":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def translate_phrase(
        self,
        phrase: str,
        *,
        variants: Iterable[str] | None = None,
        strategy: str = "prefer-balance",
        top_k: int = 3,
        llm: bool | None = None,
        llm_context: str | None = None,
        memory_update: bool = True,
        evidence_mode: SingleWordTranslationService.EvidenceMode = SingleWordTranslationService.EvidenceMode.ALL,
        weight_enabled: bool = True,
        allow_whole_word: bool = True,
        progress_reporter: PhraseProgressReporter | None = None,
    ) -> dict[str, object]:
        """Translate a phrase and expose both technical and layman-readable outputs.

        The phrase parser still chooses candidates through deterministic token
        analysis and global parse scoring. Once that source-of-truth parse is
        selected, this method can optionally add a constrained render and now
        always tries to add a separate lay translation so downstream consumers
        can inspect the technical read and the plain-English paraphrase side by
        side.
        """
        normalized_phrase = (phrase or "").strip()
        if not normalized_phrase:
            raise ValueError("Phrase must be a non-empty string.")

        tokens = tokenize_words(normalized_phrase)
        if not tokens:
            raise ValueError("Phrase must contain at least one alphabetic token.")

        llm_enabled = self.word_service.llm_enabled if llm is None else bool(llm)
        active_variants = list(variants) if variants else self.word_service.active_variants
        candidate_matrix: list[list[PhraseTokenCandidate]] = []
        token_payloads: list[dict[str, object]] = []
        footnoted_translation = ""
        translation_footnotes: list[dict[str, object]] = []
        word_result_cache: dict[str, dict[str, object]] = {}

        self._prewarm_phrase_evidence(
            tokens=tokens,
            variants=active_variants,
            evidence_mode=evidence_mode,
        )

        self._report_progress(
            progress_reporter,
            "Preparing phrase translation...",
            stage_id="prepare",
        )
        for index, token in enumerate(tokens, start=1):
            self._report_progress(
                progress_reporter,
                f"Analyzing token {index}/{len(tokens)}: {token.upper()}",
                stage_id="token_analysis",
                current=index,
                total=len(tokens),
                token=token.upper(),
            )
            cached_result = word_result_cache.get(token.upper())
            if cached_result is None:
                cached_result = self.word_service.translate_word(
                    token,
                    variants=active_variants,
                    strategy=strategy,
                    top_k=top_k,
                    llm=False,
                    llm_context=llm_context,
                    evidence_mode=evidence_mode,
                    weight_enabled=weight_enabled,
                    allow_whole_word=allow_whole_word,
                    use_beam_search=True,
                )
                word_result_cache[token.upper()] = copy.deepcopy(cached_result)
            word_result = copy.deepcopy(cached_result)
            token_candidates = self._token_candidates_from_word_result(
                token,
                word_result,
                top_k=top_k,
                allow_whole_word=allow_whole_word,
            )
            if not token_candidates:
                if not allow_whole_word:
                    diagnostics = word_result.get("diagnostics")
                    raise RuntimeError(
                        "Blind phrase translation could not resolve token "
                        f"{token.upper()} through decomposition. Diagnostics: {diagnostics!r}"
                    )
                token_candidates = self._fallback_token_candidates(token)
            candidate_matrix.append(token_candidates)
            token_payloads.append(
                {
                    "token": token,
                    "candidates": [asdict(candidate) for candidate in token_candidates],
                    "word_result": word_result,
                }
            )

        self._report_progress(
            progress_reporter,
            "Building and scoring parse candidates...",
            stage_id="parse_search",
        )
        parse_candidates = self._build_parse_candidates(
            tokens,
            candidate_matrix,
            top_k=max(3, top_k),
        )
        chosen_parse = parse_candidates[0] if parse_candidates else None
        if chosen_parse is not None:
            self._annotate_chosen_parse_traces(
                tokens=tokens,
                token_payloads=token_payloads,
                chosen_parse=chosen_parse,
            )

        rendered_translation = chosen_parse.translation_skeleton if chosen_parse else ""
        render_reasoning = "Algorithmic phrase rendering only."
        render_confidence = 0.0
        render_warnings: list[str] = []
        lay_translation = rendered_translation
        lay_reasoning = "Lay translation fell back to the algorithmic phrase skeleton."
        lay_confidence = 0.0
        poetic_translation = ""
        poetic_reasoning = ""
        poetic_confidence: float | None = None
        interpretive_translation = ""
        interpretive_reasoning = ""
        interpretive_confidence: float | None = None
        lay_warnings: list[str] = []
        if chosen_parse is not None:
            render_confidence = self._parse_confidence(chosen_parse)
            lay_confidence = render_confidence
        render_payload = (
            self._phrase_render_payload(normalized_phrase, chosen_parse)
            if chosen_parse is not None
            else None
        )
        llm_logging_context = self._llm_logging_context(active_variants)
        if llm_enabled and render_payload is not None and self.bundle_renderer is not None:
            self._report_progress(
                progress_reporter,
                "Rendering phrase translations and footnotes...",
                stage_id="render_bundle",
            )
            bundle_context = {
                "phrase": normalized_phrase,
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "use_remote": self.word_service.llm_use_remote,
                "confidence": render_confidence,
                "progress_reporter": progress_reporter,
                "progress_label": "Rendering phrase translations and footnotes",
                "progress_stage_id": "render_bundle",
                **llm_logging_context,
            }
            bundled = self.bundle_renderer(render_payload, bundle_context)
            rendered_translation = (
                bundled.technical_translation or rendered_translation
            )
            render_reasoning = bundled.technical_reasoning
            render_confidence = bundled.technical_confidence
            render_warnings = list(bundled.warnings)
            lay_translation = bundled.lay_translation or rendered_translation
            lay_reasoning = bundled.lay_reasoning
            lay_confidence = bundled.lay_confidence
            poetic_translation = bundled.poetic_translation or bundled.interpretive_translation or ""
            poetic_reasoning = bundled.poetic_reasoning or bundled.interpretive_reasoning
            poetic_confidence = bundled.poetic_confidence or bundled.interpretive_confidence
            interpretive_translation = bundled.interpretive_translation or poetic_translation
            interpretive_reasoning = bundled.interpretive_reasoning or poetic_reasoning
            interpretive_confidence = bundled.interpretive_confidence or poetic_confidence
            lay_warnings = list(bundled.warnings)
            footnoted_translation = bundled.footnoted_translation or ""
            raw_footnotes = bundled.translation_footnotes
            translation_footnotes = (
                list(raw_footnotes)
                if isinstance(raw_footnotes, list)
                else []
            )
        else:
            if llm_enabled and chosen_parse is not None:
                self._report_progress(
                    progress_reporter,
                    "Rendering technical translation...",
                    stage_id="technical_render",
                )
                render_context = {
                    "phrase": normalized_phrase,
                    "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                    "use_remote": self.word_service.llm_use_remote,
                    "confidence": render_confidence,
                    "progress_reporter": progress_reporter,
                    "progress_label": "Rendering technical translation",
                    "progress_stage_id": "technical_render",
                    **llm_logging_context,
                }
                rendered = self.llm_renderer(
                    render_payload,
                    render_context,
                )
                rendered_translation = rendered.rendered_translation or rendered_translation
                render_reasoning = rendered.reasoning
                render_confidence = rendered.confidence
                render_warnings = list(rendered.warnings)
        if (
            llm_enabled
            and render_payload is not None
            and not (self.bundle_renderer is not None)
        ):
            self._report_progress(
                progress_reporter,
                "Rendering lay translation and footnotes...",
                stage_id="lay_render",
            )
            lay_context = {
                "phrase": normalized_phrase,
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "use_remote": self.word_service.llm_use_remote,
                "confidence": render_confidence,
                "progress_reporter": progress_reporter,
                "progress_label": "Rendering lay translation and footnotes",
                "progress_stage_id": "lay_render",
                **llm_logging_context,
            }
            lay_rendered = self.lay_renderer(
                render_payload,
                lay_context,
            )
            lay_translation = lay_rendered.rendered_translation or rendered_translation
            lay_reasoning = lay_rendered.reasoning
            lay_confidence = lay_rendered.confidence
            lay_warnings = list(lay_rendered.warnings)
            footnoted_translation = lay_rendered.footnoted_translation or ""
            raw_footnotes = lay_rendered.translation_footnotes
            translation_footnotes = (
                list(raw_footnotes)
                if isinstance(raw_footnotes, list)
                else []
            )

        if render_payload is not None and (
            not footnoted_translation.strip() or not translation_footnotes
        ):
            footnoted_translation, translation_footnotes = _build_phrase_footnote_fallback(
                render_payload
            )

        memory_updates: list[dict[str, object]] = []
        if memory_update and chosen_parse is not None:
            self._report_progress(
                progress_reporter,
                "Recording provisional memory updates...",
                stage_id="memory_updates",
            )
            for index, candidate in enumerate(chosen_parse.token_choices):
                if candidate.analysis_type != "provisional":
                    continue
                glosses = [
                    gloss for gloss in [candidate.definition, *candidate.alternates]
                    if isinstance(gloss, str) and gloss.strip()
                ]
                update = self.memory_repository.record_observation(
                    word=candidate.token,
                    phrase=normalized_phrase,
                    role_hint=candidate.role_hint,
                    glosses=glosses,
                    confidence=candidate.confidence,
                    left_neighbor=tokens[index - 1] if index > 0 else None,
                    right_neighbor=tokens[index + 1] if index + 1 < len(tokens) else None,
                )
                memory_updates.append(update)

        return {
            "phrase": normalized_phrase,
            "tokens": tokens,
            "variants_queried": active_variants or self.word_service.repository.variants,
            "strategy": strategy,
            "evidence_mode": evidence_mode.value,
            "weighting_enabled": weight_enabled,
            "llm_enabled": llm_enabled,
            "llm_mode": (
                "remote"
                if llm_enabled and self.word_service.llm_use_remote
                else "local"
                if llm_enabled
                else None
            ),
            "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_analyses": token_payloads,
            "parse_candidates": [asdict(candidate) for candidate in parse_candidates],
            "chosen_parse": asdict(chosen_parse) if chosen_parse is not None else None,
            "rendered_translation": rendered_translation,
            "render_reasoning": render_reasoning,
            "render_confidence": render_confidence,
            "render_warnings": render_warnings,
            "lay_translation": lay_translation,
            "lay_reasoning": lay_reasoning,
            "lay_confidence": lay_confidence,
            "poetic_translation": poetic_translation,
            "poetic_reasoning": poetic_reasoning,
            "poetic_confidence": poetic_confidence,
            "interpretive_translation": interpretive_translation,
            "interpretive_reasoning": interpretive_reasoning,
            "interpretive_confidence": interpretive_confidence,
            "lay_warnings": lay_warnings,
            "footnoted_translation": footnoted_translation,
            "translation_footnotes": translation_footnotes,
            "lay_translation_mode": (
                "remote"
                if llm_enabled and chosen_parse is not None and self.word_service.llm_use_remote
                else "local"
                if llm_enabled and chosen_parse is not None
                else None
            ),
            "memory_updates": memory_updates,
        }

    def _prewarm_phrase_evidence(
        self,
        *,
        tokens: Sequence[str],
        variants: Sequence[str] | None,
        evidence_mode: SingleWordTranslationService.EvidenceMode,
    ) -> None:
        """Warm repository caches for the full phrase before token analysis.

        Phrase translation asks for overlapping substring evidence across many
        tokens. Priming that cache once reduces repeated repository work while
        keeping the shared single-word translator as the source of truth.
        """

        repository = getattr(self.word_service, "repository", None)
        prewarm = getattr(repository, "prewarm_translation_morphs", None)
        substring_candidates = getattr(self.word_service, "_substring_candidates", None)
        prewarm_clustered_counts = getattr(
            self.word_service,
            "_prewarm_clustered_definition_counts",
            None,
        )
        if not callable(prewarm) or not callable(substring_candidates):
            return

        morphs: set[str] = set()
        for token in tokens:
            morphs.add(token.upper())
            morphs.update(substring_candidates(token, include_singletons=True))
        prewarm(
            morphs,
            variants=variants,
            include_clusters=evidence_mode != self.word_service.EvidenceMode.RESIDUALS_ONLY,
            include_residuals=evidence_mode != self.word_service.EvidenceMode.CLUSTERS_ONLY,
        )
        if callable(prewarm_clustered_counts):
            prewarm_clustered_counts(
                morphs,
                variants=variants,
                evidence_mode=evidence_mode,
            )

    def _token_candidates_from_word_result(
        self,
        token: str,
        result: dict[str, object],
        *,
        top_k: int,
        allow_whole_word: bool,
    ) -> list[PhraseTokenCandidate]:
        """Convert single-word results into phrase-level candidate objects."""
        candidates_raw = result.get("candidates")
        candidates = candidates_raw if isinstance(candidates_raw, list) else []
        converted: list[PhraseTokenCandidate] = []
        for candidate in candidates[: max(1, top_k)]:
            if not isinstance(candidate, dict):
                continue
            meanings = candidate.get("meanings", [])
            meaning_list = meanings if isinstance(meanings, list) else []
            selected_trace = self._candidate_definition_trace(meaning_list)
            bundle = self._candidate_bundle(candidate, meaning_list)
            selected_trace = {
                **selected_trace,
                "bundle_semantic_cores": [
                    list(entry.get("semantic_core_terms") or [])
                    for entry in bundle.get("semantic_bundle", [])
                    if isinstance(entry, Mapping)
                ],
                "bundle_negative_contrast": [
                    list(entry.get("negative_contrast") or [])
                    for entry in bundle.get("semantic_bundle", [])
                    if isinstance(entry, Mapping)
                ],
                "bundle_surface_candidates": list(
                    bundle.get("bundle_surface_candidates") or []
                ),
                "bundle_selection_reason": str(
                    bundle.get("bundle_selection_reason")
                    or selected_trace.get("bundle_selection_reason")
                    or ""
                ).strip(),
            }
            primary_definition = self._candidate_definition(
                candidate,
                meaning_list,
                bundle,
            )
            raw_definition = self._candidate_raw_definition(meaning_list)
            alternates = self._candidate_alternates(
                meaning_list,
                selected_trace=selected_trace,
            )
            warnings = [
                str(warning)
                for warning in candidate.get("warnings", [])
                if isinstance(warning, str)
            ]
            if primary_definition is None and raw_definition is not None:
                warnings.append(
                    "Suppressed placeholder residual gloss in phrase output."
                )
            blind_mode_rescue_note = str(
                candidate.get("blind_mode_rescue_note") or ""
            ).strip() or None
            if blind_mode_rescue_note and blind_mode_rescue_note not in warnings:
                warnings.append(blind_mode_rescue_note)
            dictionary_rescue_gloss = self._dictionary_rescue_gloss(
                candidate,
            )
            dictionary_rescue_note = str(
                candidate.get("dictionary_rescue_note")
                or selected_trace.get("dictionary_rescue_note")
                or ""
            ).strip() or None
            role_hint = self._infer_role_hint(
                token,
                candidate,
                primary_definition or raw_definition,
                bundle=bundle,
                meanings=meaning_list,
            )
            phrase_candidate = PhraseTokenCandidate(
                token=token,
                rank=int(candidate.get("rank") or 0),
                analysis_type=str(candidate.get("analysis_type") or "compositional"),
                definition=primary_definition,
                raw_definition=raw_definition,
                alternates=alternates,
                confidence=float(candidate.get("confidence") or 0.0),
                score=float(candidate.get("score") or 0.0),
                role_hint=role_hint,
                selected_source=self._candidate_selected_source(meaning_list),
                definition_trace=selected_trace,
                morphs=[
                    str(morph)
                    for morph in candidate.get("morphs", [])
                    if isinstance(morph, str)
                ],
                warnings=warnings,
                semantic_bundle=list(bundle.get("semantic_bundle") or []),
                bundle_surface_gloss=(
                    str(bundle.get("bundle_surface_gloss") or "").strip() or None
                ),
                bundle_head_gloss=(
                    str(bundle.get("bundle_head_gloss") or "").strip() or None
                ),
                bundle_function_profile=str(
                    bundle.get("bundle_function_profile") or "unknown"
                ),
                bundle_coherence_score=float(
                    bundle.get("bundle_coherence_score") or 0.0
                ),
                blind_mode_whole_word_rescue=bool(
                    candidate.get("blind_mode_whole_word_rescue")
                ),
                blind_mode_rescue_note=blind_mode_rescue_note,
                dictionary_rescue_gloss=dictionary_rescue_gloss,
                dictionary_rescue_note=dictionary_rescue_note,
            )
            if (
                not allow_whole_word
                and phrase_candidate.blind_mode_whole_word_rescue
            ):
                continue
            if (
                not allow_whole_word
                and self._human_facing_candidate_gloss(phrase_candidate) is None
            ):
                continue
            converted.append(phrase_candidate)
        return converted

    def _dictionary_rescue_gloss(
        self,
        candidate: Mapping[str, object],
    ) -> str | None:
        """Preserve any already-selected diagnostic rescue gloss without recomputing it."""

        raw_rescue = candidate.get("dictionary_rescue_gloss")
        if not isinstance(raw_rescue, str) or not raw_rescue.strip():
            return None
        normalized = self._normalize_dictionary_rescue_gloss(raw_rescue.strip())
        return normalized or None

    @classmethod
    def _should_use_dictionary_rescue(
        cls,
        *,
        token: str,
        primary_definition: str | None,
        rescue_gloss: str,
        bundle: Mapping[str, object],
        selected_trace: Mapping[str, object],
    ) -> bool:
        """Decide when an exact dictionary gloss beats a weak compositional gloss.

        The rescue path should stay conservative: if the surviving gloss is
        already strong, we leave it alone. We only promote the exact dictionary
        entry when the current gloss is missing, just a normalized function
        label like ``in``, or obviously aligns worse with the selected semantic
        core than the exact dictionary reading does.
        """

        cleaned_primary = cls._human_facing_definition(primary_definition, token=token)
        if cleaned_primary is None:
            return True

        normalized_primary = cleaned_primary.strip().lower()
        normalized_rescue = rescue_gloss.strip().lower()
        if not normalized_rescue or normalized_primary == normalized_rescue:
            return False

        bundle_profile = str(bundle.get("bundle_function_profile") or "").strip()
        canonical_gloss = _FUNCTION_PROFILE_CANONICAL_GLOSSES.get(bundle_profile)
        if canonical_gloss and normalized_primary == canonical_gloss:
            return normalized_rescue != canonical_gloss

        primary_content_tokens = cls._meaningful_gloss_token_count(cleaned_primary)
        rescue_content_tokens = cls._meaningful_gloss_token_count(rescue_gloss)
        bundle_selection_reason = str(
            selected_trace.get("bundle_selection_reason") or ""
        ).strip().lower()
        semantic_bundle = bundle.get("semantic_bundle")
        semantic_bundle_count = (
            len(semantic_bundle)
            if isinstance(semantic_bundle, Sequence)
            and not isinstance(semantic_bundle, (str, bytes))
            else 0
        )
        if "function gloss" in bundle_selection_reason:
            if rescue_content_tokens > primary_content_tokens:
                return True
            if (
                semantic_bundle_count > 1
                and len(rescue_gloss.split()) > len(cleaned_primary.split())
            ):
                return True

        if normalized_primary in _WEAK_PRIMARY_RESCUE_GLOSSES:
            if rescue_content_tokens >= primary_content_tokens:
                return True
            if len(rescue_gloss.split()) > len(cleaned_primary.split()):
                return True

        semantic_terms = normalize_semantic_terms(
            selected_trace.get("selected_semantic_core")
        )
        if semantic_terms:
            primary_score = cls._semantic_alignment_score(cleaned_primary, semantic_terms)
            rescue_score = cls._semantic_alignment_score(rescue_gloss, semantic_terms)
            if rescue_score > primary_score + 0.15:
                return True
            if (
                rescue_score + 0.05 >= primary_score
                and len(rescue_gloss.split()) > len(cleaned_primary.split())
            ):
                return True

        return float(bundle.get("bundle_coherence_score") or 0.0) < 0.35

    @staticmethod
    def _meaningful_gloss_token_count(text: str) -> int:
        """Count content-bearing gloss tokens after discarding stopword filler.

        Render-only dictionary rescue should promote exact word entries when the
        surviving decomposition has collapsed into a tiny function-like label.
        Comparing only content-bearing tokens gives the phrase layer a cheap
        way to notice that `circle of stars` carries more lexical substance
        than `angel` or `in`, even if the raw word counts are noisy.
        """

        return len(PhraseTranslationService._gloss_alignment_tokens(text))

    @staticmethod
    def _semantic_alignment_score(gloss: str, semantic_terms: Sequence[str]) -> float:
        """Estimate whether a gloss points at the same idea as the semantic core."""

        gloss_tokens = PhraseTranslationService._gloss_alignment_tokens(gloss)
        if not gloss_tokens:
            return 0.0
        semantic_tokens: set[str] = set()
        for term in semantic_terms:
            semantic_tokens.update(PhraseTranslationService._gloss_alignment_tokens(term))
        if not semantic_tokens:
            return 0.0
        overlap = gloss_tokens & semantic_tokens
        if not overlap:
            return 0.0
        return len(overlap) / float(len(gloss_tokens | semantic_tokens))

    @classmethod
    def _weak_compositional_gloss(cls, candidate: Mapping[str, object]) -> str | None:
        """Build a best-effort gloss from concatenated morph meanings.

        Some blind-mode candidates still carry readable submorph meanings even
        when no single clean head gloss survives. This helper turns the word
        layer's surviving semantic bundle into a compact weak fallback so
        phrase output can say something grounded before giving up entirely.
        """

        cleaned_parts: list[str] = []
        seen: set[str] = set()

        def add_piece(raw_piece: object) -> None:
            if not isinstance(raw_piece, str) or not raw_piece.strip():
                return
            cleaned = cls._human_facing_definition(raw_piece.strip())
            if cleaned is None:
                return
            lowered = cleaned.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            cleaned_parts.append(cleaned)

        semantic_bundle = candidate.get("semantic_bundle")
        if isinstance(semantic_bundle, Sequence) and not isinstance(
            semantic_bundle,
            (str, bytes),
        ):
            for entry in semantic_bundle:
                if not isinstance(entry, Mapping):
                    continue
                add_piece(entry.get("head_gloss") or entry.get("surface_gloss"))

        meanings = candidate.get("meanings")
        if isinstance(meanings, Sequence) and not isinstance(meanings, (str, bytes)):
            for meaning in meanings:
                if not isinstance(meaning, Mapping):
                    continue
                add_piece(meaning.get("surface_gloss") or meaning.get("definition"))

        raw_concatenation = candidate.get("concatenated_meanings")
        if isinstance(raw_concatenation, str) and raw_concatenation.strip():
            for piece in raw_concatenation.split("+"):
                add_piece(piece.strip())

        if not cleaned_parts:
            return None
        if len(cleaned_parts) == 1:
            return cleaned_parts[0]
        return " ".join(cleaned_parts[:3]).strip()

    @staticmethod
    def _gloss_alignment_tokens(text: str) -> set[str]:
        """Tokenize a gloss for lightweight lexical alignment checks."""

        return {
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z'-]*", text)
            if token.lower() not in _GLOSS_ALIGNMENT_STOPWORDS
        }

    @staticmethod
    def _normalize_dictionary_rescue_gloss(gloss: str) -> str:
        """Flatten exact-dictionary gloss punctuation into readable phrase text.

        Exact dictionary entries are the most honest rescue source for blind
        mode, but some are written with editorial parentheses such as
        ``(within) her``. Phrase output should keep that meaning while removing
        the punctuation noise that would otherwise distract from the explicit
        rescue note.
        """

        normalized = re.sub(r"\(([^()]*)\)", r"\1", gloss)
        return " ".join(normalized.split()).strip(" ,;-")

    @staticmethod
    def _human_facing_unresolved_gloss() -> str:
        """Return the phrase-safe fallback when no grounded gloss survives.

        The final translation already carries source-token visibility in the
        parse report and footnotes. Surfacing bracketed raw tokens in the
        English line itself makes the sentence look broken, so the phrase layer
        uses one explicit generic label instead.
        """

        return "unresolved term"

    def _fallback_token_candidates(self, token: str) -> list[PhraseTokenCandidate]:
        """Provide an opaque fallback when no algorithmic word analysis exists."""
        memory_entry = self.memory_repository.fetch_entry(token)
        if memory_entry is not None:
            return [
                PhraseTokenCandidate(
                    token=token,
                    rank=1,
                    analysis_type="provisional",
                    definition=memory_entry.best_gloss,
                    raw_definition=memory_entry.best_gloss,
                    alternates=list(memory_entry.alternates),
                    confidence=memory_entry.confidence,
                    score=30.0 + memory_entry.evidence_count,
                    role_hint=memory_entry.role_hint or "unknown",
                    selected_source="memory",
                    definition_trace={
                        "selected_definition": memory_entry.best_gloss,
                        "selected_source": "memory",
                        "selected_quality": memory_entry.confidence,
                        "runner_ups": [
                            {
                                "definition": alternate,
                                "source": "memory",
                                "quality": memory_entry.confidence,
                            }
                            for alternate in memory_entry.alternates[:3]
                        ],
                        "suppressed": [],
                        "blind_dictionary_fallback": False,
                    },
                    morphs=[token],
                    warnings=["Recovered from translation memory."],
                )
            ]
        return [
            PhraseTokenCandidate(
                token=token,
                rank=1,
                analysis_type="provisional",
                definition=None,
                raw_definition=None,
                alternates=[],
                confidence=0.1,
                score=1.0,
                role_hint="unknown",
                selected_source="unknown",
                definition_trace={
                    "selected_definition": None,
                    "selected_source": "unknown",
                    "selected_quality": 0.0,
                    "runner_ups": [],
                    "suppressed": [],
                    "blind_dictionary_fallback": False,
                },
                morphs=[token],
                warnings=["No supported lexical analysis yet."],
            )
        ]

    def _build_parse_candidates(
        self,
        tokens: Sequence[str],
        candidate_matrix: Sequence[Sequence[PhraseTokenCandidate]],
        *,
        top_k: int,
    ) -> list[PhraseParseCandidate]:
        """Beam-search whole-phrase candidate parses from token analyses."""
        beams: list[dict[str, object]] = [
            {"score": 0.0, "token_choices": [], "relations": []}
        ]
        beam_width = max(6, top_k * 4)

        for index, token_candidates in enumerate(candidate_matrix):
            next_beams: list[dict[str, object]] = []
            for beam in beams:
                chosen = beam["token_choices"]
                previous = chosen[-1] if chosen else None
                for candidate in token_candidates:
                    relations = list(beam["relations"])
                    score = (
                        float(beam["score"])
                        + candidate.score
                        + (candidate.confidence * 5.0)
                        + (candidate.bundle_coherence_score * 2.0)
                    )
                    if previous is not None:
                        relation = self._relation_between(previous, candidate, index - 1, index)
                        relations.append(relation)
                        score += relation.score
                    if candidate.analysis_type == "provisional" and not candidate.definition:
                        score -= 2.0
                    if candidate.blind_mode_whole_word_rescue:
                        score -= 0.35
                    next_beams.append(
                        {
                            "score": score,
                            "token_choices": [*chosen, candidate],
                            "relations": relations,
                        }
                    )
            next_beams.sort(key=lambda item: float(item["score"]), reverse=True)
            beams = next_beams[:beam_width]

        parses: list[PhraseParseCandidate] = []
        for idx, beam in enumerate(beams[: max(1, top_k)], start=1):
            token_choices = list(beam["token_choices"])
            relations = list(beam["relations"])
            score = float(beam["score"])
            unresolved = sum(
                1 for candidate in token_choices
                if candidate.analysis_type == "provisional" and not candidate.definition
            )
            score -= unresolved * 1.5
            skeleton = self._translation_skeleton(tokens, token_choices, relations)
            parses.append(
                PhraseParseCandidate(
                    rank=idx,
                    score=score,
                    token_choices=token_choices,
                    relations=relations,
                    translation_skeleton=skeleton,
                )
            )
        parses.sort(key=lambda item: item.score, reverse=True)
        for idx, parse in enumerate(parses, start=1):
            parse.rank = idx
        return parses

    def _relation_between(
        self,
        left: PhraseTokenCandidate,
        right: PhraseTokenCandidate,
        left_index: int,
        right_index: int,
    ) -> PhraseRelation:
        """Infer a lightweight relation between adjacent token candidates."""
        if left.bundle_function_profile == "conjunction":
            relation = "coordination/apposition"
            direction = "left_to_right"
            score = 0.9
        elif right.bundle_function_profile in {
            "relative",
            "locative",
            "within_self",
            "feminine_locative_possessive",
        }:
            relation = "relational_attachment"
            direction = "left_to_right"
            score = 1.3
        elif left.bundle_function_profile == "imperative_existential":
            relation = "predicate-argument"
            direction = "left_to_right"
            score = 1.4
        elif "relational" in {left.role_hint, right.role_hint}:
            relation = "relational_attachment"
            direction = "left_to_right" if left.role_hint == "relational" else "right_to_left"
            score = 1.2
        elif left.role_hint == "modifier" and right.role_hint in {"noun", "verb"}:
            relation = "head-modifier"
            direction = "left_to_right"
            score = 0.9
        elif left.role_hint == "verb" and right.role_hint in {"noun", "unknown"}:
            relation = "predicate-argument"
            direction = "left_to_right"
            score = 1.1
        elif left.role_hint in {"noun", "unknown"} and right.role_hint == "verb":
            relation = "predicate-argument"
            direction = "right_to_left"
            score = 0.8
        elif left.role_hint == right.role_hint == "noun":
            relation = "coordination/apposition"
            direction = "left_to_right"
            score = 0.4
        else:
            relation = "coordination/apposition"
            direction = "left_to_right"
            score = 0.1
        return PhraseRelation(
            left_index=left_index,
            right_index=right_index,
            relation=relation,
            direction=direction,
            score=score,
        )

    def _infer_role_hint(
        self,
        token: str,
        candidate: dict[str, object],
        primary_definition: str | None,
        *,
        bundle: Mapping[str, object],
        meanings: Sequence[object],
    ) -> str:
        """Infer a coarse syntactic role for a token candidate."""
        dictionary_entry = self.word_service.candidate_finder.dictionary.get(token.lower())
        if dictionary_entry is not None:
            pos = dictionary_entry.get("pos")
            normalized = self._normalize_role_from_pos(pos)
            if normalized is not None:
                return normalized

        bundle_profile = str(bundle.get("bundle_function_profile") or "").strip()
        if bundle_profile == "conjunction":
            return "relational"
        if bundle_profile in {
            "relative",
            "locative",
            "within_self",
            "feminine_locative_possessive",
        }:
            return "relational"
        if bundle_profile == "imperative_existential":
            return "verb"

        definition = (primary_definition or "").strip().lower()
        if not definition:
            for meaning in meanings:
                if not isinstance(meaning, Mapping):
                    continue
                raw_definition = meaning.get("raw_definition")
                if isinstance(raw_definition, str) and raw_definition.strip().lower().startswith("to "):
                    return "verb"
            return "unknown"
        if definition.startswith("to "):
            return "verb"
        for meaning in meanings:
            if not isinstance(meaning, Mapping):
                continue
            raw_definition = meaning.get("raw_definition")
            if isinstance(raw_definition, str) and raw_definition.strip().lower().startswith("to "):
                return "verb"
        if definition.startswith(("in ", "on ", "with ", "from ", "unto ", "among", "between ", "through ")):
            return "relational"
        if any(marker in definition for marker in (" of ", " among", " within", " between", " through")):
            return "relational"
        if any(marker in definition for marker in ("-like", "able ", "ing ", "ed ")):
            return "modifier"
        analysis_type = str(candidate.get("analysis_type") or "")
        if analysis_type == "dictionary_exact":
            return "noun"
        return "noun"

    @staticmethod
    def _normalize_role_from_pos(pos: object) -> str | None:
        if not isinstance(pos, str):
            return None
        normalized = pos.strip().lower()
        if not normalized:
            return None
        if "verb" in normalized or normalized == "v":
            return "verb"
        if "adj" in normalized or "adv" in normalized or "mod" in normalized:
            return "modifier"
        if "prep" in normalized or "rel" in normalized or "conj" in normalized:
            return "relational"
        if "noun" in normalized or normalized in {"n", "pronoun", "pron"}:
            return "noun"
        return None

    @staticmethod
    def _candidate_bundle(
        candidate: Mapping[str, object],
        meanings: Sequence[object],
    ) -> dict[str, object]:
        """Return bundle metadata for one serialized token candidate.

        Phrase translation consumes real word-service payloads and lightweight
        test doubles. Deriving bundle data on the fly when it is absent keeps
        the phrase layer compatible with both shapes while still relying on the
        same ordered semantic-bundle model.
        """

        semantic_bundle = candidate.get("semantic_bundle")
        if isinstance(semantic_bundle, Sequence) and not isinstance(semantic_bundle, (str, bytes)):
            return {
                "semantic_bundle": list(semantic_bundle),
                "bundle_surface_gloss": candidate.get("bundle_surface_gloss"),
                "bundle_head_gloss": candidate.get("bundle_head_gloss"),
                "bundle_function_profile": candidate.get("bundle_function_profile"),
                "bundle_coherence_score": candidate.get("bundle_coherence_score"),
                "bundle_surface_candidates": candidate.get("bundle_surface_candidates"),
                "bundle_selection_reason": candidate.get("bundle_selection_reason"),
            }
        normalized_meanings = [
            meaning for meaning in meanings if isinstance(meaning, Mapping)
        ]
        return compose_semantic_bundle(normalized_meanings)

    @staticmethod
    def _candidate_definition(
        candidate: Mapping[str, object],
        meanings: Sequence[object],
        bundle: Mapping[str, object],
    ) -> str | None:
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            semantic_core = meaning.get("semantic_core") or meaning.get(
                "semantic_core_terms"
            )
            negative_contrast = meaning.get("negative_contrast")
            surface_gloss = meaning.get("surface_gloss")
            if isinstance(surface_gloss, str) and surface_gloss.strip():
                specific = specific_gloss_from_definition_and_semantic_core(
                    semantic_core=semantic_core,
                    definition=surface_gloss,
                    negative_contrast=negative_contrast,
                )
                cleaned = PhraseTranslationService._human_facing_definition(
                    specific or surface_gloss
                )
                if cleaned is not None:
                    return cleaned
            definition = meaning.get("definition")
            if isinstance(definition, str) and definition.strip():
                specific = specific_gloss_from_definition_and_semantic_core(
                    semantic_core=semantic_core,
                    definition=definition,
                    negative_contrast=negative_contrast,
                )
                cleaned = PhraseTranslationService._human_facing_definition(
                    specific or definition
                )
                if cleaned is not None:
                    return cleaned
        bundle_surface = bundle.get("bundle_surface_gloss")
        if isinstance(bundle_surface, str) and bundle_surface.strip():
            cleaned_bundle_surface = PhraseTranslationService._human_facing_definition(
                bundle_surface
            )
            if cleaned_bundle_surface is not None:
                return cleaned_bundle_surface
        bundle_head = bundle.get("bundle_head_gloss")
        if isinstance(bundle_head, str) and bundle_head.strip():
            cleaned_bundle_head = PhraseTranslationService._human_facing_definition(
                bundle_head
            )
            if cleaned_bundle_head is not None:
                return cleaned_bundle_head
        weak_concatenation = PhraseTranslationService._weak_compositional_gloss(candidate)
        if weak_concatenation is not None:
            return weak_concatenation
        return None

    @staticmethod
    def _candidate_raw_definition(meanings: Sequence[object]) -> str | None:
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            raw_definition = meaning.get("raw_definition")
            if isinstance(raw_definition, str) and raw_definition.strip():
                return raw_definition.strip()
        return None

    @classmethod
    def _candidate_alternates(
        cls,
        meanings: Sequence[object],
        *,
        selected_trace: Mapping[str, object] | None = None,
    ) -> list[str]:
        alternates: list[str] = []
        seen: set[str] = set()
        trace = dict(selected_trace or cls._candidate_definition_trace(meanings))
        selected_definition = trace.get("selected_definition")
        runner_ups = trace.get("runner_ups")
        if isinstance(runner_ups, Sequence) and not isinstance(runner_ups, (str, bytes)):
            for runner_up in runner_ups:
                if not isinstance(runner_up, Mapping):
                    continue
                definition = cls._human_facing_definition(runner_up.get("definition"))
                if not definition or definition == selected_definition or definition in seen:
                    continue
                seen.add(definition)
                alternates.append(definition)
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            definitions = meaning.get("definitions")
            if not isinstance(definitions, list):
                continue
            for definition in definitions:
                if not isinstance(definition, str):
                    continue
                normalized = cls._human_facing_definition(definition)
                if (
                    not normalized
                    or normalized == selected_definition
                    or normalized in seen
                ):
                    continue
                seen.add(normalized)
                alternates.append(normalized)
        return alternates

    @staticmethod
    def _candidate_selected_source(meanings: Sequence[object]) -> str:
        for meaning in meanings:
            if not isinstance(meaning, Mapping):
                continue
            provenance = meaning.get("provenance")
            if isinstance(provenance, str) and provenance.strip():
                return provenance.strip()
        return "unknown"

    @staticmethod
    def _candidate_definition_trace(meanings: Sequence[object]) -> dict[str, object]:
        for meaning in meanings:
            if not isinstance(meaning, Mapping):
                continue
            trace = meaning.get("definition_trace")
            if isinstance(trace, Mapping):
                return dict(trace)
        return {}

    @staticmethod
    def _human_facing_definition(
        definition: str | None,
        *,
        token: str | None = None,
    ) -> str | None:
        """Filter raw diagnostic residue out of phrase-facing token glosses.

        The word translator still surfaces raw residual headlines for debugging,
        but the phrase layer should only pass through glosses a human can read
        as translations. Centralizing that cleanup here keeps the chosen parse
        explicit when evidence is weak instead of echoing diagnostic strings.
        """
        cleaned = sanitize_human_gloss(definition, token=token)
        if cleaned is None:
            return None
        if PhraseTranslationService._is_phrase_meta_gloss(cleaned):
            return None
        return cleaned

    @staticmethod
    def _is_phrase_meta_gloss(definition: str) -> bool:
        """Filter lexicographic/meta senses out of phrase-facing token output.

        Phrase translation needs meanings that can participate in a sentence.
        Definitions like "the Enochian word for the digits 24" may be useful
        dictionary diagnostics, but they should not outrank lexical senses in
        normal phrase rendering.
        """

        lowered = " ".join(definition.lower().split())
        if is_meta_linguistic_gloss(lowered):
            return True
        numeric_markers = (
            "enochian word for",
            "digits ",
            "number ",
            "numeral ",
            "letter ",
        )
        return any(marker in lowered for marker in numeric_markers)

    @staticmethod
    def _human_facing_candidate_gloss(candidate: PhraseTokenCandidate) -> str | None:
        """Return the best phrase-safe gloss already carried by a token candidate."""

        for value in (
            candidate.definition,
            candidate.bundle_surface_gloss,
            candidate.bundle_head_gloss,
            *candidate.alternates,
        ):
            cleaned = PhraseTranslationService._human_facing_definition(value)
            if cleaned is not None:
                return cleaned
        for entry in candidate.semantic_bundle:
            if not isinstance(entry, Mapping):
                continue
            semantic_gloss = specific_gloss_from_definition_and_semantic_core(
                semantic_core=entry.get("semantic_core_terms"),
                definition=entry.get("surface_gloss") or entry.get("head_gloss"),
                negative_contrast=entry.get("negative_contrast"),
            )
            cleaned = PhraseTranslationService._human_facing_definition(
                semantic_gloss
                or (
                    str(entry.get("surface_gloss") or entry.get("head_gloss") or "").strip()
                    or None
                )
            )
            if cleaned is not None:
                return cleaned
        return None

    def _annotate_chosen_parse_traces(
        self,
        *,
        tokens: Sequence[str],
        token_payloads: list[dict[str, object]],
        chosen_parse: PhraseParseCandidate,
    ) -> None:
        """Attach parse-aware trace context to the winning token candidates.

        Token analysis happens before the phrase beam search picks a winner, so
        the raw token payloads do not yet know which candidate survived or what
        local relation structure helped it win. This pass stitches that context
        back onto the chosen candidates for verbose output and footnotes.
        """

        for index, candidate in enumerate(chosen_parse.token_choices):
            relation_context = self._relation_context_for_index(
                chosen_parse=chosen_parse,
                tokens=tokens,
                index=index,
            )
            trace = dict(candidate.definition_trace)
            trace["relation_context"] = relation_context
            trace["selection_reason"] = self._selection_reason_for_candidate(
                candidate=candidate,
                token_payload=token_payloads[index] if index < len(token_payloads) else None,
                relation_context=relation_context,
            )
            candidate.definition_trace = trace
            candidate.chosen_in_parse = True

            if index >= len(token_payloads):
                continue
            token_candidates = token_payloads[index].get("candidates")
            if not isinstance(token_candidates, list):
                continue
            for token_candidate in token_candidates:
                if not isinstance(token_candidate, dict):
                    continue
                token_candidate["chosen_in_parse"] = self._candidate_matches_choice(
                    token_candidate,
                    candidate,
                )
                if not token_candidate["chosen_in_parse"]:
                    continue
                token_trace = token_candidate.get("definition_trace")
                merged_trace = dict(token_trace) if isinstance(token_trace, Mapping) else {}
                merged_trace["relation_context"] = relation_context
                merged_trace["selection_reason"] = trace["selection_reason"]
                token_candidate["definition_trace"] = merged_trace

    @staticmethod
    def _candidate_matches_choice(
        token_candidate: Mapping[str, object],
        choice: PhraseTokenCandidate,
    ) -> bool:
        """Match a serialized token candidate to the chosen parse candidate."""

        rank = token_candidate.get("rank")
        if isinstance(rank, int) and rank != choice.rank:
            return False
        analysis_type = str(token_candidate.get("analysis_type") or "")
        morphs = token_candidate.get("morphs")
        serialized_morphs = [
            str(morph) for morph in morphs
        ] if isinstance(morphs, Sequence) and not isinstance(morphs, (str, bytes)) else []
        return (
            analysis_type == choice.analysis_type
            and serialized_morphs == list(choice.morphs)
        )

    @staticmethod
    def _relation_context_for_index(
        *,
        chosen_parse: PhraseParseCandidate,
        tokens: Sequence[str],
        index: int,
    ) -> dict[str, object]:
        """Summarize the adjacent parse relations for one chosen token."""

        context: dict[str, object] = {}
        if index > 0 and index - 1 < len(chosen_parse.relations):
            relation = chosen_parse.relations[index - 1]
            context["left"] = {
                "neighbor_token": tokens[index - 1].upper(),
                "relation": relation.relation,
                "direction": relation.direction,
                "score": relation.score,
            }
        if index < len(chosen_parse.relations):
            relation = chosen_parse.relations[index]
            if index + 1 < len(tokens):
                context["right"] = {
                    "neighbor_token": tokens[index + 1].upper(),
                    "relation": relation.relation,
                    "direction": relation.direction,
                    "score": relation.score,
                }
        return context

    @staticmethod
    def _selection_reason_for_candidate(
        *,
        candidate: PhraseTokenCandidate,
        token_payload: Mapping[str, object] | None,
        relation_context: Mapping[str, object],
    ) -> str:
        """Summarize why the chosen parse preferred this token reading."""

        parts: list[str] = []
        trace = candidate.definition_trace
        selected_source = str(trace.get("selected_source") or candidate.selected_source or "unknown")
        surface_gloss_strategy = str(trace.get("surface_gloss_strategy") or "").strip()
        selected_semantic_core = trace.get("selected_semantic_core")
        if isinstance(selected_semantic_core, Sequence) and not isinstance(
            selected_semantic_core, (str, bytes)
        ):
            semantic_terms = [str(term).strip() for term in selected_semantic_core if str(term).strip()]
            if semantic_terms:
                parts.append(
                    "Surface gloss came from semantic_core: "
                    + ", ".join(semantic_terms[:3])
                    + "."
                )
        elif surface_gloss_strategy == "cleaned_definition":
            parts.append("Surface gloss came from the cleaned lexical definition.")
        elif surface_gloss_strategy == "subroot_estimate":
            parts.append("Surface gloss came from the best surviving subroot estimate.")
        if selected_source != "unknown":
            parts.append(f"Selected {selected_source}-backed evidence survived for this token.")
        bundle_selection_reason = str(trace.get("bundle_selection_reason") or "").strip()
        if bundle_selection_reason:
            parts.append(bundle_selection_reason)
        negative_contrast_penalties = trace.get("negative_contrast_penalties")
        if isinstance(negative_contrast_penalties, Sequence) and not isinstance(
            negative_contrast_penalties, (str, bytes)
        ):
            penalties = [str(item).strip() for item in negative_contrast_penalties if str(item).strip()]
            if penalties:
                parts.append(
                    "Negative contrast penalized overlap with: "
                    + ", ".join(penalties[:3])
                    + "."
                )
        suppressed = trace.get("suppressed")
        if isinstance(suppressed, Sequence) and not isinstance(suppressed, (str, bytes)):
            for note in suppressed:
                if isinstance(note, str) and note.strip():
                    parts.append(note.strip())
                    break
        blind_mode_rescue_note = str(
            trace.get("blind_mode_rescue_note") or candidate.blind_mode_rescue_note or ""
        ).strip()
        if blind_mode_rescue_note:
            parts.append(blind_mode_rescue_note)
        runner_up = PhraseTranslationService._runner_up_candidate(token_payload, candidate)
        if runner_up is not None:
            parts.append(
                "Candidate score "
                f"{candidate.score:.2f} beat runner-up {float(runner_up.get('score') or 0.0):.2f}."
            )
        relation_bits: list[str] = []
        left = relation_context.get("left")
        if isinstance(left, Mapping):
            relation_bits.append(
                f"left relation {left.get('relation')} with {left.get('neighbor_token')}"
            )
        right = relation_context.get("right")
        if isinstance(right, Mapping):
            relation_bits.append(
                f"right relation {right.get('relation')} with {right.get('neighbor_token')}"
            )
        if relation_bits:
            parts.append("Parse context: " + "; ".join(relation_bits) + ".")
        return " ".join(parts).strip()

    @staticmethod
    def _runner_up_candidate(
        token_payload: Mapping[str, object] | None,
        candidate: PhraseTokenCandidate,
    ) -> Mapping[str, object] | None:
        if token_payload is None:
            return None
        candidates = token_payload.get("candidates")
        if not isinstance(candidates, list):
            return None
        for token_candidate in candidates:
            if not isinstance(token_candidate, Mapping):
                continue
            if PhraseTranslationService._candidate_matches_choice(token_candidate, candidate):
                continue
            return token_candidate
        return None

    @staticmethod
    def _report_progress(
        progress_reporter: PhraseProgressReporter | None,
        message: str,
        **metadata: object,
    ) -> None:
        """Emit a phrase-stage status update when a reporter is available.

        Phrase translation now exposes deterministic stage boundaries so the CLI
        can tell the user where a long-running request currently is. This helper
        keeps the service logic focused on translation while isolating the
        reporter's optional nature.
        """

        if progress_reporter is None:
            return
        progress_reporter.stage(message, **metadata)

    def _llm_logging_context(
        self,
        active_variants: Sequence[str] | None,
    ) -> dict[str, object]:
        """Resolve cache/logging metadata for phrase-level LLM render requests.

        Phrase renders operate on one insight variant at a time. Reusing that
        variant's SQLite connection and latest run id lets the render path share
        the existing prompt-hash cache without introducing a second store.
        """

        if not active_variants:
            return {}
        variant = str(active_variants[0])
        repository = getattr(self.word_service, "repository", None)
        llm_logging_target = getattr(repository, "llm_logging_target", None)
        if not callable(llm_logging_target):
            return {}
        conn, run_id = llm_logging_target(variant)
        context: dict[str, object] = {}
        if conn is not None:
            context["llm_query_db"] = conn
        if run_id is not None:
            context["llm_query_run_id"] = run_id
        return context

    @staticmethod
    def _translation_skeleton(
        tokens: Sequence[str],
        token_choices: Sequence[PhraseTokenCandidate],
        relations: Sequence[PhraseRelation],
    ) -> str:
        """Render a deterministic clause draft from chosen bundle-aware reads.

        The no-LLM path still needs to stay explainable and deterministic, but
        a flat token join is too weak for decomposition stress tests. This
        renderer therefore uses bundle heads and normalized function-word
        profiles to build the clause-shaped draft that both the CLI and the LLM
        refinement path consume.
        """

        chunks: list[str] = []
        for index, (token, candidate) in enumerate(zip(tokens, token_choices, strict=False)):
            left_relation = relations[index - 1] if index > 0 and index - 1 < len(relations) else None
            right_relation = relations[index] if index < len(relations) else None
            previous = token_choices[index - 1] if index > 0 else None
            following = token_choices[index + 1] if index + 1 < len(token_choices) else None
            chunk = PhraseTranslationService._clause_chunk_for_candidate(
                token=token,
                candidate=candidate,
                previous=previous,
                following=following,
                left_relation=left_relation,
                right_relation=right_relation,
            )
            if not chunk:
                continue
            if chunks and chunks[-1] in {"her", "in her"} and chunk.startswith("her "):
                chunk = chunk[4:].strip()
                if not chunk:
                    continue
            if chunks and chunks[-1] == chunk and chunk in {"and", "that", "in", "her"}:
                continue
            chunks.append(chunk)
        return " ".join(chunks).strip()

    @staticmethod
    def _clause_chunk_for_candidate(
        *,
        token: str,
        candidate: PhraseTokenCandidate,
        previous: PhraseTokenCandidate | None,
        following: PhraseTokenCandidate | None,
        left_relation: PhraseRelation | None,
        right_relation: PhraseRelation | None,
    ) -> str:
        """Choose a deterministic clause chunk for one token candidate.

        Clause composition only does local normalization. It should preserve the
        decomposition-led meaning while keeping function words compact and human
        readable instead of repeating abstract grammar labels.
        """

        profile = candidate.bundle_function_profile
        human_facing = PhraseTranslationService._human_facing_candidate_gloss(candidate)
        canonical_function_gloss = _FUNCTION_PROFILE_CANONICAL_GLOSSES.get(profile)
        if profile == "conjunction":
            return "and"
        if profile == "relative":
            return "that"
        if profile == "within_self":
            return "within itself"
        if (
            human_facing is not None
            and canonical_function_gloss is not None
            and not PhraseTranslationService._should_collapse_to_function_gloss(
                profile=profile,
                human_facing=human_facing,
                canonical_gloss=canonical_function_gloss,
            )
        ):
            return human_facing
        if profile == "locative":
            return "in"
        if profile == "imperative_existential":
            return "let there be"
        if profile == "feminine_locative_possessive":
            next_gloss = str(
                (following.bundle_surface_gloss or following.definition or "")
                if following is not None
                else ""
            ).strip().lower()
            if next_gloss.startswith("her "):
                return "in her"
            if left_relation is not None and left_relation.relation == "relational_attachment":
                return "in her"
            return "her"

        preferred = (
            candidate.bundle_surface_gloss
            or candidate.bundle_head_gloss
            or candidate.definition
        )
        if isinstance(preferred, str) and preferred.strip():
            return preferred.strip()
        if human_facing is not None:
            return human_facing
        raise RuntimeError(
            f"Phrase token {token.upper()} reached clause rendering without a grounded gloss."
        )

    @staticmethod
    def _should_collapse_to_function_gloss(
        *,
        profile: str,
        human_facing: str,
        canonical_gloss: str,
    ) -> bool:
        """Prefer compact function words when the lexical gloss is still scaffolding.

        Phrase rendering wants lexical meanings like `I am` to beat generic
        function-profile shorthand such as `in`, but it should still collapse
        verbose scaffolding like `additive conjunction that links` back to
        compact English function words.
        """

        normalized = " ".join(human_facing.strip().lower().split())
        if not normalized:
            return True
        if normalized == canonical_gloss:
            return True
        if is_meta_linguistic_gloss(normalized):
            return True
        if normalized in _WEAK_PRIMARY_RESCUE_GLOSSES:
            return True
        if profile == "conjunction":
            return "conjunction" in normalized or normalized.startswith("and ")
        if profile == "relative":
            return normalized in {"relation", "subordination", "reference"}
        if profile == "locative":
            return normalized in {"place", "position", "location", "state", "being", "existence"}
        if profile == "feminine_locative_possessive":
            return normalized in {"her", "possession", "locative relation"}
        return False

    @staticmethod
    def _parse_confidence(parse: PhraseParseCandidate) -> float:
        if not parse.token_choices:
            return 0.0
        return sum(candidate.confidence for candidate in parse.token_choices) / float(
            len(parse.token_choices)
        )

    @staticmethod
    def _phrase_render_payload(
        phrase: str,
        chosen_parse: PhraseParseCandidate,
    ) -> dict[str, object]:
        """Convert the chosen parse into a slim lay-render payload.

        Phrase rendering only needs the chosen token glosses, nearby
        alternatives, and relation structure. Keeping raw diagnostic prose out
        of the LLM payload reduces latency and avoids nudging the model back
        toward glossary-dump output.
        """

        token_choices: list[dict[str, object]] = []
        for candidate in chosen_parse.token_choices:
            trace = candidate.definition_trace
            token_choices.append(
                {
                    "token": candidate.token,
                    "definition": candidate.definition,
                    "surface_gloss": trace.get("surface_gloss") or candidate.definition,
                    "bundle_surface_gloss": candidate.bundle_surface_gloss,
                    "bundle_head_gloss": candidate.bundle_head_gloss,
                    "bundle_function_profile": candidate.bundle_function_profile,
                    "blind_mode_whole_word_rescue": candidate.blind_mode_whole_word_rescue,
                    "blind_mode_rescue_note": candidate.blind_mode_rescue_note,
                    "dictionary_rescue_gloss": candidate.dictionary_rescue_gloss,
                    "dictionary_rescue_note": candidate.dictionary_rescue_note,
                    "semantic_core": list(trace.get("selected_semantic_core") or []),
                    "negative_contrast": list(trace.get("selected_negative_contrast") or []),
                    "alternates": list(candidate.alternates[:3]),
                    "analysis_type": candidate.analysis_type,
                    "role_hint": candidate.role_hint,
                }
            )
        return {
            "phrase": phrase,
            "translation_skeleton": chosen_parse.translation_skeleton,
            "token_choices": token_choices,
            "relations": [asdict(relation) for relation in chosen_parse.relations],
            "score": chosen_parse.score,
        }
