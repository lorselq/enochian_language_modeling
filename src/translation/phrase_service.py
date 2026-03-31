from __future__ import annotations

"""Phrase-level translation built on top of the single-word translator.

This module upgrades translation from isolated word glossing to a global,
phrase-level search. Each token first receives one or more algorithmic
single-word analyses; the phrase layer then scores combinations of those
analyses jointly so anchors, unknown words, and local structure can constrain
each other before any optional LLM rendering happens.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from collections.abc import Callable, Iterable, Sequence

from enochian_lm.common.config import get_config_paths

from .llm_synthesis import (
    DEFAULT_LLM_CONTEXT,
    PhraseRenderResult,
    render_phrase_lay_translation,
    render_phrase_translation,
)
from .memory import TranslationMemoryRepository
from .progress import TranslationPhaseReporter
from .service import SingleWordTranslationService
from .tokenization import tokenize_words


@dataclass(slots=True)
class PhraseTokenCandidate:
    """Represent one phrase-level lexical reading for a token."""

    token: str
    rank: int
    analysis_type: str
    definition: str | None
    alternates: list[str]
    confidence: float
    score: float
    role_hint: str
    morphs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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
    ) -> None:
        self.word_service = word_service
        self.memory_repository = memory_repository
        self.llm_renderer = llm_renderer
        self.lay_renderer = lay_renderer

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
        progress_reporter: TranslationPhaseReporter | None = None,
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
        if progress_reporter is not None:
            progress_reporter.end("phrase.normalize", detail=f"Tokenized {len(tokens)} token(s)")

        llm_enabled = self.word_service.llm_enabled if llm is None else bool(llm)
        active_variants = list(variants) if variants else self.word_service.active_variants
        candidate_matrix: list[list[PhraseTokenCandidate]] = []
        token_payloads: list[dict[str, object]] = []

        if progress_reporter is not None:
            progress_reporter.begin("phrase.token_loop", total=len(tokens), detail="Analyzing tokens")
        for index, token in enumerate(tokens):
            if progress_reporter is not None:
                progress_reporter.advance(
                    "phrase.token_loop",
                    current=index + 1,
                    total=len(tokens),
                    detail=f"token {index + 1}/{len(tokens)}: {token}",
                )
            word_result = self.word_service.translate_word(
                token,
                variants=active_variants,
                strategy=strategy,
                top_k=top_k,
                llm=False,
                llm_context=llm_context,
                evidence_mode=evidence_mode,
                weight_enabled=weight_enabled,
                allow_whole_word=allow_whole_word,
                progress_reporter=progress_reporter,
            )
            token_candidates = self._token_candidates_from_word_result(
                token,
                word_result,
                top_k=top_k,
            )
            if not token_candidates:
                token_candidates = self._fallback_token_candidates(token)
            candidate_matrix.append(token_candidates)
            token_payloads.append(
                {
                    "token": token,
                    "candidates": [asdict(candidate) for candidate in token_candidates],
                    "word_result": word_result,
                }
            )
        if progress_reporter is not None:
            progress_reporter.end("phrase.token_loop", detail="Token analysis complete")

        if progress_reporter is not None:
            progress_reporter.begin("phrase.parse", detail="Building parse candidates")
        parse_candidates = self._build_parse_candidates(
            tokens,
            candidate_matrix,
            top_k=max(3, top_k),
        )
        if progress_reporter is not None:
            progress_reporter.end(
                "phrase.parse",
                detail=f"Built {len(parse_candidates)} parse candidate(s)",
            )
        chosen_parse = parse_candidates[0] if parse_candidates else None

        rendered_translation = chosen_parse.translation_skeleton if chosen_parse else ""
        render_reasoning = "Algorithmic phrase rendering only."
        render_confidence = 0.0
        render_warnings: list[str] = []
        lay_translation = rendered_translation
        lay_reasoning = "Lay translation fell back to the algorithmic phrase skeleton."
        lay_confidence = 0.0
        lay_warnings: list[str] = []
        if chosen_parse is not None:
            render_confidence = self._parse_confidence(chosen_parse)
            lay_confidence = render_confidence
        render_payload = (
            self._phrase_render_payload(normalized_phrase, chosen_parse)
            if chosen_parse is not None
            else None
        )
        if llm_enabled and chosen_parse is not None:
            if progress_reporter is not None:
                progress_reporter.begin("phrase.llm_render", detail="Rendering phrase with LLM")
            render_context = {
                "phrase": normalized_phrase,
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "use_remote": self.word_service.llm_use_remote,
                "confidence": render_confidence,
            }
            rendered = self.llm_renderer(
                render_payload,
                render_context,
            )
            rendered_translation = rendered.rendered_translation or rendered_translation
            render_reasoning = rendered.reasoning
            render_confidence = rendered.confidence
            render_warnings = list(rendered.warnings)
        if render_payload is not None:
            lay_context = {
                "phrase": normalized_phrase,
                "llm_context": llm_context or DEFAULT_LLM_CONTEXT,
                "use_remote": self.word_service.llm_use_remote,
                "confidence": render_confidence,
            }
            lay_rendered = self.lay_renderer(
                render_payload,
                lay_context,
            )
            lay_translation = lay_rendered.rendered_translation or rendered_translation
            lay_reasoning = lay_rendered.reasoning
            lay_confidence = lay_rendered.confidence
            lay_warnings = list(lay_rendered.warnings)

        memory_updates: list[dict[str, object]] = []
        if memory_update and chosen_parse is not None:
            if progress_reporter is not None:
                progress_reporter.begin(
                    "phrase.memory",
                    total=len(chosen_parse.token_choices),
                    detail="Writing provisional memory updates",
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
                if progress_reporter is not None:
                    progress_reporter.advance(
                        "phrase.memory",
                        current=index + 1,
                        total=len(chosen_parse.token_choices),
                        detail=f"processed token {index + 1}/{len(chosen_parse.token_choices)}",
                    )
            if progress_reporter is not None:
                progress_reporter.end(
                    "phrase.memory",
                    detail=f"Wrote {len(memory_updates)} memory update(s)",
                )

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
            "lay_warnings": lay_warnings,
            "lay_translation_mode": (
                "remote" if chosen_parse is not None and self.word_service.llm_use_remote
                else "local" if chosen_parse is not None
                else None
            ),
            "memory_updates": memory_updates,
        }

    def _token_candidates_from_word_result(
        self,
        token: str,
        result: dict[str, object],
        *,
        top_k: int,
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
            primary_definition = self._candidate_definition(meaning_list)
            alternates = self._candidate_alternates(meaning_list)
            role_hint = self._infer_role_hint(
                token,
                candidate,
                primary_definition,
            )
            converted.append(
                PhraseTokenCandidate(
                    token=token,
                    rank=int(candidate.get("rank") or 0),
                    analysis_type=str(candidate.get("analysis_type") or "compositional"),
                    definition=primary_definition,
                    alternates=alternates,
                    confidence=float(candidate.get("confidence") or 0.0),
                    score=float(candidate.get("score") or 0.0),
                    role_hint=role_hint,
                    morphs=[
                        str(morph)
                        for morph in candidate.get("morphs", [])
                        if isinstance(morph, str)
                    ],
                    warnings=[
                        str(warning)
                        for warning in candidate.get("warnings", [])
                        if isinstance(warning, str)
                    ],
                )
            )
        return converted

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
                    alternates=list(memory_entry.alternates),
                    confidence=memory_entry.confidence,
                    score=30.0 + memory_entry.evidence_count,
                    role_hint=memory_entry.role_hint or "unknown",
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
                alternates=[],
                confidence=0.1,
                score=1.0,
                role_hint="unknown",
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
                    score = float(beam["score"]) + candidate.score + (candidate.confidence * 5.0)
                    if previous is not None:
                        relation = self._relation_between(previous, candidate, index - 1, index)
                        relations.append(relation)
                        score += relation.score
                    if candidate.analysis_type == "provisional" and not candidate.definition:
                        score -= 2.0
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
            skeleton = self._translation_skeleton(tokens, token_choices)
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
        if "relational" in {left.role_hint, right.role_hint}:
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
    ) -> str:
        """Infer a coarse syntactic role for a token candidate."""
        dictionary_entry = self.word_service.candidate_finder.dictionary.get(token.lower())
        if dictionary_entry is not None:
            pos = dictionary_entry.get("pos")
            normalized = self._normalize_role_from_pos(pos)
            if normalized is not None:
                return normalized

        definition = (primary_definition or "").strip().lower()
        if not definition:
            return "unknown"
        if definition.startswith("to "):
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
    def _candidate_definition(meanings: Sequence[object]) -> str | None:
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            definition = meaning.get("definition")
            if isinstance(definition, str) and definition.strip():
                return definition.strip()
        return None

    @staticmethod
    def _candidate_alternates(meanings: Sequence[object]) -> list[str]:
        alternates: list[str] = []
        seen: set[str] = set()
        for meaning in meanings:
            if not isinstance(meaning, dict):
                continue
            definitions = meaning.get("definitions")
            if not isinstance(definitions, list):
                continue
            for definition in definitions:
                if not isinstance(definition, str):
                    continue
                normalized = definition.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                alternates.append(normalized)
        return alternates

    @staticmethod
    def _translation_skeleton(
        tokens: Sequence[str],
        token_choices: Sequence[PhraseTokenCandidate],
    ) -> str:
        """Render a deterministic translation skeleton from chosen token reads."""
        glosses: list[str] = []
        for token, candidate in zip(tokens, token_choices, strict=False):
            if candidate.definition:
                glosses.append(candidate.definition)
            else:
                glosses.append(f"[{token}]")
        return " ".join(glosses)

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
        """Convert the chosen parse into a constrained phrase-render payload."""
        return {
            "phrase": phrase,
            "translation_skeleton": chosen_parse.translation_skeleton,
            "token_choices": [asdict(candidate) for candidate in chosen_parse.token_choices],
            "relations": [asdict(relation) for relation in chosen_parse.relations],
            "score": chosen_parse.score,
        }
