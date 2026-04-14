from __future__ import annotations

"""LLM synthesis adapter for task 4.1.

This module bridges the decomposition pipeline with the existing LLM tooling
used elsewhere in the project (``QueryModelTool``). It exposes a single
function, :func:`synthesize_definition`, which requests a concise gloss for a
set of morph meanings and provides a structured fallback when LLM access fails.

The return value is intentionally explicit and type-safe to make downstream
callers resilient to LLM variability.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol
import json
import logging
import re

from enochian_lm.root_extraction.tools.query_model_tool import QueryModelTool

from .placeholder_glosses import (
    clean_lexical_gloss,
    specific_gloss_from_definition_and_semantic_core,
    sanitize_human_gloss,
    semantic_core_gloss,
    unresolved_token_gloss,
)

LOGGER = logging.getLogger(__name__)

_RAW_TOKEN_PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z' ?-]*)\]")
_HUMAN_UNRESOLVED_GLOSS = "unresolved term"

DEFAULT_LLM_CONTEXT = (
    "Use Early Modern English prose from 16th-century Britain, grounded in the "
    "mainstream Christian worldview John Dee would have known. Avoid modern "
    "terms, theology, or concepts."
)


class TranslationProgressReporter(Protocol):
    def prepare(self, progress: "LLMRequestProgress") -> None: ...

    def start_primary(self, *, phase: str, current: int, total: int) -> None: ...

    def start_validation(
        self,
        *,
        phase: str,
        current: int,
        total: int,
        kind: Literal["retry", "repair"],
    ) -> None: ...

    def llm_status(self, event: Mapping[str, object]) -> None: ...

    def done(self) -> None: ...


@dataclass(slots=True)
class LLMRequestProgress:
    """Track per-translation LLM request progress for compact status output."""

    total_primary_requests: int
    current_primary_request: int = 0
    validation_repair_budget: int = 0

    def begin_primary(self, label: str) -> str:
        self.current_primary_request += 1
        return f"LLM request {self.current_primary_request}/{self.total_primary_requests}: {label}"

    def validation_retry(self, label: str, *, kind: Literal["retry", "repair"] = "retry") -> str:
        prefix = "validation repair" if kind == "repair" else "validation retry"
        current = min(self.current_primary_request, self.total_primary_requests)
        return f"LLM {prefix} after {current}/{self.total_primary_requests}: {label}"



@dataclass(slots=True)
class SynthesisResult:
    """Structured output for synthesized definitions.

    Attributes
    ----------
    synthesized_definition:
        The LLM-composed definition when available. ``None`` when synthesis
        failed or was skipped.
    concatenated_meanings:
        Concise string produced from the provided meanings. This is always
        populated and serves as the fallback definition when synthesis is
        unavailable.
    confidence:
        Heuristic confidence in the synthesized definition (0.0–1.0). When the
        LLM returns a confidence value it is respected; otherwise, a
        coverage-weighted fallback is used.
    reasoning:
        Short explanation of the synthesis status, including failure reasons
        for transparency.
    warnings:
        Optional warnings emitted during synthesis.
    raw_response:
        Raw text returned by the LLM (useful for debugging downstream).
    best_estimations:
        Concrete 1–3 word phrases that capture the morph-level meaning.
    """

    synthesized_definition: str | None
    concatenated_meanings: str
    confidence: float
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None
    best_estimations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "synthesized_definition": self.synthesized_definition,
            "concatenated_meanings": self.concatenated_meanings,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
            "best_estimations": list(self.best_estimations),
        }


@dataclass(slots=True)
class ConsensusSynthesisResult:
    """Structured output for synthesized consensus across top candidates."""

    synthesized_definition: str | None
    best_estimations: list[str]
    confidence: float
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "synthesized_definition": self.synthesized_definition,
            "best_estimations": list(self.best_estimations),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class PhraseRenderResult:
    """Structured output for phrase-level rendering.

    Phrase rendering is intentionally downstream of the algorithmic parse. The
    LLM is allowed to improve readability and tone, but not to change the token
    inventory, structure, or meaning bundle selected by the parser.
    """

    rendered_translation: str
    confidence: float
    reasoning: str
    footnoted_translation: str | None = None
    translation_footnotes: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "rendered_translation": self.rendered_translation,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "footnoted_translation": self.footnoted_translation,
            "translation_footnotes": list(self.translation_footnotes),
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
        }


@dataclass(slots=True)
class PhraseRenderBundleResult:
    """Carry the full phrase-render package returned from one bundled LLM call.

    Phrase translation now wants a technical rendering, a grounded lay
    rendering, a more interpretive paraphrase, and a token-aligned footnote
    block. Returning all of that from one structured result lets the phrase
    service reduce remote round-trips while keeping the downstream CLI payload
    explicit.
    """

    technical_translation: str
    technical_confidence: float
    technical_reasoning: str
    lay_translation: str
    lay_confidence: float
    lay_reasoning: str
    poetic_translation: str = ""
    poetic_confidence: float = 0.0
    poetic_reasoning: str = ""
    contextual_lay_translation: str = ""
    contextual_lay_confidence: float | None = None
    contextual_lay_reasoning: str = ""
    contextual_poetic_translation: str = ""
    contextual_poetic_confidence: float | None = None
    contextual_poetic_reasoning: str = ""
    contextual_selected_parse_rank: int = 1
    contextual_interpretive_translation: str = ""
    contextual_interpretive_confidence: float | None = None
    contextual_interpretive_reasoning: str = ""
    interpretive_translation: str = ""
    interpretive_confidence: float = 0.0
    interpretive_reasoning: str = ""
    footnoted_translation: str | None = None
    translation_footnotes: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "technical_translation": self.technical_translation,
            "technical_confidence": self.technical_confidence,
            "technical_reasoning": self.technical_reasoning,
            "lay_translation": self.lay_translation,
            "lay_confidence": self.lay_confidence,
            "lay_reasoning": self.lay_reasoning,
            "poetic_translation": self.poetic_translation,
            "poetic_confidence": self.poetic_confidence,
            "poetic_reasoning": self.poetic_reasoning,
            "contextual_lay_translation": self.contextual_lay_translation,
            "contextual_lay_confidence": self.contextual_lay_confidence,
            "contextual_lay_reasoning": self.contextual_lay_reasoning,
            "contextual_poetic_translation": self.contextual_poetic_translation,
            "contextual_poetic_confidence": self.contextual_poetic_confidence,
            "contextual_poetic_reasoning": self.contextual_poetic_reasoning,
            "contextual_selected_parse_rank": self.contextual_selected_parse_rank,
            "contextual_interpretive_translation": self.contextual_interpretive_translation,
            "contextual_interpretive_confidence": self.contextual_interpretive_confidence,
            "contextual_interpretive_reasoning": self.contextual_interpretive_reasoning,
            "interpretive_translation": self.interpretive_translation,
            "interpretive_confidence": self.interpretive_confidence,
            "interpretive_reasoning": self.interpretive_reasoning,
            "footnoted_translation": self.footnoted_translation,
            "translation_footnotes": list(self.translation_footnotes),
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
        }


def synthesize_definition(
    morphs: list[str],
    meanings: list[str],
    context: dict[str, object],
) -> SynthesisResult:
    """Return an LLM-synthesized gloss for the supplied morph meanings.

    This implements Task 4.1 from ``TODO.md`` by:

    - Building a structured prompt that pairs the decomposition (``morphs``)
      with the human-readable ``meanings`` and coverage context.
    - Leveraging the existing ``QueryModelTool`` infrastructure to keep LLM
      invocation consistent with the rest of the codebase.
    - Enforcing mandatory ``best_estimations`` output with retries and
      validation to keep downstream consumers consistent.
    """

    normalized_morphs = [m.upper() for m in morphs if m]
    cleaned_meanings = [m.strip() for m in meanings if isinstance(m, str) and m.strip()]
    concatenated = " + ".join(cleaned_meanings) if cleaned_meanings else " + ".join(normalized_morphs)

    prompt = _build_prompt(normalized_morphs, cleaned_meanings, context)
    use_remote = bool(context.get("use_remote", False))

    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a precise Enochian glossator. Combine morph semantics "
            "into a single, well-formed definition without inventing "
            "unsupported etymologies. "
            f"Historical scope: {llm_context}"
        ),
        name="Definition Synthesizer",
        description="Compose a concise morph-level synthesis",
        use_remote=use_remote,
        progress_style="silent" if _get_progress_reporter(context) else "compact",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = tool._run(
                prompt=prompt,
                print_chunks=False,
                stream_callback=None,
                progress_message=_primary_progress_message(
                    context,
                    attempt=attempt,
                    phase="candidate",
                    label=f"synthesizing rank {context.get('candidate_rank', '?')}...",
                ),
            )
            parsed = _parse_response(
                response.get("response_text", ""), fallback=concatenated, context=context
            )

            synthesized_raw = parsed.get("definition")
            synthesized: str | None = synthesized_raw if isinstance(synthesized_raw, str) else None
            confidence_raw = parsed.get("confidence")
            confidence_val: float | None = (
                float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
            )
            confidence = _resolved_confidence(confidence_val, context)
            reasoning_raw = parsed.get("reasoning")
            reasoning: str = (
                reasoning_raw if isinstance(reasoning_raw, str) else "Synthesized definition via LLM."
            )
            best_estimations = _coerce_best_estimations(parsed.get("best_estimations"))
            if not best_estimations:
                best_estimations = _request_best_estimations(
                    tool,
                    context,
                    synthesized or concatenated,
                    response.get("response_text", ""),
                )
            if not best_estimations:
                raise ValueError("Missing required best_estimations list.")

            return SynthesisResult(
                synthesized_definition=synthesized,
                concatenated_meanings=concatenated,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response.get("response_text"),
                best_estimations=best_estimations,
            )
        except ValueError as exc:
            last_error = exc
            prompt = _build_prompt(
                normalized_morphs,
                cleaned_meanings,
                {**context, "validation_error": str(exc)},
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            break

    message = (
        "LLM synthesis failed to return mandatory best_estimations."
        if last_error is None
        else f"LLM synthesis failed: {last_error}"
    )
    LOGGER.error(message)
    raise ValueError(message)


def synthesize_consensus(
    candidates: list[dict[str, object]],
    context: dict[str, object],
) -> ConsensusSynthesisResult:
    """Return a consensus synthesis across top-ranked candidates."""

    prompt = _build_consensus_prompt(candidates, context)
    use_remote = bool(context.get("use_remote", False))

    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a precision Enochian glossator. Combine candidate senses "
            "into one concrete consensus without inventing unsupported etymologies. "
            f"Historical scope: {llm_context}"
        ),
        name="Consensus Synthesizer",
        description="Compose a consensus definition across candidates",
        use_remote=use_remote,
        progress_style="silent" if _get_progress_reporter(context) else "compact",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = tool._run(
                prompt=prompt,
                print_chunks=False,
                stream_callback=None,
                progress_message=_primary_progress_message(
                    context,
                    attempt=attempt,
                    phase="consensus",
                    label="building consensus...",
                ),
            )
            parsed = _parse_response(
                response.get("response_text", ""), fallback="", context=context
            )
            synthesized_raw = parsed.get("definition")
            synthesized: str | None = synthesized_raw if isinstance(synthesized_raw, str) else None
            confidence_raw = parsed.get("confidence")
            confidence_val: float | None = (
                float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
            )
            confidence = _resolved_confidence(confidence_val, context)
            reasoning_raw = parsed.get("reasoning")
            reasoning: str = (
                reasoning_raw if isinstance(reasoning_raw, str) else "Consensus synthesis via LLM."
            )
            best_estimations = _coerce_best_estimations(parsed.get("best_estimations"))
            if not best_estimations:
                best_estimations = _request_best_estimations(
                    tool,
                    context,
                    synthesized or "",
                    response.get("response_text", ""),
                )
            if not best_estimations:
                raise ValueError("Missing required best_estimations list.")

            return ConsensusSynthesisResult(
                synthesized_definition=synthesized,
                best_estimations=best_estimations,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response.get("response_text"),
            )
        except ValueError as exc:
            last_error = exc
            prompt = _build_consensus_prompt(
                candidates,
                {**context, "validation_error": str(exc)},
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            break

    message = (
        "Consensus synthesis failed to return mandatory best_estimations."
        if last_error is None
        else f"Consensus synthesis failed: {last_error}"
    )
    LOGGER.error(message)
    raise ValueError(message)


def _get_progress_reporter(
    context: dict[str, object],
) -> TranslationProgressReporter | None:
    reporter = context.get("progress_reporter")
    return reporter if reporter is not None else None


def _get_progress_tracker(context: dict[str, object]) -> LLMRequestProgress | None:
    tracker = context.get("llm_progress")
    return tracker if isinstance(tracker, LLMRequestProgress) else None


def _primary_progress_message(
    context: dict[str, object],
    *,
    attempt: int,
    phase: str,
    label: str,
) -> str:
    progress = _get_progress_tracker(context)
    reporter = _get_progress_reporter(context)
    if progress is None:
        if reporter is not None:
            reporter.start_validation(phase=phase, current=0, total=0, kind="retry")
            return ""
        return label if attempt == 0 else f"LLM validation retry {attempt}/2: {label}"
    if attempt == 0:
        message = progress.begin_primary(label)
        if reporter is not None:
            reporter.start_primary(
                phase=phase,
                current=progress.current_primary_request,
                total=progress.total_primary_requests,
            )
            return ""
        return message
    if reporter is not None:
        reporter.start_validation(
            phase=phase,
            current=progress.current_primary_request,
            total=progress.total_primary_requests,
            kind="retry",
        )
        return ""
    return progress.validation_retry(label)


def _repair_progress_message(
    context: dict[str, object],
    *,
    attempt: int,
    phase: str,
    label: str,
) -> str:
    progress = _get_progress_tracker(context)
    reporter = _get_progress_reporter(context)
    if progress is None:
        if reporter is not None:
            reporter.start_validation(phase=phase, current=0, total=0, kind="repair")
            return ""
        return f"LLM validation repair {attempt + 1}/2: {label}"
    if reporter is not None:
        reporter.start_validation(
            phase=phase,
            current=progress.current_primary_request,
            total=progress.total_primary_requests,
            kind="repair",
        )
        return ""
    return progress.validation_retry(label, kind="repair")


def _report_render_detail(context: dict[str, object], detail: str) -> None:
    """Expose long render sub-phases through the shared CLI progress reporter.

    Phrase rendering can spend noticeable time waiting on a model response and
    then validating the structured JSON payload. Surfacing those sub-steps keeps
    long-running `translate-phrase` invocations from looking frozen.
    """

    reporter = _get_progress_reporter(context)
    if reporter is None:
        return
    label = str(context.get("progress_label") or "Rendering phrase")
    reporter.stage(f"{label}... {detail}")


def _tool_progress_callback(
    context: dict[str, object],
) -> Callable[[dict[str, object]], None] | None:
    """Bridge low-level model-tool status events into the CLI renderer.

    ``QueryModelTool`` now emits structured events about retries, streaming, and
    fallback behavior. This adapter injects the higher-level phrase label so the
    CLI can present those low-level events in the context of the current
    translation stage.
    """

    reporter = _get_progress_reporter(context)
    llm_status = getattr(reporter, "llm_status", None) if reporter is not None else None
    label = str(context.get("progress_label") or "LLM request")

    def _callback(event: dict[str, object]) -> None:
        if callable(llm_status):
            llm_status({"label": label, **event})

    return _callback


def _attach_query_logging(tool: QueryModelTool, context: dict[str, object]) -> None:
    """Attach optional cache/logging metadata to phrase-render LLM requests.

    Phrase rendering now participates in the existing prompt-hash caching path
    used by the extraction tools. The phrase service provides a variant
    database handle plus a stable run id when available, and this helper keeps
    that integration local to the LLM adapter.
    """

    db = context.get("llm_query_db")
    if db is None:
        return
    tool.attach_logging(db, context.get("llm_query_run_id"))


def _build_prompt(
    morphs: list[str], meanings: list[str], context: dict[str, object]
) -> str:
    coverage_ratio = _safe_float(context.get("coverage_ratio"), default=0.0)
    residual_ratio = _safe_float(context.get("residual_ratio"), default=1.0)
    strategy = str(context.get("strategy") or "prefer-balance")
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT

    provenance_lines: list[str] = []
    provenance_raw = context.get("provenance")
    provenance_list = provenance_raw if isinstance(provenance_raw, list) else []
    for item in provenance_list:
        if not isinstance(item, dict):
            continue
        morph = item.get("morph")
        source = item.get("provenance")
        if morph:
            provenance_lines.append(f"- {morph}: {source or 'unknown'}")

    provenance_block = "\n".join(provenance_lines) or "(no explicit provenance)"
    meaning_lines = [f"- {morph} → {meaning}" for morph, meaning in zip(morphs, meanings)]
    meaning_block = "\n".join(meaning_lines) or "- No morph meanings available"

    max_len = 160
    schema_block = json.dumps(
        {
            "definition": f"<one-sentence, single-sense definition, <= {max_len} chars>",
            "best_estimations": [
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
            ],
            "confidence": 0.0,
            "reasoning": f"<one-sentence justification, <= {max_len} chars>",
        },
        ensure_ascii=False,
    )
    example_block = json.dumps(
        {
            "definition": "A pillar that bears a counted segment in shared praise.",
            "best_estimations": ["stone pillar", "praise rite", "linked support"],
            "confidence": 0.62,
            "reasoning": "Combines pillar, segment, and praise-with relation into one sense.",
        },
        ensure_ascii=False,
    )

    validation_error = context.get("validation_error")
    validation_note = (
        f"VALIDATION NOTE: {validation_error}" if isinstance(validation_error, str) else None
    )
    instruction_lines = [
        "ROLE: You are a precision Enochian glossator.",
        f"HISTORICAL CONTEXT: {llm_context}",
        "CONSTRAINTS:",
        "- Use ONLY the provided morphs/meanings/provenance; no external etymology or speculation.",
        "- Evidence trust order: clusters > residuals > hypotheses. Do not invent links.",
        "- Produce exactly one concise sense (no lists, no multiple senses, no hedging).",
        "- Provide 3-6 concrete 'best_estimations' (1-3 word phrases) that make the sense tangible.",
        "- 'best_estimations' should prefer physical objects/actions when possible.",
        "- 'best_estimations' is REQUIRED. Responses missing it are invalid.",
        "- One sentence for definition, one sentence for reasoning; trim whitespace.",
        f"- Hard max length per field: {max_len} characters; truncate gracefully if needed.",
        "- Return STRICT JSON only, matching the schema below (no prose around it).",
        "- If uncertain, restate the concatenated meanings as the definition and cap confidence accordingly.",
        "EVIDENCE:",
        f"- Morph decomposition: {' + '.join(morphs) if morphs else '(none)'}",
        "- Morph meanings:",
        meaning_block,
        "- Provenance per morph (clusters > residuals > hypotheses):",
        provenance_block,
        f"- Coverage: coverage_ratio={coverage_ratio:.2f}, residual_ratio={residual_ratio:.2f}",
        f"- Strategy hint: {strategy}",
        "SCHEMA (return exactly this shape):",
        schema_block,
        "EXAMPLE (format only; do not copy wording):",
        example_block,
    ]
    if validation_note:
        instruction_lines.append(validation_note)

    return "\n".join(instruction_lines)


def _build_consensus_prompt(
    candidates: list[dict[str, object]], context: dict[str, object]
) -> str:
    max_len = 180
    strategy = str(context.get("strategy") or "prefer-balance")
    coverage_ratio = _safe_float(context.get("coverage_ratio"), default=0.0)
    residual_ratio = _safe_float(context.get("residual_ratio"), default=1.0)
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT

    candidate_lines: list[str] = []
    synthesized_lines: list[str] = []
    concatenated_lines: list[str] = []
    best_estimation_lines: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        rank = candidate.get("rank")
        morphs = candidate.get("morphs")
        morph_list = morphs if isinstance(morphs, list) else []
        morph_label = " + ".join(str(m) for m in morph_list if m)
        meanings = candidate.get("meanings")
        meanings_list = meanings if isinstance(meanings, list) else []
        meaning_lines = []
        for meaning in meanings_list:
            if not isinstance(meaning, dict):
                continue
            morph = meaning.get("morph")
            definition = meaning.get("definition")
            if morph and definition:
                meaning_lines.append(f"{morph}: {definition}")
        synthesized = candidate.get("synthesized_definition") or candidate.get(
            "concatenated_meanings"
        )
        confidence = candidate.get("confidence")
        best_estimations = candidate.get("best_estimations")
        best_estimation_list = (
            best_estimations if isinstance(best_estimations, list) else []
        )
        best_estimation_label = ", ".join(
            str(item) for item in best_estimation_list if isinstance(item, str) and item
        )
        candidate_lines.append(
            "\n".join(
                [
                    f"- Rank {rank}: {morph_label}",
                    f"  Meanings: {', '.join(meaning_lines) if meaning_lines else 'n/a'}",
                    f"  Synthesized: {synthesized or 'n/a'}",
                    f"  Confidence: {confidence if isinstance(confidence, (int, float)) else 'n/a'}",
                ]
            )
        )
        if candidate.get("synthesized_definition"):
            synthesized_lines.append(
                f"- Rank {rank}: {candidate.get('synthesized_definition')}"
            )
        if candidate.get("concatenated_meanings"):
            concatenated_lines.append(
                f"- Rank {rank}: {candidate.get('concatenated_meanings')}"
            )
        if best_estimation_label:
            best_estimation_lines.append(f"- Rank {rank}: {best_estimation_label}")

    candidates_block = "\n".join(candidate_lines) or "- No candidate summaries available"
    synthesized_block = "\n".join(synthesized_lines) or "- No synthesized definitions available"
    concatenated_block = (
        "\n".join(concatenated_lines) or "- No concatenated meanings available"
    )
    best_estimations_block = (
        "\n".join(best_estimation_lines) or "- No best estimations available"
    )
    schema_block = json.dumps(
        {
            "definition": f"<one-sentence consensus definition, <= {max_len} chars>",
            "best_estimations": [
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
            ],
            "confidence": 0.0,
            "reasoning": f"<one-sentence justification, <= {max_len} chars>",
        },
        ensure_ascii=False,
    )
    example_block = json.dumps(
        {
            "definition": "A shared rite of praise around a counted pillar segment.",
            "best_estimations": ["praise rite", "pillar segment", "shared accord"],
            "confidence": 0.58,
            "reasoning": "Consensus favors praise, structure, and relation across candidates.",
        },
        ensure_ascii=False,
    )

    validation_error = context.get("validation_error")
    validation_note = (
        f"VALIDATION NOTE: {validation_error}" if isinstance(validation_error, str) else None
    )
    instruction_lines = [
        "ROLE: You are a precision Enochian glossator.",
        "TASK: Combine the top candidate senses into one consensus meaning.",
        f"HISTORICAL CONTEXT: {llm_context}",
        "CONSTRAINTS:",
        "- Use ONLY the provided candidate information; no external etymology.",
        "- Prefer the overlapping concrete semantics across candidates.",
        "- Produce exactly one concise sense (no lists, no multiple senses).",
        "- Provide 3-6 concrete 'best_estimations' (1-3 word phrases) that fit the consensus.",
        "- 'best_estimations' is REQUIRED. Responses missing it are invalid.",
        "- One sentence for definition, one sentence for reasoning; trim whitespace.",
        f"- Hard max length per field: {max_len} characters; truncate gracefully if needed.",
        "- Return STRICT JSON only, matching the schema below (no prose around it).",
        "CONTEXT:",
        f"- Strategy hint: {strategy}",
        f"- Coverage: coverage_ratio={coverage_ratio:.2f}, residual_ratio={residual_ratio:.2f}",
        "CANDIDATES:",
        candidates_block,
        "SYNTHESIZED DEFINITIONS:",
        synthesized_block,
        "CONCATENATED MEANINGS:",
        concatenated_block,
        "BEST ESTIMATIONS:",
        best_estimations_block,
        "SCHEMA (return exactly this shape):",
        schema_block,
        "EXAMPLE (format only; do not copy wording):",
        example_block,
    ]
    if validation_note:
        instruction_lines.append(validation_note)

    return "\n".join(instruction_lines)


def _parse_response(
    payload: str,
    *,
    fallback: str,
    context: dict[str, object],
    max_len: int = 160,
) -> dict[str, str | float]:
    """Parse LLM JSON response with validation and trimming."""

    def _trim(value: str) -> str:
        text = value.strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    parsed: dict[str, str | float | list[str]] = {}

    if not payload:
        parsed["definition"] = fallback
        parsed["reasoning"] = "LLM response empty; using fallback definition."
        parsed["confidence"] = _resolved_confidence(None, context, fallback_only=True)
        return parsed

    data = _load_json_payload(payload)
    if data is None:
        parsed["definition"] = _trim(payload)
        parsed["reasoning"] = "Non-JSON response; using raw text as definition."
        parsed["confidence"] = _resolved_confidence(None, context, fallback_only=True)
        return parsed

    definition = data.get("definition")
    if not isinstance(definition, str) or not definition.strip():
        parsed["definition"] = fallback
        parsed["reasoning"] = "Missing definition; using concatenated meanings."
    else:
        parsed["definition"] = _trim(definition)

    confidence_val = data.get("confidence")
    if isinstance(confidence_val, (int, float)):
        parsed["confidence"] = max(0.0, min(1.0, float(confidence_val)))

    reasoning_val = data.get("reasoning")
    if isinstance(reasoning_val, str) and reasoning_val.strip():
        parsed["reasoning"] = _trim(reasoning_val)

    best_estimations = data.get("best_estimations")
    if isinstance(best_estimations, list):
        parsed["best_estimations"] = [
            _trim(item)
            for item in best_estimations
            if isinstance(item, str) and item.strip()
        ]

    if "confidence" not in parsed:
        parsed["confidence"] = _resolved_confidence(None, context, fallback_only=True)
    if "reasoning" not in parsed:
        parsed["reasoning"] = "Synthesized definition via LLM."

    return parsed


def _coerce_best_estimations(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return cleaned[:6]


def _request_best_estimations(
    tool: QueryModelTool,
    context: dict[str, object],
    synthesized: str,
    raw_response: str,
) -> list[str]:
    prompt = _build_best_estimations_prompt(context, synthesized, raw_response)
    target_label = str(context.get("progress_label") or "refining best estimations...")
    for attempt in range(2):
        response = tool._run(
            prompt=prompt,
            print_chunks=False,
            stream_callback=None,
            progress_message=(
                _repair_progress_message(
                    context,
                    attempt=attempt,
                    phase=str(context.get("progress_stage") or "candidate"),
                    label=target_label,
                )
            ),
        )
        parsed = _parse_best_estimations_response(response.get("response_text", ""))
        if parsed:
            return parsed
        prompt = _build_best_estimations_prompt(
            {**context, "validation_error": "Missing or empty best_estimations list."},
            synthesized,
            raw_response,
        )
    return []


def _build_best_estimations_prompt(
    context: dict[str, object],
    synthesized: str,
    raw_response: str,
) -> str:
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    validation_error = context.get("validation_error")
    validation_note = (
        f"VALIDATION NOTE: {validation_error}" if isinstance(validation_error, str) else None
    )
    schema_block = json.dumps(
        {
            "best_estimations": [
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
                "<concrete 1-3 word phrase>",
            ]
        },
        ensure_ascii=False,
    )
    example_block = json.dumps(
        {"best_estimations": ["stone pillar", "praise rite", "linked support"]},
        ensure_ascii=False,
    )
    instruction_lines = [
        "ROLE: You are a precision Enochian glossator.",
        "TASK: Extract 3-5 concrete best_estimations from the synthesis.",
        f"HISTORICAL CONTEXT: {llm_context}",
        "CONSTRAINTS:",
        "- Use ONLY the provided synthesis or LLM output; no new etymology.",
        "- Provide 3-5 concrete 'best_estimations' (1-3 word phrases).",
        "- 'best_estimations' is REQUIRED. Responses missing it are invalid.",
        "- Return STRICT JSON only, matching the schema below (no prose around it).",
        "SYNTHESIS:",
        synthesized or "(none)",
        "RAW LLM OUTPUT:",
        raw_response or "(none)",
        "SCHEMA (return exactly this shape):",
        schema_block,
        "EXAMPLE (format only; do not copy wording):",
        example_block,
    ]
    if validation_note:
        instruction_lines.append(validation_note)
    return "\n".join(instruction_lines)


def _parse_best_estimations_response(payload: str) -> list[str]:
    if not payload:
        return []
    data = _load_json_payload(payload)
    if data is None:
        return []
    best_estimations = data.get("best_estimations")
    if not isinstance(best_estimations, list):
        return []
    cleaned = [item.strip() for item in best_estimations if isinstance(item, str) and item.strip()]
    return cleaned[:6]


def _load_json_payload(payload: str) -> dict[str, object] | None:
    """Parse JSON from direct, fenced, or prose-wrapped LLM responses."""

    text = payload.strip()
    if not text:
        return None

    candidates = [text]
    if text.startswith("```"):
        fence_lines = text.splitlines()
        if len(fence_lines) >= 3:
            candidates.append("\n".join(fence_lines[1:-1]).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _resolved_confidence(
    reported: float | None,
    context: dict[str, object],
    *,
    fallback_only: bool = False,
) -> float:
    if reported is not None and not fallback_only:
        return max(0.0, min(1.0, float(reported)))

    coverage = _safe_float(context.get("coverage_ratio"), default=0.0)
    residual = _safe_float(context.get("residual_ratio"), default=1.0)
    heuristic = max(0.0, min(1.0, 0.35 + 0.5 * coverage - 0.2 * residual))
    return heuristic


def _safe_float(value: object, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def render_phrase_translation(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> PhraseRenderResult:
    """Render a chosen phrase parse into readable English without changing it.

    The algorithmic parser remains the source of truth. This helper simply
    verbalizes the selected token meanings and relations so `translate-phrase`
    can produce smoother English while staying pinned to the chosen parse.
    """
    skeleton = str(parse_payload.get("translation_skeleton") or "").strip()
    prompt = _build_phrase_render_prompt(parse_payload, context)
    use_remote = bool(context.get("use_remote", False))
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a constrained phrase renderer for Enochian translation. "
            "You may improve fluency, but you must not introduce any meaning, "
            "relation, or token not present in the supplied parse. "
            f"Historical scope: {llm_context}"
        ),
        name="Phrase Translation Renderer",
        description="Render a chosen phrase parse into readable prose",
        use_remote=use_remote,
        progress_style="silent",
    )
    _attach_query_logging(tool, context)
    progress_callback = _tool_progress_callback(context)
    try:
        _report_render_detail(
            context,
            "waiting for remote model response"
            if use_remote
            else "running local renderer",
        )
        response = tool._run(
            prompt=prompt,
            print_chunks=False,
            stream_callback=None,
            progress_callback=progress_callback,
        )
        _report_render_detail(context, "validating renderer response")
        parsed = _parse_phrase_render_response(
            response.get("response_text", ""),
            fallback=skeleton,
        )
        return PhraseRenderResult(
            rendered_translation=parsed["rendered_translation"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            raw_response=response.get("response_text"),
        )
    except Exception as exc:
        _report_render_detail(context, "falling back to deterministic skeleton")
        return PhraseRenderResult(
            rendered_translation=skeleton,
            confidence=_safe_float(context.get("confidence"), default=0.5),
            reasoning=f"Phrase renderer unavailable; using deterministic skeleton. ({exc})",
            warnings=["LLM phrase rendering unavailable."],
        )


def refine_unknown_phrase_tokens(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> dict[str, object]:
    """Refine ``source=unknown`` token glosses using neighboring token context.

    Phrase translation keeps deterministic parse selection as the source of
    truth. This optional helper asks the LLM only for lexical replacements of
    unknown-source tokens and returns index-scoped edits so callers can merge
    valid replacements without changing parse structure.
    """

    unknown_tokens = parse_payload.get("unknown_tokens")
    if not isinstance(unknown_tokens, Sequence) or isinstance(unknown_tokens, (str, bytes)):
        return {
            "refinements": [],
            "reasoning": "No unknown-token payload was provided for contextual refinement.",
            "warnings": [],
        }
    if not unknown_tokens:
        return {
            "refinements": [],
            "reasoning": "No unknown tokens were present in the selected phrase parse.",
            "warnings": [],
        }

    prompt = _build_unknown_token_refinement_prompt(parse_payload, context)
    use_remote = bool(context.get("use_remote", False))
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a constrained Enochian token gloss refiner. "
            "Return short lexical replacements only for listed unknown-source tokens "
            "using neighboring token context. "
            f"Historical scope: {llm_context}"
        ),
        name="Unknown Token Context Refiner",
        description="Refine unknown token glosses from neighboring context",
        use_remote=use_remote,
        progress_style="silent",
    )
    _attach_query_logging(tool, context)
    progress_callback = _tool_progress_callback(context)

    try:
        _report_render_detail(
            context,
            "waiting for unknown-token context refinement"
            if use_remote
            else "running local unknown-token refinement",
        )
        response = tool._run(
            prompt=prompt,
            print_chunks=False,
            stream_callback=None,
            progress_callback=progress_callback,
        )
        _report_render_detail(context, "validating unknown-token refinement response")
        parsed = _parse_unknown_token_refinement_response(
            response.get("response_text", ""),
            parse_payload=parse_payload,
        )
        parsed["raw_response"] = response.get("response_text")
        return parsed
    except Exception as exc:
        _report_render_detail(context, "falling back to deterministic unknown-token glosses")
        return {
            "refinements": [],
            "reasoning": (
                "Unknown-token context refinement unavailable; keeping deterministic token glosses."
            ),
            "warnings": [f"Unknown-token context refinement unavailable. ({exc})"],
        }


def render_phrase_lay_translation(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> PhraseRenderResult:
    """Produce a plain-English paraphrase for the chosen phrase parse.

    The phrase parser intentionally preserves technical, token-level structure so
    debugging and corpus work can stay inspectable. This companion renderer
    always tries to convert that selected parse into everyday English, giving
    downstream callers a consistently human-friendly reading without replacing
    the parser's source-of-truth skeleton.
    """
    skeleton = str(parse_payload.get("translation_skeleton") or "").strip()
    prompt = _build_phrase_lay_render_prompt(parse_payload, context)
    use_remote = bool(context.get("use_remote", False))
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a plain-English paraphraser for Enochian translation output. "
            "You may simplify archaic or technical wording, but you must stay "
            "within the meanings and relations already present in the supplied "
            f"parse. Historical scope: {llm_context}"
        ),
        name="Phrase Lay Translation Renderer",
        description="Render a chosen phrase parse into everyday English",
        use_remote=use_remote,
        progress_style="silent",
    )
    _attach_query_logging(tool, context)
    progress_callback = _tool_progress_callback(context)
    try:
        _report_render_detail(
            context,
            "waiting for remote model response"
            if use_remote
            else "running local renderer",
        )
        response = tool._run(
            prompt=prompt,
            print_chunks=False,
            stream_callback=None,
            progress_callback=progress_callback,
        )
        _report_render_detail(context, "validating renderer response")
        parsed = _parse_phrase_lay_render_response(
            response.get("response_text", ""),
            fallback=skeleton,
            parse_payload=parse_payload,
        )
        return PhraseRenderResult(
            rendered_translation=parsed["rendered_translation"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            footnoted_translation=parsed["footnoted_translation"],
            translation_footnotes=parsed["translation_footnotes"],
            raw_response=response.get("response_text"),
        )
    except Exception as exc:
        _report_render_detail(context, "falling back to deterministic skeleton")
        fallback_footnoted, fallback_notes = _build_phrase_footnote_fallback(
            parse_payload
        )
        return PhraseRenderResult(
            rendered_translation=skeleton,
            confidence=_safe_float(context.get("confidence"), default=0.5),
            reasoning=(
                "Lay translation renderer unavailable; using deterministic "
                f"skeleton. ({exc})"
            ),
            footnoted_translation=fallback_footnoted,
            translation_footnotes=fallback_notes,
            warnings=["LLM lay translation unavailable."],
        )


def render_phrase_bundle(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> PhraseRenderBundleResult:
    """Render grounded and poetic lay phrasing beside a fixed skeleton.

    Phrase translation now treats the chosen parse skeleton as the authoritative
    technical translation. The optional model call is reserved for a grounded
    lay sentence, a more liberal poetic sentence, and token footnotes so
    remote rendering stays smaller, faster, and less likely to wander back
    into glossary prose.
    """

    skeleton = str(parse_payload.get("translation_skeleton") or "").strip()
    prompt = _build_phrase_bundle_prompt(parse_payload, context)
    use_remote = bool(context.get("use_remote", False))
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a constrained lay phrase renderer for Enochian translation. "
            "Return grounded and poetic plain-English phrase renderings "
            "with token footnotes without "
            "introducing new meaning, relations, or token claims. "
            f"Historical scope: {llm_context}"
        ),
        name="Phrase Render Bundle",
        description="Render lay phrase translation and footnotes in one pass",
        use_remote=use_remote,
        progress_style="silent",
        remote_attempts=2,
        read_timeout_seconds=45.0,
        local_fallback_enabled=False,
        stream_response=False,
    )
    _attach_query_logging(tool, context)
    progress_callback = _tool_progress_callback(context)
    try:
        _report_render_detail(
            context,
            "waiting for remote model response"
            if use_remote
            else "running local renderer",
        )
        response = tool._run(
            prompt=prompt,
            print_chunks=False,
            stream_callback=None,
            progress_callback=progress_callback,
        )
        _report_render_detail(context, "validating bundled render response")
        parsed = _parse_phrase_bundle_response(
            response.get("response_text", ""),
            fallback=skeleton,
            parse_payload=parse_payload,
        )
        confidence = _safe_float(context.get("confidence"), default=0.5)
        return PhraseRenderBundleResult(
            technical_translation=skeleton,
            technical_confidence=confidence,
            technical_reasoning="Technical translation kept deterministic from the chosen parse.",
            lay_translation=str(parsed.get("lay_translation") or skeleton),
            lay_confidence=_safe_float(parsed.get("lay_confidence"), default=confidence),
            lay_reasoning=str(
                parsed.get("lay_reasoning")
                or "Lay translation fell back to the algorithmic phrase skeleton."
            ),
            poetic_translation=str(
                parsed.get("poetic_translation")
                or parsed.get("interpretive_translation")
                or parsed.get("lay_translation")
                or skeleton
            ),
            poetic_confidence=_safe_float(
                parsed.get("poetic_confidence") or parsed.get("interpretive_confidence"),
                default=_safe_float(parsed.get("lay_confidence"), default=confidence),
            ),
            poetic_reasoning=str(
                parsed.get("poetic_reasoning")
                or parsed.get("interpretive_reasoning")
                or "Poetic translation fell back to the grounded lay reading."
            ),
            contextual_lay_translation=str(
                parsed.get("contextual_lay_translation")
                or parsed.get("lay_translation")
                or skeleton
            ),
            contextual_lay_confidence=_safe_float(
                parsed.get("contextual_lay_confidence"),
                default=_safe_float(parsed.get("lay_confidence"), default=confidence),
            ),
            contextual_lay_reasoning=str(
                parsed.get("contextual_lay_reasoning")
                or parsed.get("lay_reasoning")
                or "Contextual lay translation fell back to the rank-1 grounded reading."
            ),
            contextual_poetic_translation=str(
                parsed.get("contextual_poetic_translation")
                or parsed.get("contextual_interpretive_translation")
                or parsed.get("poetic_translation")
                or parsed.get("interpretive_translation")
                or parsed.get("lay_translation")
                or skeleton
            ),
            contextual_poetic_confidence=_safe_float(
                parsed.get("contextual_poetic_confidence")
                or parsed.get("contextual_interpretive_confidence"),
                default=_safe_float(
                    parsed.get("poetic_confidence") or parsed.get("interpretive_confidence"),
                    default=_safe_float(parsed.get("lay_confidence"), default=confidence),
                ),
            ),
            contextual_poetic_reasoning=str(
                parsed.get("contextual_poetic_reasoning")
                or parsed.get("contextual_interpretive_reasoning")
                or parsed.get("poetic_reasoning")
                or parsed.get("interpretive_reasoning")
                or "Contextual poetic translation fell back to the rank-1 poetic reading."
            ),
            contextual_selected_parse_rank=(
                int(parsed.get("contextual_selected_parse_rank"))
                if isinstance(parsed.get("contextual_selected_parse_rank"), int)
                and int(parsed.get("contextual_selected_parse_rank")) > 0
                else 1
            ),
            contextual_interpretive_translation=str(
                parsed.get("contextual_poetic_translation")
                or parsed.get("contextual_interpretive_translation")
                or parsed.get("poetic_translation")
                or parsed.get("interpretive_translation")
                or parsed.get("lay_translation")
                or skeleton
            ),
            contextual_interpretive_confidence=_safe_float(
                parsed.get("contextual_poetic_confidence")
                or parsed.get("contextual_interpretive_confidence"),
                default=_safe_float(
                    parsed.get("poetic_confidence") or parsed.get("interpretive_confidence"),
                    default=_safe_float(parsed.get("lay_confidence"), default=confidence),
                ),
            ),
            contextual_interpretive_reasoning=str(
                parsed.get("contextual_poetic_reasoning")
                or parsed.get("contextual_interpretive_reasoning")
                or parsed.get("poetic_reasoning")
                or parsed.get("interpretive_reasoning")
                or "Contextual poetic translation fell back to the rank-1 poetic reading."
            ),
            interpretive_translation=str(
                parsed.get("poetic_translation")
                or parsed.get("interpretive_translation")
                or parsed.get("lay_translation")
                or skeleton
            ),
            interpretive_confidence=_safe_float(
                parsed.get("poetic_confidence") or parsed.get("interpretive_confidence"),
                default=_safe_float(parsed.get("lay_confidence"), default=confidence),
            ),
            interpretive_reasoning=str(
                parsed.get("poetic_reasoning")
                or parsed.get("interpretive_reasoning")
                or "Poetic translation fell back to the grounded lay reading."
            ),
            footnoted_translation=parsed.get("footnoted_translation"),
            translation_footnotes=list(parsed.get("translation_footnotes") or []),
            raw_response=response.get("response_text"),
        )
    except Exception as exc:
        _report_render_detail(context, "falling back to deterministic skeleton")
        fallback_footnoted, fallback_notes = _build_phrase_footnote_fallback(
            parse_payload
        )
        fallback_lay = _strip_footnote_markers(fallback_footnoted) or skeleton
        fallback_confidence = _safe_float(context.get("confidence"), default=0.5)
        return PhraseRenderBundleResult(
            technical_translation=skeleton,
            technical_confidence=fallback_confidence,
            technical_reasoning="Technical translation kept deterministic from the chosen parse.",
            lay_translation=fallback_lay,
            lay_confidence=fallback_confidence,
            lay_reasoning=(
                "Lay renderer unavailable; using deterministic "
                f"fallback. ({exc})"
            ),
            poetic_translation=fallback_lay,
            poetic_confidence=fallback_confidence,
            poetic_reasoning=(
                "Poetic renderer unavailable; using deterministic "
                f"fallback. ({exc})"
            ),
            contextual_lay_translation=fallback_lay,
            contextual_lay_confidence=fallback_confidence,
            contextual_lay_reasoning=(
                "Contextual lay renderer unavailable; using rank-1 fallback. "
                f"({exc})"
            ),
            contextual_poetic_translation=fallback_lay,
            contextual_poetic_confidence=fallback_confidence,
            contextual_poetic_reasoning=(
                "Contextual poetic renderer unavailable; using rank-1 fallback. "
                f"({exc})"
            ),
            contextual_selected_parse_rank=1,
            contextual_interpretive_translation=fallback_lay,
            contextual_interpretive_confidence=fallback_confidence,
            contextual_interpretive_reasoning=(
                "Contextual poetic renderer unavailable; using rank-1 fallback. "
                f"({exc})"
            ),
            interpretive_translation=fallback_lay,
            interpretive_confidence=fallback_confidence,
            interpretive_reasoning=(
                "Poetic renderer unavailable; using deterministic "
                f"fallback. ({exc})"
            ),
            footnoted_translation=fallback_footnoted,
            translation_footnotes=fallback_notes,
            warnings=["LLM lay phrase rendering unavailable."],
        )


def _build_phrase_bundle_prompt(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> str:
    """Build the compact two-track lay-render prompt used by phrase translation.

    The technical translation now stays deterministic, so the model only needs
    enough context to produce a grounded lay reading, a more liberal
    poetic reading, and token footnotes. This smaller contract keeps
    phrase rendering faster and more reliable on long inputs.
    """

    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    schema = json.dumps(
        {
            "lay_translation": "<rank-1-only natural English prose reading>",
            "lay_confidence": 0.0,
            "lay_reasoning": "<brief note explaining the rank-1 grounded prose reading>",
            "poetic_translation": "<rank-1-only more expressive prose reading; may reorder/add connective helper words while staying grounded>",
            "poetic_confidence": 0.0,
            "poetic_reasoning": "<brief note explaining the rank-1 poetic choices>",
            "contextual_selected_parse_rank": 1,
            "contextual_lay_translation": "<best prose reading chosen from parse ranks 1..3>",
            "contextual_lay_confidence": 0.0,
            "contextual_lay_reasoning": "<brief note explaining which parse rank was chosen and why>",
            "contextual_poetic_translation": "<more expressive contextual prose reading from parse ranks 1..3>",
            "contextual_poetic_confidence": 0.0,
            "contextual_poetic_reasoning": "<brief note explaining contextual poetic choices and selected parse rank>",
            "footnoted_translation": "<same short translation with [^1] style markers>",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "<original token>",
                    "rendered_text": "<specific grounded English chunk, often 1-6 words>",
                    "explanation": "<brief grounded explanation for this choice>",
                }
            ],
        },
        ensure_ascii=False,
    )
    example = json.dumps(
        {
            "lay_translation": "the holy law still stands",
            "lay_confidence": 0.72,
            "lay_reasoning": "Uses rank-1 token choices only and rewrites them as readable prose.",
            "poetic_translation": "the sacred order still holds firm",
            "poetic_confidence": 0.80,
            "poetic_reasoning": "Uses rank-1 choices only but goes further stylistically.",
            "contextual_selected_parse_rank": 2,
            "contextual_lay_translation": "the sacred law yet remains in force",
            "contextual_lay_confidence": 0.77,
            "contextual_lay_reasoning": "Parse rank 2 produced a cleaner sentence with fewer awkward fragments.",
            "contextual_poetic_translation": "the holy ordinance endures and will not fail",
            "contextual_poetic_confidence": 0.83,
            "contextual_poetic_reasoning": "Chose parse rank 2 for coherence, then elevated diction while preserving meaning.",
            "footnoted_translation": "holy [^1] law [^2] still stands [^3]",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "MAD",
                    "rendered_text": "holy",
                    "explanation": "Keeps the sacred quality from the selected gloss.",
                },
                {
                    "index": 2,
                    "source_token": "CAF",
                    "rendered_text": "law",
                    "explanation": "Uses a grounded governing sense that reads naturally in English.",
                },
                {
                    "index": 3,
                    "source_token": "PRAC",
                    "rendered_text": "still stands",
                    "explanation": "Keeps the abiding sense while smoothing it into idiomatic English.",
                },
            ],
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a constrained Enochian lay phrase renderer.",
            (
                "TASK: Return four prose outputs and one footnote block: "
                "(1) rank-1 lay, (2) rank-1 poetic, "
                "(3) contextual lay from parse ranks 1..3, "
                "(4) contextual poetic from parse ranks 1..3."
            ),
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Stay grounded in supplied token choices, relations, skeletons, semantic cores, and alternates.",
            "- `lay_translation` MUST use rank-1 parse data only (`rank1_parse_payload`) and must read like normal prose English, never partial gloss fragments.",
            "- `poetic_translation` MUST also use rank-1 parse data only, but can go further stylistically while remaining readable prose.",
            "- `contextual_lay_translation` may select the best coherent option from `contextual_parse_options` (parse ranks 1..3).",
            "- `contextual_poetic_translation` may also select from parse ranks 1..3 and push farther stylistically.",
            "- Set `contextual_selected_parse_rank` to the rank used for contextual outputs.",
            "- Both lay and poetic outputs are allowed to add helper/filler words (articles, pronouns, auxiliaries, prepositions, connective tissue) when needed for grammatical English.",
            "- Prefer one coherent reading over mirroring the source token order mechanically.",
            "- Unless the phrase is extremely short, do not simply copy the technical skeleton word-for-word.",
            "- Do not return token-by-token comma lists, stacked prepositional fragments, or glossary dumps.",
            "- Repair awkward fragment chains such as repeated prepositions or noun piles into the most plausible grammatical English you can support.",
            "- Poetic outputs may reorder aggressively and smooth edges into compelling prose, but cannot invent unsupported token meanings or relations.",
            "- Poetic outputs do not need to differ from lay outputs when the lay line is already coherent and expressive.",
            "- Return exactly one footnote entry per token choice, in source order.",
            "- Footnotes MUST stay anchored to rank-1 token choices (`token_choices`) only.",
            "- Each `rendered_text` should preserve the most specific grounded sense available for that token, often in 1-6 words.",
            "- Do not flatten semantically rich glosses into weaker generic abstractions when a fuller supported gloss exists.",
            "- If a gloss contains vivid supported detail such as `ornaments of brightness`, keep that richer phrasing instead of reducing it to a blander one-word label.",
            "- Never emit raw source tokens or bracket placeholders such as `[TOKEN]` in the English output.",
            "- Do not emit `unresolved term`; use the best grounded gloss available from the payload and smooth around weak tokens.",
            "- Explanations must stay grounded in the selected token glosses, alternates, and relations.",
            "- Return STRICT JSON only.",
            "PARSE PAYLOAD:",
            payload,
            "SCHEMA:",
            schema,
            "EXAMPLE (format and brevity only; do not copy wording):",
            example,
        ]
    )


def _build_unknown_token_refinement_prompt(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> str:
    """Build a constrained prompt for unknown-token-only contextual refinement.

    This prompt intentionally limits the LLM to short lexical replacements for
    listed unknown tokens. The parse skeleton and known neighbor glosses are
    provided as context, while all structure-changing behavior is prohibited.
    """

    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    schema = json.dumps(
        {
            "refinements": [
                {
                    "index": 1,
                    "token": "TOKEN",
                    "replacement": "short lexical gloss",
                    "confidence": 0.0,
                    "reasoning": "brief grounding note",
                }
            ],
            "reasoning": "overall note",
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a constrained unknown-token gloss refiner.",
            "TASK: Suggest short lexical replacements only for listed unknown tokens.",
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Only refine tokens listed under `unknown_tokens`.",
            "- Preserve token order, parse relations, and sentence structure.",
            "- Keep each `replacement` to short lexical wording (typically 1-4 words).",
            "- Do not invent new entities, events, or theological claims.",
            "- Do not emit source tokens, bracket placeholders, or `unresolved term`.",
            "- Return STRICT JSON only.",
            "PARSE PAYLOAD:",
            payload,
            "SCHEMA:",
            schema,
        ]
    )


def _parse_unknown_token_refinement_response(
    payload: str,
    *,
    parse_payload: dict[str, object],
) -> dict[str, object]:
    """Parse and validate unknown-token refinement responses from the LLM."""

    data = _load_json_payload(payload)
    if data is None:
        return {
            "refinements": [],
            "reasoning": "Unknown-token refiner returned non-JSON output.",
            "warnings": ["Unknown-token context refinement returned non-JSON output."],
        }

    unknown_tokens = parse_payload.get("unknown_tokens")
    listed_tokens = (
        list(unknown_tokens)
        if isinstance(unknown_tokens, Sequence) and not isinstance(unknown_tokens, (str, bytes))
        else []
    )
    allowed_indexes: dict[int, str] = {}
    for item in listed_tokens:
        if not isinstance(item, Mapping):
            continue
        raw_index = item.get("index")
        token = str(item.get("token") or "").strip().upper()
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index > 0 and token:
            allowed_indexes[index] = token

    raw_refinements = data.get("refinements")
    refinements: list[dict[str, object]] = []
    used_indexes: set[int] = set()
    if isinstance(raw_refinements, Sequence) and not isinstance(raw_refinements, (str, bytes)):
        for item in raw_refinements:
            if not isinstance(item, Mapping):
                continue
            raw_index = item.get("index")
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            token = allowed_indexes.get(index)
            if token is None or index in used_indexes:
                continue

            replacement_raw = item.get("replacement")
            if not isinstance(replacement_raw, str) or not replacement_raw.strip():
                continue
            cleaned = clean_lexical_gloss(replacement_raw, token=token)
            if cleaned is None:
                cleaned = sanitize_human_gloss(replacement_raw, token=token)
            if not isinstance(cleaned, str) or not cleaned.strip():
                continue
            replacement = cleaned.strip()
            if replacement.upper() == unresolved_token_gloss(token):
                continue
            if replacement.lower() in {"unresolved term", token.lower()}:
                continue

            confidence = _safe_float(item.get("confidence"), default=0.0)
            reasoning = item.get("reasoning")
            refinements.append(
                {
                    "index": index,
                    "token": token,
                    "replacement": replacement,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reasoning": (
                        reasoning.strip()
                        if isinstance(reasoning, str) and reasoning.strip()
                        else "Contextual unknown-token refinement."
                    ),
                }
            )
            used_indexes.add(index)

    reasoning = data.get("reasoning")
    return {
        "refinements": refinements,
        "reasoning": (
            reasoning.strip()
            if isinstance(reasoning, str) and reasoning.strip()
            else (
                "Applied contextual unknown-token refinements."
                if refinements
                else "No valid unknown-token refinements were returned."
            )
        ),
        "warnings": [],
    }


def _build_phrase_render_prompt(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> str:
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    schema = json.dumps(
        {
            "rendered_translation": "<readable translation>",
            "confidence": 0.0,
            "reasoning": "<brief note anchored to supplied parse>",
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a constrained Enochian phrase renderer.",
            "TASK: Rewrite the supplied translation skeleton into readable English.",
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Use ONLY the supplied token choices and relations.",
            "- Do not add, remove, or replace meanings.",
            "- Do not infer extra syntax beyond the relation inventory already supplied.",
            "- Return STRICT JSON only.",
            "PARSE PAYLOAD:",
            payload,
            "SCHEMA:",
            schema,
        ]
    )


def _build_phrase_lay_render_prompt(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> str:
    """Build the optional layman's-speak prompt for phrase translation.

    When phrase-level LLM rendering is enabled, this prompt sits beside the
    stricter historical renderer so the pipeline can expose two complementary
    views of the same parse: one close to the selected glosses, and one
    phrased for readers who just want the idea in plain English.
    """
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    schema = json.dumps(
        {
            "rendered_translation": "<plausible, roughly grammatical English sentence or clause that expresses one coherent reading of the phrase>",
            "footnoted_translation": "<same short translation with [^1] style markers>",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "<original token>",
                    "rendered_text": "<specific grounded English chunk, often 1-6 words>",
                    "explanation": "<brief grounded explanation for this choice>",
                }
            ],
            "confidence": 0.0,
            "reasoning": "<brief note explaining the chosen reading>",
        },
        ensure_ascii=False,
    )
    example = json.dumps(
        {
            "rendered_translation": "the holy law still stands",
            "footnoted_translation": "holy [^1] law [^2] still stands [^3]",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "MAD",
                    "rendered_text": "holy",
                    "explanation": "Keeps the sacred quality from the selected gloss.",
                },
                {
                    "index": 2,
                    "source_token": "CAF",
                    "rendered_text": "law",
                    "explanation": "Uses a grounded governing sense that reads naturally in English.",
                },
                {
                    "index": 3,
                    "source_token": "PRAC",
                    "rendered_text": "still stands",
                    "explanation": "Keeps the abiding sense while smoothing it into idiomatic English.",
                },
            ],
            "confidence": 0.72,
            "reasoning": "Chooses one coherent everyday reading while staying anchored to the chosen parse.",
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a plain-English Enochian phrase explainer.",
            "TASK: Rewrite the supplied translation skeleton as one plausible, approximately grammatical English interpretation that a lay reader can immediately understand.",
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Use ONLY the supplied token choices, relations, and skeleton.",
            "- Keep the same core meaning, but replace technical or archaic phrasing with sane everyday English.",
            "- Do not add any new actors, actions, objects, or claims.",
            "- Your job is to choose one coherent reading, not to dump glosses.",
            "- `rendered_translation` must read like normal English, not like token-by-token notes.",
            "- Treat the line as a plausible hypothetical interpretation of what the sentence could mean.",
            "- You may add helper words, articles, and prepositions when they are needed to make the reading feel grammatical.",
            "- Prefer a clear sentence-level idea over mirroring the source token order mechanically.",
            "- Never restate full token definitions, example sentences, or long alternation chains in `rendered_translation`.",
            "- Return exactly one footnote entry per token choice, in source order.",
            "- Each `rendered_text` should preserve the most specific grounded sense available for that token, often in 1-6 words.",
            "- Do not flatten semantically rich glosses into weaker generic abstractions when a fuller supported gloss exists.",
            "- If a gloss contains vivid supported detail such as `ornaments of brightness`, keep that richer phrasing instead of reducing it to a blander one-word label.",
            "- Never emit raw source tokens or bracket placeholders such as `[TOKEN]` in the English output.",
            "- If a token remains weakly supported, use the best grounded gloss available from the payload and smooth around it instead of echoing the source token.",
            "- Explanations must stay grounded in the selected token glosses, alternates, and relations.",
            "- If you add helper words for readable English, explain that smoothing in the relevant footnote.",
            "- Return STRICT JSON only.",
            "PARSE PAYLOAD:",
            payload,
            "SCHEMA:",
            schema,
            "EXAMPLE (format and brevity only; do not copy wording):",
            example,
        ]
    )


def _parse_phrase_render_response(
    payload: str,
    *,
    fallback: str,
) -> dict[str, object]:
    data = _load_json_payload(payload)
    if data is None:
        return {
            "rendered_translation": fallback,
            "confidence": 0.5,
            "reasoning": "Renderer returned non-JSON output; using deterministic skeleton.",
        }
    rendered = data.get("rendered_translation")
    if not isinstance(rendered, str) or not rendered.strip():
        rendered = fallback
    confidence = _safe_float(data.get("confidence"), default=0.5)
    reasoning = data.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = "Rendered directly from the chosen parse."
    return {
        "rendered_translation": rendered.strip(),
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": reasoning.strip(),
    }


def _parse_phrase_lay_render_response(
    payload: str,
    *,
    fallback: str,
    parse_payload: dict[str, object],
) -> dict[str, object]:
    """Parse lay-translation JSON and rebuild footnotes when the model drifts.

    The lay renderer now carries both the human-readable translation and the
    footnoted closing block needed by the CLI. This parser keeps that contract
    stable by validating the footnote list against the chosen parse and falling
    back to a deterministic token-by-token view when the model omits or mangles
    the new structure.
    """

    base = _parse_phrase_render_response(payload, fallback=fallback)
    fallback_footnoted, fallback_notes = _build_phrase_footnote_fallback(parse_payload)
    data = _load_json_payload(payload)
    if data is None:
        return {
            **base,
            "footnoted_translation": fallback_footnoted,
            "translation_footnotes": fallback_notes,
        }

    notes = _coerce_phrase_footnotes(data.get("translation_footnotes"), parse_payload)
    footnoted = data.get("footnoted_translation")
    if not isinstance(footnoted, str) or not footnoted.strip():
        footnoted = _build_footnoted_translation(notes)
    normalized_markers = [f"[^{index}]" for index in range(1, len(notes) + 1)]
    if any(marker not in footnoted for marker in normalized_markers):
        footnoted = _build_footnoted_translation(notes)
    footnoted = _build_footnoted_translation(notes)
    base["rendered_translation"] = _normalize_lay_translation(
        str(base.get("rendered_translation") or ""),
        footnoted,
        fallback_translation=fallback,
    )
    base["rendered_translation"] = _replace_token_placeholders_in_translation(
        str(base.get("rendered_translation") or ""),
        parse_payload=parse_payload,
    )

    return {
        **base,
        "footnoted_translation": footnoted.strip(),
        "translation_footnotes": notes,
    }


def _parse_phrase_bundle_response(
    payload: str,
    *,
    fallback: str,
    parse_payload: dict[str, object],
) -> dict[str, object]:
    """Validate a bundled phrase-render response and rebuild drifted fields.

    The bundled render call returns technical prose, lay prose, and footnotes in
    one JSON document. This parser applies the same local validation strategy
    used elsewhere so we can avoid follow-up LLM repair calls when the model
    omits or mangles part of the schema.
    """

    data = _load_json_payload(payload)
    if data is None:
        fallback_footnoted, fallback_notes = _build_phrase_footnote_fallback(parse_payload)
        fallback_lay = _strip_footnote_markers(fallback_footnoted) or fallback
        return {
            "technical_translation": fallback,
            "technical_confidence": 0.5,
            "technical_reasoning": (
                "Bundled renderer returned non-JSON output; using deterministic skeleton."
            ),
            "lay_translation": fallback_lay,
            "lay_confidence": 0.5,
            "lay_reasoning": (
                "Bundled renderer returned non-JSON output; using deterministic lay fallback."
            ),
            "poetic_translation": fallback_lay,
            "poetic_confidence": 0.5,
            "poetic_reasoning": (
                "Bundled renderer returned non-JSON output; using deterministic poetic fallback."
            ),
            "contextual_lay_translation": fallback_lay,
            "contextual_lay_confidence": 0.5,
            "contextual_lay_reasoning": (
                "Bundled renderer returned non-JSON output; using rank-1 lay fallback."
            ),
            "contextual_poetic_translation": fallback_lay,
            "contextual_poetic_confidence": 0.5,
            "contextual_poetic_reasoning": (
                "Bundled renderer returned non-JSON output; using rank-1 poetic fallback."
            ),
            "contextual_selected_parse_rank": 1,
            "contextual_interpretive_translation": fallback_lay,
            "contextual_interpretive_confidence": 0.5,
            "contextual_interpretive_reasoning": (
                "Bundled renderer returned non-JSON output; using rank-1 poetic fallback."
            ),
            "interpretive_translation": fallback_lay,
            "interpretive_confidence": 0.5,
            "interpretive_reasoning": (
                "Bundled renderer returned non-JSON output; using deterministic poetic fallback."
            ),
            "footnoted_translation": fallback_footnoted,
            "translation_footnotes": fallback_notes,
        }

    technical_payload = json.dumps(
        {
            "rendered_translation": data.get("technical_translation"),
            "confidence": data.get("technical_confidence"),
            "reasoning": data.get("technical_reasoning"),
        },
        ensure_ascii=False,
    )
    lay_payload = json.dumps(
        {
            "rendered_translation": data.get("lay_translation"),
            "footnoted_translation": data.get("footnoted_translation"),
            "translation_footnotes": data.get("translation_footnotes"),
            "confidence": data.get("lay_confidence"),
            "reasoning": data.get("lay_reasoning"),
        },
        ensure_ascii=False,
    )
    poetic_payload = json.dumps(
        {
            "rendered_translation": data.get("poetic_translation")
            or data.get("interpretive_translation"),
            "confidence": data.get("poetic_confidence")
            or data.get("interpretive_confidence"),
            "reasoning": data.get("poetic_reasoning")
            or data.get("interpretive_reasoning"),
        },
        ensure_ascii=False,
    )
    technical = _parse_phrase_render_response(technical_payload, fallback=fallback)
    lay = _parse_phrase_lay_render_response(
        lay_payload,
        fallback=fallback,
        parse_payload=parse_payload,
    )
    poetic = _parse_phrase_render_response(
        poetic_payload,
        fallback=lay["rendered_translation"] or fallback,
    )
    normalized_poetic = _normalize_poetic_translation(
        _replace_token_placeholders_in_translation(
            poetic["rendered_translation"],
            parse_payload=parse_payload,
        ),
        lay_translation=lay["rendered_translation"],
    )
    contextual_selected_rank = _contextual_selected_parse_rank(
        data.get("contextual_selected_parse_rank"),
        parse_payload=parse_payload,
    )
    contextual_parse_payload = _contextual_parse_payload(
        parse_payload,
        parse_rank=contextual_selected_rank,
    )
    contextual_lay_payload = json.dumps(
        {
            "rendered_translation": data.get("contextual_lay_translation"),
            "confidence": data.get("contextual_lay_confidence"),
            "reasoning": data.get("contextual_lay_reasoning"),
        },
        ensure_ascii=False,
    )
    contextual_lay = _parse_phrase_render_response(
        contextual_lay_payload,
        fallback=lay["rendered_translation"] or fallback,
    )
    normalized_contextual_lay = _replace_token_placeholders_in_translation(
        contextual_lay["rendered_translation"],
        parse_payload=contextual_parse_payload,
    )
    if not normalized_contextual_lay.strip():
        normalized_contextual_lay = lay["rendered_translation"]
    contextual_poetic_payload = json.dumps(
        {
            "rendered_translation": data.get("contextual_poetic_translation")
            or data.get("contextual_interpretive_translation"),
            "confidence": data.get("contextual_poetic_confidence")
            or data.get("contextual_interpretive_confidence"),
            "reasoning": data.get("contextual_poetic_reasoning")
            or data.get("contextual_interpretive_reasoning"),
        },
        ensure_ascii=False,
    )
    contextual_poetic = _parse_phrase_render_response(
        contextual_poetic_payload,
        fallback=normalized_poetic or normalized_contextual_lay or fallback,
    )
    normalized_contextual_poetic = _normalize_poetic_translation(
        _replace_token_placeholders_in_translation(
            contextual_poetic["rendered_translation"],
            parse_payload=contextual_parse_payload,
        ),
        lay_translation=normalized_contextual_lay,
    )
    if not normalized_contextual_poetic.strip():
        normalized_contextual_poetic = normalized_poetic
    return {
        "technical_translation": technical["rendered_translation"],
        "technical_confidence": technical["confidence"],
        "technical_reasoning": technical["reasoning"],
        "lay_translation": lay["rendered_translation"],
        "lay_confidence": lay["confidence"],
        "lay_reasoning": lay["reasoning"],
        "poetic_translation": normalized_poetic,
        "poetic_confidence": poetic["confidence"],
        "poetic_reasoning": poetic["reasoning"],
        "contextual_lay_translation": normalized_contextual_lay,
        "contextual_lay_confidence": contextual_lay["confidence"],
        "contextual_lay_reasoning": contextual_lay["reasoning"],
        "contextual_poetic_translation": normalized_contextual_poetic,
        "contextual_poetic_confidence": contextual_poetic["confidence"],
        "contextual_poetic_reasoning": contextual_poetic["reasoning"],
        "contextual_selected_parse_rank": contextual_selected_rank,
        "contextual_interpretive_translation": normalized_contextual_poetic,
        "contextual_interpretive_confidence": contextual_poetic["confidence"],
        "contextual_interpretive_reasoning": contextual_poetic["reasoning"],
        "interpretive_translation": normalized_poetic,
        "interpretive_confidence": poetic["confidence"],
        "interpretive_reasoning": poetic["reasoning"],
        "footnoted_translation": lay["footnoted_translation"],
        "translation_footnotes": lay["translation_footnotes"],
    }


def _contextual_selected_parse_rank(
    value: object,
    *,
    parse_payload: Mapping[str, object],
) -> int:
    """Choose a valid contextual parse rank from the bundled payload options.

    Contextual phrase rendering may select among parse ranks 1..3. This helper
    validates model output against the provided options so downstream
    normalization always has one concrete parse payload to reference.
    """

    options = parse_payload.get("contextual_parse_options")
    available: set[int] = set()
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            parse_rank = option.get("parse_rank")
            if isinstance(parse_rank, int) and parse_rank > 0:
                available.add(parse_rank)
    if not available:
        return 1
    if isinstance(value, int) and value in available:
        return value
    return min(available)


def _contextual_parse_payload(
    parse_payload: Mapping[str, object],
    *,
    parse_rank: int,
) -> dict[str, object]:
    """Return the parse payload matching ``parse_rank`` for contextual cleanup.

    Contextual lay/poetic outputs may select rank 1..3 parse options. Placeholder
    replacement and lexical cleanup need the token inventory for the selected
    parse rank, so this helper resolves that payload deterministically.
    """

    options = parse_payload.get("contextual_parse_options")
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        for option in options:
            if not isinstance(option, Mapping):
                continue
            option_rank = option.get("parse_rank")
            if isinstance(option_rank, int) and option_rank == parse_rank:
                return dict(option)
    rank1_payload = parse_payload.get("rank1_parse_payload")
    if isinstance(rank1_payload, Mapping):
        return dict(rank1_payload)
    return dict(parse_payload)


def _coerce_phrase_footnotes(
    raw_value: object,
    parse_payload: dict[str, object],
) -> list[dict[str, object]]:
    """Normalize lay-render footnotes so downstream formatting can trust them.

    The final markdown-style output depends on one explanation per source token.
    This helper validates that shape against the chosen parse, preserving usable
    model output when possible and rebuilding the whole list when it is not.
    """

    token_choices = parse_payload.get("token_choices")
    expected = token_choices if isinstance(token_choices, Sequence) else []
    if not isinstance(raw_value, list) or len(raw_value) != len(expected):
        return _build_phrase_footnote_fallback(parse_payload)[1]

    normalized: list[dict[str, object]] = []
    for index, (entry, token_choice) in enumerate(zip(raw_value, expected, strict=False), start=1):
        if not isinstance(entry, Mapping) or not isinstance(token_choice, Mapping):
            return _build_phrase_footnote_fallback(parse_payload)[1]
        source_token = str(token_choice.get("token") or f"TOKEN_{index}").upper()
        rendered_text = _normalize_rendered_text(
            entry.get("rendered_text"),
            token_choice,
            source_token,
        )
        if rendered_text is None:
            rendered_text = _fallback_rendered_text(token_choice, source_token)
        explanation = entry.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = _fallback_footnote_explanation(token_choice, source_token)
        normalized.append(
            {
                "index": index,
                "source_token": source_token,
                "rendered_text": rendered_text,
                "explanation": explanation.strip(),
            }
        )
    return normalized


def _build_phrase_footnote_fallback(
    parse_payload: dict[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Return deterministic footnotes when lay rendering cannot provide them.

    The phrase pipeline must always be able to close with a readable
    token-aligned explanation block. When the LLM output is empty or malformed,
    this helper synthesizes a conservative fallback directly from the selected
    parse so callers still get a stable final section.
    """

    token_choices = parse_payload.get("token_choices")
    choices = token_choices if isinstance(token_choices, Sequence) else []
    notes: list[dict[str, object]] = []
    for index, token_choice in enumerate(choices, start=1):
        if not isinstance(token_choice, Mapping):
            continue
        source_token = str(token_choice.get("token") or f"TOKEN_{index}").upper()
        notes.append(
            {
                "index": index,
                "source_token": source_token,
                "rendered_text": _fallback_rendered_text(token_choice, source_token),
                "explanation": _fallback_footnote_explanation(token_choice, source_token),
            }
        )
    return _build_footnoted_translation(notes), notes


def _build_footnoted_translation(notes: Sequence[Mapping[str, object]]) -> str:
    """Assemble a markdown-footnoted line from normalized token notes.

    The CLI wants a single closing translation line with inline markers, while
    JSON consumers want the structured note list. Building the combined string
    here keeps the two representations derived from the same normalized data.
    """

    segments: list[str] = []
    for note in notes:
        rendered_text = note.get("rendered_text")
        index = note.get("index")
        if not isinstance(rendered_text, str) or not rendered_text.strip():
            continue
        if not isinstance(index, int):
            continue
        segments.append(f"{rendered_text.strip()} [^{index}]")
    return " ".join(segments).strip()


def _fallback_rendered_text(token_choice: Mapping[str, object], token: str) -> str:
    """Choose a human-facing rendered chunk for fallback footnotes.

    The lay renderer may fail precisely on difficult phrases. Falling back to
    the sanitized token definition keeps the final translation explicit while
    still respecting the chosen parse and the placeholder-cleanup rules.
    """

    primary = _preferred_primary_gloss(token_choice, token)
    if primary is not None:
        return primary
    runner_up = _preferred_runner_up_gloss(token_choice, token)
    if runner_up is not None:
        return runner_up
    alternate = _alternate_gloss_fallback(token_choice, token)
    if alternate is not None:
        return alternate
    dictionary_rescue_gloss = _dictionary_rescue_gloss(token_choice, token)
    if dictionary_rescue_gloss is not None:
        return dictionary_rescue_gloss
    return _human_facing_unresolved_gloss()


def _normalize_rendered_text(
    rendered_text: object,
    token_choice: Mapping[str, object],
    token: str,
) -> str | None:
    """Coerce footnote chunks toward short lay-readable concepts.

    The lay renderer now aims for compact token-level chunks, but remote models
    sometimes drift back into whole-definition paraphrases. This helper keeps
    those chunks concise before the CLI turns them into the final footnoted
    translation line.
    """

    compact = _compact_lay_gloss(rendered_text, token=token)
    if compact is not None:
        return compact
    return _preferred_primary_gloss(token_choice, token)


def _compact_lay_gloss(text: object, *, token: str) -> str | None:
    """Clean a gloss without truncating it into nonsense fragments.

    Surface gloss selection should already have chosen the best lexical source.
    This helper now only normalizes that wording so footnotes do not regress
    into `X signifies the`-style fragments.
    """

    if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
        semantic_gloss = semantic_core_gloss(text)
        if semantic_gloss is not None:
            return semantic_gloss
    cleaned = clean_lexical_gloss(text, token=token)
    if cleaned is not None:
        return cleaned
    sanitized = sanitize_human_gloss(text, token=token)
    if sanitized is None:
        return None
    return " ".join(sanitized.split()).strip()


def _normalize_lay_translation(
    rendered_translation: str,
    footnoted_translation: str,
    *,
    fallback_translation: str = "",
) -> str:
    """Keep sentence-level lay prose and fall back to the skeleton when needed.

    The lay translation exists to sound like an actual sentence, not just a
    marker-stripped footnote line. We therefore keep the model's sentence-level
    wording unless it is empty or clearly drifts into punctuation-heavy
    glossary prose, in which case the deterministic skeleton is a better
    fallback than a token-by-token footnote strip.
    """

    cleaned = " ".join(rendered_translation.split())
    fallback_cleaned = " ".join(fallback_translation.split())
    if not cleaned:
        return fallback_cleaned or _strip_footnote_markers(footnoted_translation)
    if not _looks_overexpanded_lay_translation(cleaned):
        return cleaned
    return fallback_cleaned or cleaned


def _normalize_poetic_translation(
    rendered_translation: str,
    *,
    lay_translation: str,
) -> str:
    """Normalize poetic output without forcing fake divergence from the lay line.

    The poetic translation is allowed to stay close to the grounded lay
    sentence when that already reads well. We only step in when the field is
    missing entirely, in which case the grounded lay translation is still more
    honest than an empty string.
    """

    cleaned = " ".join(rendered_translation.split())
    lay_cleaned = " ".join(lay_translation.split())
    if not cleaned:
        return lay_cleaned
    return cleaned


def _replace_token_placeholders_in_translation(
    text: str,
    *,
    parse_payload: Mapping[str, object],
) -> str:
    """Replace raw `[TOKEN]` placeholders with grounded glosses from the parse.

    The lay renderer is supposed to produce readable English, but remote model
    output can still occasionally surface raw bracketed source tokens. We
    already carry token-level fallback glosses in the parse payload, so this
    sanitizer swaps those placeholders out before the CLI ever prints them.
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return cleaned

    token_choices = parse_payload.get("token_choices")
    choices = token_choices if isinstance(token_choices, Sequence) else []
    choice_lookup = {
        str(choice.get("token") or "").upper(): choice
        for choice in choices
        if isinstance(choice, Mapping) and str(choice.get("token") or "").strip()
    }

    def _replacement(match: re.Match[str]) -> str:
        token = match.group(1).strip().upper()
        token_choice = choice_lookup.get(token)
        if token_choice is None:
            return _HUMAN_UNRESOLVED_GLOSS
        return _fallback_rendered_text(token_choice, token)

    return " ".join(_RAW_TOKEN_PLACEHOLDER_RE.sub(_replacement, cleaned).split())


def _looks_overexpanded_lay_translation(text: str) -> bool:
    """Detect when a lay translation has drifted into glossary-like prose."""

    if sum(text.count(marker) for marker in ";:") >= 1:
        return True
    if text.count(",") >= 3:
        return True
    lowered = text.lower()
    glossary_markers = (
        " including ",
        " namely ",
        " such as ",
        " meaning ",
        " signifying ",
        " denoting ",
    )
    return any(marker in f" {lowered} " for marker in glossary_markers)


def _strip_footnote_markers(text: str) -> str:
    """Remove markdown footnote markers while keeping the translation wording."""

    without_markers = re.sub(r"\s*\[\^\d+\]", "", text)
    return " ".join(without_markers.split()).strip()


def _fallback_footnote_explanation(
    token_choice: Mapping[str, object],
    token: str,
) -> str:
    """Explain fallback token choices in a way that mirrors the render contract.

    Even deterministic fallback output should tell the reader why a token was
    rendered in a particular way. This helper summarizes the selected analysis
    type and whether the token had to remain unresolved after placeholder
    cleanup.
    """

    rendered_text = _fallback_rendered_text(token_choice, token)
    analysis_type = str(token_choice.get("analysis_type") or "unknown")
    trace = _definition_trace_for_token_choice(token_choice)
    dictionary_rescue_gloss = _dictionary_rescue_gloss(token_choice, token)
    dictionary_rescue_note = str(
        token_choice.get("dictionary_rescue_note")
        or trace.get("dictionary_rescue_note")
        or ""
    ).strip()
    if (
        dictionary_rescue_gloss is not None
        and rendered_text == dictionary_rescue_gloss
        and dictionary_rescue_note
    ):
        return dictionary_rescue_note
    if rendered_text in {unresolved_token_gloss(token), _human_facing_unresolved_gloss()}:
        return "Only weak or placeholder evidence survived for this token."
    if trace.get("blind_dictionary_fallback"):
        return (
            "No non-dictionary definition survived in blind mode, so the fallback "
            "dictionary-backed reading was used."
        )
    blind_mode_rescue_note = str(
        trace.get("blind_mode_rescue_note")
        or token_choice.get("blind_mode_rescue_note")
        or ""
    ).strip()

    parts: list[str] = []
    selected_source = str(
        trace.get("selected_source") or token_choice.get("selected_source") or analysis_type or "unknown"
    )
    gloss_strategy = str(trace.get("surface_gloss_strategy") or "").strip()
    semantic_core = trace.get("selected_semantic_core")
    semantic_gloss = semantic_core_gloss(semantic_core)
    negative_contrast_penalties = trace.get("negative_contrast_penalties")
    if semantic_gloss is not None:
        parts.append(f'Uses the surviving semantic core "{semantic_gloss}".')
    elif gloss_strategy == "cleaned_definition":
        parts.append("Uses the cleaned lexical sense from the surviving definition evidence.")
    elif gloss_strategy == "subroot_estimate":
        parts.append("Uses the best surviving constructive subroot estimate.")
    if selected_source not in {"unknown", ""}:
        parts.append(f"Selected {_format_source_label(selected_source)} evidence survived.")
    if blind_mode_rescue_note:
        parts.append(blind_mode_rescue_note)
    if (
        isinstance(negative_contrast_penalties, Sequence)
        and not isinstance(negative_contrast_penalties, (str, bytes))
        and negative_contrast_penalties
    ):
        blocked = ", ".join(str(item) for item in negative_contrast_penalties if str(item).strip())
        if blocked:
            parts.append(f"Avoided conflicting negative-contrast terms: {blocked}.")

    suppressed = trace.get("suppressed")
    if isinstance(suppressed, Sequence) and not isinstance(suppressed, (str, bytes)):
        for note in suppressed:
            if isinstance(note, str) and note.strip():
                parts.append(note.strip())
                break

    runner_ups = trace.get("runner_ups")
    if isinstance(runner_ups, Sequence) and not isinstance(runner_ups, (str, bytes)):
        for runner_up in runner_ups:
            if not isinstance(runner_up, Mapping):
                continue
            definition = runner_up.get("definition")
            if not isinstance(definition, str) or not definition.strip():
                continue
            source = _format_source_label(str(runner_up.get("source") or "unknown"))
            parts.append(
                f'Beat nearby {source} alternative "{_compact_lay_gloss(definition, token=token) or definition.strip()}".'
            )
            break

    relation_context = trace.get("relation_context")
    relation_summary = _relation_context_summary(relation_context)
    if relation_summary:
        parts.append(relation_summary)

    selection_reason = trace.get("selection_reason")
    if isinstance(selection_reason, str) and selection_reason.strip() and not parts:
        parts.append(selection_reason.strip())

    if parts:
        return " ".join(parts)

    return (
        f'"{token}" keeps the selected {analysis_type} reading because it survived '
        "the available evidence filters."
    )


def _preferred_primary_gloss(token_choice: Mapping[str, object], token: str) -> str | None:
    """Prefer the richest grounded serialized gloss before weaker abstractions.

    Phrase lay rendering should preserve specific, human-facing lexical detail
    when the parse payload already carries it. This helper therefore tries the
    serialized bundle and surface definitions first, optionally mining a more
    specific phrase from them, before collapsing all the way down to a bare
    semantic-core label.
    """

    trace = _definition_trace_for_token_choice(token_choice)
    semantic_core = token_choice.get("semantic_core")
    negative_contrast = token_choice.get("negative_contrast")
    dictionary_rescue = _dictionary_rescue_gloss(token_choice, token)
    if dictionary_rescue is not None:
        return dictionary_rescue
    primary_candidates = [
        trace.get("surface_gloss"),
        token_choice.get("surface_gloss"),
        trace.get("selected_definition"),
        token_choice.get("definition"),
        token_choice.get("bundle_surface_gloss"),
        token_choice.get("bundle_head_gloss"),
        trace.get("raw_selected_definition"),
        token_choice.get("raw_definition"),
    ]
    for candidate in primary_candidates:
        specific = specific_gloss_from_definition_and_semantic_core(
            semantic_core=semantic_core,
            definition=candidate,
            negative_contrast=negative_contrast,
            token=token,
        )
        if specific is not None:
            return specific
        compact = _compact_lay_gloss(candidate, token=token)
        if compact is not None:
            return compact
    semantic_gloss = semantic_core_gloss(semantic_core)
    if semantic_gloss is not None:
        compact = _compact_lay_gloss(semantic_gloss, token=token)
        if compact is not None:
            return compact
    return None


def _preferred_runner_up_gloss(token_choice: Mapping[str, object], token: str) -> str | None:
    """Fallback to runner-ups from the selected evidence family before broader alternates."""

    trace = _definition_trace_for_token_choice(token_choice)
    runner_ups = trace.get("runner_ups")
    if not isinstance(runner_ups, Sequence) or isinstance(runner_ups, (str, bytes)):
        return None
    selected_source = str(trace.get("selected_source") or "")
    prioritized: list[Mapping[str, object]] = []
    others: list[Mapping[str, object]] = []
    for runner_up in runner_ups:
        if not isinstance(runner_up, Mapping):
            continue
        source = str(runner_up.get("source") or "")
        if selected_source and source == selected_source:
            prioritized.append(runner_up)
        else:
            others.append(runner_up)
    for entry in [*prioritized, *others]:
        runner_up_candidates = [
            entry.get("definition"),
            entry.get("raw_definition"),
            entry.get("semantic_core"),
        ]
        compact = None
        for runner_up_candidate in runner_up_candidates:
            specific = specific_gloss_from_definition_and_semantic_core(
                semantic_core=entry.get("semantic_core"),
                definition=runner_up_candidate,
                negative_contrast=entry.get("negative_contrast"),
                token=token,
            )
            if specific is not None:
                compact = specific
                break
            compact = _compact_lay_gloss(runner_up_candidate, token=token)
            if compact is not None:
                break
        if compact is not None:
            return compact
    return None


def _alternate_gloss_fallback(token_choice: Mapping[str, object], token: str) -> str | None:
    """Use broad alternates only after primary and traced runner-ups fail."""

    alternates = token_choice.get("alternates")
    alternate_values = (
        alternates
        if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes))
        else []
    )
    for alternate in alternate_values:
        compact = _compact_lay_gloss(alternate, token=token)
        if compact is not None:
            return compact
    return None


def _dictionary_rescue_gloss(token_choice: Mapping[str, object], token: str) -> str | None:
    """Recover a render-only exact-dictionary gloss when blind mode stayed opaque.

    Phrase rendering may need to explain a chosen decomposition that still has
    no readable gloss after placeholder cleanup. When the phrase service
    supplies an explicit exact-dictionary rescue value, this helper surfaces it
    only after the ordinary compositional and alternate gloss paths fail.
    """

    raw_rescue = token_choice.get("dictionary_rescue_gloss")
    sanitized = sanitize_human_gloss(raw_rescue, token=token)
    if sanitized is None:
        return None
    return " ".join(sanitized.split()).strip()


def _human_facing_unresolved_gloss() -> str:
    """Return the phrase-safe fallback when no grounded gloss survives."""

    return _HUMAN_UNRESOLVED_GLOSS


def _weak_fallback_gloss(token_choice: Mapping[str, object], token: str) -> str | None:
    """Return the phrase-layer weak fallback gloss when one was serialized.

    Phrase rendering now preserves a dedicated weak-evidence gloss so the
    final translation can stay readable without upgrading that gloss into the
    candidate's primary definition. This helper keeps the footnote fallback
    path aligned with that serialized phrase-layer decision.
    """

    raw_fallback = token_choice.get("weak_fallback_gloss")
    return _compact_lay_gloss(raw_fallback, token=token)


def _weak_fallback_note(token_choice: Mapping[str, object]) -> str:
    """Return the serialized weak-fallback explanation when available."""

    trace = _definition_trace_for_token_choice(token_choice)
    return str(
        token_choice.get("weak_fallback_note")
        or trace.get("weak_fallback_note")
        or ""
    ).strip()


def _definition_trace_for_token_choice(token_choice: Mapping[str, object]) -> Mapping[str, object]:
    """Return the token-level definition trace when one is present."""

    trace = token_choice.get("definition_trace")
    if isinstance(trace, Mapping):
        return trace
    return {}


def _format_source_label(source: str) -> str:
    """Convert an internal provenance label into readable footnote wording."""

    normalized = source.strip().replace("_", " ")
    if normalized.startswith("canonical "):
        return normalized
    if normalized == "cluster":
        return "cluster-backed"
    if normalized == "attested":
        return "attested"
    if normalized == "hypothesis":
        return "hypothesis-backed"
    if normalized == "residual":
        return "residual"
    if normalized == "dictionary":
        return "dictionary-backed"
    if normalized == "memory":
        return "memory-backed"
    return normalized or "unknown"


def _relation_context_summary(value: object) -> str | None:
    """Summarize adjacent parse relations for evidence-driven footnotes."""

    if not isinstance(value, Mapping):
        return None
    pieces: list[str] = []
    left = value.get("left")
    if isinstance(left, Mapping):
        pieces.append(
            f'left parse relation {left.get("relation")} with {left.get("neighbor_token")}'
        )
    right = value.get("right")
    if isinstance(right, Mapping):
        pieces.append(
            f'right parse relation {right.get("relation")} with {right.get("neighbor_token")}'
        )
    if not pieces:
        return None
    return "Parse context kept it aligned through " + " and ".join(pieces) + "."
