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
    sanitize_human_gloss,
    semantic_core_gloss,
    unresolved_token_gloss,
)

LOGGER = logging.getLogger(__name__)

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

    Phrase translation now wants a technical rendering, a lay rendering, and a
    token-aligned footnote block. Returning all of that from one structured
    result lets the phrase service reduce remote round-trips while keeping the
    downstream CLI payload explicit.
    """

    technical_translation: str
    technical_confidence: float
    technical_reasoning: str
    lay_translation: str
    lay_confidence: float
    lay_reasoning: str
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
    """Render lay phrase output while keeping the technical render deterministic.

    Phrase translation now treats the chosen parse skeleton as the authoritative
    technical translation. The optional model call is reserved for the short
    lay gist and footnotes so remote rendering stays smaller, faster, and less
    likely to wander back into glossary prose.
    """

    skeleton = str(parse_payload.get("translation_skeleton") or "").strip()
    prompt = _build_phrase_bundle_prompt(parse_payload, context)
    use_remote = bool(context.get("use_remote", False))
    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    tool = QueryModelTool(
        system_prompt=(
            "You are a constrained lay phrase renderer for Enochian translation. "
            "Return a short plain-English gist with token footnotes without "
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
            footnoted_translation=fallback_footnoted,
            translation_footnotes=fallback_notes,
            warnings=["LLM lay phrase rendering unavailable."],
        )


def _build_phrase_bundle_prompt(
    parse_payload: dict[str, object],
    context: dict[str, object],
) -> str:
    """Build the compact lay-render prompt used by phrase translation.

    The technical translation now stays deterministic, so the model only needs
    enough context to produce a short gist plus token footnotes. This smaller
    contract keeps phrase rendering faster and more reliable on long inputs.
    """

    llm_context = context.get("llm_context") or DEFAULT_LLM_CONTEXT
    schema = json.dumps(
        {
            "lay_translation": "<short plain-English gist, ideally 3-10 words>",
            "lay_confidence": 0.0,
            "lay_reasoning": "<brief note explaining the simplification>",
            "footnoted_translation": "<same short translation with [^1] style markers>",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "<original token>",
                    "rendered_text": "<compact 1-3 word English chunk>",
                    "explanation": "<brief grounded explanation for this choice>",
                }
            ],
        },
        ensure_ascii=False,
    )
    example = json.dumps(
        {
            "lay_translation": "holy rule endures",
            "lay_confidence": 0.72,
            "lay_reasoning": "Compresses the parse into a short everyday clause while preserving the same idea.",
            "footnoted_translation": "holy [^1] rule [^2] endures [^3]",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "MAD",
                    "rendered_text": "holy",
                    "explanation": "Compresses the selected sacred gloss into one everyday adjective.",
                },
                {
                    "index": 2,
                    "source_token": "CAF",
                    "rendered_text": "rule",
                    "explanation": "Uses the core governing action instead of restating the full gloss.",
                },
                {
                    "index": 3,
                    "source_token": "PRAC",
                    "rendered_text": "endures",
                    "explanation": "Picks a short everyday verb for continued abiding.",
                },
            ],
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a constrained Enochian lay phrase renderer.",
            "TASK: Return a short lay translation and token footnotes for the supplied parse.",
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Use ONLY the supplied token choices, relations, and skeleton.",
            "- Do not add, remove, or replace meanings.",
            "- `lay_translation` must be a short natural English clause or sentence, ideally 3-10 words.",
            "- Prefer a clear core idea over a token-by-token comma list or definition dump.",
            "- Return exactly one footnote entry per token choice, in source order.",
            "- Each `rendered_text` should usually be 1-3 words. Use extra filler words only when grammar truly requires them.",
            "- If a chosen gloss is verbose, compress it to the smallest everyday concept supported by that gloss.",
            "- Do not repeat the raw source token in English unless the token remains unresolved; in that case use `[TOKEN]` and explain why.",
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
            "rendered_translation": "<short plain-English gist, ideally 3-10 words>",
            "footnoted_translation": "<same short translation with [^1] style markers>",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "<original token>",
                    "rendered_text": "<compact 1-3 word English chunk>",
                    "explanation": "<brief grounded explanation for this choice>",
                }
            ],
            "confidence": 0.0,
            "reasoning": "<brief note explaining the simplification>",
        },
        ensure_ascii=False,
    )
    example = json.dumps(
        {
            "rendered_translation": "holy rule endures",
            "footnoted_translation": "holy [^1] rule [^2] endures [^3]",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "MAD",
                    "rendered_text": "holy",
                    "explanation": "Compresses the selected sacred gloss into one everyday adjective.",
                },
                {
                    "index": 2,
                    "source_token": "CAF",
                    "rendered_text": "rule",
                    "explanation": "Uses the core governing action instead of restating the full gloss.",
                },
                {
                    "index": 3,
                    "source_token": "PRAC",
                    "rendered_text": "endures",
                    "explanation": "Picks a short everyday verb for continued abiding.",
                },
            ],
            "confidence": 0.72,
            "reasoning": "Compresses each token to a short everyday concept while preserving the chosen parse.",
        },
        ensure_ascii=False,
    )
    payload = json.dumps(parse_payload, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "ROLE: You are a plain-English Enochian phrase explainer.",
            "TASK: Rewrite the supplied translation skeleton so a lay reader can understand the core idea immediately.",
            f"HISTORICAL CONTEXT: {llm_context}",
            "CONSTRAINTS:",
            "- Use ONLY the supplied token choices, relations, and skeleton.",
            "- Keep the same core meaning, but replace technical or archaic phrasing with sane everyday English.",
            "- Do not add any new actors, actions, objects, or claims.",
            "- Your job is gist, not exhaustiveness.",
            "- `rendered_translation` must be a short natural English clause or sentence, ideally 3-10 words.",
            "- Prefer a clear core idea over a token-by-token comma list or definition dump.",
            "- Never restate full token definitions, example sentences, or long alternation chains in `rendered_translation`.",
            "- Return exactly one footnote entry per token choice, in source order.",
            "- Each `rendered_text` should usually be 1-3 words. Use extra filler words only when grammar truly requires them.",
            "- If a chosen gloss is verbose, compress it to the smallest everyday concept supported by that gloss.",
            "- Do not repeat the raw source token in English unless the token remains unresolved; in that case use `[TOKEN]` and explain why.",
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
    technical = _parse_phrase_render_response(technical_payload, fallback=fallback)
    lay = _parse_phrase_lay_render_response(
        lay_payload,
        fallback=fallback,
        parse_payload=parse_payload,
    )
    return {
        "technical_translation": technical["rendered_translation"],
        "technical_confidence": technical["confidence"],
        "technical_reasoning": technical["reasoning"],
        "lay_translation": lay["rendered_translation"],
        "lay_confidence": lay["confidence"],
        "lay_reasoning": lay["reasoning"],
        "footnoted_translation": lay["footnoted_translation"],
        "translation_footnotes": lay["translation_footnotes"],
    }


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
    return unresolved_token_gloss(token)


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


def _normalize_lay_translation(rendered_translation: str, footnoted_translation: str) -> str:
    """Prefer the compact footnoted line when the lay sentence becomes bloated.

    The lay translation should communicate the idea quickly. When the raw
    `rendered_translation` balloons into a glossary dump, the marker-stripped
    footnoted line is usually the shorter and more faithful lay summary.
    """

    cleaned = " ".join(rendered_translation.split())
    if not cleaned:
        return _strip_footnote_markers(footnoted_translation)
    if not _looks_overexpanded_lay_translation(cleaned):
        return cleaned
    compact = _strip_footnote_markers(footnoted_translation)
    return compact or cleaned


def _looks_overexpanded_lay_translation(text: str) -> bool:
    """Detect when a lay translation has drifted into glossary-like prose."""

    if len(text.split()) > 12:
        return True
    return sum(text.count(marker) for marker in ",;:") >= 2


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
    if rendered_text == unresolved_token_gloss(token):
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
    """Prefer the chosen primary definition before considering alternates."""

    trace = _definition_trace_for_token_choice(token_choice)
    semantic_core_candidates = [
        trace.get("selected_semantic_core"),
        token_choice.get("semantic_core"),
    ]
    for semantic_candidate in semantic_core_candidates:
        semantic_gloss = semantic_core_gloss(semantic_candidate)
        if semantic_gloss is not None:
            compact = _compact_lay_gloss(semantic_gloss, token=token)
            if compact is not None:
                return compact
    primary_candidates = [
        token_choice.get("bundle_surface_gloss"),
        token_choice.get("bundle_head_gloss"),
        trace.get("surface_gloss"),
        token_choice.get("surface_gloss"),
        trace.get("selected_definition"),
        trace.get("raw_selected_definition"),
        token_choice.get("definition"),
        token_choice.get("raw_definition"),
    ]
    for candidate in primary_candidates:
        compact = _compact_lay_gloss(candidate, token=token)
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
            entry.get("semantic_core"),
            entry.get("definition"),
            entry.get("raw_definition"),
        ]
        compact = None
        for runner_up_candidate in runner_up_candidates:
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
    return _compact_lay_gloss(raw_rescue, token=token)


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
