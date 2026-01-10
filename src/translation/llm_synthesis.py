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
import json
import logging

from enochian_lm.root_extraction.tools.query_model_tool import QueryModelTool

LOGGER = logging.getLogger(__name__)

DEFAULT_LLM_CONTEXT = (
    "Use Early Modern English prose from 16th-century Britain, grounded in the "
    "mainstream Christian worldview John Dee would have known. Avoid modern "
    "terms, theology, or concepts."
)

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
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = tool._run(prompt=prompt, print_chunks=False, stream_callback=None)
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
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = tool._run(prompt=prompt, print_chunks=False, stream_callback=None)
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

    candidates_block = "\n".join(candidate_lines) or "- No candidate summaries available"
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

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
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
    for _ in range(2):
        response = tool._run(prompt=prompt, print_chunks=False, stream_callback=None)
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
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    best_estimations = data.get("best_estimations")
    if not isinstance(best_estimations, list):
        return []
    cleaned = [item.strip() for item in best_estimations if isinstance(item, str) and item.strip()]
    return cleaned[:6]


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
