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
from typing import Dict, List

from enochian_lm.root_extraction.tools.query_model_tool import QueryModelTool

LOGGER = logging.getLogger(__name__)


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
    """

    synthesized_definition: str | None
    concatenated_meanings: str
    confidence: float
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    raw_response: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "synthesized_definition": self.synthesized_definition,
            "concatenated_meanings": self.concatenated_meanings,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "warnings": list(self.warnings),
            "raw_response": self.raw_response,
        }


def synthesize_definition(
    morphs: List[str],
    meanings: List[str],
    context: Dict[str, object],
) -> SynthesisResult:
    """Return an LLM-synthesized gloss for the supplied morph meanings.

    This implements Task 4.1 from ``TODO.md`` by:

    - Building a structured prompt that pairs the decomposition (``morphs``)
      with the human-readable ``meanings`` and coverage context.
    - Leveraging the existing ``QueryModelTool`` infrastructure to keep LLM
      invocation consistent with the rest of the codebase.
    - Handling LLM failures gracefully by returning the concatenated meanings
      along with transparent reasoning and warnings.
    """

    normalized_morphs = [m.upper() for m in morphs if m]
    cleaned_meanings = [m.strip() for m in meanings if isinstance(m, str) and m.strip()]
    concatenated = " + ".join(cleaned_meanings) if cleaned_meanings else " + ".join(normalized_morphs)

    prompt = _build_prompt(normalized_morphs, cleaned_meanings, context)
    use_remote = bool(context.get("use_remote", False))

    try:
        tool = QueryModelTool(
            system_prompt=(
                "You are a precise Enochian glossator. Combine morph semantics "
                "into a single, well-formed definition without inventing "
                "unsupported etymologies."
            ),
            name="Definition Synthesizer",
            description="Compose a concise morph-level synthesis",
            use_remote=use_remote,
        )
        response = tool._run(prompt=prompt, print_chunks=False, stream_callback=None)
        parsed = _parse_response(response.get("response_text", ""), fallback=concatenated, context=context)

        synthesized_raw = parsed.get("definition")
        synthesized: str | None = synthesized_raw if isinstance(synthesized_raw, str) else None
        confidence_raw = parsed.get("confidence")
        confidence_val: float | None = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else None
        confidence = _resolved_confidence(confidence_val, context)
        reasoning_raw = parsed.get("reasoning")
        reasoning: str = reasoning_raw if isinstance(reasoning_raw, str) else "Synthesized definition via LLM."

        return SynthesisResult(
            synthesized_definition=synthesized,
            concatenated_meanings=concatenated,
            confidence=confidence,
            reasoning=reasoning,
            raw_response=response.get("response_text"),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("LLM synthesis failed; falling back to concatenated meanings", exc_info=exc)
        return SynthesisResult(
            synthesized_definition=None,
            concatenated_meanings=concatenated,
            confidence=_resolved_confidence(None, context, fallback_only=True),
            reasoning="LLM synthesis unavailable; returned concatenated meanings instead.",
            warnings=[str(exc)],
        )


def _build_prompt(
    morphs: List[str], meanings: List[str], context: Dict[str, object]
) -> str:
    coverage_ratio = _safe_float(context.get("coverage_ratio"), default=0.0)
    residual_ratio = _safe_float(context.get("residual_ratio"), default=1.0)
    strategy = str(context.get("strategy") or "prefer-balance")

    provenance_lines: List[str] = []
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
            "confidence": 0.0,
            "reasoning": f"<one-sentence justification, <= {max_len} chars>",
        },
        ensure_ascii=False,
    )

    instruction_lines = [
        "ROLE: You are a precision Enochian glossator.",
        "CONSTRAINTS:",
        "- Use ONLY the provided morphs/meanings/provenance; no external etymology or speculation.",
        "- Evidence trust order: clusters > residuals > hypotheses. Do not invent links.",
        "- Produce exactly one concise sense (no lists, no multiple senses, no hedging).",
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
    ]

    return "\n".join(instruction_lines)


def _parse_response(
    payload: str,
    *,
    fallback: str,
    context: Dict[str, object],
    max_len: int = 160,
) -> Dict[str, str | float]:
    """Parse LLM JSON response with validation and trimming."""

    def _trim(value: str) -> str:
        text = value.strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    parsed: Dict[str, str | float] = {}

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

    if "confidence" not in parsed:
        parsed["confidence"] = _resolved_confidence(None, context, fallback_only=True)
    if "reasoning" not in parsed:
        parsed["reasoning"] = "Synthesized definition via LLM."

    return parsed


def _resolved_confidence(
    reported: float | None,
    context: Dict[str, object],
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
