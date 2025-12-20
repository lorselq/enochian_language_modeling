from __future__ import annotations

"""
Phase 4 Tests: LLM Synthesis (Tasks 4.1 & 4.2)

These tests focus on prompt construction and response parsing/fallback for the
LLM synthesis adapter. Direct LLM calls are not exercised; instead, we validate
prompt content and parsing robustness.
"""

import json
from pathlib import Path
import sys
from typing import Dict

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.llm_synthesis import (
    SynthesisResult,
    _build_prompt,
    _parse_response,
    _resolved_confidence,
    synthesize_definition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_context() -> Dict[str, object]:
    return {
        "coverage_ratio": 0.8,
        "residual_ratio": 0.2,
        "strategy": "prefer-balance",
        "provenance": [
            {"morph": "NAZ", "provenance": "cluster"},
            {"morph": "PSAD", "provenance": "residual"},
            {"morph": "ZED", "provenance": "hypothesis"},
        ],
    }


# ---------------------------------------------------------------------------
# Prompt construction tests (Task 4.1)
# ---------------------------------------------------------------------------


def test_prompt_includes_constraints_and_evidence(sample_context: Dict[str, object]):
    prompt = _build_prompt(["NAZ", "PSAD"], ["rectangular prism", "sharp"], sample_context)

    # Constraint markers
    assert "precision Enochian glossator" in prompt
    assert "no external etymology" in prompt or "Use ONLY the provided" in prompt
    assert "Evidence trust order" in prompt or "clusters > residuals > hypotheses" in prompt
    assert "STRICT JSON" in prompt
    assert "Hard max length" in prompt

    # Evidence framing
    assert "NAZ + PSAD" in prompt
    assert "coverage_ratio=0.80" in prompt
    assert "residual_ratio=0.20" in prompt
    assert "NAZ" in prompt and "PSAD" in prompt


# ---------------------------------------------------------------------------
# Parsing robustness tests (Task 4.1)
# ---------------------------------------------------------------------------


def test_parse_response_trims_and_truncates(sample_context: Dict[str, object]):
    long_text = "A" * 300
    payload = json.dumps({
        "definition": long_text,
        "confidence": 0.9,
        "reasoning": "  reason " + long_text,
    })

    parsed = _parse_response(payload, fallback="fallback", context=sample_context, max_len=50)

    assert len(parsed["definition"]) <= 50
    assert parsed["definition"].endswith("…")
    assert parsed["reasoning"].endswith("…")
    assert 0.0 <= parsed["confidence"] <= 1.0


def test_parse_response_handles_missing_fields(sample_context: Dict[str, object]):
    payload = json.dumps({"definition": ""})
    parsed = _parse_response(payload, fallback="fallback", context=sample_context)

    assert parsed["definition"] == "fallback"
    assert parsed["confidence"] == _resolved_confidence(None, sample_context, fallback_only=True)
    assert "concatenated" in parsed["reasoning"] or "fallback" in parsed["reasoning"].lower()


def test_parse_response_handles_non_json(sample_context: Dict[str, object]):
    parsed = _parse_response("raw text", fallback="fallback", context=sample_context)

    assert parsed["definition"].startswith("raw text"[:10])
    assert parsed["confidence"] == _resolved_confidence(None, sample_context, fallback_only=True)


# ---------------------------------------------------------------------------
# Fallback clarity tests (Task 4.2 behavior via adapter surface)
# ---------------------------------------------------------------------------


def test_synthesize_definition_handles_llm_failure(monkeypatch: pytest.MonkeyPatch, sample_context: Dict[str, object]):
    # Force QueryModelTool._run to raise, triggering fallback path.
    class DummyTool:
        def __init__(self, *args, **kwargs):
            pass

        def _run(self, *args, **kwargs):
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr("translation.llm_synthesis.QueryModelTool", DummyTool)

    result = synthesize_definition(["NAZ"], ["rectangular prism"], sample_context)

    assert isinstance(result, SynthesisResult)
    assert result.synthesized_definition is None
    assert result.concatenated_meanings == "rectangular prism"
    assert 0.0 <= result.confidence <= 1.0
    assert "LLM synthesis unavailable" in result.reasoning

