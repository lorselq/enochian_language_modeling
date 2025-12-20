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
from translation.decomposition import Decomposition
from translation.repository import ClusterRecord, RawDefinition, ResidualSemanticRecord, WordEvidence
from translation.service import SingleWordTranslationService


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


# ---------------------------------------------------------------------------
# Service-level LLM toggle tests (Task 4.2)
# ---------------------------------------------------------------------------


class MockDecompositionEngine:
    """Mock decomposition engine that returns a fixed decomposition."""

    def __init__(self, decomposition: Decomposition):
        self._decomposition = decomposition

    def generate_decompositions(self, word: str, evidence: WordEvidence):
        return [self._decomposition]


class MockRepository:
    """Mock repository returning evidence with supported morphs."""

    def __init__(self):
        self.variants = ["solo"]

    def require_variants(self, variants):
        pass

    def fetch_word_evidence(self, word: str, variants=None) -> WordEvidence:
        # Provide clusters and residuals so morphs pass hard filters
        naz_cluster = ClusterRecord(
            variant="solo",
            cluster_id=1,
            run_id="r1",
            ngram="NAZ",
            cluster_index=0,
            glossator_def="rectangular prism",
            residual_explained=None,
            residual_ratio=None,
            residual_headline=None,
            residual_focus_prompt=None,
            semantic_coverage=None,
            cohesion=None,
            semantic_cohesion=None,
            best_config=None,
            residual_details=[],
            raw_definitions=[
                RawDefinition(
                    source_word="NAZ",
                    variant="solo",
                    definition=None,
                    enhanced_def="rectangular prism",
                    fasttext=None,
                    similarity=None,
                    tier=None,
                )
            ],
        )
        psad_residual = ResidualSemanticRecord(
            variant="solo",
            run_id="r1",
            residual="PSAD",
            parent_word="PSAD",
            group_index=0,
            group_size=1,
            glossator_def="sharp",
            glossator_prompt=None,
            residual_headline=None,
            residual_focus_prompt=None,
            semantic_coverage=None,
            cohesion=None,
            semantic_cohesion=None,
            residual_explained=None,
            residual_ratio=None,
            derivational_validity=None,
            rebuttal_resilience=None,
            created_at=None,
        )
        return WordEvidence(
            word=word,
            variants_queried=variants or ["solo"],
            direct_clusters=[naz_cluster],
            residual_semantics=[psad_residual],
        )

    def close(self):
        pass


class MockCandidateFinder:
    """Mock candidate finder."""

    def close(self):
        pass


class TestSingleWordTranslationServiceLLMToggle:
    """
    Task 4.2 Requirements (TODO.md):
    - --llm flag set → synthesized_definition populated
    - --no-llm flag set → synthesized_definition is None
    - LLM fails → gracefully degrade to concatenated_meanings with warning
    """

    def _make_service(self, llm_enabled: bool, llm_adapter=None):
        decomp = Decomposition(
            morphs=["NAZ", "PSAD"],
            beam_score=2.0,
            breakdown={
                "segments": [],
                "uncovered": [],
                "coverage_ratio": 1.0,
                "residual_ratio": 0.0,
            },
            morph_support={"NAZ": "cluster", "PSAD": "residual"},
        )

        service = SingleWordTranslationService(
            candidate_finder=MockCandidateFinder(),
            repository=MockRepository(),
            llm_enabled=llm_enabled,
            llm_adapter=llm_adapter or synthesize_definition,
        )
        service._decomposition_engine = MockDecompositionEngine(decomp)
        return service

    def test_llm_enabled_populates_synthesized_definition(self):
        """When llm=True, synthesized_definition should be populated."""

        def mock_adapter(morphs, meanings, context):
            return SynthesisResult(
                synthesized_definition="a cutting tool",
                concatenated_meanings=" + ".join(meanings) if meanings else "",
                confidence=0.85,
                reasoning="Synthesized via mock LLM",
            )

        service = self._make_service(llm_enabled=True, llm_adapter=mock_adapter)
        result = service.translate_word("NAZPSAD")

        top_candidate = result["candidates"][0]
        assert top_candidate["synthesized_definition"] == "a cutting tool"
        assert top_candidate["confidence"] == 0.85
        assert "mock LLM" in top_candidate["reasoning"]

    def test_llm_disabled_returns_none_synthesized(self):
        """When llm=False, synthesized_definition should be None."""
        service = self._make_service(llm_enabled=False)
        result = service.translate_word("NAZPSAD")

        top_candidate = result["candidates"][0]
        assert top_candidate["synthesized_definition"] is None
        assert "LLM disabled" in top_candidate["reasoning"]
        assert "concatenated_meanings" in top_candidate

    def test_llm_override_at_call_site(self):
        """The llm parameter to translate_word overrides llm_enabled."""

        def mock_adapter(morphs, meanings, context):
            return SynthesisResult(
                synthesized_definition="override synthesis",
                concatenated_meanings="",
                confidence=0.9,
                reasoning="Override worked",
            )

        # Service defaults to llm_enabled=False, but we override at call site
        service = self._make_service(llm_enabled=False, llm_adapter=mock_adapter)
        result = service.translate_word("NAZPSAD", llm=True)

        top_candidate = result["candidates"][0]
        assert top_candidate["synthesized_definition"] == "override synthesis"

    def test_llm_failure_degrades_gracefully_with_warning(self):
        """When LLM fails, service should return concatenated meanings with warning."""
        call_count = 0

        def failing_adapter(morphs, meanings, context):
            nonlocal call_count
            call_count += 1
            return SynthesisResult(
                synthesized_definition=None,
                concatenated_meanings=" + ".join(meanings) if meanings else "NAZ + PSAD",
                confidence=0.35,
                reasoning="LLM synthesis unavailable; returned concatenated meanings instead.",
                warnings=["Connection timeout"],
            )

        service = self._make_service(llm_enabled=True, llm_adapter=failing_adapter)
        result = service.translate_word("NAZPSAD")

        assert call_count == 1
        top_candidate = result["candidates"][0]
        assert top_candidate["synthesized_definition"] is None
        assert "concatenated_meanings" in top_candidate
        assert "LLM synthesis unavailable" in top_candidate["reasoning"]
        assert "Connection timeout" in top_candidate.get("warnings", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

