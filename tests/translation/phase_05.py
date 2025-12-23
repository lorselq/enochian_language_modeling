from __future__ import annotations

"""
Phase 5 Tests: CLI & Output Formatting (Tasks 5.1-5.3)

These tests validate output shaping, edge-case handling, and text formatting
without invoking the full database-backed pipeline.
"""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "translation" / "cli_word.py"
spec = importlib.util.spec_from_file_location("cli_word", MODULE_PATH)
cli_word = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cli_word)

_build_output_payload = cli_word._build_output_payload
_format_text_report = cli_word._format_text_report


def test_build_output_payload_marks_residual_only():
    result = {
        "word": "NAZ",
        "variants_queried": ["solo"],
        "strategy": "prefer-balance",
        "timestamp": "2025-12-10T14:23:45Z",
        "llm_enabled": False,
        "candidates": [
            {
                "rank": 1,
                "morphs": ["NAZ"],
                "score": 2.0,
                "breakdown": {"coverage_ratio": 1.0, "residual_ratio": 0.0},
                "meanings": [
                    {
                        "morph": "NAZ",
                        "definition": "rectangular prism",
                        "provenance": "residual",
                    }
                ],
                "warnings": [],
                "concatenated_meanings": "rectangular prism",
                "synthesized_definition": None,
                "confidence": 0.5,
            }
        ],
        "evidence": {
            "variants_queried": ["solo"],
            "direct_clusters": 0,
            "residual_semantics": 1,
            "morph_hypotheses": 0,
            "fasttext_neighbors": [],
        },
    }

    payload = _build_output_payload(result, variant="solo")
    sense = payload["senses"][0]

    assert sense["provenance_note"] == "residual-only (observed as remainder)"
    assert sense["confidence"] == 0.3


def test_build_output_payload_skips_residual_only_with_hypotheses():
    result = {
        "word": "NAZ",
        "variants_queried": ["solo"],
        "strategy": "prefer-balance",
        "timestamp": "2025-12-10T14:23:45Z",
        "llm_enabled": False,
        "candidates": [
            {
                "rank": 1,
                "morphs": ["NAZ"],
                "score": 2.0,
                "breakdown": {"coverage_ratio": 1.0, "residual_ratio": 0.0},
                "meanings": [
                    {
                        "morph": "NAZ",
                        "definition": "rectangular prism",
                        "provenance": "residual",
                    }
                ],
                "warnings": [],
                "concatenated_meanings": "rectangular prism",
                "synthesized_definition": None,
                "confidence": 0.5,
            }
        ],
        "evidence": {
            "variants_queried": ["solo"],
            "direct_clusters": 0,
            "residual_semantics": 1,
            "morph_hypotheses": 2,
            "fasttext_neighbors": [],
        },
    }

    payload = _build_output_payload(result, variant="solo")
    sense = payload["senses"][0]

    assert "provenance_note" not in sense
    assert sense["confidence"] == 0.5


def test_build_output_payload_no_evidence_message():
    result = {
        "word": "XYZ",
        "variants_queried": ["solo"],
        "strategy": "prefer-balance",
        "timestamp": "2025-12-10T14:23:45Z",
        "llm_enabled": False,
        "candidates": [],
        "evidence": {
            "variants_queried": ["solo"],
            "direct_clusters": 0,
            "residual_semantics": 0,
            "morph_hypotheses": 0,
            "fasttext_neighbors": [{"word": "FOO", "similarity": 0.42}],
        },
    }

    payload = _build_output_payload(result, variant="solo")

    assert payload["message"].startswith("No direct evidence found")


def test_format_text_report_includes_fasttext_neighbors():
    payload = {
        "word": "XYZ",
        "variant": "solo",
        "strategy": "prefer-balance",
        "llm_enabled": False,
        "timestamp": "2025-12-10T14:23:45Z",
        "senses": [],
        "evidence": {
            "fasttext_neighbors": [{"word": "FOO", "similarity": 0.42}],
        },
    }

    rendered = _format_text_report(payload)

    assert "FastText neighbors:" in rendered
    assert "FOO" in rendered


def test_format_text_report_includes_phase_sections():
    payload = {
        "word": "NAZPSAD",
        "variant": "solo",
        "strategy": "prefer-balance",
        "llm_enabled": True,
        "timestamp": "2025-12-10T14:23:45Z",
        "senses": [
            {
                "rank": 1,
                "morphs": ["NAZ", "PSAD"],
                "score": 7.82,
                "breakdown": {"coverage_ratio": 1.0, "residual_ratio": 0.0},
                "meanings": [
                    {
                        "morph": "NAZ",
                        "definition": "rectangular prism",
                        "provenance": "cluster",
                    },
                    {
                        "morph": "PSAD",
                        "definition": "sharp",
                        "provenance": "residual",
                    },
                ],
                "synthesized_definition": "sword, knife, cutting weapon",
                "concatenated_meanings": "rectangular prism + sharp",
                "confidence": 0.85,
                "warnings": [],
            }
        ],
        "evidence": {
            "direct_clusters": 2,
            "residual_semantics": 1,
            "morph_hypotheses": 0,
            "fasttext_neighbors": [],
        },
    }

    rendered = _format_text_report(payload)

    assert "Evidence: clusters=2, residuals=1, hypotheses=0" in rendered
    assert "Score: 7.82" in rendered
    assert "Coverage: 1.00 (residual 0.00)" in rendered
    assert "Synthesized: sword, knife, cutting weapon" in rendered
