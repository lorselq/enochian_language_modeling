from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enochian_lm.root_extraction.utils.residual_analysis import (  # noqa: E402
    build_subtraction_evidence,
    compute_word_break_subtractions,
)


def test_word_break_subtraction_naz_example():
    results = compute_word_break_subtractions("NAZPSAD", "NAZ")

    assert len(results) == 1
    assert results[0]["host_word"] == "NAZPSAD"
    assert results[0]["root"] == "NAZ"
    assert results[0]["residual"] == "PSAD"
    assert results[0]["residuals"] == [{"artifact": "PSAD", "start": 3, "end": 7}]


def test_word_break_subtraction_multiple_occurrences_returns_ambiguity_options():
    results = compute_word_break_subtractions("ANAZNAZ", "NAZ")

    assert len(results) == 2

    first = results[0]
    assert first["residual"] == "ANAZ"
    assert first["residuals"] == [
        {"artifact": "A", "start": 0, "end": 1},
        {"artifact": "NAZ", "start": 4, "end": 7},
    ]

    # ambiguity option: remove all occurrences (NAZ twice) -> A
    assert first["remove_all_occurrences"] == {
        "residual": "A",
        "residuals": [{"artifact": "A", "start": 0, "end": 1}],
        "removed_spans": [[1, 4], [4, 7]],
    }


def test_word_break_subtraction_is_case_insensitive():
    results = compute_word_break_subtractions("NazPsad", "naz")

    assert results[0]["residual"] == "Psad"
    assert results[0]["residuals"] == [{"artifact": "Psad", "start": 3, "end": 7}]
    assert results[0]["start"] == 0
    assert results[0]["end"] == 3


def test_build_subtraction_evidence_standardizes_trace_payload():
    subtraction = compute_word_break_subtractions("ANAZNAZ", "NAZ")[0]
    payload = build_subtraction_evidence(
        subtraction,
        donor_source="dictionary",
        recursion_depth=1,
        termination_reason="residual_extracted",
    )

    assert payload["equation"] == "ANAZNAZ - NAZ = ANAZ"
    assert payload["remaining_artifacts"] == [
        {"artifact": "A", "start": 0, "end": 1},
        {"artifact": "NAZ", "start": 4, "end": 7},
    ]
    assert payload["donor_source"] == "dictionary"
    assert payload["recursion_depth"] == 1
    assert payload["termination_reason"] == "residual_extracted"
