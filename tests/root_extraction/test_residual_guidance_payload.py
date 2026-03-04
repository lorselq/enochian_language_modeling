from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enochian_lm.root_extraction.utils.residual_analysis import (  # noqa: E402
    build_residual_guidance_payload,
    build_subtraction_evidence,
    compute_word_break_subtractions,
    prioritize_donor_candidates,
)


def test_residual_guidance_payload_includes_host_root_residual_and_equation():
    subtraction = compute_word_break_subtractions("NAZPSAD", "NAZ")[0]
    evidence = build_subtraction_evidence(
        subtraction,
        donor_source="dictionary",
        recursion_depth=0,
        termination_reason="root_subtracted_from_host",
    )

    payload = build_residual_guidance_payload(root="PSAD", word_breaks=[evidence])

    assert payload["root"] == "PSAD"
    assert payload["word_breaks"][0]["host_word"] == "NAZPSAD"
    assert payload["word_breaks"][0]["root"] == "NAZ"
    assert payload["word_breaks"][0]["residual"] == "PSAD"
    assert payload["subtraction_equations"] == ["NAZPSAD - NAZ = PSAD"]


def test_prioritize_donor_candidates_prefers_dictionary_then_sqlite_with_largest_match():
    # ALNAZPSAD-like fixture chain: dictionary pieces A/L and sqlite donor NAZ
    candidates = [
        {"donor": "NAZ", "source": "sqlite"},
        {"donor": "A", "source": "dictionary"},
        {"donor": "L", "source": "dictionary"},
        {"donor": "AL", "source": "dictionary"},
    ]

    ranked = prioritize_donor_candidates(candidates)

    # dictionary wins over sqlite; within dictionary, longer donor first
    assert ranked[0]["donor"] == "AL"
    assert ranked[0]["source"] == "dictionary"
    assert ranked[1]["source"] == "dictionary"
    assert ranked[2]["source"] == "dictionary"
    assert ranked[-1] == {"donor": "NAZ", "source": "sqlite"}
