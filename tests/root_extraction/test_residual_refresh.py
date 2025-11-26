import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from enochian_lm.root_extraction.utils.residual_refresh import (  # noqa: E402
    _apply_root_prefix_residual_fallback,
)


def test_root_prefix_self_match_surfaces_suffix_residual():
    """Self-matching segments should not mask suffix residuals for root-prefixed words."""

    breakdown = {
        "segments": [
            {"span": [0, 3], "canonical": "NAZ", "ngram": "NAZ"},
            {"span": [0, 7], "canonical": "NAZPSAD", "ngram": "NAZPSAD"},
        ],
        "uncovered": [],
        "coverage_ratio": 1.0,
        "residual_ratio": 0.0,
    }

    adjusted = _apply_root_prefix_residual_fallback(
        breakdown, root_norm="NAZ", target="NAZPSAD"
    )

    uncovered = adjusted.get("uncovered") or []
    assert uncovered == [{"span": [3, 7], "text": "NAZPSAD"[3:7]}]
    assert pytest.approx(adjusted.get("coverage_ratio"), rel=1e-4) == 3 / 7
    assert pytest.approx(adjusted.get("residual_ratio"), rel=1e-4) == 4 / 7
