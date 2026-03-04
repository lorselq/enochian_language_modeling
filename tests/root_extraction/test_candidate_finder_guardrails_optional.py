from __future__ import annotations

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.mark.optional_dependency
def test_candidate_finder_imports_with_full_optional_stack() -> None:
    """Optional stratum: validate real-module import when heavy deps are installed."""

    pytest.importorskip("numpy")
    pytest.importorskip("gensim")
    pytest.importorskip("rapidfuzz")

    from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder

    assert MorphemeCandidateFinder is not None
