"""
Phase 2 Tests: Decomposition & Filtering

Tests for Task 2.1 (generate_decompositions) as specified in TODO.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.decomposition import (
    DecompositionEngine,
    _build_breakdown,
    _build_support_lookup,
    _classify_support,
)
from translation.repository import (
    ClusterRecord,
    InsightsRepository,
    MorphHypothesisRecord,
    ResidualDetail,
    ResidualSemanticRecord,
    WordEvidence,
)
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.common.sqlite_bootstrap import sqlite3

# Database paths (may not exist in the repo but should be assumed available
INTERPRETATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "enochian_lm"
    / "root_extraction"
    / "interpretation"
)
SOLO_DB_PATH = INTERPRETATION_DIR / "solo_analysis_derived_definitions.sqlite3"
DEBATE_DB_PATH = INTERPRETATION_DIR / "debate_derived_definitions.sqlite3"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyVectors(dict):
    vector_size = 4

    def similar_by_word(self, _word: str, topn: int = 10):
        return []

    def __contains__(self, key: object) -> bool:  # pragma: no cover - simple shim
        return True

    def __getitem__(self, key: object):  # pragma: no cover - simple shim
        return np.zeros(self.vector_size)


class DummyFasttext:
    def __init__(self) -> None:
        self.wv = DummyVectors()


@pytest.fixture()
def monkeypatched_fasttext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "enochian_lm.root_extraction.utils.candidate_finder.get_fasttext_model",
        lambda model_path=None: DummyFasttext(),
    )


def _write_ngram_index(tmp_path: Path, tokens: Iterable[str]) -> Path:
    db_path = tmp_path / "ngram_index.sqlite3"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE ngrams (ngram TEXT PRIMARY KEY, total_occurrences INTEGER)"
        )
        conn.execute(
            "CREATE TABLE ngram_membership (ngram TEXT, canonical TEXT)"
        )
        for token in tokens:
            norm = token.lower()
            conn.execute(
                "INSERT INTO ngrams (ngram, total_occurrences) VALUES (?, ?)",
                (norm, 1),
            )
            conn.execute(
                "INSERT INTO ngram_membership (ngram, canonical) VALUES (?, ?)",
                (norm, token.upper()),
            )
    conn.close()
    return db_path


@pytest.fixture()
def candidate_finder(tmp_path: Path, monkeypatched_fasttext: None) -> MorphemeCandidateFinder:
    tokens = ["NAZ", "PSAD", "NAZP", "SAD", "A"]
    ngram_index = _write_ngram_index(tmp_path, tokens)
    dictionary_entries = [{"canonical": token.upper()} for token in tokens]
    return MorphemeCandidateFinder(
        ngram_db_path=ngram_index,
        fasttext_model_path=ngram_index,  # path unused due to monkeypatch
        dictionary_entries=dictionary_entries,
        min_n=1,
        max_n=7,
    )


@pytest.fixture()
def engine(candidate_finder: MorphemeCandidateFinder) -> DecompositionEngine:
    return DecompositionEngine(candidate_finder)


@pytest.fixture()
def solo_repo() -> InsightsRepository:
    repo = InsightsRepository(solo_path=SOLO_DB_PATH, debate_path=None)
    yield repo
    repo.close()


@pytest.fixture()
def debate_repo() -> InsightsRepository:
    repo = InsightsRepository(solo_path=None, debate_path=DEBATE_DB_PATH)
    yield repo
    repo.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_residual_detail() -> ResidualDetail:
    return ResidualDetail(
        normalized="",
        definition=None,
        coverage_ratio=None,
        residual_ratio=None,
        avg_confidence=None,
    )


# ---------------------------------------------------------------------------
# Task 2.1 Tests: generate_decompositions
# ---------------------------------------------------------------------------


class TestGenerateDecompositions:
    def test_multi_morph_word_returns_expected_splits(self, engine: DecompositionEngine):
        evidence = WordEvidence(
            word="NAZPSAD",
            variants_queried=["solo"],
            direct_clusters=[
                ClusterRecord(
                    variant="solo",
                    cluster_id=1,
                    run_id="r1",
                    ngram="NAZ",
                    cluster_index=0,
                    glossator_def=None,
                    residual_explained=None,
                    residual_ratio=None,
                    residual_headline=None,
                    residual_focus_prompt=None,
                    semantic_coverage=None,
                    cohesion=None,
                    semantic_cohesion=None,
                    best_config=None,
                    residual_details=[_dummy_residual_detail()],
                    raw_definitions=[],
                )
            ],
            residual_semantics=[
                ResidualSemanticRecord(
                    variant="solo",
                    run_id="r1",
                    residual="PSAD",
                    parent_word="NAZPSAD",
                    group_index=0,
                    group_size=1,
                    glossator_def=None,
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
            ],
        )

        decompositions = engine.generate_decompositions("NAZPSAD", evidence)

        # Expect both NAZ+PSAD and NAZP+SAD from the mock index
        morph_sets = [d.morphs for d in decompositions]
        assert ["NAZ", "PSAD"] in morph_sets
        assert ["NAZP", "SAD"] in morph_sets

        # Support labels are derived from evidence
        main = next(d for d in decompositions if d.morphs == ["NAZ", "PSAD"])
        assert main.morph_support["NAZ"] == "cluster"
        assert main.morph_support["PSAD"] == "residual"
        assert "coverage_ratio" in main.breakdown
        assert "residual_ratio" in main.breakdown

    def test_single_morph_word_returns_single_split(self, engine: DecompositionEngine):
        evidence = WordEvidence(
            word="NAZ",
            variants_queried=["solo"],
        )

        decompositions = engine.generate_decompositions("NAZ", evidence)

        assert any(d.morphs == ["NAZ"] for d in decompositions)
        for decomp in decompositions:
            assert isinstance(decomp.beam_score, float)

    def test_unknown_word_returns_empty_list(self, engine: DecompositionEngine):
        evidence = WordEvidence(word="XYZABC", variants_queried=["solo"])
        decompositions = engine.generate_decompositions("XYZABC", evidence)
        assert decompositions == []


class TestDatabaseBackedDecompositions:
    def test_solo_database_evidence_if_available(
        self, engine: DecompositionEngine, solo_repo: InsightsRepository
    ):
        if "solo" not in solo_repo.variants:
            pytest.skip("Solo database not available")

        evidence = solo_repo.fetch_word_evidence("A", variants=["solo"])
        decompositions = engine.generate_decompositions("A", evidence)

        assert isinstance(decompositions, list)
        if evidence.direct_clusters or evidence.residual_semantics:
            assert any("A" in d.morphs for d in decompositions)

    def test_debate_database_handles_missing_tables(
        self, engine: DecompositionEngine, debate_repo: InsightsRepository
    ):
        if "debate" not in debate_repo.variants:
            pytest.skip("Debate database not available")

        try:
            evidence = debate_repo.fetch_word_evidence("A", variants=["debate"])
        except Exception as exc:  # pragma: no cover - defensive skip for thin DBs
            if "no such table" in str(exc):
                pytest.skip(f"Debate database missing expected tables: {exc}")
            raise

        decompositions = engine.generate_decompositions("A", evidence)

        # If the database has no entries, we still return a list gracefully
        assert isinstance(decompositions, list)


# ---------------------------------------------------------------------------
# Task 2.1 Additional Tests: Edge Cases & Helper Functions
# ---------------------------------------------------------------------------


class TestGenerateDecompositionsEdgeCases:
    """Additional edge case tests for generate_decompositions."""

    def test_empty_word_returns_empty_list(self, engine: DecompositionEngine):
        """Empty string input should return an empty list immediately."""
        evidence = WordEvidence(word="", variants_queried=["solo"])
        decompositions = engine.generate_decompositions("", evidence)
        assert decompositions == []

    def test_case_insensitive_input(self, engine: DecompositionEngine):
        """Lowercase input should be normalized to uppercase and still work."""
        evidence = WordEvidence(word="naz", variants_queried=["solo"])
        decompositions = engine.generate_decompositions("naz", evidence)

        # Should find decompositions despite lowercase input
        assert any(d.morphs == ["NAZ"] for d in decompositions)
        # All morphs should be uppercase
        for decomp in decompositions:
            for morph in decomp.morphs:
                assert morph == morph.upper()

    def test_hypothesis_support_type(self, engine: DecompositionEngine):
        """Morph hypotheses should be classified as 'hypothesis' in morph_support."""
        evidence = WordEvidence(
            word="NAZ",
            variants_queried=["solo"],
            morph_hypotheses=[
                MorphHypothesisRecord(
                    variant="solo",
                    hyp_id=1,
                    morph="NAZ",
                    source_word="NAZPSAD",
                    anchor=None,
                    seed_glosses=["test"],
                    proposed_gloss="test gloss",
                    rationale=None,
                    delta_cosine=None,
                    residual_before=None,
                    residual_after=None,
                    created_at=None,
                )
            ],
        )

        decompositions = engine.generate_decompositions("NAZ", evidence)

        naz_decomp = next((d for d in decompositions if d.morphs == ["NAZ"]), None)
        assert naz_decomp is not None
        assert naz_decomp.morph_support["NAZ"] == "hypothesis"

    def test_unknown_morph_classification(self, engine: DecompositionEngine):
        """Morphs not in evidence should be classified as 'unknown'."""
        # Evidence has NAZ as cluster but not PSAD
        evidence = WordEvidence(
            word="NAZPSAD",
            variants_queried=["solo"],
            direct_clusters=[
                ClusterRecord(
                    variant="solo",
                    cluster_id=1,
                    run_id="r1",
                    ngram="NAZ",
                    cluster_index=0,
                    glossator_def=None,
                    residual_explained=None,
                    residual_ratio=None,
                    residual_headline=None,
                    residual_focus_prompt=None,
                    semantic_coverage=None,
                    cohesion=None,
                    semantic_cohesion=None,
                    best_config=None,
                    residual_details=[_dummy_residual_detail()],
                    raw_definitions=[],
                )
            ],
            # No residual_semantics for PSAD
        )

        decompositions = engine.generate_decompositions("NAZPSAD", evidence)
        main = next((d for d in decompositions if d.morphs == ["NAZ", "PSAD"]), None)
        assert main is not None
        assert main.morph_support["NAZ"] == "cluster"
        assert main.morph_support["PSAD"] == "unknown"

    def test_support_priority_cluster_over_residual(self, engine: DecompositionEngine):
        """When a morph appears in both clusters and residuals, cluster takes priority."""
        evidence = WordEvidence(
            word="NAZ",
            variants_queried=["solo"],
            direct_clusters=[
                ClusterRecord(
                    variant="solo",
                    cluster_id=1,
                    run_id="r1",
                    ngram="NAZ",
                    cluster_index=0,
                    glossator_def=None,
                    residual_explained=None,
                    residual_ratio=None,
                    residual_headline=None,
                    residual_focus_prompt=None,
                    semantic_coverage=None,
                    cohesion=None,
                    semantic_cohesion=None,
                    best_config=None,
                    residual_details=[_dummy_residual_detail()],
                    raw_definitions=[],
                )
            ],
            residual_semantics=[
                ResidualSemanticRecord(
                    variant="solo",
                    run_id="r1",
                    residual="NAZ",  # Same morph in residual
                    parent_word="SOMETHING",
                    group_index=0,
                    group_size=1,
                    glossator_def=None,
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
            ],
        )

        decompositions = engine.generate_decompositions("NAZ", evidence)
        naz_decomp = next((d for d in decompositions if d.morphs == ["NAZ"]), None)
        assert naz_decomp is not None
        # Cluster should take priority over residual
        assert naz_decomp.morph_support["NAZ"] == "cluster"


class TestBuildSupportLookup:
    """Direct tests for the _build_support_lookup helper function."""

    def test_empty_evidence_returns_empty_dict(self):
        """Empty evidence should produce empty support lookup."""
        evidence = WordEvidence(word="TEST", variants_queried=["solo"])
        result = _build_support_lookup(evidence)
        assert result == {}

    def test_cluster_support(self):
        """Clusters should map to 'cluster' support."""
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=[
                ClusterRecord(
                    variant="solo",
                    cluster_id=1,
                    run_id="r1",
                    ngram="foo",  # lowercase
                    cluster_index=0,
                    glossator_def=None,
                    residual_explained=None,
                    residual_ratio=None,
                    residual_headline=None,
                    residual_focus_prompt=None,
                    semantic_coverage=None,
                    cohesion=None,
                    semantic_cohesion=None,
                    best_config=None,
                    residual_details=[],
                    raw_definitions=[],
                )
            ],
        )
        result = _build_support_lookup(evidence)
        # Should be normalized to uppercase
        assert result["FOO"] == "cluster"

    def test_residual_support(self):
        """Residuals should map to 'residual' support."""
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            residual_semantics=[
                ResidualSemanticRecord(
                    variant="solo",
                    run_id="r1",
                    residual="bar",  # lowercase
                    parent_word="TESTBAR",
                    group_index=0,
                    group_size=1,
                    glossator_def=None,
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
            ],
        )
        result = _build_support_lookup(evidence)
        # Should be normalized to uppercase
        assert result["BAR"] == "residual"

    def test_hypothesis_support(self):
        """Hypotheses should map to 'hypothesis' support."""
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            morph_hypotheses=[
                MorphHypothesisRecord(
                    variant="solo",
                    hyp_id=1,
                    morph="baz",  # lowercase
                    source_word="TEST",
                    anchor=None,
                    seed_glosses=[],
                    proposed_gloss=None,
                    rationale=None,
                    delta_cosine=None,
                    residual_before=None,
                    residual_after=None,
                    created_at=None,
                )
            ],
        )
        result = _build_support_lookup(evidence)
        # Should be normalized to uppercase
        assert result["BAZ"] == "hypothesis"


class TestClassifySupport:
    """Direct tests for the _classify_support helper function."""

    def test_known_morph_returns_label(self):
        """Known morphs should return their support label."""
        lookup = {"FOO": "cluster", "BAR": "residual"}
        assert _classify_support("FOO", lookup) == "cluster"
        assert _classify_support("BAR", lookup) == "residual"

    def test_unknown_morph_returns_unknown(self):
        """Unknown morphs should return 'unknown'."""
        lookup = {"FOO": "cluster"}
        assert _classify_support("XYZ", lookup) == "unknown"

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        lookup = {"FOO": "cluster"}
        assert _classify_support("foo", lookup) == "cluster"
        assert _classify_support("Foo", lookup) == "cluster"
        assert _classify_support("FOO", lookup) == "cluster"


class TestBuildBreakdownFallback:
    """Tests for the _build_breakdown fallback path."""

    def test_fallback_with_empty_word(self):
        """Empty word should return zero coverage."""

        class MinimalCandidateFinder:
            pass

        finder = MinimalCandidateFinder()
        result = _build_breakdown(finder, "", [])  # type: ignore[arg-type]
        assert result["coverage_ratio"] == 0.0
        assert result["residual_ratio"] == 1.0
        assert result["segments"] == []
        assert result["uncovered"] == []

    def test_fallback_computes_coverage_from_segments(self):
        """Fallback should compute coverage from segment start/end fields."""

        # Create a minimal candidate finder without _build_breakdown
        class MinimalCandidateFinder:
            """Mock candidate finder without _build_breakdown method."""

            pass

        finder = MinimalCandidateFinder()
        coverage = [
            {"start": 0, "end": 3, "ngram": "NAZ"},
            {"start": 3, "end": 7, "ngram": "PSAD"},
        ]

        result = _build_breakdown(finder, "NAZPSAD", coverage)  # type: ignore[arg-type]

        assert result["coverage_ratio"] == 1.0
        assert result["residual_ratio"] == 0.0
        assert result["uncovered"] == []

    def test_fallback_identifies_uncovered_spans(self):
        """Fallback should identify gaps in coverage as uncovered spans."""

        class MinimalCandidateFinder:
            pass

        finder = MinimalCandidateFinder()
        # Coverage only for first 3 chars of a 7-char word
        coverage = [{"start": 0, "end": 3, "ngram": "NAZ"}]

        result = _build_breakdown(finder, "NAZPSAD", coverage)  # type: ignore[arg-type]

        assert result["coverage_ratio"] == 3 / 7
        assert result["residual_ratio"] == 4 / 7
        uncovered = result["uncovered"]
        assert isinstance(uncovered, list)
        assert len(uncovered) == 1
        assert uncovered[0]["span"] == [3, 7]

    def test_fallback_merges_overlapping_segments(self):
        """Overlapping segments should be merged for coverage calculation."""

        class MinimalCandidateFinder:
            pass

        finder = MinimalCandidateFinder()
        # Overlapping segments: [0,4) and [2,7) should merge to [0,7)
        coverage = [
            {"start": 0, "end": 4, "ngram": "NAZP"},
            {"start": 2, "end": 7, "ngram": "ZPSAD"},
        ]

        result = _build_breakdown(finder, "NAZPSAD", coverage)  # type: ignore[arg-type]

        assert result["coverage_ratio"] == 1.0
        assert result["residual_ratio"] == 0.0

    def test_fallback_handles_invalid_segment_values(self):
        """Fallback should gracefully handle invalid start/end values."""

        class MinimalCandidateFinder:
            pass

        finder = MinimalCandidateFinder()
        coverage = [
            {"start": "invalid", "end": "also_invalid"},  # Invalid types
            {"start": 0, "end": 3, "ngram": "NAZ"},  # Valid segment
        ]

        result = _build_breakdown(finder, "NAZPSAD", coverage)  # type: ignore[arg-type]

        # Should skip invalid segment, only count the valid one
        assert result["coverage_ratio"] == 3 / 7