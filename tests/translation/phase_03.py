from __future__ import annotations

"""
Phase 3 Tests: Strategy & Ranking (Task 3.1)

Tests for Task 3.1 (apply_strategy) as specified in TODO.md.

These tests mirror the Phase 01 / Phase 02 style:
- Add src/ to sys.path for imports
- Use real solo/debate databases when available
- Skip gracefully when a database is missing or lacks expected tables
- Keep helpers fully internal to this file
"""

import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Iterable

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.decomposition import Decomposition  # noqa: E402
from translation.repository import (  # noqa: E402
    ClusterRecord,
    InsightsRepository,
    MorphHypothesisRecord,
    ResidualSemanticRecord,
    WordEvidence,
)
from translation.strategies import apply_strategy  # noqa: E402


# ------------------------------------------------------------------------------
# Database paths (may not exist locally but are expected by design)
# ------------------------------------------------------------------------------

INTERPRETATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "enochian_lm"
    / "root_extraction"
    / "interpretation"
)
SOLO_DB_PATH = INTERPRETATION_DIR / "solo_analysis_derived_definitions.sqlite3"
DEBATE_DB_PATH = INTERPRETATION_DIR / "debate_derived_definitions.sqlite3"


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def solo_repo() -> Iterator[InsightsRepository]:
    """Repository connected to solo database only."""
    repo = InsightsRepository(solo_path=SOLO_DB_PATH, debate_path=None)
    yield repo
    repo.close()


@pytest.fixture
def debate_repo() -> Iterator[InsightsRepository]:
    """Repository connected to debate database only."""
    repo = InsightsRepository(solo_path=None, debate_path=DEBATE_DB_PATH)
    yield repo
    repo.close()


# ------------------------------------------------------------------------------
# Helpers (internal to this file)
# ------------------------------------------------------------------------------


def _cluster(morph: str, *, cluster_id: int = 1) -> ClusterRecord:
    return ClusterRecord(
        variant="solo",
        cluster_id=cluster_id,
        run_id="r1",
        ngram=morph,
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


def _residual(morph: str) -> ResidualSemanticRecord:
    return ResidualSemanticRecord(
        variant="solo",
        run_id="r1",
        residual=morph,
        parent_word=morph,
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


def _hypothesis(morph: str) -> MorphHypothesisRecord:
    return MorphHypothesisRecord(
        variant="solo",
        hyp_id=1,
        morph=morph,
        source_word=morph,
        anchor=None,
        seed_glosses=[],
        proposed_gloss=None,
        rationale=None,
        delta_cosine=0.5,
        residual_before=None,
        residual_after=None,
        created_at=None,
    )


def _decomp(
    morphs: Iterable[str],
    *,
    beam_score: float = 1.0,
    residual_ratio: float = 0.0,
    support_label: str = "unknown",
) -> Decomposition:
    upper = [m.upper() for m in morphs]
    return Decomposition(
        morphs=upper,
        beam_score=beam_score,
        breakdown={
            "segments": [],
            "uncovered": [],
            "coverage_ratio": 1.0 - residual_ratio,
            "residual_ratio": residual_ratio,
        },
        morph_support={m: support_label for m in upper},
    )


# ==============================================================================
# Task 3.1 Tests: apply_strategy
# ==============================================================================


class TestApplyStrategy:
    """
    Task 3.1 Requirements (TODO.md):
    - Implement apply_strategy(decompositions, strategy, evidence)
    - prefer-fewer: bonus = -0.5 * len(morphs)
    - prefer-known: bonus = 0.3 * (count of morphs with uses > 5)
    - prefer-balance: bonus = -0.2 * variance(lengths)
    - return list sorted by final score (descending)
    """

    def test_empty_input_returns_empty(self):
        evidence = WordEvidence(word="TEST", variants_queried=["solo"])
        assert apply_strategy([], strategy="prefer-fewer", evidence=evidence) == []

    def test_prefer_fewer_rewards_shorter_paths(self):
        evidence = WordEvidence(word="NAZPSAD", variants_queried=["solo"])

        short = _decomp(["NAZ", "PSAD"])
        long = _decomp(list("NAZPSAD"))

        ranked = apply_strategy([(short, 5.0), (long, 5.0)], "prefer-fewer", evidence)

        assert ranked[0][0].morphs == ["NAZ", "PSAD"]
        assert ranked[0][1] > ranked[1][1]

    def test_prefer_known_counts_uses_across_all_evidence_types(self):
        """
        This test is constructed to ensure "uses" are aggregated across the
        evidence bundle, not only clusters.

        KNOWN total uses:
        - direct_clusters: 4
        - residual_semantics: 1
        - morph_hypotheses: 1
        Total = 6 (>5) -> should earn bonus

        RARE total uses:
        - direct_clusters: 5
        Total = 5 (not >5) -> should NOT earn bonus
        """
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=[_cluster("KNOWN", cluster_id=i) for i in range(1, 5)]
            + [_cluster("RARE", cluster_id=i) for i in range(100, 105)],
            residual_semantics=[_residual("KNOWN")],
            morph_hypotheses=[_hypothesis("KNOWN")],
        )

        known = _decomp(["KNOWN"])
        rare = _decomp(["RARE"])

        ranked = apply_strategy([(known, 1.0), (rare, 1.0)], "prefer-known", evidence)

        assert ranked[0][0].morphs == ["KNOWN"]
        assert ranked[0][1] > ranked[1][1]

    def test_prefer_balance_penalizes_high_variance(self):
        evidence = WordEvidence(word="TESTWORD", variants_queried=["solo"])

        balanced = _decomp(["NAZ", "PSAD"], beam_score=2.0)   # lengths 3 and 4
        lopsided = _decomp(["NAZPSAD", "D"], beam_score=2.0)  # lengths 7 and 1

        ranked = apply_strategy([(balanced, 2.0), (lopsided, 2.0)], "prefer-balance", evidence)

        assert ranked[0][0].morphs == ["NAZ", "PSAD"]
        assert ranked[0][1] > ranked[1][1]

    def test_unknown_strategy_falls_back_to_base_score_sort(self):
        evidence = WordEvidence(word="TEST", variants_queried=["solo"])

        a = _decomp(["A"])
        b = _decomp(["B"])

        ranked = apply_strategy([(a, 2.0), (b, 1.0)], "unknown", evidence)

        assert [score for _, score in ranked] == [2.0, 1.0]
        assert ranked[0][0].morphs == ["A"]


class TestApplyStrategyWithDatabases:
    def test_solo_database_evidence_reranks_gracefully(self, solo_repo: InsightsRepository):
        if "solo" not in solo_repo.variants:
            pytest.skip("Solo database not available")

        evidence = solo_repo.fetch_word_evidence("A", variants=["solo"])
        decomp = _decomp([evidence.word or "A"])

        ranked = apply_strategy([(decomp, 1.0)], strategy="prefer-known", evidence=evidence)

        assert isinstance(ranked, list)
        assert len(ranked) == 1
        assert ranked[0][0].morphs[0] == (evidence.word or "A").upper()

    def test_debate_database_handles_missing_tables(self, debate_repo: InsightsRepository):
        if "debate" not in debate_repo.variants:
            pytest.skip("Debate database not available")

        try:
            evidence = debate_repo.fetch_word_evidence("A", variants=["debate"])
        except Exception as exc:  # pragma: no cover - defensive skip for thin DBs
            if "no such table" in str(exc):
                pytest.skip(f"Debate database missing expected tables: {exc}")
            raise

        decomp = _decomp([evidence.word or "A"])
        ranked = apply_strategy([(decomp, 1.0)], strategy="prefer-fewer", evidence=evidence)

        assert isinstance(ranked, list)
        assert len(ranked) == 1
        assert ranked[0][0].morphs[0] == (evidence.word or "A").upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
