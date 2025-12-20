from __future__ import annotations

"""
phase 2 Tests: Decomposition & Filtering

tests for Task 2.1 (generate_decompositions) as specified in TODO.md.
"""

import math
import sys
from pathlib import Path
from collections.abc import Iterator
from typing import Iterable
import types
import numpy as np
import pytest

# Shim external dependency used by candidate_finder without requiring heavy install
gensim_module = types.ModuleType("gensim")
gensim_utils = types.ModuleType("gensim.utils")
gensim_utils.simple_preprocess = lambda text: [str(text)]  # type: ignore[attr-defined]
gensim_models = types.ModuleType("gensim.models")


class _DummyFastText:  # pragma: no cover - simple import shim
    def __init__(self, *args, **kwargs):
        self.wv = self

    def get_vector(self, _token: str):
        return np.zeros(4)


gensim_models.FastText = _DummyFastText  # type: ignore[attr-defined]
gensim_module.utils = gensim_utils  # type: ignore[attr-defined]
gensim_module.models = gensim_models  # type: ignore[attr-defined]
sys.modules.setdefault("gensim", gensim_module)
sys.modules.setdefault("gensim.utils", gensim_utils)
sys.modules.setdefault("gensim.models", gensim_models)

sentence_module = types.ModuleType("sentence_transformers")


class _DummySentenceTransformer:  # pragma: no cover - simple import shim
    def __init__(self, *args, **kwargs):
        pass


sentence_module.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
sys.modules.setdefault("sentence_transformers", sentence_module)


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.decomposition import (
    Decomposition,
    DecompositionEngine,
    apply_hard_filters,
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
from translation.scoring import (
    ScoringWeights,
    score_decomposition,
)
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord
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
        conn.execute("CREATE TABLE ngram_membership (ngram TEXT, canonical TEXT)")
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
def candidate_finder(
    tmp_path: Path, monkeypatched_fasttext: None
) -> MorphemeCandidateFinder:
    tokens = ["NAZ", "PSAD", "NAZP", "SAD", "A"]
    ngram_index = _write_ngram_index(tmp_path, tokens)
    dictionary_entries: list[EntryRecord] = [
        {"canonical": token.upper(), "alternates": []} for token in tokens
    ]
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
def solo_repo() -> Iterator[InsightsRepository]:
    repo = InsightsRepository(solo_path=SOLO_DB_PATH, debate_path=None)
    yield repo
    repo.close()


@pytest.fixture()
def debate_repo() -> Iterator[InsightsRepository]:
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
    def test_multi_morph_word_returns_expected_splits(
        self, engine: DecompositionEngine
    ):
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


# ---------------------------------------------------------------------------
# Task 2.2 Tests: apply_hard_filters
# ---------------------------------------------------------------------------


class TestApplyHardFilters:
    def _cluster(self, morph: str, *, cluster_id: int = 1) -> ClusterRecord:
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

    def _residual(self, morph: str) -> ResidualSemanticRecord:
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

    def _hypothesis(self, morph: str, delta: float | None) -> MorphHypothesisRecord:
        return MorphHypothesisRecord(
            variant="solo",
            hyp_id=1,
            morph=morph,
            source_word=morph,
            anchor=None,
            seed_glosses=[],
            proposed_gloss=None,
            rationale=None,
            delta_cosine=delta,
            residual_before=None,
            residual_after=None,
            created_at=None,
        )

    def _make_decomposition(
        self,
        morphs: list[str],
        *,
        beam_score: float = 1.0,
        coverage_ratio: float = 1.0,
        residual_ratio: float = 0.0,
        morph_support: Iterable[tuple[str, str]] | None = None,
    ) -> Decomposition:
        return Decomposition(
            morphs=[m.upper() for m in morphs],
            beam_score=beam_score,
            breakdown={
                "segments": [],
                "uncovered": [],
                "coverage_ratio": coverage_ratio,
                "residual_ratio": residual_ratio,
            },
            morph_support={k.upper(): v for k, v in (morph_support or [])},
        )

    def test_discard_unsupported_morphs(self):
        evidence = WordEvidence(
            word="NAZPSAD",
            variants_queried=["solo"],
            direct_clusters=[self._cluster("NAZ")],
            residual_semantics=[self._residual("PSAD")],
        )

        supported = self._make_decomposition(["NAZ", "PSAD"], residual_ratio=0.0)
        unsupported = self._make_decomposition(["NAZ", "XYZ"], residual_ratio=0.0)

        filtered = apply_hard_filters([supported, unsupported], evidence)

        assert supported in filtered
        assert unsupported not in filtered

    def test_prefers_lower_residual_ratio(self):
        evidence = WordEvidence(
            word="NAZPSAD",
            variants_queried=["solo"],
            direct_clusters=[self._cluster("NAZ")],
            residual_semantics=[self._residual("PSAD")],
        )

        strong = self._make_decomposition(["NAZ", "PSAD"], residual_ratio=0.1)
        weak = self._make_decomposition(["NAZ", "PSAD"], residual_ratio=0.8)

        filtered = apply_hard_filters([weak, strong], evidence)

        assert strong in filtered
        assert weak not in filtered

    def test_prefers_attested_morphs(self):
        common_clusters = [self._cluster("COMMON", cluster_id=i) for i in range(4)]
        rare_cluster = self._cluster("RARE", cluster_id=99)

        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=common_clusters + [rare_cluster],
        )

        common_decomp = self._make_decomposition(["COMMON"], residual_ratio=0.0)
        rare_decomp = self._make_decomposition(["RARE"], residual_ratio=0.0)

        filtered = apply_hard_filters([common_decomp, rare_decomp], evidence)

        assert common_decomp in filtered
        assert rare_decomp not in filtered

    def test_hypothesis_thresholds_respected(self):
        evidence = WordEvidence(
            word="DELTAOMEGA",
            variants_queried=["solo"],
            morph_hypotheses=[
                self._hypothesis("DELTA", 0.05),
                self._hypothesis("OMEGA", 0.6),
            ],
        )

        low_confidence = self._make_decomposition(["DELTA"], residual_ratio=0.0)
        high_confidence = self._make_decomposition(["OMEGA"], residual_ratio=0.0)

        filtered = apply_hard_filters(
            [low_confidence, high_confidence], evidence, min_support_threshold=0.2
        )

        assert high_confidence in filtered
        assert low_confidence not in filtered

    def test_database_backed_filters_handle_missing_rows(
        self, engine: DecompositionEngine, solo_repo: InsightsRepository
    ):
        if "solo" not in solo_repo.variants:
            pytest.skip("Solo database not available")

        evidence = solo_repo.fetch_word_evidence("A", variants=["solo"])
        decompositions = engine.generate_decompositions("A", evidence)

        filtered = apply_hard_filters(decompositions, evidence)

        assert isinstance(filtered, list)
        if (
            evidence.direct_clusters
            or evidence.residual_semantics
            or evidence.morph_hypotheses
        ):
            assert len(filtered) <= len(decompositions)

    def test_debate_database_filters_fail_gracefully(
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

        filtered = apply_hard_filters(decompositions, evidence)

        assert isinstance(filtered, list)

# ---------------------------------------------------------------------------
# Task 2.3 Tests: Soft Scoring
# ---------------------------------------------------------------------------


class TestScoreDecomposition:
    def _cluster(self, morph: str, *, cohesion: float, coverage: float) -> ClusterRecord:
        return ClusterRecord(
            variant="solo",
            cluster_id=1,
            run_id="r1",
            ngram=morph,
            cluster_index=0,
            glossator_def=None,
            residual_explained=None,
            residual_ratio=None,
            residual_headline=None,
            residual_focus_prompt=None,
            semantic_coverage=coverage,
            cohesion=cohesion,
            semantic_cohesion=None,
            best_config=None,
            residual_details=[],
            raw_definitions=[],
        )

    def _residual(self, morph: str) -> ResidualSemanticRecord:
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

    def _hypothesis(self, morph: str) -> MorphHypothesisRecord:
        return MorphHypothesisRecord(
            variant="solo",
            hyp_id=1,
            morph=morph,
            source_word=morph,
            anchor=None,
            seed_glosses=[],
            proposed_gloss=None,
            rationale=None,
            delta_cosine=0.7,
            residual_before=None,
            residual_after=None,
            created_at=None,
        )

    def _decomp(
        self,
        morphs: Iterable[str],
        *,
        beam_score: float,
        residual_ratio: float,
    ) -> Decomposition:
        return Decomposition(
            morphs=[m.upper() for m in morphs],
            beam_score=beam_score,
            breakdown={
                "segments": [],
                "uncovered": [],
                "coverage_ratio": 1.0 - residual_ratio,
                "residual_ratio": residual_ratio,
            },
            morph_support={m.upper(): "cluster" for m in morphs},
        )

    def test_high_quality_scores_outperform_low_quality(self):
        """High-quality decompositions should score higher than low-quality ones.

        Tests that the composite score correctly combines:
        - beam_prior (0.3 weight): beam_score from TF-IDF
        - avg_cluster_quality (0.25 weight): average of cohesion and semantic_coverage
        - residual_coverage (0.25 weight): 1 - residual_ratio
        - acceptance_bonus (0.2 weight): count of evidence sources
        """
        evidence = WordEvidence(
            word="NAZPSAD",
            variants_queried=["solo"],
            direct_clusters=[
                self._cluster("NAZ", cohesion=0.8, coverage=0.8),
                self._cluster("PSAD", cohesion=0.8, coverage=0.8),
            ],
        )

        strong = self._decomp(["NAZ", "PSAD"], beam_score=5.0, residual_ratio=0.0)
        weak = self._decomp(["NAZ", "PSAD"], beam_score=1.0, residual_ratio=0.5)

        strong_score = score_decomposition(strong, evidence)
        weak_score = score_decomposition(weak, evidence)

        # Expected: beam_prior(0.3*5.0) + avg_quality(0.25*0.8) + coverage(0.25*1.0) + bonus(0.2*2)
        expected = 0.3 * 5.0 + 0.25 * 0.8 + 0.25 * 1.0 + 0.2 * 2
        assert math.isclose(strong_score, expected, rel_tol=1e-6)
        assert weak_score < strong_score

    def test_weights_normalize_before_scoring(self):
        """Custom weights should be normalized to sum to 1.0 before scoring.

        This ensures scores remain comparable across different weight configurations.
        Weights (2.0, 2.0, 0.0, 0.0) should normalize to (0.5, 0.5, 0.0, 0.0).
        """
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=[self._cluster("TEST", cohesion=0.6, coverage=0.6)],
        )

        decomp = self._decomp(["TEST"], beam_score=2.0, residual_ratio=0.25)

        custom_weights = ScoringWeights(
            beam_prior=2.0, avg_cluster_quality=2.0, residual_coverage=0.0, acceptance_bonus=0.0
        )

        score = score_decomposition(decomp, evidence, custom_weights)

        # Weights normalize: (2.0, 2.0, 0.0, 0.0) -> (0.5, 0.5, 0.0, 0.0)
        normalized_beam = 0.5
        normalized_cluster = 0.5
        expected = normalized_beam * 2.0 + normalized_cluster * 0.6
        assert math.isclose(score, expected, rel_tol=1e-6)

    def test_acceptance_bonus_counts_all_sources(self):
        """Acceptance bonus should count all evidence types with appropriate weights.

        The acceptance bonus uses weights:
        - 1.0 for clusters
        - 0.5 for residual semantics
        - 0.3 for morph hypotheses
        """
        evidence = WordEvidence(
            word="OMEGA",
            variants_queried=["solo"],
            direct_clusters=[self._cluster("OMEGA", cohesion=0.4, coverage=0.5)],
            residual_semantics=[self._residual("OMEGA")],
            morph_hypotheses=[self._hypothesis("OMEGA")],
        )

        decomp = self._decomp(["OMEGA"], beam_score=0.0, residual_ratio=0.0)
        score = score_decomposition(decomp, evidence)

        # Score components with default weights:
        # beam_prior: 0.3 * 0.0 = 0.0
        # avg_cluster_quality: 0.25 * 0.45 = 0.1125 (avg of 0.4 cohesion and 0.5 coverage)
        # residual_coverage: 0.25 * 1.0 = 0.25
        # acceptance_bonus: 0.2 * (1.0 + 0.5 + 0.3) = 0.36
        expected = 0.0 + 0.1125 + 0.25 + 0.36
        assert math.isclose(score, expected, rel_tol=1e-6)

    def test_solo_database_scoring_handles_empty_tables(self, solo_repo: InsightsRepository):
        """Scoring should work gracefully even when database tables are empty.

        Tests that score_decomposition handles real database evidence without errors,
        even when there may be no cluster/residual/hypothesis records.
        """
        if "solo" not in solo_repo.variants:
            pytest.skip("Solo database not available")

        evidence = solo_repo.fetch_word_evidence("A", variants=["solo"])
        decomp = self._decomp([evidence.word or "A"], beam_score=1.0, residual_ratio=0.0)

        score = score_decomposition(decomp, evidence)
        assert isinstance(score, float)

    def test_debate_database_scoring_fails_gracefully(self, debate_repo: InsightsRepository):
        """Scoring should work gracefully with debate database evidence.

        Tests that score_decomposition handles debate database evidence without errors,
        even when tables may be missing or empty.
        """
        if "debate" not in debate_repo.variants:
            pytest.skip("Debate database not available")

        try:
            evidence = debate_repo.fetch_word_evidence("A", variants=["debate"])
        except Exception as exc:  # pragma: no cover - defensive skip for thin DBs
            if "no such table" in str(exc):
                pytest.skip(f"Debate database missing expected tables: {exc}")
            raise

        decomp = self._decomp([evidence.word or "A"], beam_score=1.0, residual_ratio=0.0)
        score = score_decomposition(decomp, evidence)
        assert isinstance(score, float)

    def test_missing_cluster_metrics_handled_gracefully(self):
        """Clusters with missing cohesion/coverage metrics should not break scoring.

        When a cluster has None for cohesion or semantic_coverage, the scoring
        function should handle it gracefully and use available metrics.
        """
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=[
                ClusterRecord(
                    variant="solo",
                    cluster_id=1,
                    run_id="r1",
                    ngram="TEST",
                    cluster_index=0,
                    glossator_def=None,
                    residual_explained=None,
                    residual_ratio=None,
                    residual_headline=None,
                    residual_focus_prompt=None,
                    semantic_coverage=None,  # Missing coverage
                    cohesion=0.7,  # Only cohesion available
                    semantic_cohesion=None,
                    best_config=None,
                    residual_details=[],
                    raw_definitions=[],
                )
            ],
        )

        decomp = self._decomp(["TEST"], beam_score=1.0, residual_ratio=0.0)
        score = score_decomposition(decomp, evidence)

        # Should use only cohesion (0.7) for cluster quality
        # Score: 0.3*1.0 + 0.25*0.7 + 0.25*1.0 + 0.2*1.0
        expected = 0.3 * 1.0 + 0.25 * 0.7 + 0.25 * 1.0 + 0.2 * 1.0
        assert math.isclose(score, expected, rel_tol=1e-6)

    def test_empty_decomposition_scores_zero(self):
        """Empty decomposition (no morphs) should score near zero.

        This edge case should not raise an error.
        """
        evidence = WordEvidence(word="", variants_queried=["solo"])
        decomp = self._decomp([], beam_score=0.0, residual_ratio=1.0)

        score = score_decomposition(decomp, evidence)

        # Empty decomposition should have minimal score
        assert score == 0.0

    def test_multiple_clusters_per_morph_uses_best_quality(self):
        """When a morph has multiple clusters, the best quality should be used.

        This tests the _average_cluster_quality helper's logic for choosing
        the highest quality among multiple clusters for the same morph.
        """
        evidence = WordEvidence(
            word="NAZ",
            variants_queried=["solo"],
            direct_clusters=[
                self._cluster("NAZ", cohesion=0.3, coverage=0.4),  # Quality: 0.35
                self._cluster("NAZ", cohesion=0.8, coverage=0.9),  # Quality: 0.85 (best)
                self._cluster("NAZ", cohesion=0.5, coverage=0.6),  # Quality: 0.55
            ],
        )

        decomp = self._decomp(["NAZ"], beam_score=2.0, residual_ratio=0.1)
        score = score_decomposition(decomp, evidence)

        # Should use best quality (0.85) not average
        # Score: 0.3*2.0 + 0.25*0.85 + 0.25*0.9 + 0.2*3.0
        expected = 0.3 * 2.0 + 0.25 * 0.85 + 0.25 * 0.9 + 0.2 * 3.0
        assert math.isclose(score, expected, rel_tol=1e-6)

    def test_invalid_beam_score_defaults_to_zero(self):
        """Non-numeric beam_score values should default to 0.0.

        Tests the _safe_number helper's error handling.
        """
        evidence = WordEvidence(
            word="TEST",
            variants_queried=["solo"],
            direct_clusters=[self._cluster("TEST", cohesion=0.5, coverage=0.5)],
        )

        # Create decomposition with invalid beam_score
        decomp = Decomposition(
            morphs=["TEST"],
            beam_score="invalid",  # type: ignore - intentionally invalid
            breakdown={
                "segments": [],
                "uncovered": [],
                "coverage_ratio": 1.0,
                "residual_ratio": 0.0,
            },
            morph_support={"TEST": "cluster"},
        )

        score = score_decomposition(decomp, evidence)

        # Should still compute a score, treating beam_score as 0.0
        # Score: 0.3*0.0 + 0.25*0.5 + 0.25*1.0 + 0.2*1.0
        expected = 0.3 * 0.0 + 0.25 * 0.5 + 0.25 * 1.0 + 0.2 * 1.0
        assert math.isclose(score, expected, rel_tol=1e-6)


class TestScoringWeights:
    """Tests for the ScoringWeights dataclass and its normalization logic."""

    def test_default_weights_sum_to_one(self):
        """Default weights should already sum to 1.0."""
        weights = ScoringWeights()
        total = (
            weights.beam_prior
            + weights.avg_cluster_quality
            + weights.residual_coverage
            + weights.acceptance_bonus
        )
        assert math.isclose(total, 1.0, rel_tol=1e-6)

    def test_normalized_weights_sum_to_one(self):
        """Normalized weights should always sum to 1.0."""
        weights = ScoringWeights(
            beam_prior=10.0,
            avg_cluster_quality=5.0,
            residual_coverage=3.0,
            acceptance_bonus=2.0,
        )
        normalized = weights.normalized()
        total = (
            normalized.beam_prior
            + normalized.avg_cluster_quality
            + normalized.residual_coverage
            + normalized.acceptance_bonus
        )
        assert math.isclose(total, 1.0, rel_tol=1e-6)

    def test_zero_weights_fallback_to_equal(self):
        """When all weights are zero or negative, should fallback to equal weights.

        This prevents division by zero and keeps the scoring function usable.
        """
        zero_weights = ScoringWeights(
            beam_prior=0.0,
            avg_cluster_quality=0.0,
            residual_coverage=0.0,
            acceptance_bonus=0.0,
        )
        normalized = zero_weights.normalized()

        # Should fall back to equal weights (0.25 each)
        assert math.isclose(normalized.beam_prior, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.avg_cluster_quality, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.residual_coverage, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.acceptance_bonus, 0.25, rel_tol=1e-6)

    def test_negative_weights_fallback_to_equal(self):
        """Negative weights should also trigger the equal weights fallback."""
        negative_weights = ScoringWeights(
            beam_prior=-1.0,
            avg_cluster_quality=-2.0,
            residual_coverage=-3.0,
            acceptance_bonus=-4.0,
        )
        normalized = negative_weights.normalized()

        # Should fall back to equal weights (0.25 each)
        assert math.isclose(normalized.beam_prior, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.avg_cluster_quality, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.residual_coverage, 0.25, rel_tol=1e-6)
        assert math.isclose(normalized.acceptance_bonus, 0.25, rel_tol=1e-6)

    def test_proportional_normalization(self):
        """Normalization should preserve proportions between weights."""
        weights = ScoringWeights(
            beam_prior=2.0,  # 2/6 = 1/3
            avg_cluster_quality=4.0,  # 4/6 = 2/3
            residual_coverage=0.0,
            acceptance_bonus=0.0,
        )
        normalized = weights.normalized()

        # Proportions should be preserved
        assert math.isclose(normalized.beam_prior, 1.0 / 3.0, rel_tol=1e-6)
        assert math.isclose(normalized.avg_cluster_quality, 2.0 / 3.0, rel_tol=1e-6)
        assert math.isclose(normalized.residual_coverage, 0.0, rel_tol=1e-6)
        assert math.isclose(normalized.acceptance_bonus, 0.0, rel_tol=1e-6)