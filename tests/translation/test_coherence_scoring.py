"""Tests for semantic coherence scoring of decompositions."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Shim external dependencies
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.scoring import (
    CoherenceResult,
    compute_semantic_coherence,
    _compute_avg_pairwise_similarity,
    _cosine_similarity,
)


class MockFastTextModel:
    """Mock FastText model that returns predictable vectors for testing."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self._vectors = {k.lower(): v for k, v in vectors.items()}
        self.wv = self

    def get_word_vector(self, word: str) -> np.ndarray:
        return self._vectors.get(word.lower(), np.zeros(4))


class TestCosineSimlarity:
    """Tests for the _cosine_similarity helper."""

    def test_identical_vectors_return_one(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        assert abs(_cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        vec_a = np.array([1.0, 0.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(vec_a, vec_b)) < 1e-6

    def test_opposite_vectors_return_negative_one(self):
        vec_a = np.array([1.0, 2.0, 3.0, 4.0])
        vec_b = -vec_a
        assert abs(_cosine_similarity(vec_a, vec_b) + 1.0) < 1e-6

    def test_zero_vector_returns_zero(self):
        vec_a = np.array([1.0, 2.0, 3.0, 4.0])
        vec_b = np.zeros(4)
        assert _cosine_similarity(vec_a, vec_b) == 0.0


class TestAvgPairwiseSimilarity:
    """Tests for the _compute_avg_pairwise_similarity helper."""

    def test_single_morph_returns_zero(self):
        vectors = {"A": np.array([1.0, 0.0, 0.0, 0.0])}
        result = _compute_avg_pairwise_similarity(["A"], vectors)
        assert result == 0.0

    def test_two_identical_morphs_returns_one(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0])
        vectors = {"A": vec, "B": vec}
        result = _compute_avg_pairwise_similarity(["A", "B"], vectors)
        assert abs(result - 1.0) < 1e-6

    def test_two_orthogonal_morphs_returns_zero(self):
        vectors = {
            "A": np.array([1.0, 0.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        result = _compute_avg_pairwise_similarity(["A", "B"], vectors)
        assert abs(result) < 1e-6

    def test_missing_vectors_are_skipped(self):
        vectors = {"A": np.array([1.0, 0.0, 0.0, 0.0])}
        result = _compute_avg_pairwise_similarity(["A", "B", "C"], vectors)
        # Only A has a vector, so no pairs can be compared
        assert result == 0.0


class TestComputeSemanticCoherence:
    """Tests for the compute_semantic_coherence function."""

    def test_empty_morphs_returns_neutral_score(self):
        model = MockFastTextModel({})
        result = compute_semantic_coherence([], model)
        assert result.score == 0.5
        assert result.singleton_count == 0
        assert result.large_morph_count == 0

    def test_none_model_returns_neutral_score(self):
        result = compute_semantic_coherence(["NAZ", "PSAD"], None)
        assert result.score == 0.5

    def test_counts_singletons_and_large_morphs(self):
        # Vectors where singletons are similar, large morphs are dissimilar
        model = MockFastTextModel({
            "P": np.array([1.0, 0.0, 0.0, 0.0]),
            "D": np.array([0.9, 0.1, 0.0, 0.0]),  # Similar to P
            "NAZ": np.array([0.0, 1.0, 0.0, 0.0]),
            "SA": np.array([0.0, 0.0, 1.0, 0.0]),  # Orthogonal to NAZ
        })
        result = compute_semantic_coherence(["NAZ", "P", "SA", "D"], model)

        assert result.singleton_count == 2
        assert result.large_morph_count == 2

    def test_singletons_should_cohere(self):
        """When singletons are similar to each other, cohesion should be high."""
        # P and D have nearly identical vectors
        model = MockFastTextModel({
            "P": np.array([1.0, 0.0, 0.0, 0.0]),
            "D": np.array([1.0, 0.0, 0.0, 0.0]),
        })
        result = compute_semantic_coherence(["P", "D"], model)

        # High singleton cohesion
        assert result.singleton_cohesion > 0.9

    def test_large_morphs_should_be_diverse(self):
        """When large morphs are dissimilar, diversity should be high."""
        # NAZ and SA are orthogonal
        model = MockFastTextModel({
            "NAZ": np.array([1.0, 0.0, 0.0, 0.0]),
            "SA": np.array([0.0, 1.0, 0.0, 0.0]),
        })
        result = compute_semantic_coherence(["NAZ", "SA"], model)

        # High diversity (low similarity)
        assert result.large_morph_diversity > 0.9

    def test_single_singleton_returns_neutral(self):
        """Single singleton can't compute cohesion - returns neutral."""
        model = MockFastTextModel({
            "P": np.array([1.0, 0.0, 0.0, 0.0]),
            "NAZ": np.array([0.0, 1.0, 0.0, 0.0]),
            "SA": np.array([0.0, 0.0, 1.0, 0.0]),
        })
        result = compute_semantic_coherence(["NAZ", "P", "SA"], model)

        # Only one singleton, so no cohesion to compute
        assert result.singleton_count == 1
        assert result.singleton_cohesion == 0.0  # Can't compute with 1

    def test_good_decomposition_scores_higher(self):
        """A decomposition with cohering singletons and diverse large morphs scores well."""
        # Good decomposition: P and D are similar, NAZ and SA are dissimilar
        good_model = MockFastTextModel({
            "P": np.array([1.0, 0.0, 0.0, 0.0]),
            "D": np.array([0.95, 0.05, 0.0, 0.0]),  # Similar to P
            "NAZ": np.array([0.0, 1.0, 0.0, 0.0]),
            "SA": np.array([0.0, 0.0, 1.0, 0.0]),  # Orthogonal to NAZ
        })

        # Bad decomposition: P and D are dissimilar (incoherent singletons)
        bad_model = MockFastTextModel({
            "P": np.array([1.0, 0.0, 0.0, 0.0]),
            "D": np.array([0.0, 1.0, 0.0, 0.0]),  # Orthogonal to P
            "NAZ": np.array([0.0, 0.0, 1.0, 0.0]),
            "SA": np.array([0.0, 0.0, 0.0, 1.0]),
        })

        good_result = compute_semantic_coherence(["NAZ", "P", "SA", "D"], good_model)
        bad_result = compute_semantic_coherence(["NAZ", "P", "SA", "D"], bad_model)

        assert good_result.singleton_cohesion > bad_result.singleton_cohesion
