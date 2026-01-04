"""Shared type definitions for the Enochian modeling toolchain.

This module provides consistent type aliases and protocols used across
the codebase, ensuring type safety without excessive use of Any.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    Vector: TypeAlias = NDArray[np.floating]
else:
    try:
        import numpy as np
        from numpy.typing import NDArray

        Vector: TypeAlias = NDArray[np.floating]
        _HAS_NUMPY = True
    except ImportError:
        _HAS_NUMPY = False
        Vector: TypeAlias = list[float]


# Type for values that are known to be convertible to float
NumberConvertible: TypeAlias = float | int | str | None

# Type for values that may or may not be convertible to numbers.
# Used by _safe_number() functions that accept untrusted/dynamic values
# (e.g., from dict.get() calls where the value type is object | None).
MaybeNumber: TypeAlias = float | int | str | object | None


@runtime_checkable
class FastTextModel(Protocol):
    """Protocol for FastText-like embedding models.

    This protocol defines the minimal interface expected by functions
    that compute semantic similarity using FastText embeddings. It
    supports multiple implementations:

    - Facebook fasttext (C++/python bindings)
    - gensim.models.FastText
    - gensim KeyedVectors
    - FastTextWrapper from embeddings.py
    """

    def get_word_vector(self, word: str) -> Vector:
        """Return the embedding vector for a word."""
        ...


@runtime_checkable
class KeyedVectorsLike(Protocol):
    """Protocol for gensim KeyedVectors-like objects.

    Used when accessing the .wv attribute of a FastText model.
    """

    def get_word_vector(self, word: str) -> Vector:
        """Return the embedding vector for a word."""
        ...

    def get_vector(self, word: str) -> Vector:
        """Return the embedding vector for a word (alternative method)."""
        ...


@runtime_checkable
class SentenceEmbedder(Protocol):
    """Protocol for sentence embedding models like SentenceTransformer."""

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        normalize_embeddings: bool = False,
    ) -> Vector:
        """Encode sentences into embedding vectors."""
        ...


__all__ = [
    "FastTextModel",
    "KeyedVectorsLike",
    "MaybeNumber",
    "NumberConvertible",
    "SentenceEmbedder",
    "Vector",
]
