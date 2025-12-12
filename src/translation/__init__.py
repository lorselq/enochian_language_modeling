"""Utilities for post-run interpretation of unseen text."""

from .decomposition import DecompositionEngine, Decomposition
from .scoring import ScoringWeights, score_decomposition
from .service import InterpretationService

__all__ = [
    "InterpretationService",
    "DecompositionEngine",
    "Decomposition",
    "ScoringWeights",
    "score_decomposition",
]