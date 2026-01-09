"""Utilities for post-run interpretation of unseen text."""

from __future__ import annotations

from .decomposition import DecompositionEngine, Decomposition
from .scoring import ScoringWeights, score_decomposition
from .llm_synthesis import (
    ConsensusSynthesisResult,
    SynthesisResult,
    synthesize_consensus,
    synthesize_definition,
)
from .service import InterpretationService, SingleWordTranslationService
from .strategies import apply_strategy

__all__ = [
    "InterpretationService",
    "SingleWordTranslationService",
    "DecompositionEngine",
    "Decomposition",
    "ScoringWeights",
    "score_decomposition",
    "apply_strategy",
    "SynthesisResult",
    "synthesize_definition",
    "synthesize_consensus",
    "ConsensusSynthesisResult",
]
