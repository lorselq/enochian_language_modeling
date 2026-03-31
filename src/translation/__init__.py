"""Utilities for post-run interpretation of unseen text."""

from __future__ import annotations

from .decomposition import DecompositionEngine, Decomposition
from .scoring import ScoringWeights, score_decomposition
from .llm_synthesis import (
    ConsensusSynthesisResult,
    PhraseRenderResult,
    SynthesisResult,
    render_phrase_translation,
    synthesize_consensus,
    synthesize_definition,
)
from .memory import ProvisionalLexiconEntry, TranslationMemoryRepository
from .phrase_service import PhraseTranslationService
from .service import InterpretationService, SingleWordTranslationService
from .strategies import apply_strategy

__all__ = [
    "InterpretationService",
    "SingleWordTranslationService",
    "PhraseTranslationService",
    "DecompositionEngine",
    "Decomposition",
    "ScoringWeights",
    "score_decomposition",
    "apply_strategy",
    "SynthesisResult",
    "synthesize_definition",
    "synthesize_consensus",
    "ConsensusSynthesisResult",
    "PhraseRenderResult",
    "render_phrase_translation",
    "TranslationMemoryRepository",
    "ProvisionalLexiconEntry",
]
