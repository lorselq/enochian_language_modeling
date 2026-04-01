"""Regression tests for the translation strategy upgrade work.

These tests focus on the new behavior introduced for single-word and
phrase-level translation:

- debate DB path selection prefers the populated extraction-only artifact;
- nested glossator payloads are parsed correctly;
- rejected residual pieces are treated as negative evidence;
- exact-word and provisional analyses are surfaced explicitly;
- phrase parsing uses global scoring rather than a greedy left-to-right read;
- provisional phrase observations accumulate in translation memory.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest


def _install_dependency_shims() -> None:
    """Install lightweight import shims for optional heavy dependencies.

    The translation stack imports NumPy, gensim, sentence-transformers,
    RapidFuzz, and the query-model wrapper at module import time. These tests
    only exercise the orchestration logic, so small stand-ins keep the unit
    suite focused and runnable in minimal environments.
    """

    if "numpy" not in sys.modules:
        numpy_module = types.ModuleType("numpy")
        numpy_module.ndarray = list  # type: ignore[attr-defined]
        numpy_module.array = lambda values, dtype=None: list(values)  # type: ignore[attr-defined]
        numpy_module.asarray = lambda values, dtype=None: list(values)  # type: ignore[attr-defined]
        numpy_module.zeros = lambda size, dtype=float: [0.0] * int(size)  # type: ignore[attr-defined]
        numpy_module.dot = lambda left, right: sum(  # type: ignore[attr-defined]
            float(a) * float(b) for a, b in zip(left, right, strict=False)
        )
        numpy_module.linalg = types.SimpleNamespace(
            norm=lambda values: sum(float(value) ** 2 for value in values) ** 0.5
        )
        sys.modules["numpy"] = numpy_module

    gensim_module = sys.modules.setdefault("gensim", types.ModuleType("gensim"))
    gensim_utils = sys.modules.setdefault(
        "gensim.utils",
        types.ModuleType("gensim.utils"),
    )
    gensim_utils.simple_preprocess = lambda text, deacc=True, min_len=1: [  # type: ignore[attr-defined]
        str(text)
    ]

    gensim_models = sys.modules.setdefault(
        "gensim.models",
        types.ModuleType("gensim.models"),
    )

    class _DummyFastText:
        """Provide the minimal FastText surface used by embeddings helpers."""

        def __init__(self, *args, **kwargs) -> None:
            self.wv = self

        @classmethod
        def load(cls, _path: str) -> "_DummyFastText":
            return cls()

        def get_vector(self, _token: str) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0]

        def similar_by_word(self, _word: str, topn: int = 5) -> list[tuple[str, float]]:
            return []

    gensim_models.FastText = _DummyFastText  # type: ignore[attr-defined]
    gensim_module.models = gensim_models  # type: ignore[attr-defined]
    gensim_module.utils = gensim_utils  # type: ignore[attr-defined]

    sentence_module = sys.modules.setdefault(
        "sentence_transformers",
        types.ModuleType("sentence_transformers"),
    )

    class _DummySentenceTransformer:
        """Return a predictable embedding vector for definition clustering."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, _text: str, normalize_embeddings: bool = True) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0]

    sentence_module.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]

    rapidfuzz_module = sys.modules.setdefault(
        "rapidfuzz",
        types.ModuleType("rapidfuzz"),
    )
    rapidfuzz_module.process = types.SimpleNamespace(extract=lambda *args, **kwargs: [])  # type: ignore[attr-defined]
    rapidfuzz_module.fuzz = types.SimpleNamespace(ratio=lambda *args, **kwargs: 0)  # type: ignore[attr-defined]

    query_tool_module = types.ModuleType(
        "enochian_lm.root_extraction.tools.query_model_tool"
    )

    class _DummyQueryModelTool:
        """Return empty JSON so rendering code can fall back deterministically."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def _run(self, *args, **kwargs) -> dict[str, str]:
            return {"response_text": "{}"}

    query_tool_module.QueryModelTool = _DummyQueryModelTool  # type: ignore[attr-defined]
    sys.modules.setdefault(
        "enochian_lm.root_extraction.tools.query_model_tool",
        query_tool_module,
    )

    pydantic_module = sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))

    class _DummyBaseModel:
        """Satisfy the dictionary loader's lightweight model declarations."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    pydantic_module.BaseModel = _DummyBaseModel  # type: ignore[attr-defined]
    pydantic_module.Field = lambda default=None, **kwargs: default  # type: ignore[attr-defined]
    pydantic_module.field_validator = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: (lambda func: func)
    )

    dotenv_module = sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
    dotenv_module.find_dotenv = lambda *args, **kwargs: ""  # type: ignore[attr-defined]
    dotenv_module.load_dotenv = lambda *args, **kwargs: False  # type: ignore[attr-defined]


_install_dependency_shims()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils import candidate_finder as root_candidate_finder_module
from translation.decomposition import Decomposition
from translation.llm_synthesis import (
    PhraseRenderBundleResult,
    PhraseRenderResult,
    _build_phrase_bundle_prompt,
    _build_phrase_footnote_fallback,
    _build_phrase_lay_render_prompt,
    _build_phrase_render_prompt,
    _compact_lay_gloss,
    _parse_phrase_bundle_response,
    _parse_phrase_lay_render_response,
    render_phrase_bundle,
    render_phrase_lay_translation,
    render_phrase_translation,
)
from translation import strategies as translation_strategies
from translation import scoring as translation_scoring
from translation import service as translation_service_module
from translation.memory import TranslationMemoryRepository
from translation.phrase_service import PhraseTranslationService
from translation import cli as translation_cli
from translation.repository import (
    AttestedDefinition,
    ClusterRecord,
    DictionaryMorph,
    InsightsRepository,
    ResidualSemanticRecord,
    WordEvidence,
    _parse_glossator_definition,
)
from translation.strategies import (
    compose_semantic_bundle,
    compute_complementarity_band_similarity,
    extract_definition_candidates,
)
from translation.service import SingleWordTranslationService


class FakeCandidateFinder:
    """Expose the tiny candidate-finder interface the services need.

    These tests target translation orchestration, not n-gram search quality, so
    a small in-memory stub keeps the service under test while avoiding the
    heavyweight real candidate-finder setup.
    """

    def __init__(self, dictionary: dict[str, dict[str, object]] | None = None) -> None:
        self.dictionary = dictionary or {}
        self.min_n = 1
        self.max_n = 7
        self.beam_width = 5
        self.fasttext_model = None

    def close(self) -> None:
        """Mirror the real candidate-finder cleanup surface."""


class FakeDecompositionEngine:
    """Return a deterministic list of decompositions for one test scenario."""

    def __init__(self, decompositions: list[Decomposition]) -> None:
        self._decompositions = decompositions

    def generate_decompositions(
        self,
        word: str,
        evidence: WordEvidence,
        **_kwargs: object,
    ) -> tuple[list[Decomposition], dict[str, object]]:
        """Act like the beam-search layer without invoking real segmentation."""
        return copy.deepcopy(self._decompositions), {"decomposition_count": len(self._decompositions)}


class FakeRepository:
    """Supply deterministic evidence to the translation services.

    The upgrade work is mainly about how the services rank and combine evidence,
    so the repository fake focuses on returning clean, pre-shaped evidence
    payloads while still exposing the same public API as the real repository.
    """

    def __init__(
        self,
        *,
        evidence_by_word: dict[str, WordEvidence],
        support_clusters: dict[str, ClusterRecord] | None = None,
        rejected_morphs: set[str] | None = None,
    ) -> None:
        self.variants = ["solo"]
        self._evidence_by_word = {
            word.upper(): copy.deepcopy(evidence)
            for word, evidence in evidence_by_word.items()
        }
        self._support_clusters = {
            morph.upper(): copy.deepcopy(record)
            for morph, record in (support_clusters or {}).items()
        }
        self._rejected_morphs = {morph.upper() for morph in (rejected_morphs or set())}

    def close(self) -> None:
        """Mirror the real repository cleanup surface."""

    def require_variants(self, variants: list[str] | tuple[str, ...]) -> None:
        """Accept requested variants so service setup can proceed."""

    def fetch_word_evidence(
        self,
        word: str,
        variants=None,
        **_kwargs: object,
    ) -> WordEvidence:
        """Return a fresh evidence object so service mutation stays isolated."""
        normalized = word.upper()
        if normalized in self._evidence_by_word:
            return copy.deepcopy(self._evidence_by_word[normalized])
        return WordEvidence(word=normalized, variants_queried=["solo"])

    def fetch_morph_support(self, morphs, variants=None):
        """Return only accepted support records for the requested morphs."""
        clusters = []
        for morph in morphs:
            cluster = self._support_clusters.get(str(morph).upper())
            if cluster is not None:
                clusters.append(copy.deepcopy(cluster))
        return clusters, [], []

    def fetch_rejected_morphs(self, morphs, variants=None) -> set[str]:
        """Return the subset of requested morphs that are explicitly rejected."""
        return {
            str(morph).upper()
            for morph in morphs
            if str(morph).upper() in self._rejected_morphs
        }

    def fetch_accepted_definition_counts(self, morphs, variants=None) -> dict[str, int]:
        """Keep ambiguity scoring inert so tests stay focused on ranking policy."""
        return {}

    def fetch_accepted_definition_glosses(
        self,
        morphs,
        variants=None,
        include_clusters: bool = True,
        include_residuals: bool = True,
    ) -> dict[str, list[tuple[str, float | None]]]:
        """Keep definition clustering inert for these orchestration tests."""
        return {}

    def fasttext_diagnostics(self) -> dict[str, object]:
        """Return empty model diagnostics for stable assertions."""
        return {}

    def path_diagnostics(self) -> dict[str, object]:
        """Return empty path diagnostics for stable assertions."""
        return {}

    def word_lookup_diagnostics(self, word: str, variants=None) -> dict[str, object]:
        """Return a minimal lookup diagnostic payload."""
        return {"word": word, "variants": list(variants or self.variants)}


class FakeWordService:
    """Provide predictable token analyses to the phrase translation service."""

    def __init__(
        self,
        *,
        results_by_word: dict[str, dict[str, object]],
        dictionary: dict[str, dict[str, object]] | None = None,
    ) -> None:
        self._results_by_word = {
            word.upper(): copy.deepcopy(result)
            for word, result in results_by_word.items()
        }
        self.candidate_finder = types.SimpleNamespace(dictionary=dictionary or {})
        self.repository = types.SimpleNamespace(variants=["solo"])
        self.active_variants = ["solo"]
        self.llm_enabled = False
        self.llm_use_remote = False

    def translate_word(self, word: str, **_kwargs: object) -> dict[str, object]:
        """Return the prebuilt single-word result for the requested token."""
        return copy.deepcopy(self._results_by_word[word.upper()])

    def close(self) -> None:
        """Mirror the real word-service cleanup surface."""


def _cluster_record(
    morph: str,
    definition: str,
    *,
    cluster_id: int = 1,
    semantic_core: list[str] | None = None,
    negative_contrast: list[str] | None = None,
) -> ClusterRecord:
    """Build a compact accepted cluster record for a target morph."""
    glossator_payload: dict[str, object] = {"DEFINITION": definition, "REJECTED": False}
    if semantic_core:
        glossator_payload["SEMANTIC_CORE"] = list(semantic_core)
    if negative_contrast:
        glossator_payload["NEGATIVE_CONTRAST"] = list(negative_contrast)
    return ClusterRecord(
        variant="solo",
        cluster_id=cluster_id,
        run_id="run-1",
        ngram=morph,
        cluster_index=0,
        glossator_def=json.dumps(glossator_payload),
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


def _residual_record(
    residual: str,
    definition: str,
    *,
    parent_word: str | None = None,
) -> ResidualSemanticRecord:
    """Build a residual-semantic record for placeholder-anchor regressions.

    The placeholder demotion tests need the same raw residual headline shape the
    production pipeline sees when a whole-word anchor is derived from residual
    evidence rather than a grounded dictionary or cluster gloss.
    """

    normalized_parent = parent_word or residual
    return ResidualSemanticRecord(
        variant="solo",
        run_id="run-1",
        residual=residual,
        parent_word=normalized_parent,
        group_index=0,
        group_size=1,
        glossator_def=json.dumps({"DEFINITION": definition, "REJECTED": False}),
        glossator_prompt=None,
        residual_headline=definition,
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


def _word_candidate(
    definition: str | None,
    *,
    analysis_type: str,
    score: float,
    confidence: float,
    morphs: list[str] | None = None,
    definitions: list[str] | None = None,
    warnings: list[str] | None = None,
    provenance: str | None = None,
    definition_trace: dict[str, object] | None = None,
    semantic_core: list[str] | None = None,
    negative_contrast: list[str] | None = None,
    surface_gloss_strategy: str | None = None,
    meanings: list[dict[str, object]] | None = None,
    semantic_bundle: list[dict[str, object]] | None = None,
    bundle_surface_gloss: str | None = None,
    bundle_head_gloss: str | None = None,
    bundle_function_profile: str | None = None,
    bundle_coherence_score: float | None = None,
    blind_mode_whole_word_rescue: bool = False,
    blind_mode_rescue_note: str | None = None,
    concatenated_meanings: str | None = None,
) -> dict[str, object]:
    """Build a translation candidate payload in the service's public schema."""
    surface = morphs or ["TOKEN"]
    normalized_definitions = []
    if definition is not None:
        normalized_definitions.append(definition)
    if definitions:
        normalized_definitions.extend(definitions)
    unique_definitions = list(dict.fromkeys(normalized_definitions))
    candidate_meanings = meanings
    if candidate_meanings is None:
        candidate_meanings = [
            {
                "morph": surface[0],
                "canonical": surface[0],
                "definition": definition,
                "definitions": unique_definitions,
                "provenance": provenance or analysis_type,
                "anchor_strength": 1.0,
                "semantic_core": list(semantic_core or []),
                "semantic_core_terms": list(semantic_core or []),
                "negative_contrast": list(negative_contrast or []),
                "surface_gloss": definition,
                "surface_gloss_strategy": surface_gloss_strategy or "cleaned_definition",
                "definition_trace": definition_trace
                or {
                    "selected_definition": definition,
                    "raw_selected_definition": definition,
                    "selected_source": provenance or analysis_type,
                    "selected_quality": 1.0,
                    "selected_semantic_core": list(semantic_core or []),
                    "selected_negative_contrast": list(negative_contrast or []),
                    "surface_gloss": definition,
                    "surface_gloss_strategy": surface_gloss_strategy or "cleaned_definition",
                    "runner_ups": [],
                    "suppressed": [],
                    "blind_dictionary_fallback": False,
                    "negative_contrast_penalties": [],
                    "meta_linguistic_rejections": [],
                },
            }
        ]
    return {
        "rank": 1,
        "analysis_type": analysis_type,
        "morphs": list(surface),
        "score": score,
        "confidence": confidence,
        "meanings": candidate_meanings,
        "warnings": list(warnings or []),
        "semantic_bundle": list(semantic_bundle or []),
        "bundle_surface_gloss": bundle_surface_gloss,
        "bundle_head_gloss": bundle_head_gloss,
        "bundle_function_profile": bundle_function_profile,
        "bundle_coherence_score": bundle_coherence_score,
        "blind_mode_whole_word_rescue": blind_mode_whole_word_rescue,
        "blind_mode_rescue_note": blind_mode_rescue_note,
        "concatenated_meanings": concatenated_meanings,
    }


def _word_result(
    word: str,
    *candidates: dict[str, object],
    fallback_morphs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Wrap candidate payloads in the single-word result schema."""
    return {
        "word": word,
        "variants_queried": ["solo"],
        "strategy": "prefer-balance",
        "evidence_mode": "all",
        "weighting_enabled": True,
        "llm_enabled": False,
        "llm_mode": None,
        "llm_context": None,
        "timestamp": "2026-03-29T00:00:00Z",
        "candidates": list(candidates),
        "evidence": {},
        "fallback_morphs": list(fallback_morphs or []),
        "diagnostics": {},
    }


def test_config_prefers_populated_debate_extraction_only_database() -> None:
    """Keep debate translation pointed at the populated on-disk SQLite file."""
    paths = get_config_paths()

    assert paths["debate_extraction_only"].exists()
    assert paths["debate_primary"].exists()
    assert paths["debate_primary"].stat().st_size == 0
    assert paths["debate"] == paths["debate_extraction_only"]


def test_debate_repository_opens_the_selected_extraction_only_database() -> None:
    """Verify the configured debate path is immediately usable by translation."""
    paths = get_config_paths()
    repository = InsightsRepository(solo_path=None, debate_path=paths["debate"])
    try:
        assert repository.variants == ["debate"]
    finally:
        repository.close()


def test_parse_glossator_definition_reads_nested_raw_text_payload() -> None:
    """Recover accepted definitions even when the payload is nested in RAW_TEXT."""
    payload = json.dumps(
        {
            "RAW_TEXT": "```json\n"
            '{"DEFINITION":"sword","REJECTED":false}'
            "\n```"
        }
    )

    assert _parse_glossator_definition(payload) == ("sword", False)


def test_fetch_rejected_morphs_uses_nested_rejected_payload(tmp_path: Path) -> None:
    """Surface rejected residual pieces so translation can penalize them."""
    db_path = tmp_path / "rejected.sqlite3"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE root_residual_semantics (
                residual TEXT,
                glossator_def TEXT,
                group_idx INTEGER
            );
            """
        )
        conn.execute(
            """
            INSERT INTO root_residual_semantics (residual, glossator_def, group_idx)
            VALUES (?, ?, ?);
            """,
            (
                "PSAD",
                json.dumps(
                    {
                        "RAW_TEXT": "```json\n"
                        '{"DEFINITION":"","REJECTED":true}'
                        "\n```"
                    }
                ),
                0,
            ),
        )
    conn.close()

    repository = InsightsRepository(solo_path=db_path, debate_path=None)
    try:
        assert repository.fetch_rejected_morphs(["PSAD"], variants=["solo"]) == {"PSAD"}
    finally:
        repository.close()


def test_translate_word_prioritizes_dictionary_exact_candidates() -> None:
    """Expose exact dictionary items as the lead non-provisional reading."""
    evidence = WordEvidence(
        word="OD",
        variants_queried=["solo"],
        dictionary_morphs={
            "OD": DictionaryMorph(
                morph="OD",
                definition="self",
                senses=["self", "the selfsame"],
                part_of_speech="pronoun",
            )
        },
    )
    repository = FakeRepository(evidence_by_word={"OD": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(
            {"od": {"canonical": "OD", "pos": "pronoun"}}
        ),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word("od", llm=False, use_beam_search=True)

    assert result["candidates"]
    lead = result["candidates"][0]
    assert lead["analysis_type"] == "dictionary_exact"
    assert lead["meanings"][0]["definition"] == "self"
    assert float(lead["confidence"]) >= 0.98


def test_translate_word_no_whole_word_suppresses_dictionary_exact_and_prefers_split() -> None:
    """Blind retranslation should not let exact dictionary entries short-circuit parsing.

    The broadened `--no-whole-word(s)` mode is meant for blind retranslations,
    where dictionary exact matches would otherwise leak the canonical answer.
    This regression proves the exact dictionary read is removed and the
    productive decomposition can win instead.
    """

    evidence = WordEvidence(
        word="MICAOLZ",
        variants_queried=["solo"],
        dictionary_morphs={
            "MICAOLZ": DictionaryMorph(
                morph="MICAOLZ",
                definition="mighty",
                senses=["mighty"],
                part_of_speech="adjective",
            )
        },
    )
    repository = FakeRepository(
        evidence_by_word={"MICAOLZ": evidence},
        support_clusters={
            "MI": _cluster_record("MI", "force", cluster_id=10),
            "CA": _cluster_record("CA", "center", cluster_id=11),
            "OLZ": _cluster_record("OLZ", "brightness", cluster_id=12),
        },
    )
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["MI", "CA", "OLZ"],
                canonicals=["MI", "CA", "OLZ"],
                beam_score=8.5,
                breakdown={
                    "segments": [
                        {"start": 0, "end": 2, "ngram": "MI", "canonical": "MI"},
                        {"start": 2, "end": 4, "ngram": "CA", "canonical": "CA"},
                        {"start": 4, "end": 7, "ngram": "OLZ", "canonical": "OLZ"},
                    ],
                    "uncovered": [],
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                },
                morph_support={"MI": "cluster", "CA": "cluster", "OLZ": "cluster"},
            )
        ]
    )

    result = service.translate_word(
        "micaolz",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"]
    assert result["candidates"][0]["analysis_type"] == "compositional"
    assert all(
        candidate["analysis_type"] != "dictionary_exact"
        for candidate in result["candidates"]
    )
    blind = result["diagnostics"]["blind_retranslation"]
    assert blind["enabled"] is True
    assert blind["short_root_max_len"] == 2
    assert any(
        "suppressed exact dictionary match for MICAOLZ" in note
        for note in blind["suppressed"]
    )


def test_translate_word_no_whole_word_keeps_short_db_root_anchor() -> None:
    """Blind retranslation still allows very short DB-backed root anchors.

    The new mode should remain usable for root-like tokens such as `NA`, where
    exact dictionary suppression should not force an unnecessary decomposition
    if the DB already strongly supports the short form itself.
    """

    evidence = WordEvidence(
        word="NA",
        variants_queried=["solo"],
        direct_clusters=[_cluster_record("NA", "divine essence", cluster_id=20)],
    )
    repository = FakeRepository(evidence_by_word={"NA": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word(
        "na",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"]
    assert result["candidates"][0]["analysis_type"] == "whole_word_anchor"
    assert result["candidates"][0]["meanings"][0]["definition"] == "divine essence"
    assert result["diagnostics"]["blind_retranslation"]["suppressed"] == []


def test_translate_word_no_whole_word_suppresses_long_db_anchor() -> None:
    """Blind retranslation should drop long DB-backed whole-word anchors.

    Long exact DB anchors such as `OVOARS` are precisely the kind of
    pre-solved readings that make blind retranslations feel like cheating. This
    regression proves those anchors disappear when the broadened flag is used.
    """

    evidence = WordEvidence(
        word="OVOARS",
        variants_queried=["solo"],
        direct_clusters=[_cluster_record("OVOARS", "center", cluster_id=30)],
    )
    repository = FakeRepository(evidence_by_word={"OVOARS": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word(
        "ovoars",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"] == []
    assert any(
        "suppressed long whole-word DB anchor for OVOARS" in note
        for note in result["diagnostics"]["blind_retranslation"]["suppressed"]
    )


def test_translate_word_rejected_piece_does_not_survive_split_filter() -> None:
    """Prefer a supported whole-word anchor over a split containing rejected evidence."""
    whole_word_cluster = _cluster_record("NAZPSAD", "sword", cluster_id=10)
    supported_piece = _cluster_record("NAZ", "pillar", cluster_id=11)
    evidence = WordEvidence(
        word="NAZPSAD",
        variants_queried=["solo"],
        direct_clusters=[whole_word_cluster],
        rejected_morphs={"PSAD"},
    )
    repository = FakeRepository(
        evidence_by_word={"NAZPSAD": evidence},
        support_clusters={"NAZ": supported_piece},
        rejected_morphs={"PSAD"},
    )
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["NAZ", "PSAD"],
                canonicals=["NAZ", "PSAD"],
                beam_score=5.0,
                breakdown={
                    "segments": [
                        {"start": 0, "end": 3, "ngram": "NAZ", "canonical": "NAZ"},
                        {"start": 3, "end": 7, "ngram": "PSAD", "canonical": "PSAD"},
                    ],
                    "uncovered": [],
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                },
                morph_support={"NAZ": "cluster", "PSAD": "unknown"},
            )
        ]
    )

    result = service.translate_word("NAZPSAD", llm=False, use_beam_search=True)

    assert result["candidates"][0]["analysis_type"] == "whole_word_anchor"
    assert all(
        candidate["analysis_type"] != "compositional"
        for candidate in result["candidates"]
    )
    assert result["evidence"]["rejected_morphs"] == ["PSAD"]


def test_translate_word_returns_provisional_fallback_when_support_is_missing() -> None:
    """Return an explicit provisional reading instead of failing silently."""
    evidence = WordEvidence(
        word="GAMFABURO",
        variants_queried=["solo"],
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="GAMFABURO",
                definition="banner of arising",
                cluster_id=5,
                root_ngram="GAM",
            )
        ],
    )
    repository = FakeRepository(evidence_by_word={"GAMFABURO": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word("GAMFABURO", llm=False, use_beam_search=True)

    assert result["candidates"]
    lead = result["candidates"][0]
    assert lead["analysis_type"] == "provisional"
    assert lead["meanings"][0]["definition"] == "banner of arising"
    assert any(
        "Provisional whole-word reading" in warning
        for warning in lead.get("warnings", [])
    )
    assert float(lead["confidence"]) <= 0.55


def test_translate_word_no_whole_word_allows_provisional_without_dictionary_collision() -> None:
    """Blind retranslation may still surface provisional exact reads for unknown words.

    `ELO`-style cases should stay available when the system has raw attestation
    but no exact dictionary entry. This keeps blind mode from becoming an
    artificial decomposition-only failure mode for genuinely unknown forms.
    """

    evidence = WordEvidence(
        word="ELO",
        variants_queried=["solo"],
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="ELO",
                definition="shining being",
                cluster_id=6,
                root_ngram="EL",
            )
        ],
    )
    repository = FakeRepository(evidence_by_word={"ELO": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word(
        "elo",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"]
    assert result["candidates"][0]["analysis_type"] == "provisional"
    assert result["candidates"][0]["meanings"][0]["definition"] == "shining being"
    assert result["diagnostics"]["blind_retranslation"]["suppressed"] == []


def test_translate_word_no_whole_word_blocks_provisional_when_dictionary_entry_exists() -> None:
    """Blind retranslation should not smuggle back a blocked exact read via provisional fallback.

    If an exact dictionary entry exists for the token, blind mode must suppress
    both the exact dictionary candidate and any provisional whole-word fallback
    that would effectively restore the same shortcut.
    """

    evidence = WordEvidence(
        word="MICAOLZ",
        variants_queried=["solo"],
        dictionary_morphs={
            "MICAOLZ": DictionaryMorph(
                morph="MICAOLZ",
                definition="mighty",
                senses=["mighty"],
                part_of_speech="adjective",
            )
        },
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="MICAOLZ",
                definition="strong one",
                cluster_id=7,
                root_ngram="MI",
            )
        ],
    )
    repository = FakeRepository(evidence_by_word={"MICAOLZ": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word(
        "micaolz",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"] == []
    blind = result["diagnostics"]["blind_retranslation"]["suppressed"]
    assert any(
        "suppressed exact dictionary match for MICAOLZ" in note
        for note in blind
    )
    assert any(
        "suppressed provisional full-word fallback for MICAOLZ" in note
        for note in blind
    )


def test_extract_definition_candidates_blind_mode_excludes_dictionary_provenance() -> None:
    """Blind mode should keep compositional definition pools dictionary-blind.

    Exact whole-word suppression is not enough for retranslations if the
    decomposition selector can still quietly pull dictionary definitions back
    in at the morph level. This regression keeps richer non-dictionary evidence
    in play while dropping dictionary provenance from the candidate pool.
    """

    evidence = WordEvidence(
        word="BLIORS",
        variants_queried=["solo"],
        direct_clusters=[_cluster_record("BLIORS", "comfort", cluster_id=80)],
        dictionary_morphs={
            "BLIORS": DictionaryMorph(
                morph="BLIORS",
                definition="state",
                senses=["state"],
                part_of_speech="noun",
            )
        },
    )

    candidates = extract_definition_candidates(
        ["BLIORS"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["BLIORS"]
    assert all(
        candidate["source"] != "dictionary"
        for candidate in candidates["BLIORS"]
    )
    assert candidates["BLIORS"][0]["definition"] == "comfort"


def test_extract_definition_candidates_blind_mode_reintroduces_dictionary_as_last_resort() -> None:
    """Blind mode may reintroduce dictionary evidence only when nothing else survives."""

    evidence = WordEvidence(
        word="CAOSGO",
        variants_queried=["solo"],
        dictionary_morphs={
            "CAOSGO": DictionaryMorph(
                morph="CAOSGO",
                definition="earth",
                senses=["earth"],
                part_of_speech="noun",
            )
        },
    )

    candidates = extract_definition_candidates(
        ["CAOSGO"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["CAOSGO"]
    assert candidates["CAOSGO"][0]["source"] == "dictionary"
    assert candidates["CAOSGO"][0]["blind_dictionary_fallback"] is True


def test_extract_definition_candidates_prefers_semantic_core_over_verbose_definition() -> None:
    """Human-facing morph glosses should prefer semantic_core over prose scaffolding.

    The selected candidate may carry a rich explanatory definition, but the
    translation surface should use the lexical `semantic_core` when it is
    available. This keeps CAOS/BLI-family outputs aligned to meaning in speech
    instead of meta-linguistic commentary.
    """

    evidence = WordEvidence(
        word="CAOSGO",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "CAOSGO",
                "CAOS signifies the physical earth as a governable entity walked upon, watered, visited, and possessing moss, creatures, and a center.",
                cluster_id=91,
                semantic_core=["earth", "ground"],
                negative_contrast=["govern", "making"],
            )
        ],
    )

    candidates = extract_definition_candidates(
        ["CAOSGO"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["CAOSGO"]
    assert candidates["CAOSGO"][0]["definition"] == "earth"
    assert candidates["CAOSGO"][0]["surface_gloss_strategy"] == "semantic_core"
    assert candidates["CAOSGO"][0]["semantic_core_terms"] == ["earth", "ground"]
    assert candidates["CAOSGO"][0]["negative_contrast"] == ["govern", "making"]
    assert candidates["CAOSGO"][0]["negative_contrast_penalties"] == []


def test_extract_definition_candidates_can_promote_specific_definition_span() -> None:
    """A semantically aligned definition span may outrank the bare semantic core.

    `semantic_core` should stay authoritative, but when the full definition
    contains a short, more specific lexical phrase that clearly tracks that
    core, the surface gloss may use the richer phrase instead.
    """

    evidence = WordEvidence(
        word="NACRO",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "NACRO",
                "The divine source or godly essence underlying lordship and spirit.",
                cluster_id=95,
                semantic_core=["source"],
            )
        ],
    )

    candidates = extract_definition_candidates(
        ["NACRO"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["NACRO"]
    assert candidates["NACRO"][0]["definition"] == "divine source"
    assert candidates["NACRO"][0]["surface_gloss_strategy"] == "semantic_core_guided_definition"


def test_extract_definition_candidates_keeps_semantic_core_when_specific_span_hits_negative_contrast() -> None:
    """Negative contrast should block an over-specific upgrade that conflicts."""

    evidence = WordEvidence(
        word="CAOSGO",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "CAOSGO",
                "Burning earth as a scourged land.",
                cluster_id=96,
                semantic_core=["earth"],
                negative_contrast=["burning"],
            )
        ],
    )

    candidates = extract_definition_candidates(
        ["CAOSGO"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["CAOSGO"]
    assert candidates["CAOSGO"][0]["definition"] == "earth"
    assert candidates["CAOSGO"][0]["surface_gloss_strategy"] == "semantic_core"
    assert candidates["CAOSGO"][0]["negative_contrast_penalties"] == []


def test_extract_definition_candidates_penalizes_negative_contrast_overlap() -> None:
    """Negative contrast should demote candidates that echo their own exclusions."""

    evidence = WordEvidence(
        word="OIAD",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "OIAD",
                "The proximate sacred referent; this.",
                cluster_id=101,
                semantic_core=["this"],
                negative_contrast=["this"],
            ),
            _cluster_record(
                "OIAD",
                "The proximate sacred referent; presence.",
                cluster_id=102,
                semantic_core=["presence"],
                negative_contrast=["eternal god"],
            ),
        ],
    )

    candidates = extract_definition_candidates(
        ["OIAD"],
        evidence,
        max_per_morph=5,
        allow_dictionary=False,
    )

    assert candidates["OIAD"]
    penalties_by_cluster = {
        int(candidate["cluster_id"]): list(candidate["negative_contrast_penalties"])
        for candidate in candidates["OIAD"]
    }
    assert penalties_by_cluster[101] == ["this"]
    assert penalties_by_cluster[102] == []


def test_compute_complementarity_band_similarity_stays_offline_safe(monkeypatch) -> None:
    """Definition scoring should degrade gracefully when embeddings are unavailable.

    Phrase/token trace generation must still work in offline or sandboxed
    environments. If the sentence-transformer loader fails, the selector should
    return a neutral score instead of crashing the translation flow.
    """

    monkeypatch.setattr(
        translation_strategies,
        "get_sentence_transformer_if_available",
        lambda _name, *, local_files_only=False: None,
    )

    assert compute_complementarity_band_similarity(["comfort", "earth"]) == 0.0


def test_compact_lay_gloss_prefers_meaning_over_meta_linguistic_scaffolding() -> None:
    """Gloss compaction should recover meaning, not the analysis label."""

    assert (
        _compact_lay_gloss(
            "A prefixal deictic morpheme meaning 'this', used to specify a particular entity.",
            token="OIAD",
        )
        == "this"
    )
    assert (
        _compact_lay_gloss(
            "CAOS signifies the physical earth as a governable entity walked upon, watered, visited, and possessing moss, creatures, and a center.",
            token="CAOSGO",
        )
        == "earth"
    )
    assert (
        _compact_lay_gloss(
            "BLI encodes the state or provision of comfort as solace, reassurance, or supportive consolation.",
            token="BLIORS",
        )
        == "comfort"
    )
    assert (
        _compact_lay_gloss(
            "DS is a derivational morpheme that functions as a relativizing prefix, combining with verb stems to form compound relative pronouns meaning 'which [verb]'. As an independent word, it means 'which' or 'that' in relative clauses.",
            token="DS",
        )
        == "which"
    )


def test_phrase_footnote_fallback_prefers_selected_trace_over_shorter_alternates() -> None:
    """Keep family-consistent token glosses instead of chasing shorter alternates.

    BLIORS/CAOSGO-style failures showed the old fallback happily swapping in
    shorter alternates like `state` or `act` even when the chosen surviving
    definition trace already pointed at `comfort` and `earth`.
    """

    parse_payload = {
        "phrase": "bliors caosgo",
        "translation_skeleton": "comfort earth",
        "token_choices": [
            {
                "token": "BLIORS",
                "definition": "comfort",
                "raw_definition": "comfort",
                "analysis_type": "compositional",
                "role_hint": "noun",
                "selected_source": "cluster",
                "alternates": ["darkness", "state"],
                "definition_trace": {
                    "selected_definition": "comfort",
                    "selected_source": "cluster",
                    "selected_quality": 0.93,
                    "runner_ups": [
                        {"definition": "state", "source": "attested", "quality": 0.20},
                        {"definition": "darkness", "source": "hypothesis", "quality": 0.10},
                    ],
                    "suppressed": [
                        "Blind mode suppressed dictionary-backed definition candidates for BLIORS."
                    ],
                    "blind_dictionary_fallback": False,
                    "relation_context": {
                        "right": {
                            "neighbor_token": "CAOSGO",
                            "relation": "coordination/apposition",
                            "direction": "left_to_right",
                            "score": 0.4,
                        }
                    },
                    "selection_reason": "Selected cluster-backed evidence survived for this token.",
                },
            },
            {
                "token": "CAOSGO",
                "definition": "earth",
                "raw_definition": "earth",
                "analysis_type": "compositional",
                "role_hint": "noun",
                "selected_source": "cluster",
                "alternates": ["act", "govern"],
                "definition_trace": {
                    "selected_definition": "earth",
                    "selected_source": "cluster",
                    "selected_quality": 0.89,
                    "runner_ups": [
                        {"definition": "govern", "source": "attested", "quality": 0.18},
                        {"definition": "act", "source": "hypothesis", "quality": 0.11},
                    ],
                    "suppressed": [
                        "Blind mode suppressed dictionary-backed definition candidates for CAOSGO."
                    ],
                    "blind_dictionary_fallback": False,
                },
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    translation, notes = _build_phrase_footnote_fallback(parse_payload)

    assert translation == "comfort [^1] earth [^2]"
    assert notes[0]["rendered_text"] == "comfort"
    assert notes[1]["rendered_text"] == "earth"
    assert "cluster-backed" in notes[0]["explanation"]
    assert "dictionary-backed definition candidates for BLIORS" in notes[0]["explanation"]
    assert 'alternative "state"' in notes[0]["explanation"]


def test_phrase_translation_surfaces_definition_trace_in_json_payload(tmp_path: Path) -> None:
    """Phrase JSON should expose the token-level evidence trace for inspection."""

    word_service = FakeWordService(
        results_by_word={
            "BLIORS": _word_result(
                "BLIORS",
                _word_candidate(
                    "comfort",
                    analysis_type="compositional",
                    score=12.0,
                    confidence=0.86,
                    morphs=["BLIORS"],
                    definitions=["comfort", "state", "darkness"],
                    provenance="cluster",
                    definition_trace={
                        "selected_definition": "comfort",
                        "selected_source": "cluster",
                        "selected_quality": 0.86,
                        "runner_ups": [
                            {"definition": "state", "source": "attested", "quality": 0.20},
                        ],
                        "suppressed": [
                            "Blind mode suppressed dictionary-backed definition candidates for BLIORS."
                        ],
                        "blind_dictionary_fallback": False,
                    },
                ),
            ),
        },
        dictionary={"bliors": {"canonical": "BLIORS", "pos": "noun"}},
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-trace.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase("bliors", top_k=1, llm=False)

    candidate = result["token_analyses"][0]["candidates"][0]
    assert candidate["chosen_in_parse"] is True
    assert candidate["definition_trace"]["selected_source"] == "cluster"
    assert candidate["definition_trace"]["runner_ups"][0]["definition"] == "state"
    assert "Blind mode suppressed dictionary-backed definition candidates for BLIORS." in (
        candidate["definition_trace"]["suppressed"][0]
    )
    assert "selection_reason" in candidate["definition_trace"]


def test_compose_semantic_bundle_keeps_ordered_morph_cores_in_one_word_reading() -> None:
    """Preserve ordered morph semantics instead of collapsing to one flat core.

    The decomposition-first phrase work needs the word layer to remember that a
    token can carry several morph meanings at once. This regression proves the
    bundle model preserves order while still choosing one usable lexical head
    for downstream phrase assembly.
    """

    meanings = [
        {
            "morph": "CAOS",
            "definition": "earth",
            "raw_definition": "The terrestrial ground or planet; earth.",
            "surface_gloss": "earth",
            "semantic_core_terms": ["earth", "ground"],
            "negative_contrast": ["celestial"],
            "provenance": "cluster",
            "anchor_strength": 1.0,
            "definition_trace": {"selected_quality": 0.92},
        },
        {
            "morph": "GA",
            "definition": "relation",
            "raw_definition": (
                "Bound morpheme marking relational reference in subordinate clauses."
            ),
            "surface_gloss": "relation",
            "semantic_core_terms": ["relation", "subordination"],
            "negative_contrast": ["concrete-noun"],
            "provenance": "cluster",
            "anchor_strength": 1.0,
            "definition_trace": {"selected_quality": 0.61},
        },
    ]

    bundle = compose_semantic_bundle(meanings)

    assert [entry["morph"] for entry in bundle["semantic_bundle"]] == ["CAOS", "GA"]
    assert bundle["bundle_head_gloss"] == "earth"
    assert bundle["bundle_surface_candidates"][0] == "earth"
    assert bundle["bundle_function_profile"] == "content"
    assert bundle["bundle_coherence_score"] > 0.3


def test_translate_word_blind_mode_reserves_whole_word_reads_when_bundle_is_usable() -> None:
    """Keep blind mode decomposition-first when a lexical bundle already works.

    `--no-whole-word(s)` is meant to stress-test decomposition quality. This
    regression proves that even when an attested whole-word gloss exists, the
    word result stays compositional-first as long as the decomposition yields a
    usable lexical bundle.
    """

    evidence = WordEvidence(
        word="CAFOD",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "CAFOD",
                "Shortcut governing phrase.",
                cluster_id=200,
                semantic_core=["shortcut"],
            )
        ],
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="CAFOD",
                definition="shortcut whole-word reading",
                cluster_id=200,
                root_ngram="CAFOD",
            )
        ],
    )
    repository = FakeRepository(
        evidence_by_word={"CAFOD": evidence},
        support_clusters={
            "CAF": _cluster_record(
                "CAF",
                "To govern or rule a domain.",
                cluster_id=201,
                semantic_core=["govern", "rule"],
            ),
            "OD": _cluster_record(
                "OD",
                "A conjunction meaning and.",
                cluster_id=202,
                semantic_core=["and", "conjunction"],
            ),
        },
    )
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["CAF", "OD"],
                canonicals=["CAF", "OD"],
                beam_score=0.9,
                breakdown={
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                    "segments": [],
                    "uncovered": [],
                },
                morph_support={"CAF": "cluster", "OD": "cluster"},
            )
        ]
    )

    result = service.translate_word(
        "cafod",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert result["candidates"]
    assert all(
        candidate["analysis_type"] == "compositional"
        for candidate in result["candidates"]
    )
    assert any(
        "reserved whole-word readings for CAFOD" in note
        for note in result["diagnostics"]["blind_retranslation"]["suppressed"]
    )


def test_translate_word_blind_mode_keeps_whole_word_rescue_suppressed_when_bundle_fails() -> None:
    """Keep blind mode decomposition-only even when the first bundle is weak.

    The updated blind-mode contract no longer reintroduces whole-word rescue
    once decomposition has been attempted. If the surviving compositional read
    is still weak, the word result should report that bug diagnostically rather
    than sneaking an exact whole-word reading back into the user-facing pool.
    """

    placeholder_cluster = ClusterRecord(
        variant="solo",
        cluster_id=300,
        run_id="run-1",
        ngram="ZI",
        cluster_index=0,
        glossator_def=None,
        residual_explained=None,
        residual_ratio=None,
        residual_headline="Top residuals: zi:1.00",
        residual_focus_prompt=None,
        semantic_coverage=None,
        cohesion=None,
        semantic_cohesion=None,
        best_config=None,
        residual_details=[],
        raw_definitions=[],
    )
    placeholder_cluster_lna = ClusterRecord(
        variant="solo",
        cluster_id=301,
        run_id="run-1",
        ngram="LNA",
        cluster_index=0,
        glossator_def=None,
        residual_explained=None,
        residual_ratio=None,
        residual_headline="Top residuals: lna:1.00",
        residual_focus_prompt=None,
        semantic_coverage=None,
        cohesion=None,
        semantic_cohesion=None,
        best_config=None,
        residual_details=[],
        raw_definitions=[],
    )
    evidence = WordEvidence(
        word="ZILNA",
        variants_queried=["solo"],
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="ZILNA",
                definition="within itself",
                cluster_id=302,
                root_ngram="ZILNA",
            )
        ],
    )
    repository = FakeRepository(
        evidence_by_word={"ZILNA": evidence},
        support_clusters={
            "ZI": placeholder_cluster,
            "LNA": placeholder_cluster_lna,
        },
    )
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["ZI", "LNA"],
                canonicals=["ZI", "LNA"],
                beam_score=0.8,
                breakdown={
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                    "segments": [],
                    "uncovered": [],
                },
                morph_support={"ZI": "cluster", "LNA": "cluster"},
            )
        ]
    )

    result = service.translate_word(
        "zilna",
        llm=False,
        allow_whole_word=False,
        use_beam_search=True,
    )

    assert not any(
        candidate.get("blind_mode_whole_word_rescue")
        for candidate in result["candidates"]
    )
    assert any(
        "kept whole-word readings suppressed for ZILNA" in note
        for note in result["diagnostics"]["blind_retranslation"]["suppressed"]
    )


def test_translate_word_whole_word_enabled_still_allows_whole_word_candidates() -> None:
    """Keep default translation mode whole-word friendly outside blind mode.

    The new decomposition-first behavior is only for `--no-whole-word(s)`. This
    regression proves the default mode still lets exact whole-word evidence
    compete when the user is translating novel material.
    """

    evidence = WordEvidence(
        word="CAFOD",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "CAFOD",
                "Whole-word governing phrase.",
                cluster_id=400,
                semantic_core=["governance"],
            )
        ],
        attested_definitions=[
            AttestedDefinition(
                variant="solo",
                source_word="CAFOD",
                definition="whole-word shortcut",
                cluster_id=400,
                root_ngram="CAFOD",
            )
        ],
    )
    repository = FakeRepository(
        evidence_by_word={"CAFOD": evidence},
        support_clusters={
            "CAF": _cluster_record(
                "CAF",
                "To govern or rule a domain.",
                cluster_id=401,
                semantic_core=["govern", "rule"],
            ),
            "OD": _cluster_record(
                "OD",
                "A conjunction meaning and.",
                cluster_id=402,
                semantic_core=["and", "conjunction"],
            ),
        },
    )
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["CAF", "OD"],
                canonicals=["CAF", "OD"],
                beam_score=0.9,
                breakdown={
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                    "segments": [],
                    "uncovered": [],
                },
                morph_support={"CAF": "cluster", "OD": "cluster"},
            )
        ]
    )

    result = service.translate_word(
        "cafod",
        llm=False,
        allow_whole_word=True,
        use_beam_search=True,
    )

    assert any(
        candidate["analysis_type"] == "whole_word_anchor"
        for candidate in result["candidates"]
    )


def test_phrase_translation_known_sentence_uses_clause_like_blind_rendering_in_solo(
    tmp_path: Path,
) -> None:
    """Render the known Dee sentence with compact clause English in solo mode.

    The deterministic phrase path should now sound much more like a modernized
    clause and much less like a glossary dump. This synthetic regression
    focuses on the wording quality of the clause assembler rather than the live
    database contents.
    """

    word_service = FakeWordService(
        results_by_word={
            "CAOSGA": _word_result(
                "CAOSGA",
                _word_candidate(
                    "earth",
                    analysis_type="compositional",
                    score=12.0,
                    confidence=0.88,
                    morphs=["CAOS", "GA"],
                    semantic_core=["earth", "ground"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "CAOS",
                            "surface_gloss": "earth",
                            "semantic_core_terms": ["earth", "ground"],
                            "negative_contrast": ["celestial"],
                            "provenance": "cluster",
                            "kind": "content",
                            "head_gloss": "earth",
                            "quality": 0.9,
                        },
                        {
                            "morph": "GA",
                            "surface_gloss": "that",
                            "semantic_core_terms": ["relation", "subordination"],
                            "negative_contrast": ["concrete-noun"],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "relative",
                            "head_gloss": "that",
                            "quality": 0.5,
                        },
                    ],
                    bundle_surface_gloss="earth",
                    bundle_head_gloss="earth",
                    bundle_function_profile="content",
                    bundle_coherence_score=0.92,
                ),
            ),
            "TABAORD": _word_result(
                "TABAORD",
                _word_candidate(
                    "govern",
                    analysis_type="compositional",
                    score=11.8,
                    confidence=0.86,
                    morphs=["TABA", "ORD"],
                    definitions=["govern", "governor"],
                    semantic_core=["govern", "rule"],
                    provenance="cluster",
                    surface_gloss_strategy="semantic_core_guided_definition",
                ),
            ),
            "SAANIR": _word_result(
                "SAANIR",
                _word_candidate(
                    "parts",
                    analysis_type="compositional",
                    score=10.9,
                    confidence=0.8,
                    morphs=["SAAN", "IR"],
                    semantic_core=["parts", "members"],
                    provenance="cluster",
                ),
            ),
            "OD": _word_result(
                "OD",
                _word_candidate(
                    "additive conjunction that links",
                    analysis_type="whole_word_anchor",
                    score=150.0,
                    confidence=0.92,
                    morphs=["OD"],
                    semantic_core=["and", "conjunction"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "OD",
                            "surface_gloss": "and",
                            "semantic_core_terms": ["and", "conjunction"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "conjunction",
                            "head_gloss": "and",
                            "quality": 1.0,
                        }
                    ],
                    bundle_surface_gloss="and",
                    bundle_head_gloss="and",
                    bundle_function_profile="conjunction",
                    bundle_coherence_score=1.0,
                ),
            ),
            "CHRISTEOS": _word_result(
                "CHRISTEOS",
                _word_candidate(
                    "let there be",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["CHRISTEOS"],
                    provenance="attested",
                    semantic_bundle=[
                        {
                            "morph": "CHRISTEOS",
                            "surface_gloss": "let there be",
                            "semantic_core_terms": ["let there be"],
                            "negative_contrast": [],
                            "provenance": "attested",
                            "kind": "function",
                            "function_profile": "imperative_existential",
                            "head_gloss": "let there be",
                            "quality": 1.0,
                        }
                    ],
                    bundle_surface_gloss="let there be",
                    bundle_head_gloss="let there be",
                    bundle_function_profile="imperative_existential",
                    bundle_coherence_score=1.0,
                ),
            ),
            "IRPOIL": _word_result(
                "IRPOIL",
                _word_candidate(
                    "division",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["IRPOIL"],
                    provenance="attested",
                    semantic_core=["division"],
                ),
            ),
            "TIOBL": _word_result(
                "TIOBL",
                _word_candidate(
                    "possession",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.77,
                    morphs=["TI", "OBL"],
                    provenance="cluster",
                    semantic_core=["possession", "locative relation"],
                    semantic_bundle=[
                        {
                            "morph": "TI",
                            "surface_gloss": "her",
                            "semantic_core_terms": ["possession", "locative relation"],
                            "negative_contrast": ["copular function"],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "feminine_locative_possessive",
                            "head_gloss": "her",
                            "quality": 0.9,
                        }
                    ],
                    bundle_surface_gloss="her",
                    bundle_head_gloss="her",
                    bundle_function_profile="feminine_locative_possessive",
                    bundle_coherence_score=0.88,
                ),
            ),
            "BUSDIRTILB": _word_result(
                "BUSDIRTILB",
                _word_candidate(
                    "her glory",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["BUSDIRTILB"],
                    provenance="attested",
                ),
            ),
            "NOALN": _word_result(
                "NOALN",
                _word_candidate(
                    "always",
                    analysis_type="compositional",
                    score=9.4,
                    confidence=0.76,
                    morphs=["NOA", "LN"],
                    semantic_core=["always", "continually"],
                    provenance="cluster",
                ),
            ),
            "PAID": _word_result(
                "PAID",
                _word_candidate(
                    "drunken",
                    analysis_type="compositional",
                    score=9.2,
                    confidence=0.75,
                    morphs=["PAI", "D"],
                    semantic_core=["drunken", "inebriated"],
                    provenance="cluster",
                ),
            ),
            "ORSBA": _word_result(
                "ORSBA",
                _word_candidate(
                    "vexed",
                    analysis_type="compositional",
                    score=9.1,
                    confidence=0.74,
                    morphs=["ORS", "BA"],
                    semantic_core=["vexed", "troubled"],
                    provenance="cluster",
                ),
            ),
            "DODRMNI": _word_result(
                "DODRMNI",
                _word_candidate(
                    "vexing",
                    analysis_type="compositional",
                    score=8.9,
                    confidence=0.72,
                    morphs=["DOD", "RMNI"],
                    semantic_core=["vexing", "torment"],
                    provenance="cluster",
                ),
            ),
            "ZILNA": _word_result(
                "ZILNA",
                _word_candidate(
                    "within itself",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["ZILNA"],
                    provenance="attested",
                    semantic_bundle=[
                        {
                            "morph": "ZILNA",
                            "surface_gloss": "within itself",
                            "semantic_core_terms": ["within itself"],
                            "negative_contrast": [],
                            "provenance": "attested",
                            "kind": "function",
                            "function_profile": "within_self",
                            "head_gloss": "within itself",
                            "quality": 1.0,
                        }
                    ],
                    bundle_surface_gloss="within itself",
                    bundle_head_gloss="within itself",
                    bundle_function_profile="within_self",
                    bundle_coherence_score=1.0,
                ),
            ),
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-known-solo.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "CAOSGA TABAORD SAANIR OD CHRISTEOS IRPOIL TIOBL BUSDIRTILB NOALN PAID ORSBA OD DODRMNI ZILNA",
        top_k=2,
        llm=False,
        allow_whole_word=False,
    )

    translation = result["rendered_translation"]
    assert "earth govern parts and let there be division" in translation
    assert "in her glory" in translation
    assert "within itself" in translation
    assert "additive conjunction" not in translation
    assert "parallel elements" not in translation


def test_phrase_translation_known_sentence_normalizes_debate_function_words(
    tmp_path: Path,
) -> None:
    """Use bundle/function metadata to avoid debate-mode glossary fragments.

    The debate path was especially prone to abstract nouns like `parallel
    elements` or `possession`. This regression keeps the deterministic phrase
    render pinned to concise English function words and lexical heads.
    """

    word_service = FakeWordService(
        results_by_word={
            "CAOSGA": _word_result(
                "CAOSGA",
                _word_candidate(
                    "relation",
                    analysis_type="compositional",
                    score=9.0,
                    confidence=0.78,
                    morphs=["CAOS", "GA"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "CAOS",
                            "surface_gloss": "earth",
                            "semantic_core_terms": ["earth", "ground"],
                            "negative_contrast": ["celestial"],
                            "provenance": "cluster",
                            "kind": "content",
                            "head_gloss": "earth",
                            "quality": 0.8,
                        },
                        {
                            "morph": "GA",
                            "surface_gloss": "relation",
                            "semantic_core_terms": ["relation", "subordination"],
                            "negative_contrast": ["concrete-noun"],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "relative",
                            "head_gloss": "that",
                            "quality": 0.5,
                        },
                    ],
                    bundle_surface_gloss="earth",
                    bundle_head_gloss="earth",
                    bundle_function_profile="content",
                    bundle_coherence_score=0.86,
                ),
            ),
            "OD": _word_result(
                "OD",
                _word_candidate(
                    "parallel elements",
                    analysis_type="whole_word_anchor",
                    score=150.0,
                    confidence=0.92,
                    morphs=["OD"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "OD",
                            "surface_gloss": "parallel elements",
                            "semantic_core_terms": ["and", "conjunction"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "conjunction",
                            "head_gloss": "and",
                            "quality": 1.0,
                        }
                    ],
                    bundle_surface_gloss="and",
                    bundle_head_gloss="and",
                    bundle_function_profile="conjunction",
                    bundle_coherence_score=1.0,
                ),
            ),
            "CHRISTEOS": _word_result(
                "CHRISTEOS",
                _word_candidate(
                    "let there be",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["CHRISTEOS"],
                    provenance="attested",
                ),
            ),
            "IRPOIL": _word_result(
                "IRPOIL",
                _word_candidate(
                    "division",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["IRPOIL"],
                    provenance="attested",
                ),
            ),
            "TIOBL": _word_result(
                "TIOBL",
                _word_candidate(
                    "possession",
                    analysis_type="compositional",
                    score=9.2,
                    confidence=0.75,
                    morphs=["TI", "OBL"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "TI",
                            "surface_gloss": "possession",
                            "semantic_core_terms": ["possession", "locative relation"],
                            "negative_contrast": ["copular function"],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "feminine_locative_possessive",
                            "head_gloss": "her",
                            "quality": 0.9,
                        }
                    ],
                    bundle_surface_gloss="her",
                    bundle_head_gloss="her",
                    bundle_function_profile="feminine_locative_possessive",
                    bundle_coherence_score=0.88,
                ),
            ),
            "BUSDIRTILB": _word_result(
                "BUSDIRTILB",
                _word_candidate(
                    "her glory",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["BUSDIRTILB"],
                    provenance="attested",
                ),
            ),
            "DODRMNI": _word_result(
                "DODRMNI",
                _word_candidate(
                    "vexing",
                    analysis_type="compositional",
                    score=8.7,
                    confidence=0.72,
                    morphs=["DOD", "RMNI"],
                    semantic_core=["vexing", "torment"],
                    provenance="cluster",
                ),
            ),
            "ZILNA": _word_result(
                "ZILNA",
                _word_candidate(
                    "within itself",
                    analysis_type="provisional",
                    score=41.0,
                    confidence=0.55,
                    morphs=["ZILNA"],
                    provenance="attested",
                    semantic_bundle=[
                        {
                            "morph": "ZILNA",
                            "surface_gloss": "within itself",
                            "semantic_core_terms": ["within itself"],
                            "negative_contrast": [],
                            "provenance": "attested",
                            "kind": "function",
                            "function_profile": "within_self",
                            "head_gloss": "within itself",
                            "quality": 1.0,
                        }
                    ],
                    bundle_surface_gloss="within itself",
                    bundle_head_gloss="within itself",
                    bundle_function_profile="within_self",
                    bundle_coherence_score=1.0,
                ),
            ),
        },
        dictionary={},
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-known-debate.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "CAOSGA OD CHRISTEOS IRPOIL TIOBL BUSDIRTILB OD DODRMNI ZILNA",
        top_k=2,
        llm=False,
        allow_whole_word=False,
    )

    translation = result["rendered_translation"]
    assert translation == "earth and let there be division in her glory and vexing within itself"
    assert "parallel elements" not in translation
    assert "possession" not in translation
    assert "relation" not in translation


def test_phrase_translation_uses_global_parse_scores_across_tokens(tmp_path: Path) -> None:
    """Let neighboring token roles tip the winning parse away from greedy local choice."""
    word_service = FakeWordService(
        results_by_word={
            "MIRC": _word_result(
                "MIRC",
                _word_candidate(
                    "stone",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.6,
                    morphs=["MIRC"],
                ),
                _word_candidate(
                    "to strike",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.6,
                    morphs=["MIRC"],
                ),
            ),
            "CICASB": _word_result(
                "CICASB",
                _word_candidate(
                    "enemy",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.8,
                    morphs=["CICASB"],
                ),
            ),
        },
        dictionary={
            "cicasb": {"canonical": "CICASB", "pos": "noun"},
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-memory.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase("mirc cicasb", top_k=2, llm=False)

    assert result["chosen_parse"] is not None
    chosen = result["chosen_parse"]
    assert chosen["token_choices"][0]["definition"] == "to strike"
    assert chosen["relations"][0]["relation"] == "predicate-argument"
    assert result["rendered_translation"] == "to strike enemy"
    assert result["lay_translation"] == "to strike enemy"


def test_phrase_translation_without_llm_skips_all_phrase_renderers(tmp_path: Path) -> None:
    """Honor `--no-llm` by keeping phrase translation fully deterministic.

    Phrase translation still performs algorithmic token analysis and parse
    scoring when LLM rendering is disabled. This regression proves the service
    does not call either the constrained renderer, the lay renderer, or the
    bundled renderer when `llm=False`.
    """
    word_service = FakeWordService(
        results_by_word={
            "MIRC": _word_result(
                "MIRC",
                _word_candidate(
                    "to strike",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.6,
                    morphs=["MIRC"],
                ),
            ),
            "CICASB": _word_result(
                "CICASB",
                _word_candidate(
                    "enemy",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.8,
                    morphs=["CICASB"],
                ),
            ),
        },
        dictionary={
            "cicasb": {"canonical": "CICASB", "pos": "noun"},
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-memory.sqlite3")

    def _unexpected_phrase_renderer(*args, **kwargs) -> PhraseRenderResult:
        raise AssertionError("Phrase LLM renderers should stay disabled when llm=False.")

    def _unexpected_bundle_renderer(*args, **kwargs) -> PhraseRenderBundleResult:
        raise AssertionError("Bundled phrase renderer should stay disabled when llm=False.")

    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
        llm_renderer=_unexpected_phrase_renderer,
        lay_renderer=_unexpected_phrase_renderer,
        bundle_renderer=_unexpected_bundle_renderer,
    )

    result = service.translate_phrase("mirc cicasb", top_k=2, llm=False)

    assert result["llm_enabled"] is False
    assert result["rendered_translation"] == "to strike enemy"
    assert result["lay_translation"] == "to strike enemy"
    assert result["lay_confidence"] == result["render_confidence"]
    assert (
        result["lay_reasoning"]
        == "Lay translation fell back to the algorithmic phrase skeleton."
    )
    assert result["lay_translation_mode"] is None
    assert result["footnoted_translation"] == "strike [^1] enemy [^2]"
    assert result["translation_footnotes"][0]["source_token"] == "MIRC"
    assert result["translation_footnotes"][1]["source_token"] == "CICASB"


def test_translate_word_no_llm_only_uses_local_only_embedding_loader(monkeypatch) -> None:
    """Deterministic translation should never trigger network-backed embedder loads."""

    loader_calls: list[bool] = []

    def _fake_loader(_model_name: str, *, local_files_only: bool = False):
        loader_calls.append(local_files_only)
        return None

    monkeypatch.setattr(
        translation_service_module,
        "get_sentence_transformer_if_available",
        _fake_loader,
    )
    monkeypatch.setattr(
        translation_strategies,
        "get_sentence_transformer_if_available",
        _fake_loader,
    )
    monkeypatch.setattr(
        translation_scoring,
        "get_sentence_transformer_if_available",
        _fake_loader,
    )

    evidence = WordEvidence(
        word="CAOSGO",
        variants_queried=["solo"],
        direct_clusters=[
            _cluster_record(
                "CAOSGO",
                "CAOS signifies the physical earth as a governable entity.",
                cluster_id=120,
                semantic_core=["earth"],
            )
        ],
    )
    repository = FakeRepository(evidence_by_word={"CAOSGO": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    service._decomposition_engine = FakeDecompositionEngine([])

    result = service.translate_word(
        "caosgo",
        llm=False,
        allow_whole_word=True,
        use_beam_search=True,
    )

    assert result["candidates"]
    assert loader_calls
    assert all(loader_calls)


def test_phrase_translation_reuses_repeated_token_results_and_prewarms_evidence(
    tmp_path: Path,
) -> None:
    """Phrase translation should avoid redoing identical token work inside one run."""

    class _CountingWordService(FakeWordService):
        def __init__(self) -> None:
            super().__init__(
                results_by_word={
                    "MIRC": _word_result(
                        "MIRC",
                        _word_candidate(
                            "watchman",
                            analysis_type="dictionary_exact",
                            score=10.0,
                            confidence=0.9,
                            morphs=["MIRC"],
                        ),
                    ),
                },
                dictionary={"mirc": {"canonical": "MIRC", "pos": "noun"}},
            )
            self.translate_calls: list[str] = []
            self.prewarm_calls: list[set[str]] = []

            class _Repository:
                variants = ["solo"]

                def __init__(repo_self, outer: "_CountingWordService") -> None:
                    repo_self.outer = outer

                def prewarm_translation_morphs(
                    repo_self,
                    morphs,
                    *,
                    variants=None,
                    include_clusters: bool = True,
                    include_residuals: bool = True,
                ) -> None:
                    repo_self.outer.prewarm_calls.append(
                        {str(morph).upper() for morph in morphs}
                    )

            self.repository = _Repository(self)
            self._substring_candidates = lambda word, include_singletons=False: {
                word.upper()
            }
            self.EvidenceMode = SingleWordTranslationService.EvidenceMode

        def translate_word(self, word: str, **kwargs: object) -> dict[str, object]:
            self.translate_calls.append(word.upper())
            return super().translate_word(word, **kwargs)

    word_service = _CountingWordService()
    memory = TranslationMemoryRepository(tmp_path / "phrase-prewarm.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase("mirc mirc", top_k=2, llm=False)

    assert result["rendered_translation"] == "watchman watchman"
    assert word_service.translate_calls == ["MIRC"]
    assert len(word_service.prewarm_calls) == 1
    assert "MIRC" in word_service.prewarm_calls[0]


def test_phrase_translation_memory_accumulates_repeated_provisional_reads(tmp_path: Path) -> None:
    """Persist cautious unknown-word evidence across repeated phrase runs."""
    word_service = FakeWordService(
        results_by_word={
            "GITHULCAG": _word_result(
                "GITHULCAG",
                _word_candidate(
                    "storm-born",
                    analysis_type="provisional",
                    score=5.0,
                    confidence=0.4,
                    morphs=["GITHULCAG"],
                    definitions=["storm-born", "wind-made"],
                ),
            ),
            "MIRC": _word_result(
                "MIRC",
                _word_candidate(
                    "watchman",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.9,
                    morphs=["MIRC"],
                ),
            ),
        },
        dictionary={
            "githulcag": {"canonical": "GITHULCAG", "pos": "adjective"},
            "mirc": {"canonical": "MIRC", "pos": "noun"},
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "translation-memory.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    first = service.translate_phrase("githulcag mirc", top_k=2, llm=False)
    second = service.translate_phrase("githulcag mirc", top_k=2, llm=False)
    entry = memory.fetch_entry("GITHULCAG")

    assert first["memory_updates"][0]["evidence_count"] == 1
    assert second["memory_updates"][0]["evidence_count"] == 2
    assert entry is not None
    assert entry.best_gloss == "storm-born"
    assert entry.evidence_count == 2
    assert "githulcag mirc" in entry.examples


def test_phrase_render_prompt_forbids_unsupported_semantics() -> None:
    """Keep the phrase renderer pinned to the chosen parse payload."""
    prompt = _build_phrase_render_prompt(
        {
            "phrase": "ol sonf vorsg",
            "translation_skeleton": "I reign among",
            "token_choices": [],
            "relations": [],
            "score": 1.0,
        },
        {"llm_context": "Use only the supplied parse."},
    )

    assert "Use ONLY the supplied token choices and relations." in prompt
    assert "Do not add, remove, or replace meanings." in prompt
    assert "Do not infer extra syntax" in prompt


def test_phrase_lay_render_prompt_requires_plain_english_without_new_claims() -> None:
    """Ground the lay translation prompt in the chosen parse instead of freewheeling."""
    prompt = _build_phrase_lay_render_prompt(
        {
            "phrase": "ol sonf vorsg",
            "translation_skeleton": "I reign among",
            "token_choices": [],
            "relations": [],
            "score": 1.0,
        },
        {"llm_context": "Use only the supplied parse."},
    )

    assert "lay reader" in prompt
    assert "Keep the same core meaning" in prompt
    assert "Do not add any new actors, actions, objects, or claims." in prompt


def test_translation_cli_progress_renderer_shows_elapsed_and_honest_token_eta() -> None:
    """Render elapsed time and token ETA for deterministic phrase analysis.

    Token analysis is one of the few stages where we genuinely know how much
    work remains. This regression keeps the CLI renderer honest by showing an
    elapsed timer plus a moving average and ETA only for that deterministic
    loop.
    """

    class _Buffer:
        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, value: str) -> None:
            self.parts.append(value)

        def flush(self) -> None:
            return None

        def isatty(self) -> bool:
            return False

        def getvalue(self) -> str:
            return "".join(self.parts)

    timeline = iter([0.0, 0.0, 5.0, 14.0, 26.0])
    buffer = _Buffer()
    renderer = translation_cli.TranslationCLIProgressRenderer(
        stream=buffer,
        clock=lambda: next(timeline),
    )

    renderer.stage(
        "Starting variant solo (1/2)...",
        stage_id="variant_start",
        variant="solo",
        variant_index=1,
        variant_total=2,
    )
    renderer.stage(
        "Analyzing token 1/3: ILS",
        stage_id="token_analysis",
        variant="solo",
        variant_index=1,
        variant_total=2,
        current=1,
        total=3,
    )
    renderer.stage(
        "Analyzing token 2/3: MICAOLZ",
        stage_id="token_analysis",
        current=2,
        total=3,
    )
    renderer.llm_status(
        {
            "label": "Rendering phrase translations and footnotes",
            "stage_id": "render_bundle",
            "state": "waiting for first token",
            "source": "remote",
            "attempt": 1,
            "max_attempts": 5,
            "elapsed_seconds": 12.0,
        }
    )

    rendered = buffer.getvalue()

    assert "Starting variant solo (1/2)... | solo 1/2 | elapsed 00:00" in rendered
    assert "Analyzing token 1/3: ILS | solo 1/2 | elapsed 00:00 | avg 0.0s/token | eta 00:00" in rendered
    assert "Analyzing token 2/3: MICAOLZ | solo 1/2 | elapsed 00:09 | avg 4.5s/token | eta 00:04" in rendered
    assert (
        "Rendering phrase translations and footnotes | waiting for first token | "
        "solo 1/2 | remote attempt 1/5 | elapsed 00:12"
    ) in rendered


def test_parse_phrase_lay_render_response_accepts_structured_footnotes() -> None:
    """Preserve structured lay footnotes when the renderer returns valid JSON.

    The final phrase report now depends on a token-aligned footnote payload in
    addition to the plain lay translation. This regression keeps the parser from
    discarding valid structured explanations when the LLM follows the new
    schema.
    """

    parse_payload = {
        "phrase": "ol sonf vorsg",
        "translation_skeleton": "I reign above",
        "token_choices": [
            {
                "token": "OL",
                "definition": "I",
                "raw_definition": "I",
                "analysis_type": "dictionary_exact",
                "role_hint": "noun",
            },
            {
                "token": "SONF",
                "definition": "rule",
                "raw_definition": "rule",
                "analysis_type": "compositional",
                "role_hint": "verb",
            },
            {
                "token": "VORSG",
                "definition": "above",
                "raw_definition": "above",
                "analysis_type": "compositional",
                "role_hint": "relational",
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    parsed = _parse_phrase_lay_render_response(
        json.dumps(
            {
                "rendered_translation": "I rule above",
                "footnoted_translation": "I [^1] rule [^2] above [^3]",
                "translation_footnotes": [
                    {
                        "index": 1,
                        "source_token": "OL",
                        "rendered_text": "I",
                        "explanation": "OL contributes the first-person subject.",
                    },
                    {
                        "index": 2,
                        "source_token": "SONF",
                        "rendered_text": "rule",
                        "explanation": "SONF supplies the governing action.",
                    },
                    {
                        "index": 3,
                        "source_token": "VORSG",
                        "rendered_text": "above",
                        "explanation": "VORSG keeps the upward relational sense.",
                    },
                ],
                "confidence": 0.81,
                "reasoning": "Simplified the phrase while keeping the same parse.",
            }
        ),
        fallback="I reign above",
        parse_payload=parse_payload,
    )

    assert parsed["rendered_translation"] == "I rule above"
    assert parsed["footnoted_translation"] == "I [^1] rule [^2] above [^3]"
    assert parsed["translation_footnotes"][1]["rendered_text"] == "rule"
    assert parsed["translation_footnotes"][2]["explanation"] == (
        "VORSG keeps the upward relational sense."
    )


def test_parse_phrase_lay_render_response_rebuilds_missing_footnotes() -> None:
    """Fall back to deterministic token notes when structured footnotes are absent.

    The lay renderer may return valid JSON without the new footnote fields. The
    CLI still needs a stable closing section, so the parser must reconstruct the
    token-aligned output directly from the chosen parse in that case.
    """

    parse_payload = {
        "phrase": "mirc cicasb",
        "translation_skeleton": "to strike enemy",
        "token_choices": [
            {
                "token": "MIRC",
                "definition": "to strike",
                "raw_definition": "to strike",
                "analysis_type": "compositional",
                "role_hint": "verb",
            },
            {
                "token": "CICASB",
                "definition": "enemy",
                "raw_definition": "enemy",
                "analysis_type": "dictionary_exact",
                "role_hint": "noun",
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    parsed = _parse_phrase_lay_render_response(
        json.dumps(
            {
                "rendered_translation": "hit the enemy",
                "confidence": 0.82,
                "reasoning": "Smoothed the phrase for a lay reader.",
            }
        ),
        fallback="to strike enemy",
        parse_payload=parse_payload,
    )

    assert parsed["rendered_translation"] == "hit the enemy"
    assert parsed["footnoted_translation"] == "strike [^1] enemy [^2]"
    assert parsed["translation_footnotes"][0]["source_token"] == "MIRC"
    assert "chosen parse" not in parsed["translation_footnotes"][0]["explanation"].lower()


def test_parse_phrase_lay_render_response_compacts_overlong_lay_chunks() -> None:
    """Fall back to the deterministic skeleton for glossary-like lay dumps.

    The lay renderer should return actual sentence-level prose. When remote
    output drifts into a comma-heavy mini-glossary, the parser should keep the
    grounded footnotes but prefer the deterministic skeleton as the visible lay
    sentence instead of collapsing everything to token salad.
    """

    parse_payload = {
        "phrase": "gamph caf",
        "translation_skeleton": "that which is not to rule",
        "token_choices": [
            {
                "token": "GAMPH",
                "definition": "that which is not",
                "raw_definition": "that which is not",
                "analysis_type": "provisional",
                "role_hint": "modifier",
                "alternates": [],
            },
            {
                "token": "CAF",
                "definition": "to rule, as a law that rules the holy ones",
                "raw_definition": "to rule, as a law that rules the holy ones",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    parsed = _parse_phrase_lay_render_response(
        json.dumps(
            {
                "rendered_translation": (
                    "That which is not, including absence, to rule, as a law "
                    "that rules the holy ones"
                ),
                "footnoted_translation": (
                    "That which is not, including absence [^1] to rule, as a law "
                    "that rules the holy ones [^2]"
                ),
                "translation_footnotes": [
                    {
                        "index": 1,
                        "source_token": "GAMPH",
                        "rendered_text": "That which is not, including absence",
                        "explanation": "Keeps the negating sense.",
                    },
                    {
                        "index": 2,
                        "source_token": "CAF",
                        "rendered_text": "to rule, as a law that rules the holy ones",
                        "explanation": "Keeps the governing sense.",
                    },
                ],
                "confidence": 0.67,
                "reasoning": "Compressed the phrase for a lay reader.",
            }
        ),
        fallback="that which is not to rule",
        parse_payload=parse_payload,
    )

    assert parsed["rendered_translation"] == "that which is not to rule"
    assert parsed["footnoted_translation"] == "not [^1] rule [^2]"
    assert parsed["translation_footnotes"][0]["rendered_text"] == "not"
    assert parsed["translation_footnotes"][1]["rendered_text"] == "rule"


def test_phrase_report_includes_technical_and_lay_translations() -> None:
    """Show both translation styles clearly in CLI text output."""
    report = translation_cli._format_phrase_report(
        {
            "phrase": "ol sonf vorsg",
            "variant": "solo",
            "strategy": "prefer-balance",
            "llm_enabled": False,
            "lay_translation_mode": "remote",
            "rendered_translation": "I reign among",
            "render_confidence": 0.72,
            "render_reasoning": "Algorithmic phrase rendering only.",
            "lay_translation": "I rule in the middle of them",
            "lay_confidence": 0.88,
            "lay_reasoning": "Simplified the phrasing for a modern reader.",
            "interpretive_translation": "I hold the center of their order.",
            "interpretive_confidence": 0.81,
            "interpretive_reasoning": "Pushes the line into a more expressive idiom.",
        }
    )

    assert "Technical translation: I reign among" in report
    assert "Lay translation: I rule in the middle of them" in report
    assert "Poetic translation: I hold the center of their order." in report
    assert "Constrained render enabled: False" in report
    assert "Lay translation mode: remote" in report


def test_phrase_report_appends_markdown_style_footnotes_last() -> None:
    """Print the new footnoted closing translation after all summary sections.

    The final phrase report should still show the existing technical and lay
    translations, but now end with a token-aligned markdown-style explanation
    block so readers can inspect how each source token contributed to the final
    wording.
    """

    report = translation_cli._format_phrase_report(
        {
            "phrase": "ol sonf vorsg",
            "variant": "solo",
            "strategy": "prefer-balance",
            "llm_enabled": False,
            "rendered_translation": "I reign above",
            "render_confidence": 0.72,
            "render_reasoning": "Algorithmic phrase rendering only.",
            "lay_translation": "I rule above",
            "lay_confidence": 0.88,
            "lay_reasoning": "Simplified the phrasing for a modern reader.",
            "footnoted_translation": "I [^1] rule [^2] above [^3]",
            "translation_footnotes": [
                {
                    "index": 1,
                    "source_token": "OL",
                    "rendered_text": "I",
                    "explanation": "OL supplies the first-person subject.",
                },
                {
                    "index": 2,
                    "source_token": "SONF",
                    "rendered_text": "rule",
                    "explanation": "SONF contributes the governing action.",
                },
                {
                    "index": 3,
                    "source_token": "VORSG",
                    "rendered_text": "above",
                    "explanation": "VORSG keeps the upward relational reading.",
                },
            ],
        }
    )

    assert "Final translation: I [^1] rule [^2] above [^3]" in report
    assert '[^1]: "OL" rendered as "I". OL supplies the first-person subject.' in report
    assert report.rfind("Final translation:") > report.rfind("Lay reasoning:")
    assert report.strip().endswith(
        '[^3]: "VORSG" rendered as "above". VORSG keeps the upward relational reading.'
    )


def test_phrase_report_verbose_includes_definition_trace_details() -> None:
    """Verbose phrase output should expose token-level evidence traces."""

    report = translation_cli._format_phrase_report(
        {
            "phrase": "bliors caosgo",
            "variant": "solo",
            "strategy": "prefer-balance",
            "llm_enabled": False,
            "token_analyses": [
                {
                    "token": "bliors",
                    "candidates": [
                        {
                            "rank": 1,
                            "analysis_type": "compositional",
                            "role_hint": "noun",
                            "definition": "comfort",
                            "chosen_in_parse": True,
                            "definition_trace": {
                                "surface_gloss": "comfort",
                                "surface_gloss_strategy": "semantic_core",
                                "raw_selected_definition": (
                                    "BLI encodes the state or provision of comfort as solace."
                                ),
                                "selected_semantic_core": ["comfort", "solace"],
                                "selected_negative_contrast": ["darkness"],
                                "selected_source": "cluster",
                                "runner_ups": [
                                    {"definition": "state", "source": "attested", "quality": 0.20}
                                ],
                                "suppressed": [
                                    "Blind mode suppressed dictionary-backed definition candidates for BLIORS."
                                ],
                                "negative_contrast_penalties": ["darkness"],
                                "meta_linguistic_rejections": [
                                    "A root denoting comfort as solace."
                                ],
                                "selection_reason": (
                                    "Selected cluster-backed evidence survived for this token. "
                                    "Candidate score 12.00 beat runner-up 8.00."
                                ),
                            },
                        }
                    ],
                }
            ],
        },
        verbose=True,
    )

    assert "definition=comfort [chosen parse]" in report
    assert "Selected source: cluster" in report
    assert "Surface gloss: comfort (semantic_core)" in report
    assert "Raw selected definition:" in report
    normalized_report = " ".join(report.split())
    assert "BLI encodes the state or provision of comfort as solace." in normalized_report
    assert "Semantic core: comfort; solace" in report
    assert "Negative contrast: darkness" in report
    assert "Runner-ups: state (attested, q=0.20)" in report
    assert "Suppressed: Blind mode suppressed dictionary-backed definition" in report
    assert "Negative-contrast penalties: darkness" in report
    assert "Meta-linguistic rejections: A root denoting comfort as solace." in report
    assert "BLIORS" in report
    assert "Why this won: Selected cluster-backed evidence survived for this token." in report


def test_phrase_report_verbose_includes_bundle_and_blind_rescue_details() -> None:
    """Show bundle selection and blind rescue notes in verbose phrase output."""

    report = translation_cli._format_phrase_report(
        {
            "phrase": "zilna",
            "variant": "solo",
            "strategy": "prefer-balance",
            "llm_enabled": False,
            "token_analyses": [
                {
                    "token": "zilna",
                    "candidates": [
                        {
                            "rank": 1,
                            "analysis_type": "provisional",
                            "role_hint": "relational",
                            "definition": "within itself",
                            "chosen_in_parse": True,
                            "definition_trace": {
                                "surface_gloss": "within itself",
                                "surface_gloss_strategy": "whole_word",
                                "raw_selected_definition": "within itself",
                                "selected_semantic_core": [],
                                "selected_negative_contrast": [],
                                "selected_source": "attested",
                                "bundle_surface_candidates": ["within itself"],
                                "bundle_selection_reason": (
                                    "Ordered bundle normalized 1 morph reading(s) into the function gloss "
                                    "\"within itself\"."
                                ),
                                "blind_mode_rescue_note": (
                                    "Blind mode whole-word rescue activated because no usable "
                                    "decomposition-level lexical bundle survived."
                                ),
                                "runner_ups": [],
                                "suppressed": [],
                                "negative_contrast_penalties": [],
                                "meta_linguistic_rejections": [],
                                "selection_reason": "Selected attested-backed evidence survived for this token.",
                            },
                        }
                    ],
                }
            ],
        },
        verbose=True,
    )

    assert "Bundle surface candidates: within itself" in report
    assert "Bundle selection: Ordered bundle normalized 1 morph reading(s)" in report
    assert "Blind-mode rescue: Blind mode whole-word rescue activated" in report


def test_translate_phrase_cli_emits_progress_updates_to_stderr(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Expose deterministic phrase-stage progress lines during CLI translation.

    Phrase translation can take noticeable time even without LLM rendering
    because token analysis and parse search still run. This regression proves
    the CLI now tells the user where it is in that longer workflow.
    """

    word_service = FakeWordService(
        results_by_word={
            "MIRC": _word_result(
                "MIRC",
                _word_candidate(
                    "to strike",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.7,
                    morphs=["MIRC"],
                ),
            ),
            "CICASB": _word_result(
                "CICASB",
                _word_candidate(
                    "enemy",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.8,
                    morphs=["CICASB"],
                ),
            ),
        },
        dictionary={
            "mirc": {"canonical": "MIRC", "pos": "verb"},
            "cicasb": {"canonical": "CICASB", "pos": "noun"},
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-progress.sqlite3")

    def _unexpected_phrase_renderer(*args, **kwargs) -> PhraseRenderResult:
        raise AssertionError("Phrase renderers should not run when --no-llm is used.")

    def _unexpected_bundle_renderer(*args, **kwargs) -> PhraseRenderBundleResult:
        raise AssertionError("Bundled phrase renderer should not run when --no-llm is used.")

    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
        llm_renderer=_unexpected_phrase_renderer,
        lay_renderer=_unexpected_phrase_renderer,
        bundle_renderer=_unexpected_bundle_renderer,
    )

    monkeypatch.setattr(
        translation_cli.PhraseTranslationService,
        "from_config",
        classmethod(lambda cls, **_kwargs: service),
    )
    monkeypatch.setattr(translation_cli, "_missing_db_paths", lambda _variants: [])
    monkeypatch.setattr(translation_cli, "_configure_llm_env", lambda _mode: None)

    args = translation_cli.build_parser().parse_args(
        ["translate-phrase", "mirc cicasb", "--no-llm"]
    )

    exit_code = translation_cli.translate_phrase_from_args(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Preparing phrase translation..." in captured.err
    assert "Analyzing token 1/2: MIRC" in captured.err
    assert "Building and scoring parse candidates..." in captured.err
    assert "Rendering lay translation and footnotes..." not in captured.err
    assert "Done." in captured.err


def test_translate_phrase_cli_reports_variant_boundaries_once(
    capsys,
    monkeypatch,
) -> None:
    """Explain the built-in solo/debate loop instead of looking like a rerun.

    `translate-phrase` defaults to running both insight variants. The CLI should
    announce those boundaries explicitly and emit a single final completion line
    so the second pass does not look like the command restarted from scratch.
    """

    class _FakePhraseService:
        def __enter__(self) -> "_FakePhraseService":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def translate_phrase(self, phrase: str, **kwargs) -> dict[str, object]:
            reporter = kwargs.get("progress_reporter")
            if reporter is not None:
                reporter.stage(f"Preparing phrase translation for {phrase.upper()}...")
            return {
                "phrase": phrase.upper(),
                "variant": kwargs["variants"][0],
                "strategy": kwargs["strategy"],
                "llm_enabled": kwargs["llm"],
                "rendered_translation": "technical",
                "render_confidence": 0.5,
                "render_reasoning": "stub",
                "lay_translation": "plain",
                "lay_confidence": 0.5,
                "lay_reasoning": "stub",
                "chosen_parse": {"score": 1.0},
                "translation_footnotes": [],
                "footnoted_translation": "",
            }

    monkeypatch.setattr(
        translation_cli.PhraseTranslationService,
        "from_config",
        classmethod(lambda cls, **_kwargs: _FakePhraseService()),
    )
    monkeypatch.setattr(translation_cli, "_missing_db_paths", lambda _variants: [])
    monkeypatch.setattr(translation_cli, "_configure_llm_env", lambda _mode: None)

    args = translation_cli.build_parser().parse_args(
        ["translate-phrase", "ol sonf", "--no-llm"]
    )

    exit_code = translation_cli.translate_phrase_from_args(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Running 2 translation variants: solo, debate" in captured.err
    assert "Starting variant solo (1/2)..." in captured.err
    assert "Completed variant solo (1/2)." in captured.err
    assert "Starting variant debate (2/2)..." in captured.err
    assert "Completed variant debate (2/2)." in captured.err
    assert captured.err.count("Done.") == 1


def test_phrase_renderers_report_wait_and_validation_steps(monkeypatch) -> None:
    """Expose extra detail during the long technical and lay render phases.

    The phrase service now surfaces coarse stage boundaries, but the slow part
    is often the remote render itself. These regressions prove the renderers now
    announce the wait-for-model and validation sub-steps that happen inside
    those longer phases.
    """

    class _FakeTool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _run(self, *args, **kwargs) -> dict[str, str]:
            return {
                "response_text": json.dumps(
                    {
                        "rendered_translation": "holy rule",
                        "footnoted_translation": "holy [^1] rule [^2]",
                        "translation_footnotes": [
                            {
                                "index": 1,
                                "source_token": "MAD",
                                "rendered_text": "holy",
                                "explanation": "Sacred sense.",
                            },
                            {
                                "index": 2,
                                "source_token": "CAF",
                                "rendered_text": "rule",
                                "explanation": "Governing sense.",
                            },
                        ],
                        "confidence": 0.74,
                        "reasoning": "Compact everyday phrasing.",
                    }
                )
            }

    class _StageReporter:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def stage(self, message: str) -> None:
            self.messages.append(message)

    monkeypatch.setattr("translation.llm_synthesis.QueryModelTool", _FakeTool)
    parse_payload = {
        "phrase": "mad caf",
        "translation_skeleton": "holy rule",
        "token_choices": [
            {
                "token": "MAD",
                "definition": "holy",
                "raw_definition": "holy",
                "analysis_type": "dictionary_exact",
                "role_hint": "modifier",
                "alternates": [],
            },
            {
                "token": "CAF",
                "definition": "to rule",
                "raw_definition": "to rule",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    technical_reporter = _StageReporter()
    render_phrase_translation(
        parse_payload,
        {
            "use_remote": True,
            "progress_reporter": technical_reporter,
            "progress_label": "Rendering technical translation",
        },
    )

    lay_reporter = _StageReporter()
    render_phrase_lay_translation(
        parse_payload,
        {
            "use_remote": True,
            "progress_reporter": lay_reporter,
            "progress_label": "Rendering lay translation and footnotes",
        },
    )

    assert technical_reporter.messages == [
        "Rendering technical translation... waiting for remote model response",
        "Rendering technical translation... validating renderer response",
    ]
    assert lay_reporter.messages == [
        "Rendering lay translation and footnotes... waiting for remote model response",
        "Rendering lay translation and footnotes... validating renderer response",
    ]


def test_render_phrase_bundle_returns_both_outputs_from_one_model_call(monkeypatch) -> None:
    """Bundle technical and lay phrase rendering into one remote request.

    The phrase CLI used to spend two model calls per variant just to get the
    technical and lay views of the same chosen parse. This regression keeps the
    new bundled renderer honest by proving it can return both outputs, plus the
    final footnote block, from a single tool invocation.
    """

    calls: list[dict[str, object]] = []
    init_kwargs: list[dict[str, object]] = []

    class _FakeTool:
        def __init__(self, *args, **kwargs) -> None:
            init_kwargs.append(dict(kwargs))

        def attach_logging(self, db, run_id) -> None:
            calls.append({"db": db, "run_id": run_id, "attached": True})

        def _run(self, *args, **kwargs) -> dict[str, str]:
            calls.append({"progress_callback": kwargs.get("progress_callback")})
            return {
                "response_text": json.dumps(
                    {
                        "lay_translation": "holy rule endures",
                        "lay_confidence": 0.72,
                        "lay_reasoning": "Compact everyday phrasing.",
                        "poetic_translation": "the sacred law holds firm",
                        "poetic_confidence": 0.8,
                        "poetic_reasoning": "More expressive paraphrase.",
                        "footnoted_translation": "holy [^1] rule [^2] endures [^3]",
                        "translation_footnotes": [
                            {
                                "index": 1,
                                "source_token": "MAD",
                                "rendered_text": "holy",
                                "explanation": "Sacred sense.",
                            },
                            {
                                "index": 2,
                                "source_token": "CAF",
                                "rendered_text": "rule",
                                "explanation": "Governing sense.",
                            },
                            {
                                "index": 3,
                                "source_token": "PRAC",
                                "rendered_text": "endures",
                                "explanation": "Abiding sense.",
                            },
                        ],
                    }
                )
            }

    monkeypatch.setattr("translation.llm_synthesis.QueryModelTool", _FakeTool)
    parse_payload = {
        "phrase": "mad caf prac",
        "translation_skeleton": "holy rule abides",
        "token_choices": [
            {
                "token": "MAD",
                "definition": "holy",
                "raw_definition": "holy",
                "analysis_type": "dictionary_exact",
                "role_hint": "modifier",
                "alternates": [],
            },
            {
                "token": "CAF",
                "definition": "to rule",
                "raw_definition": "to rule",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
            {
                "token": "PRAC",
                "definition": "to abide",
                "raw_definition": "to abide",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    bundled = render_phrase_bundle(
        parse_payload,
        {
            "use_remote": True,
            "progress_label": "Rendering phrase translations and footnotes",
            "llm_query_db": object(),
            "llm_query_run_id": "run-123",
        },
    )

    assert bundled.technical_translation == "holy rule abides"
    assert bundled.technical_reasoning == "Technical translation kept deterministic from the chosen parse."
    assert bundled.lay_translation == "holy rule endures"
    assert bundled.poetic_translation == "the sacred law holds firm"
    assert bundled.interpretive_translation == "the sacred law holds firm"
    assert bundled.footnoted_translation == "holy [^1] rule [^2] endures [^3]"
    assert len(calls) == 2
    assert calls[0]["attached"] is True
    assert callable(calls[1]["progress_callback"])
    assert init_kwargs[0]["remote_attempts"] == 2
    assert init_kwargs[0]["read_timeout_seconds"] == 45.0
    assert init_kwargs[0]["local_fallback_enabled"] is False
    assert init_kwargs[0]["stream_response"] is False


def test_parse_phrase_bundle_response_allows_poetic_copy_of_technical() -> None:
    """Allow the poetic line to stay close when the sentence is already coherent.

    The redesigned poetic track is free to diverge, but it is not required to
    invent a second phrasing when the grounded or technical line already reads
    well. The parser should therefore preserve a valid poetic line even when it
    matches the technical sentence exactly.
    """

    parse_payload = {
        "phrase": "mad caf prac",
        "translation_skeleton": "holy rule abides",
        "token_choices": [
            {
                "token": "MAD",
                "definition": "holy",
                "raw_definition": "holy",
                "analysis_type": "dictionary_exact",
                "role_hint": "modifier",
                "alternates": [],
            },
            {
                "token": "CAF",
                "definition": "to rule",
                "raw_definition": "to rule",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
            {
                "token": "PRAC",
                "definition": "to abide",
                "raw_definition": "to abide",
                "analysis_type": "compositional",
                "role_hint": "verb",
                "alternates": [],
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    parsed = _parse_phrase_bundle_response(
        json.dumps(
            {
                "lay_translation": "the holy law still stands",
                "lay_confidence": 0.74,
                "lay_reasoning": "Keeps a grounded English reading.",
                "interpretive_translation": "holy rule abides",
                "interpretive_confidence": 0.71,
                "interpretive_reasoning": "Copied the technical skeleton.",
                "footnoted_translation": "holy [^1] law [^2] still stands [^3]",
                "translation_footnotes": [
                    {
                        "index": 1,
                        "source_token": "MAD",
                        "rendered_text": "holy",
                        "explanation": "Sacred quality.",
                    },
                    {
                        "index": 2,
                        "source_token": "CAF",
                        "rendered_text": "law",
                        "explanation": "Grounded governing sense.",
                    },
                    {
                        "index": 3,
                        "source_token": "PRAC",
                        "rendered_text": "still stands",
                        "explanation": "Abiding sense in plain English.",
                    },
                ],
            }
        ),
        fallback="holy rule abides",
        parse_payload=parse_payload,
    )

    assert parsed["lay_translation"] == "the holy law still stands"
    assert parsed["poetic_translation"] == "holy rule abides"
    assert parsed["interpretive_translation"] == "holy rule abides"


def test_parse_phrase_bundle_response_replaces_raw_token_placeholders() -> None:
    """Strip raw `[TOKEN]` placeholders back to grounded glosses.

    Even if the remote renderer misbehaves and echoes source tokens in bracket
    form, the phrase parser should replace them with the best grounded glosses
    already present in the parse payload before the CLI prints the line.
    """

    parse_payload = {
        "phrase": "vansax naz",
        "translation_skeleton": "circle of stars and rectangular prism",
        "token_choices": [
            {
                "token": "VANSAX",
                "definition": "power",
                "dictionary_rescue_gloss": "circle of stars",
                "alternates": ["circle of stars"],
                "analysis_type": "compositional",
                "role_hint": "noun",
            },
            {
                "token": "NAZ",
                "definition": "shape",
                "dictionary_rescue_gloss": "rectangular prism",
                "alternates": ["rectangular prism"],
                "analysis_type": "compositional",
                "role_hint": "noun",
            },
        ],
        "relations": [],
        "score": 1.0,
    }

    parsed = _parse_phrase_bundle_response(
        json.dumps(
            {
                "lay_translation": "In [VANSAX] and in [NAZ].",
                "lay_confidence": 0.8,
                "lay_reasoning": "Remote renderer left raw tokens in place.",
                "interpretive_translation": "Within [VANSAX] and [NAZ], the image abides.",
                "interpretive_confidence": 0.77,
                "interpretive_reasoning": "Remote renderer left raw tokens in place.",
                "footnoted_translation": "circle of stars [^1] rectangular prism [^2]",
                "translation_footnotes": [
                    {
                        "index": 1,
                        "source_token": "VANSAX",
                        "rendered_text": "circle of stars",
                        "explanation": "Exact dictionary rescue keeps the stellar sense.",
                    },
                    {
                        "index": 2,
                        "source_token": "NAZ",
                        "rendered_text": "rectangular prism",
                        "explanation": "Exact dictionary rescue keeps the geometric sense.",
                    },
                ],
            }
        ),
        fallback="circle of stars and rectangular prism",
        parse_payload=parse_payload,
    )

    assert "[VANSAX]" not in parsed["lay_translation"]
    assert "[NAZ]" not in parsed["lay_translation"]
    assert "circle of stars" in parsed["lay_translation"]
    assert "rectangular prism" in parsed["lay_translation"]
    assert "[VANSAX]" not in parsed["interpretive_translation"]
    assert "[NAZ]" not in parsed["interpretive_translation"]


def test_translate_phrase_uses_bundle_renderer_once_when_llm_enabled(
    tmp_path: Path,
) -> None:
    """Use one bundled render call instead of separate technical and lay calls.

    The new bundle renderer is the main API-call reduction for `translate-phrase`
    when `--llm` is enabled. This regression proves the phrase service consumes
    the bundled result directly and does not fall back to the legacy two-call
    render path in that mode.
    """

    word_service = FakeWordService(
        results_by_word={
            "MAD": _word_result(
                "MAD",
                _word_candidate(
                    "holy",
                    analysis_type="dictionary_exact",
                    score=10.0,
                    confidence=0.9,
                    morphs=["MAD"],
                ),
            ),
            "CAF": _word_result(
                "CAF",
                _word_candidate(
                    "to rule",
                    analysis_type="compositional",
                    score=9.7,
                    confidence=0.82,
                    morphs=["CAF"],
                ),
            ),
        },
        dictionary={
            "mad": {"canonical": "MAD", "pos": "adj"},
            "caf": {"canonical": "CAF", "pos": "verb"},
        },
    )
    word_service.llm_enabled = True
    memory = TranslationMemoryRepository(tmp_path / "phrase-bundle.sqlite3")
    bundle_calls: list[dict[str, object]] = []

    def _fail_renderer(*args, **kwargs) -> PhraseRenderResult:
        raise AssertionError("legacy renderer should not be used when bundle renderer is active")

    def _fake_bundle_renderer(
        parse_payload: dict[str, object],
        context: dict[str, object],
    ) -> PhraseRenderBundleResult:
        bundle_calls.append({"payload": parse_payload, "context": context})
        return PhraseRenderBundleResult(
            technical_translation="holy rule",
            technical_confidence=0.8,
            technical_reasoning="Close render.",
            lay_translation="holy rule",
            lay_confidence=0.78,
            lay_reasoning="Compact render.",
            poetic_translation="the sacred order stands firm",
            poetic_confidence=0.83,
            poetic_reasoning="Expressive render.",
            interpretive_translation="the sacred order stands firm",
            interpretive_confidence=0.83,
            interpretive_reasoning="Expressive render.",
            footnoted_translation="holy [^1] rule [^2]",
            translation_footnotes=[
                {
                    "index": 1,
                    "source_token": "MAD",
                    "rendered_text": "holy",
                    "explanation": "Sacred sense.",
                },
                {
                    "index": 2,
                    "source_token": "CAF",
                    "rendered_text": "rule",
                    "explanation": "Governing sense.",
                },
            ],
        )

    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
        llm_renderer=_fail_renderer,
        lay_renderer=_fail_renderer,
        bundle_renderer=_fake_bundle_renderer,
    )

    result = service.translate_phrase(
        "mad caf",
        variants=["solo"],
        top_k=2,
        llm=True,
    )

    assert len(bundle_calls) == 1
    assert result["rendered_translation"] == "holy rule"
    assert result["lay_translation"] == "holy rule"
    assert result["poetic_translation"] == "the sacred order stands firm"
    assert result["interpretive_translation"] == "the sacred order stands firm"
    assert result["footnoted_translation"] == "holy [^1] rule [^2]"


def test_phrase_bundle_prompt_requests_grounded_and_poetic_renders() -> None:
    """Spell out the grounded-plus-poetic contract in the bundle prompt."""

    prompt = _build_phrase_bundle_prompt(
        {
            "phrase": "mad caf prac",
            "translation_skeleton": "holy rule abides",
            "token_choices": [],
            "relations": [],
            "score": 1.0,
        },
        {},
    )

    lowered = prompt.lower()
    assert "poetic_translation" in prompt
    assert "lay_translation` should not simply copy the technical skeleton" in prompt
    assert "never emit raw source tokens or bracket placeholders" in lowered
    assert "do not emit `unresolved term`" in lowered
    assert "repair awkward fragment chains" in lowered
    assert "reorder aggressively" in lowered
    assert "does not need to differ" in lowered


def test_translate_word_demotes_residual_placeholder_anchor_below_composition() -> None:
    """Prefer grounded full-cover decomposition over residual-only whole-word text.

    NACRO and ASCLAD both showed that raw residual headlines can surface as
    high-scoring whole-word anchors even when the word already has complete
    compositional analyses. This regression targets the shared merge/enrichment
    path that now demotes those anchors before final ranking and confidence
    reporting.
    """

    evidence = WordEvidence(
        word="NACRO",
        variants_queried=["solo"],
        residual_semantics=[
            _residual_record("NACRO", "Top residuals: nacro:1.00", parent_word="NACRO")
        ],
    )
    repository = FakeRepository(evidence_by_word={"NACRO": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    placeholder_anchor = service._build_single_morph_candidate(
        "NACRO",
        definition="Top residuals: nacro:1.00",
        definitions=["Top residuals: nacro:1.00"],
        provenance="residual",
        score=150.0,
        analysis_type="whole_word_anchor",
        warnings=[],
    )
    compositional_candidate = {
        "rank": 1,
        "analysis_type": "compositional",
        "morphs": ["NA", "CRO"],
        "canonicals": ["NA", "CRO"],
        "score": 1.5,
        "breakdown": {
            "segments": [
                {"start": 0, "end": 2, "ngram": "NA", "canonical": "NA"},
                {"start": 2, "end": 5, "ngram": "CRO", "canonical": "CRO"},
            ],
            "uncovered": [],
            "coverage_ratio": 1.0,
            "residual_ratio": 0.0,
        },
        "score_breakdown": None,
        "meanings": [
            {
                "morph": "NA",
                "canonical": "NA",
                "definition": "divine essence",
                "definitions": ["divine essence"],
                "provenance": "cluster",
                "anchor_strength": 1.0,
            },
            {
                "morph": "CRO",
                "canonical": "CRO",
                "definition": "origin",
                "definitions": ["origin"],
                "provenance": "cluster",
                "anchor_strength": 1.0,
            },
        ],
        "warnings": [],
    }

    merged = service._merge_candidate_pool(
        [placeholder_anchor],
        [compositional_candidate],
        top_k=3,
    )
    enriched = service._enrich_candidates(
        merged,
        evidence=evidence,
        strategy="prefer-balance",
        llm_enabled=False,
        llm_context=None,
    )

    assert merged[0]["analysis_type"] == "compositional"
    assert merged[0]["morphs"] == ["NA", "CRO"]
    placeholder = next(
        candidate
        for candidate in enriched
        if candidate["analysis_type"] == "whole_word_anchor"
    )
    assert float(placeholder["score"]) < float(merged[0]["score"])
    assert float(placeholder["confidence"]) <= 0.2
    assert any(
        "Residual-only whole-word placeholder anchor demoted" in warning
        for warning in placeholder.get("warnings", [])
    )


def test_translate_word_demotes_opaque_placeholder_anchor_below_composition() -> None:
    """Treat bracket placeholders like `[ASCLAD]` as unresolved, not preferred.

    Solo analysis for ASCLAD was still ranking an opaque whole-word placeholder
    above grounded compositional readings because the old demotion logic only
    recognized residual headlines. This regression keeps unresolved bracket
    anchors from winning once a full-cover compositional candidate exists.
    """

    evidence = WordEvidence(word="ASCLAD", variants_queried=["solo"])
    repository = FakeRepository(evidence_by_word={"ASCLAD": evidence})
    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    placeholder_anchor = service._build_single_morph_candidate(
        "ASCLAD",
        definition="[ASCLAD]",
        definitions=["[ASCLAD]"],
        provenance="cluster",
        score=150.0,
        analysis_type="whole_word_anchor",
        warnings=[],
    )
    compositional = {
        "word": "ASCLAD",
        "rank": 1,
        "analysis_type": "compositional",
        "morphs": ["AS", "CL", "AD"],
        "meanings": [
            {
                "morph": "AS",
                "definition": "living spirit",
                "definitions": ["living spirit"],
                "provenance": "cluster",
            },
            {
                "morph": "CL",
                "definition": "word",
                "definitions": ["word"],
                "provenance": "dictionary",
            },
            {
                "morph": "AD",
                "definition": "before",
                "definitions": ["before"],
                "provenance": "cluster",
            },
        ],
        "score": 91.0,
        "warnings": [],
        "breakdown": {
            "segments": [
                {"start": 0, "end": 2, "ngram": "AS", "canonical": "AS"},
                {"start": 2, "end": 4, "ngram": "CL", "canonical": "CL"},
                {"start": 4, "end": 6, "ngram": "AD", "canonical": "AD"},
            ],
            "uncovered": [],
            "coverage_ratio": 1.0,
            "residual_ratio": 0.0,
        },
    }

    merged = service._merge_candidate_pool(
        [placeholder_anchor],
        [compositional],
        [],
        top_k=5,
    )
    enriched = service._enrich_candidates(
        merged,
        evidence=evidence,
        strategy="prefer-balance",
        llm_enabled=False,
        llm_context=None,
    )

    placeholder = next(
        candidate
        for candidate in enriched
        if candidate["analysis_type"] == "whole_word_anchor"
    )
    assert enriched[0]["analysis_type"] == "compositional"
    assert float(placeholder["score"]) < float(merged[0]["score"])
    assert float(placeholder["confidence"]) <= 0.25
    assert any(
        "Opaque whole-word placeholder anchor demoted" in warning
        for warning in placeholder.get("warnings", [])
    )


def test_phrase_translation_hides_placeholder_leftovers_when_no_clean_gloss_exists(
    tmp_path: Path,
) -> None:
    """Treat placeholder-only phrase output as a bug instead of faking a gloss."""

    word_service = FakeWordService(
        results_by_word={
            "NACRO": _word_result(
                "NACRO",
                _word_candidate(
                    "Top residuals: nacro:1.00",
                    analysis_type="whole_word_anchor",
                    score=10.0,
                    confidence=0.92,
                    morphs=["NACRO"],
                    warnings=["Raw residual placeholder surfaced."],
                ),
            ),
        },
        dictionary={"nacro": {"canonical": "NACRO", "pos": "noun"}},
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-placeholder.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    with pytest.raises(
        RuntimeError,
        match="reached clause rendering without a grounded gloss",
    ):
        service.translate_phrase("nacro", top_k=1, llm=False)


def test_phrase_translation_raises_for_unresolved_blind_mode_token(
    tmp_path: Path,
) -> None:
    """Raise a blind-mode bug when decomposition still cannot resolve a token."""

    word_service = FakeWordService(
        results_by_word={
            "TIOBL": _word_result(
                "TIOBL",
                _word_candidate(
                    "Top residuals: tiobl:1.00",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.42,
                    morphs=["TI", "OBL"],
                    warnings=["Raw residual placeholder surfaced."],
                    definition_trace={
                        "selected_definition": None,
                        "raw_selected_definition": "Top residuals: tiobl:1.00",
                        "selected_source": "cluster",
                        "selected_quality": 0.2,
                        "selected_semantic_core": [],
                        "selected_negative_contrast": [],
                        "surface_gloss": None,
                        "surface_gloss_strategy": "unresolved",
                        "runner_ups": [],
                        "suppressed": [],
                        "blind_dictionary_fallback": False,
                        "negative_contrast_penalties": [],
                        "meta_linguistic_rejections": [],
                    },
                ),
            ),
        },
        dictionary={
            "tiobl": {
                "canonical": "TIOBL",
                "definition": "(within) her",
                "senses": [{"definition": "(within) her"}],
                "pos": "pronoun",
            }
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-dictionary-rescue.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    with pytest.raises(
        RuntimeError,
        match="could not resolve token TIOBL through decomposition",
    ):
        service.translate_phrase(
            "tiobl",
            top_k=1,
            llm=False,
            allow_whole_word=False,
        )


def test_phrase_translation_uses_stronger_lexical_gloss_for_weak_function_bundle_in_blind_mode(
    tmp_path: Path,
) -> None:
    """Prefer the lexical meaning over a generic bundle-profile shorthand."""

    word_service = FakeWordService(
        results_by_word={
            "ZIRDO": _word_result(
                "ZIRDO",
                _word_candidate(
                    "I am",
                    analysis_type="compositional",
                    score=9.5,
                    confidence=0.56,
                    morphs=["ZIR", "DO"],
                    semantic_core=["being", "existence"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "ZIR",
                            "surface_gloss": None,
                            "semantic_core_terms": ["being", "existence", "identity"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "locative",
                            "head_gloss": "in",
                            "quality": 0.2,
                        }
                    ],
                    bundle_surface_gloss=None,
                    bundle_head_gloss="in",
                    bundle_function_profile="locative",
                    bundle_coherence_score=0.22,
                ),
            ),
        },
        dictionary={
            "zirdo": {
                "canonical": "ZIRDO",
                "definition": "I am",
                "senses": [{"definition": "I am"}],
                "pos": "verb",
            }
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-zirdo-rescue.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "zirdo",
        top_k=1,
        llm=False,
        allow_whole_word=False,
    )

    assert result["rendered_translation"] == "I am"
    assert result["lay_translation"] == "I am"
    assert result["translation_footnotes"][0]["rendered_text"] == "I am"
    assert result["chosen_parse"]["token_choices"][0]["dictionary_rescue_gloss"] is None


def test_phrase_translation_uses_specific_lexical_gloss_when_semantic_core_beats_generic_gloss(
    tmp_path: Path,
) -> None:
    """Prefer a specific lexical gloss over a generic abstraction in blind mode."""

    word_service = FakeWordService(
        results_by_word={
            "PIRIPSOL": _word_result(
                "PIRIPSOL",
                _word_candidate(
                    "heavens",
                    analysis_type="compositional",
                    score=9.4,
                    confidence=0.58,
                    morphs=["PIRI", "PSOL"],
                    semantic_core=["heavens", "celestial tiers", "stratification"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "PIRI",
                            "surface_gloss": "heavens",
                            "semantic_core_terms": [
                                "heavens",
                                "celestial tiers",
                                "stratification",
                            ],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "content",
                            "head_gloss": "heavens",
                            "quality": 0.9,
                        }
                    ],
                    bundle_surface_gloss="heavens",
                    bundle_head_gloss="heavens",
                    bundle_function_profile="content",
                    bundle_coherence_score=0.91,
                ),
            ),
        },
        dictionary={
            "piripsol": {
                "canonical": "PIRIPSOL",
                "definition": "heavens",
                "senses": [{"definition": "heavens"}],
                "pos": "noun",
            }
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-piripsol-rescue.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "piripsol",
        top_k=1,
        llm=False,
        allow_whole_word=False,
    )

    assert result["rendered_translation"] == "heavens"
    assert result["lay_translation"] == "heavens"
    assert result["translation_footnotes"][0]["rendered_text"] == "heavens"
    assert result["chosen_parse"]["token_choices"][0]["dictionary_rescue_gloss"] is None


def test_phrase_translation_prefers_definition_when_bundle_head_is_misleading(
    tmp_path: Path,
) -> None:
    """Prefer the candidate's lexical definition over a misleading bundle head."""

    word_service = FakeWordService(
        results_by_word={
            "VANSAX": _word_result(
                "VANSAX",
                _word_candidate(
                    "the circle of stars",
                    analysis_type="compositional",
                    score=9.8,
                    confidence=0.72,
                    morphs=["VA", "NS", "AX"],
                    semantic_core=["circle", "stars", "celestial ring"],
                    provenance="cluster",
                    semantic_bundle=[
                        {
                            "morph": "VA",
                            "surface_gloss": None,
                            "semantic_core_terms": ["circle", "stars", "celestial ring"],
                            "negative_contrast": ["non-descriptive", "non-ordinary noun"],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "locative",
                            "head_gloss": "angel",
                            "quality": 0.98,
                        },
                        {
                            "morph": "NS",
                            "surface_gloss": "holy pentagram",
                            "semantic_core_terms": ["pentagram", "holy symbol", "divine emblem"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "locative",
                            "head_gloss": "holy pentagram",
                            "quality": 1.04,
                        },
                        {
                            "morph": "AX",
                            "surface_gloss": "surround",
                            "semantic_core_terms": ["surround", "encircle", "enclose"],
                            "negative_contrast": [
                                "non-penetrative",
                                "non-dispersive",
                                "non-internal",
                            ],
                            "provenance": "cluster",
                            "kind": "function",
                            "function_profile": "locative",
                            "head_gloss": "surround",
                            "quality": 0.85,
                        },
                    ],
                    bundle_surface_gloss=None,
                    bundle_head_gloss="angel",
                    bundle_function_profile="locative",
                    bundle_coherence_score=1.0,
                    definition_trace={
                        "selected_definition": "the circle of stars",
                        "raw_selected_definition": "the circle of stars",
                        "selected_source": "cluster",
                        "selected_quality": 0.98,
                        "selected_semantic_core": ["circle", "stars", "celestial ring"],
                        "selected_negative_contrast": [],
                        "surface_gloss": None,
                        "surface_gloss_strategy": "cleaned_definition",
                        "runner_ups": [],
                        "suppressed": [],
                        "blind_dictionary_fallback": False,
                        "negative_contrast_penalties": [],
                        "meta_linguistic_rejections": [],
                        "bundle_selection_reason": (
                            'Ordered bundle normalized 3 morph reading(s) into the '
                            'function gloss "angel".'
                        ),
                    },
                ),
            ),
        },
        dictionary={
            "vansax": {
                "canonical": "VANSAX",
                "definition": "the circle of stars",
                "senses": [{"definition": "the circle of stars"}],
                "pos": "noun",
            }
        },
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-vansax-rescue.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "vansax",
        top_k=1,
        llm=False,
        allow_whole_word=False,
    )

    assert result["rendered_translation"] == "circle of stars"
    assert result["lay_translation"] == "circle of stars"
    assert result["translation_footnotes"][0]["rendered_text"] == "circle of stars"
    assert result["chosen_parse"]["token_choices"][0]["dictionary_rescue_gloss"] is None


def test_phrase_translation_uses_semantic_bundle_fallback_before_unresolved_term(
    tmp_path: Path,
) -> None:
    """Build a weak gloss from surviving bundle evidence before giving up.

    Some blind decompositions do not have an exact dictionary rescue, but they
    still preserve enough semantic-bundle detail to say something better than
    `unresolved term`. This keeps the phrase layer grounded in surviving morph
    evidence even when no single clean head gloss survives.
    """

    word_service = FakeWordService(
        results_by_word={
            "QVX": _word_result(
                "QVX",
                _word_candidate(
                    None,
                    analysis_type="compositional",
                    score=7.2,
                    confidence=0.44,
                    morphs=["QV", "X"],
                    meanings=[
                        {
                            "morph": "QV",
                            "canonical": "QV",
                            "definition": None,
                            "definitions": [],
                            "provenance": "cluster",
                            "anchor_strength": 0.7,
                            "semantic_core": ["throne"],
                            "semantic_core_terms": ["throne"],
                            "negative_contrast": [],
                            "surface_gloss": None,
                            "surface_gloss_strategy": "semantic_core",
                            "definition_trace": {
                                "selected_definition": None,
                                "raw_selected_definition": None,
                                "selected_source": "cluster",
                                "selected_quality": 0.7,
                                "selected_semantic_core": ["throne"],
                                "selected_negative_contrast": [],
                                "surface_gloss": None,
                                "surface_gloss_strategy": "semantic_core",
                                "runner_ups": [],
                                "suppressed": [],
                                "blind_dictionary_fallback": False,
                                "negative_contrast_penalties": [],
                                "meta_linguistic_rejections": [],
                            },
                        },
                        {
                            "morph": "X",
                            "canonical": "X",
                            "definition": None,
                            "definitions": [],
                            "provenance": "cluster",
                            "anchor_strength": 0.66,
                            "semantic_core": ["flame"],
                            "semantic_core_terms": ["flame"],
                            "negative_contrast": [],
                            "surface_gloss": None,
                            "surface_gloss_strategy": "semantic_core",
                            "definition_trace": {
                                "selected_definition": None,
                                "raw_selected_definition": None,
                                "selected_source": "cluster",
                                "selected_quality": 0.66,
                                "selected_semantic_core": ["flame"],
                                "selected_negative_contrast": [],
                                "surface_gloss": None,
                                "surface_gloss_strategy": "semantic_core",
                                "runner_ups": [],
                                "suppressed": [],
                                "blind_dictionary_fallback": False,
                                "negative_contrast_penalties": [],
                                "meta_linguistic_rejections": [],
                            },
                        },
                    ],
                    semantic_bundle=[
                        {
                            "morph": "QV",
                            "surface_gloss": "throne",
                            "semantic_core_terms": ["throne"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "content",
                            "head_gloss": "throne",
                            "quality": 0.7,
                        },
                        {
                            "morph": "X",
                            "surface_gloss": "flame",
                            "semantic_core_terms": ["flame"],
                            "negative_contrast": [],
                            "provenance": "cluster",
                            "kind": "content",
                            "head_gloss": "flame",
                            "quality": 0.66,
                        },
                    ],
                    bundle_surface_gloss=None,
                    bundle_head_gloss=None,
                    bundle_function_profile="content",
                    bundle_coherence_score=0.31,
                ),
            ),
        },
        dictionary={},
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-semantic-bundle-fallback.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "qvx",
        top_k=1,
        llm=False,
        allow_whole_word=False,
    )

    assert result["rendered_translation"] == "throne flame"
    assert result["lay_translation"] == "throne flame"
    assert result["translation_footnotes"][0]["rendered_text"] == "throne flame"


def test_phrase_translation_no_whole_word_inherits_blind_dictionary_suppression(
    tmp_path: Path,
) -> None:
    """Phrase translation should inherit blind-mode whole-word suppression from word analysis.

    The phrase pipeline delegates token analysis to the single-word service, so
    the broadened `--no-whole-word(s)` behavior must remove dictionary-backed
    exact reads there as well. This regression proves a dictionary-heavy token
    like `MICAOLZ` no longer surfaces its exact match inside phrase analysis.
    """

    evidence_by_word = {
        "MICAOLZ": WordEvidence(
            word="MICAOLZ",
            variants_queried=["solo"],
            dictionary_morphs={
                "MICAOLZ": DictionaryMorph(
                    morph="MICAOLZ",
                    definition="mighty",
                    senses=["mighty"],
                    part_of_speech="adjective",
                )
            },
        ),
        "ELO": WordEvidence(
            word="ELO",
            variants_queried=["solo"],
            attested_definitions=[
                AttestedDefinition(
                    variant="solo",
                    source_word="ELO",
                    definition="shining being",
                    cluster_id=8,
                    root_ngram="EL",
                )
            ],
        ),
    }
    repository = FakeRepository(
        evidence_by_word=evidence_by_word,
        support_clusters={
            "MI": _cluster_record("MI", "force", cluster_id=40),
            "CA": _cluster_record("CA", "center", cluster_id=41),
            "OLZ": _cluster_record("OLZ", "brightness", cluster_id=42),
        },
    )
    word_service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=repository,
        llm_enabled=False,
    )
    word_service._decomposition_engine = FakeDecompositionEngine(
        [
            Decomposition(
                morphs=["MI", "CA", "OLZ"],
                canonicals=["MI", "CA", "OLZ"],
                beam_score=8.2,
                breakdown={
                    "segments": [
                        {"start": 0, "end": 2, "ngram": "MI", "canonical": "MI"},
                        {"start": 2, "end": 4, "ngram": "CA", "canonical": "CA"},
                        {"start": 4, "end": 7, "ngram": "OLZ", "canonical": "OLZ"},
                    ],
                    "uncovered": [],
                    "coverage_ratio": 1.0,
                    "residual_ratio": 0.0,
                },
                morph_support={"MI": "cluster", "CA": "cluster", "OLZ": "cluster"},
            )
        ]
    )
    memory = TranslationMemoryRepository(tmp_path / "phrase-blind.sqlite3")
    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
    )

    result = service.translate_phrase(
        "micaolz elo",
        top_k=2,
        llm=False,
        allow_whole_word=False,
    )

    micaolz = next(
        token_payload
        for token_payload in result["token_analyses"]
        if str(token_payload["token"]).upper() == "MICAOLZ"
    )
    assert all(
        candidate["analysis_type"] != "dictionary_exact"
        for candidate in micaolz["word_result"]["candidates"]
    )
    assert micaolz["candidates"][0]["definition"] != "mighty"


def test_translation_cli_registers_translate_phrase_command() -> None:
    """Expose the new phrase-translation command through the public CLI parser."""
    parser = translation_cli.build_parser()
    args = parser.parse_args(["translate-phrase", "ol sonf vorsg"])

    assert args.command == "translate-phrase"
    assert args.phrase == "ol sonf vorsg"
    assert args.handler == translation_cli._run_translate_phrase


def test_translation_cli_accepts_plural_no_whole_words_alias_for_phrase() -> None:
    """Support the more natural plural alias for the phrase-level whole-word flag."""
    parser = translation_cli.build_parser()
    args = parser.parse_args(
        ["translate-phrase", "ol sonf vorsg", "--no-whole-words"]
    )

    assert args.allow_whole_word is False


def test_translation_cli_accepts_plural_no_whole_words_alias_for_word() -> None:
    """Keep the single-word parser aligned with the phrase parser's flag alias."""
    parser = translation_cli.build_parser()
    args = parser.parse_args(["translate-word", "OL", "--no-whole-words"])

    assert args.allow_whole_word is False


def test_translation_cli_help_describes_blind_retranslation_semantics() -> None:
    """Document the broadened flag semantics for both word and phrase workflows.

    Once `--no-whole-word(s)` grows into the blind-retranslation switch, the
    CLI help text needs to spell out the new behavior so users understand why
    exact dictionary matches and long whole-word anchors disappear.
    """

    parser = translation_cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    word_help = subparsers.choices["translate-word"].format_help()
    phrase_help = subparsers.choices["translate-phrase"].format_help()

    assert "Blind retranslation mode" in word_help
    assert "suppress exact dictionary" in word_help
    assert "Short 1-2 character DB roots" in word_help
    assert "Blind retranslation mode" in phrase_help
    assert "whole-word DB anchors longer than 2" in phrase_help


def test_candidate_finder_segment_target_uses_precomputed_definition_counts_without_reclustering(
    tmp_path: Path,
) -> None:
    """Trust precomputed clustered counts instead of reclustering inside beam search.

    Phrase translation already computes clustered definition counts upstream.
    This regression keeps `segment_target` from re-running semantic clustering
    for the same gloss buckets once those counts are available.
    """

    ngram_db = tmp_path / "ngrams.sqlite3"
    conn = sqlite3.connect(ngram_db)
    conn.execute("CREATE TABLE ngrams (ngram TEXT, total_occurrences INTEGER)")
    conn.execute("CREATE TABLE ngram_membership (ngram TEXT, canonical TEXT)")
    conn.commit()
    conn.close()

    finder = root_candidate_finder_module.MorphemeCandidateFinder(
        ngram_db_path=ngram_db,
        fasttext_model_path=tmp_path / "dummy.fasttext",
        dictionary_entries=[],
        min_n=1,
        max_n=2,
        beam_width=5,
    )

    original_cluster_definitions = root_candidate_finder_module.cluster_definitions
    root_candidate_finder_module.cluster_definitions = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[assignment]
        AssertionError("segment_target should reuse provided definition counts")
    )
    try:
        parses = finder.segment_target(
            "AB",
            extra_ngrams={
                "a": [("A", 1, 1)],
                "b": [("B", 1, 1)],
            },
            restrict_to_attested=True,
            min_n=1,
            definition_counts={"A": 1, "B": 1},
            definition_glosses={
                "A": [("alpha", 0.9)],
                "B": [("beta", 0.9)],
            },
        )
    finally:
        root_candidate_finder_module.cluster_definitions = original_cluster_definitions  # type: ignore[assignment]
        finder.close()

    assert parses
    assert parses[0][0]


def test_single_word_service_caches_clustered_definition_counts_between_calls() -> None:
    """Reuse semantic cluster counts for identical gloss buckets across tokens.

    The phrase layer revisits the same morph gloss families many times. This
    cache regression keeps semantic deduplication from reclustering identical
    gloss buckets every time a token requests them.
    """

    service = SingleWordTranslationService(
        candidate_finder=FakeCandidateFinder(),
        repository=FakeRepository(evidence_by_word={}),
        llm_enabled=False,
    )
    calls: list[dict[str, list[tuple[str, float | None]]]] = []
    original_cluster_counts = translation_service_module.cluster_definition_counts

    def _fake_cluster_counts(glosses, embedder, **_kwargs):
        calls.append(copy.deepcopy(glosses))
        return {str(morph).upper(): 1 for morph in glosses}

    translation_service_module.cluster_definition_counts = _fake_cluster_counts
    try:
        first = service._cluster_definition_counts_cached(
            {"CAOS": [("earth", 0.9), ("ground", 0.8)]},
            embedder=object(),
        )
        second = service._cluster_definition_counts_cached(
            {"CAOS": [("earth", 0.9), ("ground", 0.8)]},
            embedder=object(),
        )
    finally:
        translation_service_module.cluster_definition_counts = original_cluster_counts
        service.close()

    assert first == {"CAOS": 1}
    assert second == {"CAOS": 1}
    assert len(calls) == 1


def test_repository_prewarm_translation_morphs_batches_phrase_cache_warming(
    tmp_path: Path,
) -> None:
    """Warm phrase caches in bulk so later token fetches stay query-free.

    This regression protects the new phrase-prewarm path from sliding back to
    one-query-per-morph behavior. The exact query count is less important than
    the two guarantees we care about: the warm step stays compact, and later
    fetches are true cache hits.
    """

    db_path = tmp_path / "insights.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE clusters (
            cluster_id INTEGER,
            run_id TEXT,
            ngram TEXT,
            cluster_index INTEGER,
            glossator_def TEXT,
            residual_explained REAL,
            residual_ratio REAL,
            residual_headline TEXT,
            residual_focus_prompt TEXT,
            semantic_coverage REAL,
            cohesion REAL,
            semantic_cohesion REAL,
            best_config TEXT,
            action TEXT,
            verdict TEXT
        );
        CREATE TABLE residual_details (
            residual_id INTEGER,
            cluster_id INTEGER,
            normalized TEXT,
            definition TEXT,
            coverage_ratio REAL,
            residual_ratio REAL,
            avg_confidence REAL,
            uncovered_json TEXT,
            low_conf_json TEXT
        );
        CREATE TABLE raw_defs (
            def_id INTEGER,
            cluster_id INTEGER,
            source_word TEXT,
            variant TEXT,
            definition TEXT,
            enhanced_def TEXT,
            fasttext REAL,
            similarity REAL,
            tier TEXT
        );
        CREATE TABLE root_residual_semantics (
            run_id TEXT,
            residual TEXT,
            parent_word TEXT,
            group_idx INTEGER,
            group_size INTEGER,
            glossator_def TEXT,
            glossator_prompt TEXT,
            residual_headline TEXT,
            residual_focus_prompt TEXT,
            semantic_coverage REAL,
            cohesion REAL,
            semantic_cohesion REAL,
            residual_explained REAL,
            residual_ratio REAL,
            derivational_validity REAL,
            rebuttal_resilience REAL,
            created_at TEXT
        );
        CREATE TABLE morph_hypotheses (
            hyp_id INTEGER,
            morph TEXT,
            source_word TEXT,
            anchor TEXT,
            seed_glosses TEXT,
            proposed_gloss TEXT,
            rationale TEXT,
            delta_cosine REAL,
            residual_before REAL,
            residual_after REAL,
            created_at TEXT,
            accepted INTEGER
        );
        """
    )
    accepted_payload = json.dumps(
        {
            "DEFINITION": "earth",
            "SEMANTIC_CORE": ["earth", "ground"],
            "REJECTED": False,
        }
    )
    residual_payload = json.dumps(
        {
            "DEFINITION": "earth-related",
            "SEMANTIC_CORE": ["earth"],
            "REJECTED": False,
        }
    )
    for offset, morph in enumerate(("CA", "OS"), start=1):
        conn.execute(
            """INSERT INTO clusters
               (cluster_id, run_id, ngram, cluster_index, glossator_def, residual_explained,
                residual_ratio, residual_headline, residual_focus_prompt, semantic_coverage,
                cohesion, semantic_cohesion, best_config, action, verdict)
               VALUES (?, 'run-1', ?, 0, ?, NULL, NULL, NULL, NULL, 0.9, 0.8, 0.8,
                       NULL, 'escalate', 'True')""",
            (offset, morph, accepted_payload),
        )
        conn.execute(
            """INSERT INTO root_residual_semantics
               (run_id, residual, parent_word, group_idx, group_size, glossator_def,
                glossator_prompt, residual_headline, residual_focus_prompt, semantic_coverage,
                cohesion, semantic_cohesion, residual_explained, residual_ratio,
                derivational_validity, rebuttal_resilience, created_at)
               VALUES ('run-1', ?, 'CAOS', 0, 1, ?, NULL, NULL, NULL, 0.7, 0.6, 0.6,
                       NULL, NULL, 0.5, 0.5, NULL)""",
            (morph, residual_payload),
        )
        conn.execute(
            """INSERT INTO morph_hypotheses
               (hyp_id, morph, source_word, anchor, seed_glosses, proposed_gloss,
                rationale, delta_cosine, residual_before, residual_after, created_at, accepted)
               VALUES (?, ?, 'CAOS', NULL, '[]', 'earth', NULL, 0.8, NULL, NULL, NULL, 1)""",
            (offset, morph),
        )
    conn.commit()
    conn.close()

    repository = InsightsRepository(solo_path=db_path, debate_path=None)
    statements: list[str] = []
    repository._connections["solo"].set_trace_callback(statements.append)

    repository.prewarm_translation_morphs(["CA", "OS"], variants=["solo"])

    warm_query_count = len(statements)
    statements.clear()
    repository.fetch_morph_support(["CA", "OS"], variants=["solo"])
    repository.fetch_rejected_morphs(["CA", "OS"], variants=["solo"])
    repository.fetch_accepted_definition_counts(["CA", "OS"], variants=["solo"])
    repository.fetch_accepted_definition_glosses(["CA", "OS"], variants=["solo"])
    cache_hit_query_count = len(statements)
    repository.close()

    assert warm_query_count <= 10
    assert cache_hit_query_count == 0
