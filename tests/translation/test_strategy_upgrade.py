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

import copy
import json
import sqlite3
import sys
import types
from pathlib import Path


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
from translation.decomposition import Decomposition
from translation.llm_synthesis import (
    PhraseRenderResult,
    _build_phrase_lay_render_prompt,
    _build_phrase_render_prompt,
    _parse_phrase_lay_render_response,
    render_phrase_lay_translation,
    render_phrase_translation,
)
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


def _cluster_record(morph: str, definition: str, *, cluster_id: int = 1) -> ClusterRecord:
    """Build a compact accepted cluster record for a target morph."""
    return ClusterRecord(
        variant="solo",
        cluster_id=cluster_id,
        run_id="run-1",
        ngram=morph,
        cluster_index=0,
        glossator_def=json.dumps({"DEFINITION": definition, "REJECTED": False}),
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
) -> dict[str, object]:
    """Build a translation candidate payload in the service's public schema."""
    surface = morphs or ["TOKEN"]
    normalized_definitions = []
    if definition is not None:
        normalized_definitions.append(definition)
    if definitions:
        normalized_definitions.extend(definitions)
    unique_definitions = list(dict.fromkeys(normalized_definitions))
    return {
        "rank": 1,
        "analysis_type": analysis_type,
        "morphs": list(surface),
        "score": score,
        "confidence": confidence,
        "meanings": [
            {
                "morph": surface[0],
                "canonical": surface[0],
                "definition": definition,
                "definitions": unique_definitions,
                "provenance": analysis_type,
                "anchor_strength": 1.0,
            }
        ],
        "warnings": list(warnings or []),
    }


def _word_result(word: str, *candidates: dict[str, object]) -> dict[str, object]:
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
        "fallback_morphs": [],
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


def test_phrase_translation_always_provides_lay_translation(tmp_path: Path) -> None:
    """Keep a plain-English phrase paraphrase available even without the strict render pass.

    The phrase service now exposes two parallel outputs: a technical translation
    that stays close to the selected glosses, and an always-on lay translation
    that is meant for human readers. This regression ensures the lay path still
    runs when the stricter `--llm` render is disabled.
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

    def _unexpected_constrained_renderer(
        _parse_payload: dict[str, object],
        _context: dict[str, object],
    ) -> PhraseRenderResult:
        raise AssertionError("The constrained phrase renderer should stay disabled.")

    def _fake_lay_renderer(
        parse_payload: dict[str, object],
        context: dict[str, object],
    ) -> PhraseRenderResult:
        assert parse_payload["translation_skeleton"] == "to strike enemy"
        assert context["phrase"] == "mirc cicasb"
        return PhraseRenderResult(
            rendered_translation="hit the enemy",
            confidence=0.84,
            reasoning="Simplified the technical gloss into plain English.",
        )

    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
        llm_renderer=_unexpected_constrained_renderer,
        lay_renderer=_fake_lay_renderer,
    )

    result = service.translate_phrase("mirc cicasb", top_k=2, llm=False)

    assert result["llm_enabled"] is False
    assert result["rendered_translation"] == "to strike enemy"
    assert result["lay_translation"] == "hit the enemy"
    assert result["lay_confidence"] == 0.84
    assert result["lay_reasoning"] == "Simplified the technical gloss into plain English."


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
    """Collapse glossary-like lay chunks back into short token-level concepts.

    The lay renderer should return short, sane phrasing, but remote runs can
    drift into comma-heavy mini-definitions. This regression keeps the parser
    from passing those long chunks straight through to the final footnoted
    translation block.
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

    assert parsed["rendered_translation"] == "not rule"
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
        }
    )

    assert "Technical translation: I reign among" in report
    assert "Lay translation: I rule in the middle of them" in report
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


def test_translate_phrase_cli_emits_progress_updates_to_stderr(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Expose deterministic phrase-stage progress lines during CLI translation.

    Phrase translation can take noticeable time even without the strict render
    pass because token analysis, parse search, and the always-on lay renderer
    still run. This regression proves the CLI now tells the user where it is in
    that longer workflow.
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

    def _fake_lay_renderer(
        _parse_payload: dict[str, object],
        _context: dict[str, object],
    ) -> PhraseRenderResult:
        return PhraseRenderResult(
            rendered_translation="hit the enemy",
            confidence=0.84,
            reasoning="Simplified the phrase for a modern reader.",
            footnoted_translation="hit [^1] the enemy [^2]",
            translation_footnotes=[
                {
                    "index": 1,
                    "source_token": "MIRC",
                    "rendered_text": "hit",
                    "explanation": "MIRC contributes the striking action.",
                },
                {
                    "index": 2,
                    "source_token": "CICASB",
                    "rendered_text": "the enemy",
                    "explanation": "CICASB names the object of that action.",
                },
            ],
        )

    service = PhraseTranslationService(
        word_service=word_service,
        memory_repository=memory,
        lay_renderer=_fake_lay_renderer,
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
    assert "Rendering lay translation and footnotes..." in captured.err
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
    """Render unresolved tokens cleanly instead of echoing residual dump text.

    Even after ranking demotion, some phrases will still contain tokens whose
    only surviving evidence is a placeholder residual headline. The phrase layer
    should mark those tokens as unresolved in human-facing output while keeping
    the raw single-word diagnostics intact.
    """

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

    result = service.translate_phrase("nacro", top_k=1, llm=False)

    assert result["rendered_translation"] == "[NACRO]"
    assert result["lay_translation"] == "[NACRO]"
    assert result["footnoted_translation"] == "[NACRO] [^1]"
    assert result["translation_footnotes"][0]["source_token"] == "NACRO"
    assert "Top residuals:" not in result["footnoted_translation"]
    assert result["token_analyses"][0]["word_result"]["candidates"][0]["meanings"][0]["definition"] == (
        "Top residuals: nacro:1.00"
    )


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
