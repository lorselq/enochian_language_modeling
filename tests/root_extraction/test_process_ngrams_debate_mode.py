from __future__ import annotations

import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lightweight stubs keep this regression test focused on orchestration style
# gating, not optional runtime packages.
if "enochian_lm.root_extraction.tools.debate_semantic_subtraction_engine" not in sys.modules:
    debate_stub = types.ModuleType(
        "enochian_lm.root_extraction.tools.debate_semantic_subtraction_engine"
    )
    debate_stub.debate_semantic_subtraction = lambda **_kwargs: {}
    sys.modules[
        "enochian_lm.root_extraction.tools.debate_semantic_subtraction_engine"
    ] = debate_stub

if "enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine" not in sys.modules:
    solo_stub = types.ModuleType(
        "enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine"
    )
    solo_stub.solo_semantic_subtraction = lambda **_kwargs: {}
    sys.modules[
        "enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine"
    ] = solo_stub

if "enochian_lm.root_extraction.utils.semantic_search" not in sys.modules:
    semantic_stub = types.ModuleType("enochian_lm.root_extraction.utils.semantic_search")
    semantic_stub.find_semantically_similar_words = lambda *a, **k: []
    semantic_stub.compute_cluster_cohesion = lambda *a, **k: 0.0
    semantic_stub.cluster_definitions = lambda *a, **k: {"clusters": [], "config": {}}
    sys.modules["enochian_lm.root_extraction.utils.semantic_search"] = semantic_stub

if "enochian_lm.root_extraction.utils.candidate_finder" not in sys.modules:
    candidate_stub = types.ModuleType("enochian_lm.root_extraction.utils.candidate_finder")

    class _DummyFinder:
        def __init__(self, *args, **kwargs):
            pass

    candidate_stub.MorphemeCandidateFinder = _DummyFinder
    sys.modules["enochian_lm.root_extraction.utils.candidate_finder"] = candidate_stub

if "enochian_lm.root_extraction.utils.types_lexicon" not in sys.modules:
    lex_stub = types.ModuleType("enochian_lm.root_extraction.utils.types_lexicon")
    lex_stub.EntryRecord = dict
    sys.modules["enochian_lm.root_extraction.utils.types_lexicon"] = lex_stub

if "enochian_lm.root_extraction.utils.embeddings" not in sys.modules:
    embeddings_stub = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
    embeddings_stub.get_fasttext_model = lambda *a, **k: object()
    embeddings_stub.get_sentence_transformer = lambda *a, **k: object()
    embeddings_stub.stream_text = lambda *a, **k: None
    sys.modules["enochian_lm.root_extraction.utils.embeddings"] = embeddings_stub

if "enochian_lm.root_extraction.utils.dictionary_loader" not in sys.modules:
    dict_stub = types.ModuleType("enochian_lm.root_extraction.utils.dictionary_loader")
    dict_stub.load_dictionary = lambda *a, **k: []
    sys.modules["enochian_lm.root_extraction.utils.dictionary_loader"] = dict_stub

import enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction as pipeline  # noqa: E402
from enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction import (  # noqa: E402
    RemainderExtractionCrew,
)


class _DummyCursor:
    def execute(self, *args, **kwargs):
        return None

    def executemany(self, *args, **kwargs):
        return None


class _DummyDB:
    def cursor(self):
        return _DummyCursor()

    def commit(self):
        return None

    def close(self):
        return None


class _DummyFinderInstance:
    def get_all_ngram_candidates(self, *_args, **_kwargs):
        return []


def test_process_ngrams_allows_debate_mode_and_reaches_evaluate_dispatch(monkeypatch):
    """Regression for Addendum F orchestration entrypoint behavior.

    What: run process_ngrams in debate mode through one minimal single-ngram pass.
    Why: guard against reintroducing the old solo-only ValueError at the queue layer.
    Big picture: ensures end-to-end debate runs are reachable from main orchestration.
    """

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)

    # Keep the loop deterministic and dependency-light for this style-gating check.
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._refresh_ngram_inventory = lambda: None
    crew._load_queue_order = lambda: []
    crew._get_current_cycle = lambda: 0
    crew._is_root_processed = lambda *_args, **_kwargs: False
    crew._get_dictionary_entry = lambda _token: {"enhanced_definition": "defined"}
    crew._load_accepted_glosses = lambda _token: []
    crew._insert_skip = lambda *_args, **_kwargs: None
    crew._build_parent_entries = lambda *_args, **_kwargs: []
    crew._get_field_value = (
        lambda item, field, default="": item.get(field, default)
        if isinstance(item, dict)
        else default
    )
    crew.is_ngram_in_variants = lambda *_args, **_kwargs: True
    crew.get_matching_variant = lambda *_args, **_kwargs: None
    crew._dynamic_coh_floor = lambda *_args, **_kwargs: 0.0
    crew._build_stats_summary = lambda *_args, **_kwargs: "stub-summary"
    crew._mark_root_complete = lambda *_args, **_kwargs: None
    crew._record_preanalysis_consumed = lambda *_args, **_kwargs: None

    crew.entries = []
    crew.fasttext = object()
    crew.sentence_model = object()
    crew.subst_map = {}
    crew.candidate_finder = _DummyFinderInstance()
    crew.new_definitions_db = _DummyDB()
    crew.ngram_db = _DummyDB()

    monkeypatch.setattr(
        pipeline,
        "find_semantically_similar_words",
        lambda **_kwargs: [
            {
                "word": "NAZPSAD",
                "normalized": "nazpsad",
                "canonical": "NAZPSAD",
                "definition": "host",
                "enhanced_definition": "host",
                "source": "semantic",
                "citations": [],
            }
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "cluster_definitions",
        lambda candidates, _sentence_model: {"clusters": [candidates], "config": {}},
    )
    monkeypatch.setattr(pipeline, "compute_cluster_cohesion", lambda *_a, **_k: 0.7)
    monkeypatch.setattr(pipeline, "stream_text", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(pipeline, "save_log", lambda *_a, **_k: None)

    seen_styles: list[str] = []

    def _fake_evaluate_ngram(**kwargs):
        seen_styles.append(str(kwargs.get("style")))
        return {
            "Glossator": "accept",
            "Archivist": "audit",
            "raw_output": {"Glossator": "accept", "Archivist": "audit"},
        }

    crew.evaluate_ngram = _fake_evaluate_ngram

    # This call used to raise ValueError for style='debate'.
    crew.process_ngrams(single_ngram="NAZ", style="debate", max_words=1)

    assert seen_styles == ["debate"]
