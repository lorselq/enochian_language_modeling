from __future__ import annotations

import pathlib
import sqlite3
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lightweight import stubs keep this test focused on DB persistence mapping.
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


class _DBHandle:
    """Keep an in-memory sqlite handle open for post-run assertions.

    What: thin wrapper exposing the subset of DB methods the pipeline uses.
    Why: `process_ngrams` calls `close()`, but this test must still query rows.
    Big picture: isolates persistence-mapping verification from connection lifecycle noise.
    """

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")

    def cursor(self):
        return self.conn.cursor()

    def execute(self, *args, **kwargs):
        return self.conn.execute(*args, **kwargs)

    def commit(self):
        return self.conn.commit()

    def close(self):
        # Keep connection open for post-run assertions in this test.
        return None


def test_process_ngrams_persists_semantic_subtraction_traces(monkeypatch):
    """DB-focused Addendum J regression.

    What: run one ngram through process_ngrams and persist analytics traces.
    Why: ensures guidance payloads are durably stored, not only passed to prompts.
    Big picture: makes HOST-ROOT-RESIDUAL evidence auditable in insights DB.
    """

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    db = _DBHandle()

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
    crew.new_definitions_db = db
    crew.ngram_db = db
    crew.run_id = "run-test"

    class _DummyFinderInstance:
        def get_all_ngram_candidates(self, *_args, **_kwargs):
            return []

    crew.candidate_finder = _DummyFinderInstance()

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

    def _fake_evaluate_ngram(**_kwargs):
        return {
            "Glossator": "accept",
            "Archivist": "audit",
            "raw_output": {"Glossator": "accept", "Archivist": "audit"},
            "analytics_summary": {
                "word_breaks": [
                    {
                        "host_word": "NAZPSAD",
                        "root": "NAZ",
                        "residual": "PSAD",
                        "equation": "NAZPSAD - NAZ = PSAD",
                        "donor_source": "host_subtraction",
                        "recursion_depth": 0,
                        "termination_reason": "root_subtracted_from_host",
                    }
                ],
                "hierarchy_traces": [
                    {
                        "host_word": "ALNAZPSAD",
                        "root": "A",
                        "residual": "LNAZ",
                        "equation": "ALNAZ - A = LNAZ",
                        "donor_source": "dictionary",
                        "recursion_depth": 1,
                        "termination_reason": "residual_extracted",
                    }
                ],
                "subtraction_equations": [
                    "NAZPSAD - NAZ = PSAD",
                    "ALNAZ - A = LNAZ",
                ],
            },
        }

    crew.evaluate_ngram = _fake_evaluate_ngram

    crew.process_ngrams(single_ngram="NAZ", style="solo", max_words=1)

    rows = db.conn.execute(
        """
        SELECT ngram, host_word, root, residual, equation,
               donor_source, recursion_depth, termination_reason, trace_role
        FROM semantic_subtraction_traces
        WHERE run_id = ?
        ORDER BY trace_id
        """,
        ("run-test",),
    ).fetchall()

    assert rows, "expected persisted semantic_subtraction_traces rows"
    assert any(r[8] == "word_break" for r in rows)
    assert any(r[8] == "hierarchy_trace" for r in rows)
    assert any(r[8] == "equation" for r in rows)

    # Required persisted fields from Addendum J mapping.
    assert any(r[0] == "NAZ" for r in rows)
    assert any(r[1] == "NAZPSAD" and r[2] == "NAZ" and r[3] == "PSAD" for r in rows)
    assert any(r[4] == "NAZPSAD - NAZ = PSAD" for r in rows)
    assert any(r[5] == "dictionary" and int(r[6]) == 1 and r[7] == "residual_extracted" for r in rows)
