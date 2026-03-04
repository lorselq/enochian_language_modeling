from __future__ import annotations

import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lightweight import stubs keep this test focused on real hierarchy traversal
# inside the pipeline module instead of optional runtime dependency setup.
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
    semantic_stub.cluster_definitions = lambda *a, **k: []
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


def test_evaluate_ngram_preserves_multiple_dictionary_branches_in_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Addendum G regression for multi-branch hierarchy evidence.

    What: run evaluate_ngram with real donor-hierarchy traversal enabled.
    Why: ensure recursion keeps multiple ranked donor branches for ambiguous hosts.
    Big picture: downstream solo/debate prompts need competing subtraction paths,
    not a single collapsed donor guess.
    """

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew.use_remote = False
    crew.new_definitions_db = object()
    crew.run_id = "run-test"
    crew._breakdown_diag_cache = {}

    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._get_candidate_breakdown = lambda _norm: None
    crew._find_host_words_for_residual = lambda _residual: [{"canonical": "ALNAZPSAD"}]

    dictionary_defs = {"a": "in, with, on", "l": "the one"}

    def _get_dictionary_entry(token: str):
        key = str(token or "").strip().lower()
        if key in dictionary_defs:
            return {"enhanced_definition": dictionary_defs[key]}
        return None

    crew._get_dictionary_entry = _get_dictionary_entry
    crew._dictionary_definition = (
        lambda entry: str(entry.get("enhanced_definition", "")).strip() if entry else ""
    )

    def _load_accepted_glosses(token: str):
        key = str(token or "").strip().lower()
        if key == "naz":
            return ['{"ROOT":"NAZ","DEFINITION":"rectangular prism"}']
        return []

    crew._load_accepted_glosses = _load_accepted_glosses

    monkeypatch.setattr(
        pipeline,
        "gather_morph_evidence",
        lambda *_args, **_kwargs: {"summary_lines": ["stub analytics"]},
    )
    monkeypatch.setattr(
        pipeline,
        "fetch_preanalysis_summary",
        lambda *_args, **_kwargs: None,
    )

    captured: dict[str, object] = {}

    def _fake_solo_engine(**kwargs):
        captured.update(kwargs)
        return {
            "Glossator": "ok",
            "Model": "test-model",
            "raw_output": {"Model": "test-model"},
        }

    monkeypatch.setattr(pipeline, "solo_semantic_subtraction", _fake_solo_engine)

    cluster = [
        {
            "word": "ALNAZPSAD",
            "normalized": "alnazpsad",
            "definition": "within the sword of the One",
            "enhanced_definition": "within the sword of the One",
            "source": "semantic",
            "tier": "T1",
            "priority": 1,
        }
    ]

    crew.evaluate_ngram(
        ngram="PSAD",
        cluster=cluster,
        cohesion_score=0.5,
        semantic_hits=1,
        semantic_coverage=0.5,
        sem_count=1,
        idx_count=1,
        overlap_count=1,
        stats_summary="unused",
        style="solo",
    )

    guidance = captured["residual_guidance"]
    hierarchy_traces = guidance.get("hierarchy_traces") or []
    word_breaks = guidance.get("word_breaks") or []

    # At depth 1, both competing dictionary donors (A and L) should survive.
    depth_one_dictionary_roots = {
        str(row.get("root", "")).upper()
        for row in hierarchy_traces
        if str(row.get("donor_source", "")).lower() == "dictionary"
        and int(row.get("recursion_depth", 0)) == 1
    }
    assert {"A", "L"}.issubset(depth_one_dictionary_roots)

    # Guidance payload should include the same competing branches in word_breaks.
    depth_one_break_roots = {
        str(row.get("root", "")).upper()
        for row in word_breaks
        if str(row.get("donor_source", "")).lower() == "dictionary"
        and int(row.get("recursion_depth", 0)) == 1
    }
    assert {"A", "L"}.issubset(depth_one_break_roots)
