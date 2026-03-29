from __future__ import annotations

import pathlib
import sys

import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lightweight import stubs keep this hierarchy test runnable without optional
# runtime deps while still exercising real recursion logic in the pipeline.
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

from enochian_lm.root_extraction.pipeline.run_residual_semantic_extraction import (  # noqa: E402
    RemainderExtractionCrew,
)


def test_collect_donor_glosses_for_residual_emits_hierarchy_traces():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)

    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._find_host_words_for_residual = lambda residual: [{"canonical": "ALNAZPSAD"}]

    dictionary_defs = {
        "a": "in, with, on",
        "l": "the one",
    }

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

    donor_glosses, traces = crew._collect_donor_glosses_for_residual("PSAD")

    assert traces, "expected hierarchy traces"
    assert any(t.get("donor_source") == "host_subtraction" for t in traces)
    assert any(t.get("donor_source") == "dictionary" for t in traces)
    assert any(int(t.get("recursion_depth", 0)) >= 1 for t in traces)

    # The sqlite accepted gloss should still be discoverable in recursion.
    assert "naz" in donor_glosses


def test_resolve_donor_hierarchy_emits_empty_token_terminal_trace():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="ALNAZPSAD",
        token="",
        target_residual="PSAD",
        depth=1,
        visited=set(),
        donor_glosses={},
        traces=traces,
    )

    assert any(t.get("termination_reason") == "empty_token" for t in traces)


def test_resolve_donor_hierarchy_emits_max_depth_terminal_trace():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="ALNAZPSAD",
        token="naz",
        target_residual="PSAD",
        depth=999,
        visited=set(),
        donor_glosses={},
        traces=traces,
    )

    assert any(t.get("termination_reason") == "max_depth_reached" for t in traces)


def test_resolve_donor_hierarchy_emits_no_viable_donor_terminal_trace():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._get_dictionary_entry = lambda _token: None
    crew._dictionary_definition = lambda _entry: ""
    crew._load_accepted_glosses = lambda _token: []

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="ALNAZPSAD",
        token="naz",
        target_residual="A",
        depth=1,
        visited=set(),
        donor_glosses={},
        traces=traces,
    )

    assert any(t.get("termination_reason") == "no_viable_donor" for t in traces)


def test_resolve_donor_hierarchy_emits_cycle_detected_terminal_trace():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()

    # Only donor 'a' is viable for token 'ba', and it is pre-marked visited.
    crew._get_dictionary_entry = (
        lambda token: {"enhanced_definition": "in"}
        if str(token).strip().lower() == "a"
        else None
    )
    crew._dictionary_definition = (
        lambda entry: str(entry.get("enhanced_definition", "")).strip() if entry else ""
    )
    crew._load_accepted_glosses = lambda _token: []

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="BA",
        token="ba",
        target_residual="B",
        depth=1,
        visited={("ba", "b", "a", "b", "single_occurrence")},
        donor_glosses={},
        traces=traces,
    )

    assert any(t.get("termination_reason") == "cycle_detected" for t in traces)


def test_resolve_donor_hierarchy_prefers_longest_donor_for_same_coverage():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()

    dictionary_defs = {
        "n": "letter n",
        "a": "letter a",
        "z": "letter z",
        "na": "sound na",
        "az": "they",
        "naz": "rectangular prism",
    }

    def _get_dictionary_entry(token: str):
        key = str(token or "").strip().lower()
        if key in dictionary_defs:
            return {"enhanced_definition": dictionary_defs[key]}
        return None

    crew._get_dictionary_entry = _get_dictionary_entry
    crew._dictionary_definition = (
        lambda entry: str(entry.get("enhanced_definition", "")).strip() if entry else ""
    )
    crew._load_accepted_glosses = lambda _token: []

    traces: list[dict[str, object]] = []
    donor_glosses: dict[str, list[str]] = {}
    crew._resolve_donor_hierarchy(
        host_word="NAZPSAD",
        token="nazpsad",
        target_residual="psad",
        depth=1,
        visited=set(),
        donor_glosses=donor_glosses,
        traces=traces,
    )

    equations = [str(t.get("equation", "")) for t in traces]
    assert "NAZPSAD - NAZ = PSAD" in equations

    top_level_equations = [
        str(t.get("equation", ""))
        for t in traces
        if int(t.get("recursion_depth", 0)) == 1
    ]
    assert not any("NAZPSAD - N =" in eq for eq in top_level_equations)
    assert not any("NAZPSAD - A =" in eq for eq in top_level_equations)
    assert not any("NAZPSAD - Z =" in eq for eq in top_level_equations)


def test_resolve_donor_hierarchy_keeps_remove_all_branch_for_repeated_roots():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._get_dictionary_entry = (
        lambda token: {"enhanced_definition": "rectangular prism"}
        if str(token or "").strip().lower() == "naz"
        else None
    )
    crew._dictionary_definition = (
        lambda entry: str(entry.get("enhanced_definition", "")).strip() if entry else ""
    )
    crew._load_accepted_glosses = lambda _token: []

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="ANAZNAZ",
        token="anaznaz",
        target_residual="a",
        depth=0,
        visited=set(),
        donor_glosses={},
        traces=traces,
    )

    assert any(
        str(trace.get("equation", "")) == "ANAZNAZ - NAZ = A"
        and bool(trace.get("remove_all"))
        and str(trace.get("selected_variant", "")).lower() == "remove_all_occurrences"
        for trace in traces
    )
    assert any(
        str(trace.get("equation", "")) == "ANAZNAZ - NAZ = ANAZ"
        and str(trace.get("selected_variant", "")).lower() == "single_occurrence"
        for trace in traces
    )


def test_resolve_donor_hierarchy_recursively_boils_off_intermediate_residual():
    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew._normalize_root = lambda token: str(token or "").strip().lower()

    dictionary_defs = {
        "de": "undo / remove",
        "forest": "woodland mass",
    }

    def _get_dictionary_entry(token: str):
        key = str(token or "").strip().lower()
        if key in dictionary_defs:
            return {"enhanced_definition": dictionary_defs[key]}
        return None

    crew._get_dictionary_entry = _get_dictionary_entry
    crew._dictionary_definition = (
        lambda entry: str(entry.get("enhanced_definition", "")).strip() if entry else ""
    )
    crew._load_accepted_glosses = lambda _token: []

    traces: list[dict[str, object]] = []
    crew._resolve_donor_hierarchy(
        host_word="FODERESTA",
        token="foderesta",
        target_residual="a",
        depth=0,
        visited=set(),
        donor_glosses={},
        traces=traces,
    )

    equations = [str(trace.get("equation", "")) for trace in traces]
    assert "FODERESTA - DE = FORESTA" in equations
    assert "FORESTA - FOREST = A" in equations
