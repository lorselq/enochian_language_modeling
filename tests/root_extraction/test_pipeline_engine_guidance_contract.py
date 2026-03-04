from __future__ import annotations

import pathlib
import sys

import pytest
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Provide lightweight stubs so importing the pipeline module does not require
# heavyweight optional runtime dependencies in this integration-contract test.
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
from enochian_lm.root_extraction.utils.residual_analysis import (  # noqa: E402
    build_residual_guidance_payload,
    build_subtraction_evidence,
    compute_word_break_subtractions,
)


def _build_cluster(root: str, hosts: list[str]) -> list[dict[str, object]]:
    """Create a minimal candidate cluster used to exercise evaluate_ngram.

    What: build synthetic entries with required fields only.
    Why: integration-ish tests should verify pipeline->engine payload contracts
    without loading the full runtime stack.
    Big picture: keeps these tests deterministic while covering real orchestration.
    """

    entries: list[dict[str, object]] = [
        {
            "word": root,
            "normalized": root.lower(),
            "definition": "base root",
            "enhanced_definition": "base root",
            "source": "both",
            "tier": "T1",
            "priority": 1,
        }
    ]
    for idx, host in enumerate(hosts):
        entries.append(
            {
                "word": host,
                "normalized": host.lower(),
                "definition": f"host-{idx}",
                "enhanced_definition": f"host-{idx}",
                "source": "semantic",
                "tier": "T1",
                "priority": 1,
            }
        )
    return entries


def _expected_host_subtractions(hosts: list[str], root: str) -> list[dict[str, object]]:
    """Mirror the host-level subtraction rows produced by evaluate_ngram."""

    rows: list[dict[str, object]] = []
    for host in hosts:
        for subtraction in compute_word_break_subtractions(host, root.lower()):
            residual = str(subtraction.get("residual", "")).strip().lower()
            if not residual or residual == root.lower():
                continue
            payload = dict(subtraction)
            payload.setdefault("host_word", host)
            payload.setdefault("root", root.lower())
            payload["residual"] = residual
            rows.append(
                build_subtraction_evidence(
                    payload,
                    donor_source="host_subtraction",
                    recursion_depth=0,
                    termination_reason="root_subtracted_from_host",
                )
            )
    return rows


@pytest.mark.parametrize("style", ["solo", "debate"])
def test_pipeline_sends_full_residual_guidance_contract_to_engine(
    monkeypatch: pytest.MonkeyPatch,
    style: str,
) -> None:
    """Ensure evaluate_ngram forwards the complete subtraction guidance contract.

    This test intentionally runs one ngram through both engine modes (solo/debate)
    and validates the exact payload shape expected by downstream prompt layers.
    """

    root = "NAZ"

    # Five northstar-style hosts + five infix/fragment hosts.
    northstar_hosts = ["NAZPSAD", "NAZMICAL", "NAZOD", "NAZABRAX", "NAZTOR"]
    infix_hosts = ["ALNAZPS", "TORNAZIM", "ANAZNAZ", "QENAZP", "PALNAZOR"]
    hosts = northstar_hosts + infix_hosts

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew.use_remote = False
    crew.new_definitions_db = object()
    crew.run_id = "run-test"
    crew._breakdown_diag_cache = {}

    # Keep decomposition deterministic: no external candidate finder required.
    crew._get_candidate_breakdown = lambda _norm: None
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._find_host_words_for_residual = lambda _residual: [{"canonical": w} for w in hosts]
    crew._load_accepted_glosses = lambda _token: []

    donor_traces = [
        {
            "host_word": "NAZPSAD",
            "root": "PSAD",
            "residual": "",
            "equation": "NAZPSAD - PSAD = ",
            "remaining_artifacts": [],
            "donor_source": "dictionary",
            "recursion_depth": 1,
            "termination_reason": "resolved",
        },
        {
            "host_word": "ANAZNAZ",
            "root": "A",
            "residual": "",
            "equation": "ANAZNAZ - A = ",
            "remaining_artifacts": [],
            "donor_source": "sqlite",
            "recursion_depth": 2,
            "termination_reason": "resolved",
        },
    ]
    crew._collect_donor_glosses_for_residual = lambda _residual: ({}, donor_traces)

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

    def _fake_engine(**kwargs):
        captured.update(kwargs)
        return {"Glossator": "ok", "Model": "test-model", "raw_output": {"Model": "test-model"}}

    if style == "solo":
        monkeypatch.setattr(pipeline, "solo_semantic_subtraction", _fake_engine)
    else:
        monkeypatch.setattr(pipeline, "debate_semantic_subtraction", _fake_engine)

    cluster = _build_cluster(root=root, hosts=hosts)
    crew.evaluate_ngram(
        ngram=root,
        cluster=cluster,
        cohesion_score=0.5,
        semantic_hits=5,
        semantic_coverage=0.5,
        sem_count=5,
        idx_count=5,
        overlap_count=3,
        stats_summary="unused",
        style=style,
    )

    expected_word_breaks = _expected_host_subtractions(hosts=hosts, root=root)
    merged_word_breaks = expected_word_breaks + donor_traces
    expected_guidance = {"summary_lines": ["stub analytics"]}
    expected_guidance.update(
        build_residual_guidance_payload(root=root.lower(), word_breaks=merged_word_breaks)
    )
    expected_guidance["hierarchy_traces"] = donor_traces

    assert captured["residual_guidance"] == expected_guidance

    guidance = captured["residual_guidance"]
    word_breaks = guidance["word_breaks"]

    # Contract checks required by Addendum B.
    assert all({"host_word", "root", "residual"}.issubset(row) for row in word_breaks)
    assert guidance["subtraction_equations"] == [str(row["equation"]).strip() for row in word_breaks]

    assert any(int(row.get("recursion_depth", 0)) >= 1 for row in word_breaks)
    assert any(str(row.get("donor_source", "")) in {"dictionary", "sqlite"} for row in word_breaks)
    assert all("termination_reason" in row for row in word_breaks)

    # Coverage check: we explicitly include 5 northstar + 5 infix/fragment hosts.
    observed_hosts = {str(row.get("host_word", "")) for row in word_breaks}
    assert all(host in observed_hosts for host in northstar_hosts)
    assert all(host in observed_hosts for host in infix_hosts)
