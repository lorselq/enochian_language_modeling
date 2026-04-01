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

    candidate_stub.CoverageSegment = dict
    candidate_stub.MorphemeCandidateFinder = _DummyFinder
    sys.modules["enochian_lm.root_extraction.utils.candidate_finder"] = candidate_stub

if "enochian_lm.root_extraction.utils.types_lexicon" not in sys.modules:
    lex_stub = types.ModuleType("enochian_lm.root_extraction.utils.types_lexicon")
    lex_stub.AltRecord = dict
    lex_stub.EntryRecord = dict
    lex_stub.SenseRecord = dict
    sys.modules["enochian_lm.root_extraction.utils.types_lexicon"] = lex_stub

if "enochian_lm.root_extraction.utils.embeddings" not in sys.modules:
    embeddings_stub = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
    embeddings_stub.get_fasttext_model = lambda *a, **k: object()
    embeddings_stub.get_sentence_transformer = lambda *a, **k: object()
    embeddings_stub.get_sentence_transformer_if_available = (
        lambda *a, **k: object()
    )
    embeddings_stub.cluster_definitions = lambda *a, **k: []
    embeddings_stub.select_definitions = lambda defs, max_words=300: defs
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


def _build_host_subtraction(
    host_word: str,
    donor_root: str,
    residual: str,
    *,
    termination_reason: str = "resolved",
    host_definition: str = "",
    host_source: str = "",
) -> dict[str, object]:
    """Build one corrected HOST - DONOR = TARGET/INTERMEDIATE trace row."""

    return build_subtraction_evidence(
        {
            "host_word": host_word,
            "root": donor_root,
            "residual": residual,
            "host_definition": host_definition,
            "host_source": host_source,
        },
        donor_source="host_subtraction",
        recursion_depth=0,
        termination_reason=termination_reason,
    )


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
    def _load_accepted_glosses(token: str):
        key = str(token or "").strip().lower()
        if key == "lcordzi":
            return ['{"ROOT":"LCORDZI","EVALUATION":"accepted","DEFINITION":"stub"}']
        return []

    crew._load_accepted_glosses = _load_accepted_glosses

    host_metadata = {
        host.lower(): {"host_definition": f"host-{idx}", "host_source": "semantic"}
        for idx, host in enumerate(hosts)
    }

    host_rows = {
        "nazpsad": _build_host_subtraction("NAZPSAD", "PSAD", "NAZ", **host_metadata["nazpsad"]),
        "nazmical": _build_host_subtraction("NAZMICAL", "MICAL", "NAZ", **host_metadata["nazmical"]),
        "nazod": _build_host_subtraction("NAZOD", "OD", "NAZ", **host_metadata["nazod"]),
        "nazabrax": _build_host_subtraction("NAZABRAX", "ABRAX", "NAZ", **host_metadata["nazabrax"]),
        "naztor": _build_host_subtraction("NAZTOR", "TOR", "NAZ", **host_metadata["naztor"]),
        "alnazps": _build_host_subtraction(
            "ALNAZPS",
            "AL",
            "NAZPS",
            termination_reason="residual_extracted",
            **host_metadata["alnazps"],
        ),
        "tornazim": _build_host_subtraction(
            "TORNAZIM",
            "IM",
            "TORNAZ",
            termination_reason="residual_extracted",
            **host_metadata["tornazim"],
        ),
        "anaznaz": _build_host_subtraction(
            "ANAZNAZ",
            "A",
            "NAZNAZ",
            termination_reason="residual_extracted",
            **host_metadata["anaznaz"],
        ),
        "qenazp": _build_host_subtraction(
            "QENAZP",
            "QE",
            "NAZP",
            termination_reason="residual_extracted",
            **host_metadata["qenazp"],
        ),
        "palnazor": _build_host_subtraction(
            "PALNAZOR",
            "PAL",
            "NAZOR",
            termination_reason="residual_extracted",
            **host_metadata["palnazor"],
        ),
    }

    def _fake_resolve_donor_hierarchy(
        *,
        host_word: str,
        token: str,
        target_residual: str,
        depth: int,
        visited: set[tuple[str, str, str, str, str]],
        donor_glosses: dict[str, list[str]],
        traces: list[dict[str, object]],
    ) -> None:
        del token, target_residual, depth, visited, donor_glosses
        row = host_rows.get(str(host_word or "").strip().lower())
        if row:
            traces.append(dict(row))

    crew._resolve_donor_hierarchy = _fake_resolve_donor_hierarchy

    donor_traces = [
        {
            "host_word": "ALNAZPS",
            "root": "AL",
            "residual": "NAZPS",
            "equation": "ALNAZPS - AL = NAZPS",
            "remaining_artifacts": [],
            "donor_source": "dictionary",
            "recursion_depth": 1,
            "termination_reason": "residual_extracted",
        },
        {
            "host_word": "NAZPS",
            "root": "PS",
            "residual": "NAZ",
            "equation": "NAZPS - PS = NAZ",
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

    expected_word_breaks = list(host_rows.values())
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


def test_stats_summary_prioritizes_direct_host_remainder_dictionary_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: direct host subtraction remainder (e.g., NAZ) must be surfaced first."""

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew.use_remote = False
    crew.new_definitions_db = object()
    crew.run_id = "run-test"
    crew._breakdown_diag_cache = {}

    crew._get_candidate_breakdown = lambda _norm: None
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._find_host_words_for_residual = lambda _residual: [{"canonical": "NAZPSAD"}]

    dictionary_defs = {
        "naz": "(something shaped like a rectangular prism)",
        "na": "(the Enochian word for the letter 'H')",
        "az": "they",
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
        if key == "psad":
            return [
                "{\"ROOT\":\"NA\",\"EVALUATION\":\"accepted\",\"DEFINITION\":\"(the Enochian word for the letter 'H')\"}",
                '{"ROOT":"AZ","EVALUATION":"accepted","DEFINITION":"they"}',
            ]
        if key == "naz":
            return [
                '{"ROOT":"NAZ","EVALUATION":"accepted","DEFINITION":"Linear-edged geometric form","DECODING_GUIDE":"^NAZ-","SEMANTIC_CORE":["linear-edged geometry"],"NEGATIVE_CONTRAST":["non-edged"],"EXAMPLE":["blade form"],"POS_BIAS":{"nounness":0.8}}'
            ]
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

    def _fake_engine(**kwargs):
        captured.update(kwargs)
        return {"Glossator": "ok", "Model": "test-model", "raw_output": {"Model": "test-model"}}

    monkeypatch.setattr(pipeline, "debate_semantic_subtraction", _fake_engine)

    cluster = [
        {
            "word": "NAZPSAD",
            "normalized": "nazpsad",
            "definition": "sword",
            "enhanced_definition": "sword",
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
        semantic_coverage=1.0,
        sem_count=1,
        idx_count=1,
        overlap_count=1,
        stats_summary="unused",
        style="debate",
    )

    stats_summary = str(captured.get("stats_summary", ""))
    assert "Prioritized donor dictionary anchors:" in stats_summary
    assert "- NAZ (⭐ key donor; direct subtraction from NAZPSAD via NAZPSAD - NAZ = PSAD):" in stats_summary
    assert "- AZ (⭐ key donor; direct subtraction from NAZPSAD via NAZPSAD - AZ = NPSAD): they" in stats_summary
    assert (
        "- NA (⭐ key donor; direct subtraction from NAZPSAD via NAZPSAD - NA = ZPSAD): "
        "(the Enochian word for the letter 'H')"
    ) in stats_summary
    assert "\n- A (⭐ key donor;" not in stats_summary
    assert "\n- Z (⭐ key donor;" not in stats_summary


def test_stats_summary_keeps_nonadjacent_host_remainder_artifacts_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: non-adjacent remainder artifacts should not be merged into one key donor."""

    crew = RemainderExtractionCrew.__new__(RemainderExtractionCrew)
    crew.use_remote = False
    crew.new_definitions_db = object()
    crew.run_id = "run-test"
    crew._breakdown_diag_cache = {}

    crew._get_candidate_breakdown = lambda _norm: None
    crew._normalize_root = lambda token: str(token or "").strip().lower()
    crew._find_host_words_for_residual = lambda _residual: [{"canonical": "OLCORDZIZ"}]

    dictionary_defs = {
        "o": "five",
        "z": "they",
        "oz": "active force or agency",
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
        if key == "lcordzi":
            return ['{"ROOT":"LCORDZI","EVALUATION":"accepted","DEFINITION":"stub"}']
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

    def _fake_engine(**kwargs):
        captured.update(kwargs)
        return {"Glossator": "ok", "Model": "test-model", "raw_output": {"Model": "test-model"}}

    monkeypatch.setattr(pipeline, "debate_semantic_subtraction", _fake_engine)

    cluster = [
        {
            "word": "OLCORDZIZ",
            "normalized": "olcordziz",
            "definition": "made mankind",
            "enhanced_definition": "made mankind",
            "source": "semantic",
            "tier": "T1",
            "priority": 1,
        }
    ]

    crew.evaluate_ngram(
        ngram="LCORDZI",
        cluster=cluster,
        cohesion_score=0.0,
        semantic_hits=0,
        semantic_coverage=1.0,
        sem_count=1,
        idx_count=1,
        overlap_count=0,
        stats_summary="unused",
        style="debate",
    )

    stats_summary = str(captured.get("stats_summary", ""))
    assert "- O (⭐ key donor; direct subtraction from OLCORDZIZ via OLCORDZIZ - O = LCORDZIZ): five" in stats_summary
    assert "- Z (⭐ key donor; direct subtraction from OLCORDZIZ via OLCORDZIZ - Z = OLCORDZI): they" in stats_summary
    assert "- OZ (⭐ key donor; direct subtraction from OLCORDZIZ" not in stats_summary
