from __future__ import annotations

import sys
import types
from types import SimpleNamespace


def _install_fasttext_shim() -> None:
    """Install minimal gensim shims before importing translation CLI modules.

    The broader translation tests install similar optional-dependency shims at
    module import time. These tests also import the public CLI, so they need the
    same lightweight FastText surface to avoid order-dependent real model loads.
    """

    gensim_module = sys.modules.setdefault("gensim", types.ModuleType("gensim"))
    gensim_utils = sys.modules.setdefault(
        "gensim.utils",
        types.ModuleType("gensim.utils"),
    )
    gensim_utils.simple_preprocess = lambda text, deacc=True, min_len=1: [str(text)]
    gensim_models = sys.modules.setdefault(
        "gensim.models",
        types.ModuleType("gensim.models"),
    )

    class _DummyFastText:
        """Provide the minimal FastText surface used by candidate lookups."""

        def __init__(self, *args, **kwargs) -> None:
            self.wv = self

        @classmethod
        def load(cls, _path: str) -> "_DummyFastText":
            return cls()

        def get_vector(self, _token: str) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0]

        def similar_by_word(self, _word: str, topn: int = 5) -> list[tuple[str, float]]:
            return []

    gensim_models.FastText = _DummyFastText
    gensim_module.models = gensim_models
    gensim_module.utils = gensim_utils


_install_fasttext_shim()

from translation import cli as translation_cli
from translation.agentic_service import AgenticTranslationService


class StubRepository:
    """Provide read-only evidence fixtures for the agentic harness tests.

    The production service talks to the large solo/debate SQLite databases.
    These fixtures keep unit tests fast while preserving the same curated tool
    shape: word evidence, accepted clusters, and FNP profile lookups.
    """

    def fetch_word_evidence(self, word: str, *, variants=None, **_kwargs):
        """Return empty residual support unless a test fixture overrides it."""

        return SimpleNamespace(
            word=word,
            variants_queried=list(variants or []),
            residual_semantics=[],
        )

    def fetch_clusters(self, morph: str, *, variants=None):
        """Return one compact accepted-cluster fixture for follow-up evidence."""

        source = list(variants or ["solo"])[0]
        return [
            SimpleNamespace(
                variant=source,
                cluster_id=101,
                run_id="run-1",
                ngram=morph,
                cluster_index=0,
                glossator_def=f"{morph} fixture gloss",
                residual_explained=None,
                residual_ratio=None,
                residual_headline=None,
                residual_focus_prompt=None,
                semantic_coverage=1.0,
                cohesion=0.8,
                semantic_cohesion=0.75,
                best_config=None,
                semantic_core=["fixture"],
                negative_contrast=["non-fixture"],
                residual_details=[],
                raw_definitions=[],
            )
        ]

    def fetch_root_fnp_profiles(self, morphs, *, variants=None):
        """Return no FNP profiles so tests focus on dossier orchestration."""

        return {}


class StubWordService:
    """Expose source-specific baseline translations to the harness.

    The harness must not merge solo and debate evidence. This stub returns
    intentionally different lower-ranked candidates so tests can assert source
    comparison and otherwise exploration behavior.
    """

    def __init__(self) -> None:
        self.repository = StubRepository()
        self.candidate_finder = SimpleNamespace(dictionary={"nazpsad": "dictionary-entry"})
        self.closed = False
        self.calls: list[dict[str, object]] = []

    def translate_word(self, word: str, **kwargs):
        """Return a source-aware word baseline with dictionary and compositional rows."""

        call = dict(kwargs)
        call["dictionary_snapshot"] = dict(self.candidate_finder.dictionary)
        self.calls.append(call)
        source = list(kwargs.get("variants") or ["solo"])[0]
        if word.upper() == "XNON":
            return {
                "word": word.upper(),
                "evidence": {"direct_clusters": 1, "residual_semantics": 0, "attested_definitions": 0},
                "candidates": [
                    {
                        "rank": 1,
                        "analysis_type": "whole_word_anchor",
                        "morphs": ["XNON"],
                        "confidence": 0.72,
                        "score": 1.0,
                        "synthesized_definition": "non-canonical object",
                        "concatenated_meanings": "non-canon fixture",
                        "meanings": [
                            {
                                "morph": "XNON",
                                "definition": "non-canon fixture",
                                "provenance": "cluster",
                            }
                        ],
                    }
                ],
            }
        compositional = {
            "rank": 2,
            "analysis_type": "compositional",
            "morphs": ["NAZ", "PS", "AD"],
            "confidence": 0.84,
            "score": 1.8 if source == "debate" else 1.7,
            "synthesized_definition": "radiant judgment",
            "concatenated_meanings": "radiant justice",
            "score_breakdown": {"definition_counts": {"AD": 60, "PS": 2}},
            "decision_trace": {"selection_reason": "Compositional fixture retained."},
            "meanings": [
                {"morph": "NAZ", "definition": "linear form", "provenance": "cluster"},
                {
                    "morph": "PS",
                    "definition": "celestial",
                    "provenance": "cluster",
                    "definition_trace": {
                        "bundle_selection_reason": "Fixture selected the dominant PS bundle.",
                        "runner_ups": [
                            {
                                "definition": "divisive",
                                "source": "cluster",
                                "quality": "medium",
                                "cluster_id": 202,
                            }
                        ],
                    },
                },
                {
                    "morph": "AD",
                    "definition": "divine justice",
                    "provenance": "cluster",
                    "definition_trace": {
                        "bundle_selection_reason": "Fixture selected the dominant AD bundle.",
                        "runner_ups": [
                            {
                                "definition": "power",
                                "source": "cluster",
                                "quality": "high",
                                "cluster_id": 303,
                            }
                        ],
                    },
                },
            ],
        }
        result = {
            "word": word.upper(),
            "evidence": {
                "direct_clusters": 2,
                "residual_semantics": 0,
                "attested_definitions": 1,
                "rejected_morphs": ["AZP"] if source == "solo" else [],
            },
            "candidates": [
                {
                    "rank": 1,
                    "analysis_type": "dictionary_exact",
                    "morphs": [word.upper()],
                    "confidence": 0.98,
                    "score": 2.0,
                    "synthesized_definition": "cutting weapon"
                    if source == "solo"
                    else "edged weapon",
                    "concatenated_meanings": "sword" if source == "solo" else "blade",
                    "meanings": [
                        {
                            "morph": word.upper(),
                            "definition": "sword" if source == "solo" else "blade",
                            "provenance": "dictionary",
                        }
                    ],
                },
                compositional,
                {
                    "rank": 3,
                    "analysis_type": "compositional",
                    "morphs": ["PS", "AD"],
                    "confidence": 0.73,
                    "score": 1.5,
                    "synthesized_definition": "divisive power",
                    "concatenated_meanings": "divisive power",
                    "decision_trace": {"selection_reason": "Alternate PS/AD fixture retained."},
                    "meanings": [
                        {"morph": "PS", "definition": "divisive", "provenance": "cluster"},
                        {"morph": "AD", "definition": "power", "provenance": "cluster"},
                    ],
                },
            ],
        }
        if kwargs.get("speculative"):
            result["speculative_hypotheses"] = [
                {
                    "rank": 1,
                    "analysis_type": "speculative_compositional",
                    "morphs": ["NAZ", "P", "SA", "D"],
                    "confidence": 0.61,
                    "score": 1.25,
                    "bundle_surface_gloss": "pillars separation distinction disturbance",
                    "concatenated_meanings": "pillars + separation + distinction + disturbance",
                    "decision_trace": {
                        "selection_reason": "Speculative split retained by shared separation motifs.",
                        "head_modifier_analysis": {
                            "head_roots": ["NAZ"],
                            "modifier_roots": ["P", "SA", "D"],
                            "distributed_motif_count": 3,
                            "transformational_gloss": (
                                "NAZ + P + SA + D: rigid/linear head + "
                                "separative modifiers => sword-like artifact"
                            ),
                        },
                    },
                    "head_modifier_analysis": {
                        "head_roots": ["NAZ"],
                        "modifier_roots": ["P", "SA", "D"],
                        "distributed_motif_count": 3,
                        "transformational_gloss": (
                            "NAZ + P + SA + D: rigid/linear head + "
                            "separative modifiers => sword-like artifact"
                        ),
                    },
                    "meanings": [
                        {"morph": "NAZ", "definition": "pillars", "provenance": "cluster"},
                        {"morph": "P", "definition": "separation", "provenance": "cluster"},
                        {"morph": "SA", "definition": "distinction", "provenance": "cluster"},
                        {"morph": "D", "definition": "disturbance", "provenance": "cluster"},
                    ],
                }
            ]
        return result

    def close(self) -> None:
        """Record close calls without touching external resources."""

        self.closed = True


class StubPhraseService:
    """Return a compact phrase baseline for kind=phrase tests."""

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict[str, object]] = []

    def translate_phrase(self, phrase: str, **kwargs):
        """Return a chosen parse shaped like the production phrase service."""

        self.calls.append(dict(kwargs))
        source = list(kwargs.get("variants") or ["solo"])[0]
        return {
            "phrase": phrase,
            "evidence": {"direct_clusters": 1, "residual_semantics": 0, "attested_definitions": 1},
            "chosen_parse": {
                "rank": 1,
                "translation_skeleton": f"{source} phrase skeleton",
                "confidence": 0.8,
                "token_choices": [
                    {
                        "rank": 1,
                        "token": "OL",
                        "translation_gloss": "I",
                        "morphs": ["OL"],
                        "confidence": 0.8,
                    }
                ],
            },
        }

    def close(self) -> None:
        """Record close calls without touching external resources."""

        self.closed = True


class StubAgenticService:
    """Act as a CLI-safe context manager for command tests.

    CLI tests should verify parser/renderer behavior without opening the real
    insights databases, so this object supplies the two service methods the CLI
    needs.
    """

    def __enter__(self):
        """Return the service stub as the context payload."""

        return self

    def __exit__(self, *_args):
        """Leave the stub context without side effects."""

        return None

    def build_dossier(self, text: str, **kwargs):
        """Return the requested fields so tests can assert CLI propagation."""

        return {
            "request": {
                "text": text,
                "kind": kwargs["kind"],
                "analysis_source": kwargs["analysis_source"],
                "dictionary_enabled": kwargs["allow_dictionary"],
                "whole_word_enabled": kwargs["allow_whole_word"],
            },
            "final_recommendation": {"translation": "stub reading"},
        }

    def render_markdown(self, dossier):
        """Render a stable Markdown line for CLI text assertions."""

        return f"# Agentic Translation Dossier: {dossier['request']['text']}\n"

    def compact_dossier(self, dossier):
        """Mirror the production summary hook used by JSON CLI output."""

        return dossier


def _service() -> AgenticTranslationService:
    """Build a harness fixture with canon and non-canon dictionary metadata."""

    return AgenticTranslationService(
        word_service=StubWordService(),
        phrase_service=StubPhraseService(),
        raw_dictionary_entries=[
            {
                "word": "NAZPSAD",
                "normalized": "nazpsad",
                "canon_word": True,
                "definition": "sword",
            },
            {
                "word": "XNON",
                "normalized": "xnon",
                "canon_word": False,
                "definition": "non-canon fixture",
            },
        ],
    )


def test_agentic_service_preserves_source_specific_baselines() -> None:
    """Keep solo/debate baselines separate in the public dossier contract."""

    dossier = _service().build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="both",
        top_k=2,
        exploration_budget=4,
    )

    assert set(dossier["baseline_by_source"]) == {"solo", "debate"}
    assert dossier["source_comparison"]["disagreements"]
    assert any(
        packet["analysis_source"] == "solo"
        for packet in dossier["evidence_packets"]
    )
    assert any(
        exploration["trigger"] == "dictionary-vs-compositional conflict"
        for exploration in dossier["otherwise_explorations"]
    )
    assert dossier["translation_summary"]["final_definition"] == "cutting weapon"
    assert dossier["final_recommendation"]["translation"] == "cutting weapon"
    considerations = dossier["translation_summary"]["root_considerations"]
    considered_morphs = {row["morph"] for row in considerations}
    assert {"NAZPSAD", "NAZ", "PS", "AD"}.issubset(considered_morphs)
    ps = next(row for row in considerations if row["morph"] == "PS")
    ad = next(row for row in considerations if row["morph"] == "AD")
    assert ps["candidate_role"] == "alternate_candidate"
    assert ps["alternate_definitions"][0]["definition"] == "divisive"
    assert ad["alternate_definitions"][0]["definition"] == "power"
    assert ad["definition_counts"] == {"AD": 60}


def test_agentic_root_considerations_dedupe_repeated_morphs() -> None:
    """Repeated morphs across top candidates should aggregate, not spam rows."""

    dossier = _service().build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="solo",
        top_k=3,
        exploration_budget=4,
    )

    considerations = dossier["translation_summary"]["root_considerations"]
    ps_rows = [row for row in considerations if row["morph"] == "PS"]
    ad_rows = [row for row in considerations if row["morph"] == "AD"]

    assert len(ps_rows) == 1
    assert len(ad_rows) == 1
    assert len(ps_rows[0]["candidate_occurrences"]) == 2


def test_agentic_markdown_is_bounded_but_shows_roots_considered() -> None:
    """Default text output should answer the root-tuning questions directly."""

    service = _service()
    dossier = service.build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="solo",
        top_k=2,
        exploration_budget=4,
    )

    rendered = service.render_markdown(dossier)

    assert "## Final Definition" in rendered
    assert "## Etymological Breakdown" in rendered
    assert "## Roots Considered" in rendered
    assert "`PS`: celestial [solo rank 2; alternate_candidate]" in rendered
    assert "Other possibilities: divisive" in rendered
    assert "`AD`: divine justice [solo rank 2; alternate_candidate]" in rendered


def test_agentic_markdown_shows_exploratory_hypotheses() -> None:
    """Speculative branches should be readable without replacing the winner."""

    service = _service()
    dossier = service.build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="solo",
        top_k=2,
        exploration_budget=4,
        speculative=True,
    )

    rendered = service.render_markdown(dossier)

    assert "## Exploratory Hypotheses" in rendered
    assert "`NAZ + P + SA + D`" in rendered
    assert "pillars separation distinction disturbance" in rendered
    assert "Head/modifiers: head=NAZ; modifiers=P+SA+D" in rendered
    assert "Distributed motif roots: 3" in rendered
    assert "sword-like artifact" in rendered
    assert dossier["translation_summary"]["final_definition"] == "cutting weapon"


def test_agentic_compact_json_summarizes_trace_instead_of_dumping_it() -> None:
    """Summary JSON should stay small while preserving tool-use accounting."""

    service = _service()
    dossier = service.build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="solo",
        top_k=2,
        exploration_budget=4,
    )

    compact = service.compact_dossier(dossier)

    assert "research_trace" not in compact
    assert compact["research_trace_summary"]["event_count"] > 0


def test_agentic_service_labels_non_canon_dictionary_support() -> None:
    """Expose solo/non-canon support instead of treating it as canon evidence."""

    dossier = _service().build_dossier(
        "XNON",
        kind="word",
        analysis_source="solo",
        exploration_budget=2,
    )

    dictionary_packets = [
        packet
        for packet in dossier["evidence_packets"]
        if packet["packet_type"] == "dictionary_entry"
    ]
    assert dictionary_packets
    assert dictionary_packets[0]["canon_status"] == "non_canon"


def test_agentic_service_suppresses_dictionary_packets_and_labels_when_disabled() -> None:
    """Blind dossiers must not leak dictionary packets or dictionary labels."""

    service = _service()
    dossier = service.build_dossier(
        "NAZPSAD",
        kind="word",
        analysis_source="solo",
        exploration_budget=2,
        allow_dictionary=False,
    )

    assert dossier["request"]["dictionary_enabled"] is False
    assert dossier["request"]["blind_mode_enabled"] is True
    assert service.word_service.calls[0]["allow_dictionary"] is False
    assert service.word_service.calls[0]["dictionary_snapshot"] == {}
    assert service.word_service.candidate_finder.dictionary == {
        "nazpsad": "dictionary-entry"
    }
    assert not any(
        packet["packet_type"] == "dictionary_entry"
        for packet in dossier["evidence_packets"]
    )
    assert any(
        packet.get("canon_status") == "unclassified_without_dictionary"
        for packet in dossier["evidence_packets"]
        if packet["packet_type"] != "baseline_summary"
    )


def test_agentic_service_supports_phrase_kind_with_stubbed_phrase_service() -> None:
    """Route phrase dossiers through the phrase baseline tool."""

    dossier = _service().build_dossier(
        "OL SONF",
        kind="phrase",
        analysis_source="debate",
        exploration_budget=1,
    )

    assert list(dossier["baseline_by_source"]) == ["debate"]
    assert dossier["hypotheses"][0]["analysis_source"] == "debate"


def test_agentic_service_passes_blind_flags_to_phrase_baseline() -> None:
    """Phrase dossiers should reuse the same blind controls as word dossiers."""

    service = _service()
    dossier = service.build_dossier(
        "OL SONF",
        kind="phrase",
        analysis_source="debate",
        allow_dictionary=False,
        allow_whole_word=False,
    )

    assert dossier["request"]["whole_word_enabled"] is False
    assert dossier["request"]["blind_mode_enabled"] is True
    assert service.phrase_service.calls[0]["allow_dictionary"] is False
    assert service.phrase_service.calls[0]["allow_whole_word"] is False


def test_translation_cli_registers_agentic_translate_command() -> None:
    """Expose agentic-translate through the translation CLI parser."""

    parser = translation_cli.build_parser()
    args = parser.parse_args(
        [
            "agentic-translate",
            "NAZPSAD",
            "--analysis-source",
            "solo",
            "--no-dictionary",
            "--no-whole-words",
        ]
    )

    assert args.command == "agentic-translate"
    assert args.text == "NAZPSAD"
    assert args.analysis_source == "solo"
    assert args.allow_dictionary is False
    assert args.allow_whole_word is False
    assert args.handler == translation_cli._run_agentic_translate


def test_translation_cli_accepts_agentic_markdown_format_alias() -> None:
    """Make the Markdown renderer discoverable as an explicit format."""

    parser = translation_cli.build_parser()
    args = parser.parse_args(
        ["agentic-translate", "NAZPSAD", "--format", "markdown"]
    )

    assert args.format == "markdown"


def test_agentic_cli_accepts_speculative_toggle() -> None:
    """Expose speculative exploration controls on the dossier command."""

    parser = translation_cli.build_parser()
    args = parser.parse_args(
        [
            "agentic-translate",
            "NAZPSAD",
            "--no-speculative",
            "--translation-profile",
            "separation_artifact",
        ]
    )

    assert args.speculative is False
    assert args.translation_profile == "separation_artifact"


def test_agentic_translate_from_args_renders_markdown(monkeypatch, capsys) -> None:
    """Render Markdown-oriented text without opening real databases in CLI tests."""

    monkeypatch.setattr(translation_cli, "_missing_db_paths", lambda _variants: [])
    monkeypatch.setattr(
        translation_cli.AgenticTranslationService,
        "from_config",
        classmethod(lambda cls, **_kwargs: StubAgenticService()),
    )

    args = translation_cli.build_parser().parse_args(
        ["agentic-translate", "NAZPSAD", "--kind", "word", "--analysis-source", "solo"]
    )
    exit_code = translation_cli.agentic_translate_from_args(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Agentic Translation Dossier: NAZPSAD" in captured.out


def test_agentic_translate_from_args_renders_json(monkeypatch, capsys) -> None:
    """Keep JSON output available as the stable machine-readable contract."""

    monkeypatch.setattr(translation_cli, "_missing_db_paths", lambda _variants: [])
    monkeypatch.setattr(
        translation_cli.AgenticTranslationService,
        "from_config",
        classmethod(lambda cls, **_kwargs: StubAgenticService()),
    )

    args = translation_cli.build_parser().parse_args(
        ["agentic-translate", "NAZPSAD", "--format", "json", "--pretty"]
    )
    exit_code = translation_cli.agentic_translate_from_args(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"final_recommendation"' in captured.out
    assert '"translation": "stub reading"' in captured.out


def test_agentic_translate_output_file_prints_done(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    """Confirm successful file output with a short human-facing notice."""

    monkeypatch.setattr(translation_cli, "_missing_db_paths", lambda _variants: [])
    monkeypatch.setattr(
        translation_cli.AgenticTranslationService,
        "from_config",
        classmethod(lambda cls, **_kwargs: StubAgenticService()),
    )

    output_path = tmp_path / "dossier.md"
    args = translation_cli.build_parser().parse_args(
        [
            "agentic-translate",
            "NAZPSAD",
            "--output",
            str(output_path),
        ]
    )
    exit_code = translation_cli.agentic_translate_from_args(args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8").startswith(
        "# Agentic Translation Dossier: NAZPSAD"
    )
    assert captured.out == f"Done. Wrote dossier to {output_path}\n"


def test_enlm_parser_registers_agentic_translate_command() -> None:
    """Expose the dossier command through `poetry run enlm agentic-translate`."""

    from enochian_lm.analysis import cli as enlm_cli

    args = enlm_cli._build_parser().parse_args(
        ["agentic-translate", "NAZPSAD", "--analysis-source", "debate"]
    )

    assert args.command == "agentic-translate"
    assert args.analysis_source == "debate"
