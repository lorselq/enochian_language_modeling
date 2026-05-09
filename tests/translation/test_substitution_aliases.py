from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_dependency_shims() -> None:
    """Install tiny stand-ins for optional imports used by translation modules.

    These tests exercise substitution alias orchestration, not vector search or
    model loading. The shims keep the unit tests focused and runnable in a
    lightweight environment where the optional NLP stack may be unavailable.
    """

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
        """Provide the minimal FastText API touched during module import."""

        def __init__(self, *args, **kwargs) -> None:
            self.wv = self

        @classmethod
        def load(cls, _path: str) -> "_DummyFastText":
            return cls()

    gensim_models.FastText = _DummyFastText  # type: ignore[attr-defined]
    gensim_module.utils = gensim_utils  # type: ignore[attr-defined]
    gensim_module.models = gensim_models  # type: ignore[attr-defined]

    sentence_module = sys.modules.setdefault(
        "sentence_transformers",
        types.ModuleType("sentence_transformers"),
    )

    class _DummySentenceTransformer:
        """Stand in for sentence-transformers when imported indirectly."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    sentence_module.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
    sentence_module.util = getattr(
        sentence_module,
        "util",
        types.SimpleNamespace(cos_sim=lambda *_args, **_kwargs: [[1.0]]),
    )


_install_dependency_shims()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from translation.repository import ClusterRecord, WordEvidence
from translation.service import SingleWordTranslationService
from translation.strategies import extract_definition_candidates
from translation.substitution_aliases import (
    LookupSubstitutionRule,
    build_lookup_aliases,
    build_lookup_variants,
    collect_lookup_substitution_rules,
)


def test_collect_lookup_substitution_rules_supports_directional_expansions() -> None:
    """The parser should include directional and multi-letter substitutions."""

    payload = {
        "y": {
            "canonical": "y",
            "alternates": [
                {"value": "i", "direction": "from"},
                {"value": "ee", "direction": "both"},
                {"value": "z", "direction": "to"},
            ],
        },
        "q": {
            "canonical": "q",
            "alternates": [{"value": "qu", "direction": "both"}],
        },
        "u": {
            "canonical": "u",
            "alternates": [{"value": "v", "direction": "both"}],
        },
    }

    rules = collect_lookup_substitution_rules(payload)

    rule_pairs = {(rule.source, rule.target) for rule in rules}
    assert ("Y", "I") in rule_pairs
    assert ("Y", "EE") in rule_pairs
    assert ("EE", "Y") in rule_pairs
    assert ("Z", "Y") in rule_pairs
    assert ("Q", "QU") in rule_pairs
    assert ("QU", "Q") in rule_pairs
    assert ("U", "V") in rule_pairs
    assert ("V", "U") in rule_pairs


def test_build_lookup_variants_applies_all_configured_letters() -> None:
    """Variant generation should substitute every applicable letter position."""

    rules = (
        LookupSubstitutionRule(source="Y", target="I"),
        LookupSubstitutionRule(source="U", target="V"),
    )

    assert build_lookup_variants("YU", rules) == ("YU", "IU", "IV", "YV")


def test_build_lookup_variants_handles_bidirectional_multi_letter_rule() -> None:
    """Lookup variants should expand and collapse multi-letter substitutions."""

    rules = (
        LookupSubstitutionRule(source="Q", target="QU"),
        LookupSubstitutionRule(source="QU", target="Q"),
    )

    assert build_lookup_variants("Q", rules) == ("Q", "QU")
    assert build_lookup_variants("QURLST", rules) == ("QURLST", "QRLST")


def test_build_lookup_aliases_maps_lookup_forms_to_surface_spans() -> None:
    """Alias mapping should let one fetched lookup form support a surface span."""

    rules = (LookupSubstitutionRule(source="Y", target="I"),)

    aliases = build_lookup_aliases(["Y", "LS", "YLS"], rules)

    assert aliases == {"I": {"Y"}, "ILS": {"YLS"}}


def test_service_clones_lookup_support_onto_surface_morph() -> None:
    """Support cloned from a lookup form should become usable surface evidence."""

    service = SingleWordTranslationService.__new__(SingleWordTranslationService)
    service._lookup_substitution_rules = (
        LookupSubstitutionRule(source="Y", target="I"),
    )
    evidence = WordEvidence(word="YLS", variants_queried=["solo"])
    lookup_cluster = _cluster_record("I", "identity; self")

    aliases = service._substitution_aliases_for_substrings(["Y", "LS"])
    alias_clusters, alias_residuals, alias_hypotheses = (
        service._alias_support_records_for_substrings(
            aliases,
            [lookup_cluster],
            [],
            [],
        )
    )
    service._merge_support_evidence(
        evidence,
        alias_clusters,
        alias_residuals,
        alias_hypotheses,
    )

    candidates = extract_definition_candidates(["Y"], evidence)

    assert [cluster.ngram for cluster in evidence.direct_clusters] == ["Y"]
    assert candidates["Y"][0]["raw_definition"] == "identity; self"


def test_service_exposes_dictionary_lookup_aliases_on_surface_morph() -> None:
    """Dictionary aliases should preserve evidence while using the surface key."""

    evidence = WordEvidence(word="YLS", variants_queried=["solo"])
    dictionary_entries = {
        "i": {
            "canonical": "i",
            "definition": "I, self, me",
            "senses": [],
        }
    }

    SingleWordTranslationService._merge_substitution_dictionary_aliases(
        evidence,
        {"I": {"Y"}},
        dictionary_entries,
    )

    assert evidence.dictionary_morphs["Y"].morph == "Y"
    assert evidence.dictionary_morphs["Y"].definition == "I, self, me"


def test_service_multi_letter_alias_clones_lookup_support() -> None:
    """Surface spans should gain support from multi-letter lookup aliases."""

    service = SingleWordTranslationService.__new__(SingleWordTranslationService)
    service._lookup_substitution_rules = (
        LookupSubstitutionRule(source="QU", target="Q"),
    )
    evidence = WordEvidence(word="QU", variants_queried=["solo"])
    lookup_cluster = _cluster_record("Q", "vibrating force")

    aliases = service._substitution_aliases_for_substrings(["QU"])
    alias_clusters, alias_residuals, alias_hypotheses = (
        service._alias_support_records_for_substrings(
            aliases,
            [lookup_cluster],
            [],
            [],
        )
    )
    service._merge_support_evidence(
        evidence,
        alias_clusters,
        alias_residuals,
        alias_hypotheses,
    )

    candidates = extract_definition_candidates(["QU"], evidence)

    assert [cluster.ngram for cluster in evidence.direct_clusters] == ["QU"]
    assert candidates["QU"][0]["raw_definition"] == "vibrating force"


def _cluster_record(ngram: str, definition: str) -> ClusterRecord:
    """Build the minimal accepted cluster payload needed by strategy tests."""

    return ClusterRecord(
        variant="solo",
        cluster_id=1,
        run_id="test",
        ngram=ngram,
        cluster_index=0,
        glossator_def=f'{{"definition": "{definition}"}}',
        residual_explained=1.0,
        residual_ratio=0.0,
        residual_headline=None,
        residual_focus_prompt=None,
        semantic_coverage=1.0,
        cohesion=1.0,
        semantic_cohesion=1.0,
        best_config=None,
    )
