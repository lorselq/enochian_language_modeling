import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.datasets import enrich_dictionary_pos as module


class MockCitationTagger(module.CitationTagger):
    def tag_phrase(self, phrase: str) -> list[str]:
        return ["VERB"] if phrase else []


def test_citation_pos_overrides_default_noun():
    entry = {
        "word": "MOVE",
        "senses": [
            {
                "sense_id": "1",
                "definition": "move",
                "key_citations": [
                    {"context": "... *move* therefore ..."},
                ],
            }
        ],
    }
    domain_config = module.DomainConfig(
        domains={}, headword_to_domains={}, headword_stopwords=set()
    )

    module.enrich_senses(entry, domain_config, citation_tagger=MockCitationTagger())

    sense = entry["senses"][0]
    assert sense["parts_of_speech"][0] == "VERB"
    assert "citation:VERB" in (sense["notes_pos"] or "")


def _require_wordnet():
    nltk = pytest.importorskip("nltk")
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        pytest.skip("WordNet corpus is not available. Run `python -m nltk.downloader wordnet`.")
    return wn


def test_wordnet_synonym_infers_domain():
    _require_wordnet()
    domain_config = module.DomainConfig(
        domains={"SOCIAL": "people"},
        headword_to_domains={"monarch": ["SOCIAL"]},
        headword_stopwords=set(),
    )

    result, notes = domain_config.lookup("sovereigns", heuristic_pos=["NOUN"])

    assert result == ["SOCIAL"]
    assert notes is None


def test_wordnet_lexname_mapping_used_when_no_synonym_match():
    _require_wordnet()
    domain_config = module.DomainConfig(
        domains={"SOCIAL": "people"},
        headword_to_domains={},
        headword_stopwords=set(),
        wordnet_lexname_to_domains={"noun.person": ["SOCIAL"]},
    )

    result, notes = domain_config.lookup("astronaut", heuristic_pos=["NOUN"])

    assert result == ["SOCIAL"]
    assert notes == "wordnet_lexname"


def test_wordnet_angelic_indicators_promote_divine_domains():
    _require_wordnet()
    domain_config = module.DomainConfig(
        domains={
            "DIVINE": "sacred",
            "CELESTIAL": "sky",
            "SOCIAL": "people",
        },
        headword_to_domains={},
        headword_stopwords=set(),
        wordnet_lexname_to_domains={"noun.person": ["SOCIAL"]},
        angelic_lexnames={"noun.person"},
        sacred_indicators={"angel", "cherub"},
    )

    result, notes = domain_config.lookup("angel", heuristic_pos=["NOUN"])

    assert result == ["DIVINE", "CELESTIAL", "SOCIAL"]
    assert notes == "wordnet_lexname"


def test_wordnet_gloss_similarity_used_when_enabled():
    _require_wordnet()
    domain_config = module.DomainConfig(
        domains={"SPACE": "space travel cosmos", "ACTION": "movement work"},
        headword_to_domains={},
        headword_stopwords=set(),
        wordnet_lexname_to_domains={},
        use_wordnet_gloss_similarity=True,
        wordnet_gloss_similarity_threshold=0.05,
    )

    result, notes = domain_config.lookup("astronaut", heuristic_pos=["NOUN"])

    assert result == ["SPACE"]
    assert notes and notes.startswith("wordnet_gloss:")
