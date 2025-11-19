import pathlib
import sys

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
    domain_config = module.DomainConfig(domains={}, headword_to_domains={}, headword_stopwords=set())

    module.enrich_senses(entry, domain_config, citation_tagger=MockCitationTagger())

    sense = entry["senses"][0]
    assert sense["parts_of_speech"][0] == "VERB"
    assert "citation:VERB" in (sense["notes_pos"] or "")
