from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.datasets.generate_canon_translation_profiles import generate_profiles


def test_generate_profiles_uses_raw_canon_set_and_optional_enriched_metadata(
    tmp_path: Path,
) -> None:
    dictionary_path = tmp_path / "dictionary.json"
    enriched_path = tmp_path / "dictionary_enriched.json"
    dictionary_path.write_text(
        json.dumps(
            [
                {
                    "word": "NAZ",
                    "normalized": "naz",
                    "canon_word": True,
                    "definition": "pillars",
                    "senses": [{"definition": "pillars"}],
                },
                {
                    "word": "PSAD",
                    "normalized": "psad",
                    "canon_word": False,
                    "definition": "sword",
                },
                {
                    "word": "D",
                    "normalized": "d",
                    "canon_word": True,
                    "definition": "force",
                    "senses": [{"definition": "force"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    enriched_path.write_text(
        json.dumps(
            [
                {
                    "word": "NAZ",
                    "normalized": "naz",
                    "canon_word": True,
                    "definition": "pillars",
                    "senses": [
                        {
                            "definition": "pillars",
                            "parts_of_speech": ["NOUN"],
                            "semantic_domains": ["PHYSICAL"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    payload = generate_profiles(
        dictionary_path=dictionary_path,
        enriched_path=enriched_path,
    )
    profiles = payload["canon_dictionary_profiles"]

    assert payload["metadata"]["canon_profile_count"] == 2
    assert payload["metadata"]["missing_enriched_count"] == 1
    assert sorted(profiles) == ["D", "NAZ"]
    assert profiles["NAZ"]["parts_of_speech"] == ["NOUN"]
    assert profiles["NAZ"]["semantic_domains"] == ["PHYSICAL"]
    assert profiles["D"]["enriched"] is False
