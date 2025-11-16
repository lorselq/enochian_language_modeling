import json
from pathlib import Path

from enochian_translation_team.utils.preanalysis import load_trusted_ngrams


def _write_dictionary(tmp_path: Path, data: list[dict]) -> Path:
    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(json.dumps(data))
    return dictionary_path


def test_load_trusted_ngrams_falls_back_to_dictionary(tmp_path: Path) -> None:
    dictionary_path = _write_dictionary(
        tmp_path,
        [
            {"canonical": "naz", "alternates": [{"value": "nazaz"}]},
            {"canonical": "ab"},
        ],
    )

    trusted_path = tmp_path / "trusted.json"
    result = load_trusted_ngrams(trusted_path, dictionary_path=dictionary_path)

    assert result == ["AB", "NAZ", "NAZAZ"]


def test_load_trusted_ngrams_honors_max_length(tmp_path: Path) -> None:
    dictionary_path = _write_dictionary(
        tmp_path,
        [
            {"canonical": "naz", "alternates": [{"value": "nazaz"}]},
            {"canonical": "chi"},
            {"canonical": "longer"},
        ],
    )

    trusted_path = tmp_path / "trusted.json"
    result = load_trusted_ngrams(
        trusted_path,
        dictionary_path=dictionary_path,
        max_length=3,
    )

    assert result == ["CHI", "NAZ"]
