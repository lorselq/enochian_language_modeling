import json
import sqlite3
from pathlib import Path

from enochian_translation_team.utils.preanalysis import load_trusted_ngrams


def _write_dictionary(tmp_path: Path, data: list[dict]) -> Path:
    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(json.dumps(data))
    return dictionary_path


def _write_ngram_index(tmp_path: Path, tokens: list[str]) -> Path:
    db_path = tmp_path / "ngrams.sqlite3"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("CREATE TABLE ngrams (ngram TEXT PRIMARY KEY, total_occurrences INTEGER)")
        for token in tokens:
            conn.execute(
                "INSERT INTO ngrams (ngram, total_occurrences) VALUES (?, ?)",
                (token.lower(), 1),
            )
    conn.close()
    return db_path


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


def test_load_trusted_ngrams_merges_dictionary_and_manual(tmp_path: Path) -> None:
    dictionary_path = _write_dictionary(
        tmp_path,
        [
            {"canonical": "ipam"},
            {"canonical": "ita", "normalized": "ita"},
        ],
    )
    ngram_index = _write_ngram_index(tmp_path, ["ipam", "ita"])

    trusted_path = tmp_path / "trusted.json"
    trusted_path.write_text(json.dumps(["manual"]))

    result = load_trusted_ngrams(
        trusted_path,
        dictionary_path=dictionary_path,
        ngram_index_path=ngram_index,
    )

    assert result == ["IPAM", "ITA", "MANUAL"]
