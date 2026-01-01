from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

# Shim external dependency used by candidate_finder without requiring heavy install
gensim_module = types.ModuleType("gensim")
gensim_utils = types.ModuleType("gensim.utils")
gensim_utils.simple_preprocess = lambda text: [str(text)]  # type: ignore[attr-defined]
gensim_models = types.ModuleType("gensim.models")


class _DummyFastText:  # pragma: no cover - simple import shim
    def __init__(self, *args, **kwargs):
        self.wv = self

    def get_vector(self, _token: str):
        return np.zeros(4)


gensim_models.FastText = _DummyFastText  # type: ignore[attr-defined]
gensim_module.utils = gensim_utils  # type: ignore[attr-defined]
gensim_module.models = gensim_models  # type: ignore[attr-defined]
sys.modules.setdefault("gensim", gensim_module)
sys.modules.setdefault("gensim.utils", gensim_utils)
sys.modules.setdefault("gensim.models", gensim_models)

sentence_module = types.ModuleType("sentence_transformers")


class _DummySentenceTransformer:  # pragma: no cover - simple import shim
    def __init__(self, *args, **kwargs):
        pass


sentence_module.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
sys.modules.setdefault("sentence_transformers", sentence_module)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord


class DummyVectors(dict):
    vector_size = 4

    def similar_by_word(self, _word: str, topn: int = 10):
        return []

    def __contains__(self, key: object) -> bool:  # pragma: no cover - simple shim
        return True

    def __getitem__(self, key: object):  # pragma: no cover - simple shim
        return np.zeros(self.vector_size)


class DummyFasttext:
    def __init__(self) -> None:
        self.wv = DummyVectors()


@pytest.fixture()
def monkeypatched_fasttext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "enochian_lm.root_extraction.utils.candidate_finder.get_fasttext_model",
        lambda model_path=None: DummyFasttext(),
    )


def _write_ngram_index(tmp_path: Path, tokens: Iterable[str]) -> Path:
    db_path = tmp_path / "ngram_index.sqlite3"
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE ngrams (ngram TEXT PRIMARY KEY, total_occurrences INTEGER)"
        )
        conn.execute("CREATE TABLE ngram_membership (ngram TEXT, canonical TEXT)")
        for token in tokens:
            norm = token.lower()
            conn.execute(
                "INSERT INTO ngrams (ngram, total_occurrences) VALUES (?, ?)",
                (norm, 3),
            )
            conn.execute(
                "INSERT INTO ngram_membership (ngram, canonical) VALUES (?, ?)",
                (norm, token.upper()),
            )
    conn.close()
    return db_path


def _build_finder(
    tmp_path: Path,
    tokens: Iterable[str],
    *,
    min_n: int,
    max_n: int = 7,
    beam_width: int = 5,
) -> MorphemeCandidateFinder:
    ngram_index = _write_ngram_index(tmp_path, tokens)
    dictionary_entries: list[EntryRecord] = [
        {"canonical": token.upper(), "alternates": []} for token in tokens
    ]
    return MorphemeCandidateFinder(
        ngram_db_path=ngram_index,
        fasttext_model_path=ngram_index,  # path unused due to monkeypatch
        dictionary_entries=dictionary_entries,
        min_n=min_n,
        max_n=max_n,
        beam_width=beam_width,
    )


def test_nazpsad_prefers_morpheme_splits(
    tmp_path: Path, monkeypatched_fasttext: None
) -> None:
    tokens = ["NAZ", "PSAD", "PS", "AD", "NAZP", "SAD"]
    finder = _build_finder(tmp_path, tokens, min_n=2)

    parses = finder.segment_target("NAZPSAD")
    assert parses
    best_path = parses[0][0]
    assert best_path in (["NAZ", "PSAD"], ["NAZ", "PS", "AD"])


def test_debuheka_does_not_invent_unknown_morphemes(
    tmp_path: Path, monkeypatched_fasttext: None
) -> None:
    tokens = ["DE", "BU", "HE", "KA"]
    finder = _build_finder(tmp_path, tokens, min_n=2)

    parses = finder.segment_target("DEBUHEKA")
    assert parses
    for path, *_ in parses:
        assert "DEBU" not in path
        assert "HEKA" not in path
        assert all(segment in tokens for segment in path)


def test_whole_word_match_allowed_when_known(
    tmp_path: Path, monkeypatched_fasttext: None
) -> None:
    tokens = ["OD"]
    finder = _build_finder(tmp_path, tokens, min_n=2)

    parses = finder.segment_target("OD")
    assert parses
    path, _, _, coverage = parses[0]
    breakdown = finder._build_breakdown("OD", coverage)
    assert path == ["OD"]
    assert breakdown["coverage_ratio"] == 1.0
    assert breakdown["residual_ratio"] == 0.0


def test_single_letter_i_l_allowed_in_fallback(
    tmp_path: Path, monkeypatched_fasttext: None
) -> None:
    tokens = ["I", "L"]
    finder = _build_finder(tmp_path, tokens, min_n=1, max_n=1)

    parses = finder.segment_target("IL")
    assert parses
    best_path = parses[0][0]
    assert best_path == ["I", "L"]
