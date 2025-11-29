from enochian_lm.analysis.utils.token_morphs import (
    fetch_token_morphs,
    fetch_token_segments,
    summarize_token_decomp,
    tokens_with_morph,
)
from enochian_lm.common.sqlite_bootstrap import sqlite3


def test_token_morph_helpers_roundtrip():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE token_morph_decomp (
          run_id      TEXT NOT NULL,
          token       TEXT NOT NULL,
          seg_index   INTEGER NOT NULL,
          morph       TEXT NOT NULL,
          span_start  INTEGER NOT NULL,
          span_end    INTEGER NOT NULL,
          score       REAL,
          source      TEXT,
          PRIMARY KEY (run_id, token, seg_index)
        );
        """
    )
    rows = [
        ("demo", "nazpsad", 0, "naz", 0, 3, 0.5, "seg_v1"),
        ("demo", "nazpsad", 1, "ps", 3, 5, 0.4, "seg_v1"),
        ("demo", "nazpsad", 2, "ad", 5, 7, 0.8, "seg_v1"),
    ]
    conn.executemany(
        "INSERT INTO token_morph_decomp VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )

    morphs = fetch_token_morphs(conn, "nazpsad", run_id="demo")
    assert morphs == ["naz", "ps", "ad"]

    segments = fetch_token_segments(conn, "nazpsad", run_id="demo")
    assert segments[0]["span"] == [0, 3]
    assert segments[-1]["score"] == 0.8

    tokens = tokens_with_morph(conn, "ps", run_id="demo")
    assert tokens == ["nazpsad"]

    summary = summarize_token_decomp(conn, run_id="demo")
    assert summary == {"zero": 0, "single": 0, "multi": 1, "tokens": 1}
