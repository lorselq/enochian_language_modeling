from __future__ import annotations

import json

from enochian_lm.analysis.find_ngram import (
    collect_citations,
    find_ngram_matches,
    format_ngram_matches_json,
    format_ngram_matches_text,
    normalize_ngram,
)


def _sample_entries() -> list[dict]:
    """Provide compact dictionary records for ngram lookup tests.

    Why:
    the production dictionary is large and mutable, while these tests need a
    tiny fixture that covers canon flags, multiple senses, citations,
    alternates, and verbose metadata.

    How:
    return raw dictionary-shaped dictionaries that mirror the fields consumed by
    the lookup helper.

    Responsibility:
    keep each behavior test focused on lookup logic rather than fixture setup.
    """

    return [
        {
            "word": "ABBA",
            "normalized": "abba",
            "canon_word": True,
            "definition": "father",
            "senses": [
                {
                    "sense_id": 1,
                    "definition": "father",
                    "key_citations": [
                        {"location": "1.1", "context": "...*father*..."}
                    ],
                },
                {"sense_id": 2, "definition": "source"},
            ],
            "key_citations": [{"location": "1.1", "context": "...*father*..."}],
            "commentary": "kinship note",
            "alternates": [{"value": "AVVA", "confidence": 0.8}],
            "pos": "NOUN",
            "context_tags": ["kinship"],
        },
        {
            "word": "CAB",
            "normalized": "cab",
            "canon_word": False,
            "senses": [{"sense_id": 1, "definition": "non-canon container"}],
        },
        {
            "word": "MASSING",
            "normalized": "massing",
            "senses": [{"sense_id": 1, "definition": "missing canon flag"}],
        },
    ]


def test_normalize_ngram_keeps_letters_only() -> None:
    """Verify the CLI's promised query normalization.

    Why:
    users may paste ngrams with punctuation or spaces, and the command should
    still perform a letter-only lookup.

    How:
    normalize a mixed string and assert only lowercase letters remain.

    Responsibility:
    lock the search normalization contract used by all lookup paths.
    """

    assert normalize_ngram(" A-B 1! ") == "ab"


def test_find_ngram_filters_to_explicit_canon_entries() -> None:
    """Confirm `--canon-only` keeps only `canon_word: true` records.

    Why:
    entries missing `canon_word` should not accidentally become canon in
    analysis output.

    How:
    search for ngrams that hit explicit canon, explicit non-canon, and missing
    canon-flag records.

    Responsibility:
    protect the command's canon filtering semantics.
    """

    all_matches = find_ngram_matches(_sample_entries(), "ab")
    canon_matches = find_ngram_matches(_sample_entries(), "ab", canon_only=True)
    missing_flag_matches = find_ngram_matches(
        _sample_entries(), "ss", canon_only=True
    )

    assert [match["word"] for match in all_matches] == ["ABBA", "CAB"]
    assert [match["word"] for match in canon_matches] == ["ABBA"]
    assert missing_flag_matches == []


def test_find_ngram_returns_all_senses_for_matching_word() -> None:
    """Ensure multi-sense words are not collapsed.

    Why:
    dictionary analysis depends on seeing every sense attached to a matching
    word, not just the top-level definition.

    How:
    match a two-sense fixture and inspect the serialized match record.

    Responsibility:
    keep the command faithful to dictionary sense inventory.
    """

    matches = find_ngram_matches(_sample_entries(), "bb")

    assert len(matches) == 1
    assert [sense["definition"] for sense in matches[0]["senses"]] == [
        "father",
        "source",
    ]


def test_collect_citations_deduplicates_entry_and_sense_evidence() -> None:
    """Check citation gathering across entry and sense levels.

    Why:
    the dictionary can repeat the same citation in both places, and duplicated
    evidence would make `--citations` noisy.

    How:
    collect citations from a fixture with the same citation at both levels.

    Responsibility:
    keep citation output concise without losing first-seen evidence.
    """

    match = find_ngram_matches(_sample_entries(), "bb")[0]

    assert collect_citations(match) == [
        {"location": "1.1", "context": "...*father*..."}
    ]


def test_text_output_includes_verbose_metadata_and_citations() -> None:
    """Render readable text with optional metadata enabled.

    Why:
    `--verbose --citations` is the richest human-facing output path and should
    expose match surfaces, alternates, commentary, and evidence.

    How:
    format one match with both flags enabled and assert representative lines.

    Responsibility:
    guard the default text renderer against dropping analysis metadata.
    """

    matches = find_ngram_matches(_sample_entries(), "bb", include_alternates=True)
    output = format_ngram_matches_text(
        matches, query="B-B", citations=True, verbose=True
    )

    assert 'Matches for "B-B" (normalized: "bb"): 1' in output
    assert "ABBA [canon]" in output
    assert "normalized: abba" in output
    assert "match: canonical abba at [1]" in output
    assert "alternates: AVVA" in output
    assert "commentary: kinship note" in output
    assert "1. father" in output
    assert "2. source" in output
    assert "1.1: ...*father*..." in output


def test_include_alternates_expands_matching_surfaces() -> None:
    """Verify alternate spellings are opt-in search surfaces.

    Why:
    default searches should match canonical dictionary words, while
    `--include-alternates` should explain alternate-surface hits.

    How:
    search for an ngram that appears only in the alternate spelling.

    Responsibility:
    protect the distinction between canonical and alternate matching.
    """

    assert find_ngram_matches(_sample_entries(), "vv") == []

    matches = find_ngram_matches(_sample_entries(), "vv", include_alternates=True)

    assert [match["word"] for match in matches] == ["ABBA"]
    assert matches[0]["matched_surfaces"][0]["kind"] == "alternate"
    assert matches[0]["matched_surfaces"][0]["positions"] == [1]


def test_json_output_wraps_query_and_matches() -> None:
    """Confirm machine-readable output contains the lookup envelope.

    Why:
    downstream analysis scripts need a predictable JSON shape instead of parsing
    text output.

    How:
    render a match payload and parse it back into Python data.

    Responsibility:
    lock the JSON contract for automation users.
    """

    matches = find_ngram_matches(_sample_entries(), "bb")
    payload = json.loads(format_ngram_matches_json(matches, query="B-B"))

    assert payload["query"] == "B-B"
    assert payload["normalized_query"] == "bb"
    assert payload["match_count"] == 1
    assert payload["matches"][0]["word"] == "ABBA"
