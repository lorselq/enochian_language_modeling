from __future__ import annotations

import json

from enochian_lm.analysis import find_ngram_cli


def test_standalone_parser_accepts_requested_options() -> None:
    """Verify the lightweight top-level parser exposes lookup options.

    Why:
    `poetry run find-ngram` points at a standalone module so it can avoid heavy
    analytics imports, and that parser must still match the public command
    contract.

    How:
    parse a representative invocation and inspect the resulting namespace.

    Responsibility:
    protect the direct console-script interface.
    """

    parser = find_ngram_cli.build_parser()
    args = parser.parse_args(
        ["A-B", "--verbose", "--citations", "--canon-only", "--format", "json"]
    )

    assert args.ngram == "A-B"
    assert args.verbose is True
    assert args.citations is True
    assert args.canon_only is True
    assert args.format == "json"


def test_standalone_main_writes_output_file(tmp_path, capsys) -> None:
    """Exercise the installed script target's output path.

    Why:
    the Poetry entry point now targets this lightweight module directly, so its
    runtime path needs coverage independent of the larger analysis CLI.

    How:
    run `main` against a tiny temporary dictionary and inspect the written file.

    Responsibility:
    ensure the real top-level command target can load, match, and save output.
    """

    dictionary_path = tmp_path / "dictionary.json"
    output_path = tmp_path / "matches.json"
    dictionary_path.write_text(
        json.dumps(
            [
                {
                    "word": "ABBA",
                    "normalized": "abba",
                    "canon_word": True,
                    "senses": [{"definition": "father"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = find_ngram_cli.main(
        [
            "bb",
            "--dictionary",
            str(dictionary_path),
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["match_count"] == 1
    assert capsys.readouterr().out == ""


def test_standalone_main_returns_one_when_no_matches(tmp_path, capsys) -> None:
    """Confirm standalone no-match behavior.

    Why:
    empty searches should be script-friendly with exit status 1 while still
    telling a human what was searched.

    How:
    run the lightweight entry point against a dictionary with no matching word.

    Responsibility:
    keep the direct `find-ngram` command aligned with requested no-match output.
    """

    dictionary_path = tmp_path / "dictionary.json"
    dictionary_path.write_text(
        json.dumps(
            [
                {
                    "word": "ABBA",
                    "normalized": "abba",
                    "canon_word": True,
                    "senses": [{"definition": "father"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = find_ngram_cli.main(["zz", "--dictionary", str(dictionary_path)])

    assert exit_code == 1
    assert 'Matches for "zz" (normalized: "zz"): 0' in capsys.readouterr().out
