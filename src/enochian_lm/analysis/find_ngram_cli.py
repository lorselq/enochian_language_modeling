"""Lightweight top-level CLI for dictionary ngram lookup."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from enochian_lm.common.config import get_config_paths

from .find_ngram import (
    find_ngram_matches,
    format_ngram_matches_json,
    format_ngram_matches_text,
    load_dictionary_entries,
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure command logging without importing the full analysis CLI.

    Why:
    the top-level `find-ngram` script should start quickly and avoid the heavy
    imports used by unrelated analytics commands.

    How:
    select DEBUG for verbose mode and INFO otherwise using the shared project
    log format.

    Responsibility:
    provide enough logging setup for this standalone read-only command.
    """

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write command output atomically to a user-selected file.

    Why:
    `--output` should not leave partial files if the process is interrupted
    while writing lookup results.

    How:
    write to a temporary file in the destination directory, fsync it, then
    replace the target path.

    Responsibility:
    keep file output reliable without depending on the broader CLI helpers.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    def writer(tmp: NamedTemporaryFile) -> None:
        tmp.write(content)

    _with_temp_file(path, writer)


def _with_temp_file(path: Path, writer: Callable[[NamedTemporaryFile], None]) -> None:
    """Run a text writer against a temporary sibling file.

    Why:
    the standalone command needs the same safe-write behavior as the larger
    analysis CLI but should not import that module just to access it.

    How:
    create a named temporary file next to the target, delegate writing, flush and
    sync, then replace the target.

    Responsibility:
    isolate the small amount of file-system plumbing needed for `--output`.
    """

    with NamedTemporaryFile(
        mode="w", delete=False, dir=str(path.parent), encoding="utf-8", newline="\n"
    ) as tmp:
        writer(tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `find-ngram` arguments to the standalone parser.

    Why:
    the top-level command needs the same public options as the `enlm`
    subcommand while staying import-light.

    How:
    define the query, dictionary source, filters, formatting flags, and output
    destination directly in this small wrapper.

    Responsibility:
    maintain the direct `poetry run find-ngram "NGRAM"` command surface.
    """

    parser.add_argument("ngram", metavar="NGRAM", help="Ngram to find in words")
    parser.add_argument(
        "--dictionary",
        default=str(get_config_paths()["dictionary"]),
        help="Dictionary JSON path",
    )
    parser.add_argument(
        "--canon-only",
        action="store_true",
        help="Return only entries explicitly marked canon_word=true",
    )
    parser.add_argument(
        "--include-alternates",
        action="store_true",
        help="Also match alternate spellings when present",
    )
    parser.add_argument(
        "--citations",
        action="store_true",
        help="Include entry-level and sense-level citations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include match surfaces and dictionary metadata",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        help="Optional file path to write output instead of printing to stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone `find-ngram` argument parser.

    Why:
    Poetry invokes this module directly for the top-level console script.

    How:
    create a parser named `find-ngram` and attach the command's shared options.

    Responsibility:
    provide a small, testable parser factory for the standalone entry point.
    """

    parser = argparse.ArgumentParser(
        prog="find-ngram",
        description="Find dictionary words containing an ngram",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configure_parser(parser)
    return parser


def execute(args: argparse.Namespace) -> int:
    """Run a parsed standalone ngram lookup.

    Why:
    parser tests and the script entry point both need a direct execution helper
    that avoids the heavyweight analysis CLI module.

    How:
    load dictionary JSON, find matches, render text or JSON, and write to stdout
    or `--output`.

    Responsibility:
    implement the top-level command's read-only runtime behavior and exit code.
    """

    dictionary_path = Path(args.dictionary).expanduser().resolve()
    matches = find_ngram_matches(
        load_dictionary_entries(dictionary_path),
        args.ngram,
        canon_only=args.canon_only,
        include_alternates=args.include_alternates,
    )
    if args.format == "json":
        rendered = format_ngram_matches_json(matches, query=args.ngram)
    else:
        rendered = format_ngram_matches_text(
            matches,
            query=args.ngram,
            citations=args.citations,
            verbose=args.verbose,
        )

    if args.output:
        _atomic_write_text(Path(args.output).expanduser().resolve(), rendered)
    else:
        print(rendered, end="")

    if not matches:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the top-level `find-ngram` console script.

    Why:
    users need the exact `poetry run find-ngram "NGRAM"` command without paying
    the startup cost of unrelated analytics imports.

    How:
    parse standalone arguments, configure logging, execute the lookup, and
    convert expected file/input errors into status code 2.

    Responsibility:
    serve as the lightweight Poetry entry point for dictionary ngram lookup.
    """

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - delegated to argparse
        return exc.code

    _configure_logging(args.verbose)
    try:
        return execute(args)
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error(
            f"Command failed. Reason: {str(error)}",
            exc_info=False,
            extra={"error": str(error)},
        )
        return 2
