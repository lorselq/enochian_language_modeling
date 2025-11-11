from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .service import InterpretationService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpret unseen text using stored enochian analysis insights.",
    )
    parser.add_argument(
        "--text",
        help="Text to interpret. Use --input-file or stdin for longer passages.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to a UTF-8 text file (or '-' for stdin).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=["solo", "debate"],
        help="Restrict interpretation to specific insight variants. May be repeated.",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=7,
        help="Maximum character length for generated n-grams (default: 7).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON report. Defaults to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    text = _resolve_text(args, parser)
    if not text.strip():
        parser.error("Provided text is empty after stripping whitespace.")

    variants = _dedupe_variants(args.variant)
    max_ngram = max(1, args.max_ngram)

    with InterpretationService.from_config(
        variants=variants,
        max_ngram_len=max_ngram,
    ) as service:
        report = service.interpret_text(
            text,
            max_ngram_len=max_ngram,
            variants=variants,
        )

    json_kwargs = {"ensure_ascii": False}
    if args.pretty:
        json_kwargs["indent"] = 2
    output_payload = json.dumps(report, **json_kwargs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_payload + ("\n" if args.pretty else ""), encoding="utf-8")
    else:
        print(output_payload)


def _resolve_text(args, parser: argparse.ArgumentParser) -> str:
    if args.text:
        return args.text
    if args.input_file:
        if str(args.input_file) == "-":
            return sys.stdin.read()
        try:
            return args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            parser.error(f"Input file not found: {exc.filename}")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    parser.error("Provide --text, --input-file, or pipe text via stdin.")
    return ""


def _dedupe_variants(variants: Optional[List[str]]) -> Optional[List[str]]:
    if not variants:
        return None
    seen = set()
    deduped: List[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


if __name__ == "__main__":  # pragma: no cover
    main()
