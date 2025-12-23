"""Command-line interface for interpretation and single-word translation.

This module deliberately houses both the interpret-text workflow (legacy
entrypoint behavior) and the single-word translation CLI. Keeping the two
together reduces drift in shared behaviors such as output rendering, variant
selection, and error handling.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv

from enochian_lm.common.config import get_config_paths

from .service import InterpretationService, SingleWordTranslationService

INTERPRET_COMMAND = "interpret-text"
TRANSLATE_WORD_COMMAND = "translate-word"
COMMAND_ALIASES = {
    "interpret": INTERPRET_COMMAND,
    INTERPRET_COMMAND: INTERPRET_COMMAND,
    TRANSLATE_WORD_COMMAND: TRANSLATE_WORD_COMMAND,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands.

    The subcommand layout makes intent explicit:
    - ``interpret-text`` for sentence/phrase inspection
    - ``translate-word`` for single-word translation

    We still keep a legacy mode for scripts that call the entry point without a
    subcommand.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Interpret unseen Enochian text or translate individual words using "
            "stored analysis insights."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    interpret_parser = subparsers.add_parser(
        INTERPRET_COMMAND,
        help="Interpret unseen text using stored analysis insights.",
    )
    _add_interpret_arguments(interpret_parser)
    interpret_parser.set_defaults(handler=_run_interpret_text)

    translate_word_parser = subparsers.add_parser(
        TRANSLATE_WORD_COMMAND,
        help="Translate a single Enochian word using stored insights.",
    )
    configure_translate_word_parser(translate_word_parser)
    translate_word_parser.set_defaults(handler=_run_translate_word)

    return parser


def configure_translate_word_parser(parser: argparse.ArgumentParser) -> None:
    """Register translate-word CLI arguments on an existing parser.

    This helper exists so the translation CLI and the main ``enlm`` CLI share
    the same flags, defaults, and help text. It keeps the single-word UX stable
    across entry points without copying argument definitions in multiple places.
    """
    parser.add_argument("word", help="Single word to translate.")
    parser.add_argument(
        "--variant",
        choices=["solo", "debate", "both"],
        default="both",
        help="Insights variant to query (default: both).",
    )
    parser.add_argument(
        "--strategy",
        choices=["prefer-fewer", "prefer-known", "prefer-balance"],
        default="prefer-balance",
        help="Reranking strategy to apply (default: prefer-balance).",
    )

    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM synthesis for the top-ranked candidate.",
    )
    llm_group.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM synthesis (default).",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["local", "remote"],
        default="remote",
        help=(
            "Choose which LLM backend to use when --llm is enabled. "
            "'local' loads .env_local; 'remote' loads .env_remote and falls back "
            "to .env_local if available (default: remote)."
        ),
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the output. Defaults to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate definitions to return (default: 3).",
    )


def build_interpret_parser() -> argparse.ArgumentParser:
    """Return a standalone parser for text interpretation.

    The interpret-text command predates the consolidated CLI. Keeping a dedicated
    parser here preserves backward compatibility for tooling that still invokes
    ``enochian-interpret`` without a subcommand.
    """
    parser = argparse.ArgumentParser(
        description="Interpret unseen text using stored enochian analysis insights.",
    )
    _add_interpret_arguments(parser)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Dispatch the CLI while preserving backward compatibility.

    The entry point supports two invocation styles:
    1. Modern subcommands (``interpret-text`` / ``translate-word``).
    2. Legacy interpret-only usage where no subcommand is provided.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list or args_list[0] in {"-h", "--help"}:
        parser = build_parser()
        args = parser.parse_args(args_list)
        return args.handler(args)

    command = _resolve_command(args_list[0])
    if command:
        args_list[0] = command
        parser = build_parser()
        args = parser.parse_args(args_list)
        return args.handler(args)

    parser = build_interpret_parser()
    args = parser.parse_args(args_list)
    return _run_interpret_text(args)


def _resolve_command(raw: str) -> str | None:
    """Resolve a raw CLI token into a supported command.

    Aliases are mapped to canonical command names to keep CLI usage flexible
    without duplicating parser registrations.
    """
    return COMMAND_ALIASES.get(raw)


def _add_interpret_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach interpretation flags to the supplied parser.

    These are extracted into a helper so both the legacy parser and the
    subcommand parser share identical flag definitions.
    """
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


def _run_interpret_text(args: argparse.Namespace) -> int:
    """Run the text interpretation pipeline and emit a JSON report.

    Interpretation is designed for exploratory analysis of longer spans (e.g.,
    sentences) where n-gram overlaps provide contextual hints. It always emits
    JSON so the downstream consumer can inspect per-span details.
    """
    text = _resolve_text(args)
    if not text.strip():
        raise SystemExit("Provided text is empty after stripping whitespace.")

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

    rendered = _render_json(report, pretty=args.pretty)
    _emit_output(rendered, output_path=args.output, newline=bool(args.pretty))
    return 0


def _run_translate_word(args: argparse.Namespace) -> int:
    """Run the single-word translation pipeline and emit text or JSON output."""
    return translate_word_from_args(args)


def translate_word_from_args(args: argparse.Namespace) -> int:
    """Translate a single word based on parsed CLI arguments.

    The translation entry point is used both by the dedicated translation CLI
    and by the main ``enlm`` CLI. This helper keeps the pipeline composable and
    ensures both entry points behave identically.
    """
    try:
        word = _normalize_word(args.word)
    except ValueError as exc:
        _emit_error(str(exc))
        return 2

    variants = _resolve_variants(args.variant)
    missing = _missing_db_paths(variants)
    if missing:
        _emit_error(
            "Missing insights database file(s): "
            + ", ".join(str(path) for path in missing)
        )
        return 2

    llm_enabled = bool(args.llm) if args.llm or args.no_llm else False
    llm_use_remote = args.llm_mode == "remote"
    if llm_enabled:
        try:
            _configure_llm_env(args.llm_mode)
        except FileNotFoundError as exc:
            _emit_error(str(exc))
            return 2
    top_k = max(1, int(args.top_k)) if args.top_k is not None else 3

    try:
        with SingleWordTranslationService.from_config(
            variants=variants,
            llm_enabled=llm_enabled,
            llm_use_remote=llm_use_remote,
        ) as service:
            outputs: List[dict[str, object]] = []
            for variant in variants:
                result = service.translate_word(
                    word,
                    variants=[variant],
                    strategy=args.strategy,
                    top_k=top_k,
                    llm=llm_enabled,
                )
                outputs.append(_build_output_payload(result, variant=variant))
    except FileNotFoundError as exc:
        _emit_error(str(exc))
        return 2
    except ValueError as exc:
        _emit_error(str(exc))
        return 2

    payload: dict[str, object] | List[dict[str, object]]
    if args.variant == "both":
        payload = outputs
    else:
        payload = outputs[0] if outputs else {}

    exit_code = _determine_exit_code(outputs)

    if args.format == "json":
        rendered = _render_json(payload, pretty=args.pretty)
        _emit_output(rendered, output_path=args.output)
    else:
        rendered = _format_text_report(payload)
        _emit_output(rendered, output_path=args.output)

    return exit_code


def _normalize_word(word: str) -> str:
    """Normalize and validate the input word for translation.

    The translation pipeline expects uppercase alphabetic morphemes. Normalizing
    here ensures the rest of the pipeline can assume consistent casing and
    reject malformed inputs early, avoiding confusing downstream errors.
    """
    normalized = (word or "").strip().upper()
    if not normalized:
        raise ValueError("Word must be a non-empty string.")
    if not normalized.isalpha():
        raise ValueError("Word must contain only alphabetic characters (A-Z).")
    return normalized


def _configure_llm_env(llm_mode: str) -> None:
    """Load environment variables required for LLM synthesis.

    Single-word translation supports optional LLM synthesis. We load the
    relevant environment file(s) here so the rest of the pipeline can remain
    model-agnostic and focus purely on the translation logic.
    """
    if llm_mode == "local":
        env_local = find_dotenv(".env_local")
        if not env_local:
            raise FileNotFoundError("Missing .env_local for local LLM configuration.")
        load_dotenv(env_local, override=True)
        return

    env_remote = find_dotenv(".env_remote")
    if not env_remote:
        raise FileNotFoundError("Missing .env_remote for remote LLM configuration.")
    load_dotenv(env_remote, override=True)
    env_local = find_dotenv(".env_local")
    if env_local:
        load_dotenv(env_local, override=True)
    else:
        _emit_error(
            "Warning: .env_local not found; remote fallback to a local LLM is disabled."
        )


def _resolve_variants(variant: str) -> List[str]:
    """Expand a variant selector into the concrete variant list.

    The ``both`` option is expanded to ``["solo", "debate"]`` to keep downstream
    code simple and consistent.
    """
    if variant == "both":
        return ["solo", "debate"]
    return [variant]


def _missing_db_paths(variants: Sequence[str]) -> List[Path]:
    """Return any insight database paths that are missing on disk.

    Missing databases are a hard failure for translation, so we preflight them
    up front to provide immediate, clear feedback.
    """
    paths = get_config_paths()
    missing: List[Path] = []
    for variant in variants:
        path = paths.get(variant)
        if path and not path.exists():
            missing.append(path)
    return missing


def _determine_exit_code(outputs: Sequence[dict[str, object]]) -> int:
    """Return exit codes for word translation.

    We intentionally mirror the task definitions:
    - ``0`` when evidence is found,
    - ``1`` when only FastText fallback evidence exists,
    - ``2`` for hard failures (handled earlier in the flow).
    """
    if not outputs:
        return 1

    def has_evidence(report: dict[str, object]) -> bool:
        evidence = report.get("evidence")
        if not isinstance(evidence, dict):
            return False
        keys = ("direct_clusters", "residual_semantics", "morph_hypotheses")
        return any(evidence.get(key, 0) for key in keys)

    if all(not has_evidence(report) for report in outputs):
        return 1
    return 0


def _build_output_payload(
    result: dict[str, object], *, variant: str
) -> dict[str, object]:
    """Normalize translation results into a CLI-friendly output payload.

    This step intentionally reshapes internal service output into a stable,
    documented schema so downstream consumers (text renderer, JSON output, tests)
    can rely on consistent keys.
    """
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    payload: dict[str, object] = {
        "word": result.get("word"),
        "variant": variant,
        "variants_queried": result.get("variants_queried", []),
        "strategy": result.get("strategy"),
        "timestamp": result.get("timestamp"),
        "llm_enabled": result.get("llm_enabled"),
        "llm_mode": result.get("llm_mode"),
        "senses": [],
        "evidence": evidence,
    }

    for candidate in result.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        payload["senses"].append(
            {
                "rank": candidate.get("rank"),
                "variant": variant,
                "morphs": list(candidate.get("morphs", [])),
                "score": candidate.get("score"),
                "breakdown": candidate.get("breakdown"),
                "meanings": list(candidate.get("meanings", [])),
                "synthesized_definition": candidate.get("synthesized_definition"),
                "concatenated_meanings": candidate.get("concatenated_meanings"),
                "confidence": candidate.get("confidence"),
                "warnings": list(candidate.get("warnings", [])),
            }
        )

    _apply_residual_only_adjustment(payload)
    if _no_direct_evidence(evidence):
        payload[
            "message"
        ] = "No direct evidence found. Showing FastText neighbors as heuristic."

    return payload


def _apply_residual_only_adjustment(payload: dict[str, object]) -> None:
    """Annotate residual-only matches with a caution note and lower confidence.

    Residual-only evidence indicates a weak link to established roots. We label
    these cases explicitly and reduce confidence to discourage over-interpretation.
    """
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        return
    if evidence.get("direct_clusters", 0) != 0:
        return
    if evidence.get("morph_hypotheses", 0) != 0:
        return
    if evidence.get("residual_semantics", 0) <= 0:
        return

    senses = payload.get("senses")
    if not isinstance(senses, list):
        return

    for sense in senses:
        meanings = sense.get("meanings") if isinstance(sense, dict) else None
        if not isinstance(meanings, list) or not meanings:
            continue
        if not all(
            isinstance(entry, dict) and entry.get("provenance") == "residual"
            for entry in meanings
        ):
            continue
        sense["provenance_note"] = "residual-only (observed as remainder)"
        confidence = sense.get("confidence")
        if isinstance(confidence, (int, float)):
            sense["confidence"] = max(0.0, min(1.0, float(confidence) - 0.2))


def _no_direct_evidence(evidence: dict[str, object]) -> bool:
    """Return True when no direct evidence exists.

    Direct evidence includes clusters, residual semantics, or accepted morph
    hypotheses. Without them we fall back to FastText neighbors.
    """
    keys = ("direct_clusters", "residual_semantics", "morph_hypotheses")
    return not any(evidence.get(key, 0) for key in keys)


def _render_json(
    payload: dict[str, object] | List[dict[str, object]], *, pretty: bool
) -> str:
    """Render output payloads as JSON, optionally pretty-printed.

    JSON output is the canonical machine-readable format for this CLI, so we
    keep serialization centralized to avoid formatting drift.
    """
    json_kwargs = {"ensure_ascii": False}
    if pretty:
        json_kwargs["indent"] = 2
    return json.dumps(payload, **json_kwargs)


def _format_text_report(payload: dict[str, object] | List[dict[str, object]]) -> str:
    """Render translation output in a human-readable, wrapped text format.

    This format is designed for terminal inspection: short labels, 80-column
    wrapping, and clear grouping by candidate rank.
    """
    if isinstance(payload, list):
        blocks = [_format_variant_report(item) for item in payload]
        return "\n\n".join(blocks)
    return _format_variant_report(payload)


def _format_variant_report(payload: dict[str, object]) -> str:
    """Render a single-variant translation payload into a text report.

    The output emphasizes the evidence hierarchy: first the high-level summary,
    then candidate senses, then any FastText fallback neighbors.
    """
    lines: List[str] = []
    word = payload.get("word", "")
    variant = payload.get("variant", "")
    strategy = payload.get("strategy", "")
    llm_enabled = payload.get("llm_enabled", False)
    llm_mode = payload.get("llm_mode")
    llm_mode_label = llm_mode if llm_mode else "n/a"

    lines.append(f"Word: {word}")
    lines.append(f"Variant: {variant}")
    lines.append(f"Strategy: {strategy}")
    lines.append(f"LLM enabled: {llm_enabled}")
    lines.append(f"LLM mode: {llm_mode_label}")

    message = payload.get("message")
    if isinstance(message, str) and message:
        lines.append(_wrap_text(message, indent=0))

    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        clusters = evidence.get("direct_clusters", 0)
        residuals = evidence.get("residual_semantics", 0)
        hypotheses = evidence.get("morph_hypotheses", 0)
        lines.append(
            f"Evidence: clusters={clusters}, residuals={residuals}, hypotheses={hypotheses}"
        )

    senses = payload.get("senses")
    if isinstance(senses, list) and senses:
        for sense in senses:
            if not isinstance(sense, dict):
                continue
            rank = sense.get("rank")
            morphs = " + ".join(sense.get("morphs", []))
            lines.append(f"\nRank {rank}: {morphs}")
            score = sense.get("score")
            if isinstance(score, (int, float)):
                lines.append(f"Score: {score:.2f}")

            provenance_note = sense.get("provenance_note")
            if isinstance(provenance_note, str) and provenance_note:
                lines.append(_wrap_text(f"Note: {provenance_note}", indent=0))

            breakdown = (
                sense.get("breakdown") if isinstance(sense.get("breakdown"), dict) else {}
            )
            coverage = breakdown.get("coverage_ratio")
            residual = breakdown.get("residual_ratio")
            if isinstance(coverage, (int, float)) and isinstance(residual, (int, float)):
                lines.append(f"Coverage: {coverage:.2f} (residual {residual:.2f})")

            meanings = sense.get("meanings")
            if isinstance(meanings, list) and meanings:
                lines.append("Meanings:")
                for meaning in meanings:
                    if not isinstance(meaning, dict):
                        continue
                    morph = meaning.get("morph", "")
                    provenance = meaning.get("provenance", "unknown")
                    definition = meaning.get("definition") or "unknown"
                    line = f"{morph} ({provenance}): {definition}"
                    lines.append(_wrap_text(line, indent=2, bullet=True))

            synthesized = sense.get("synthesized_definition")
            concatenated = sense.get("concatenated_meanings")
            if synthesized:
                lines.append(_wrap_text(f"Synthesized: {synthesized}", indent=0))
                if concatenated:
                    lines.append(_wrap_text(f"Concatenated: {concatenated}", indent=0))
            elif concatenated:
                lines.append(
                    _wrap_text(f"Concatenated meanings: {concatenated}", indent=0)
                )

            confidence = sense.get("confidence")
            if isinstance(confidence, (int, float)):
                lines.append(f"Confidence: {float(confidence):.2f}")

            warnings = sense.get("warnings")
            if isinstance(warnings, list) and warnings:
                lines.append(_wrap_text("Warnings: " + "; ".join(warnings), indent=0))
    else:
        lines.append("No candidate decompositions found.")

    if isinstance(evidence, dict):
        neighbors = evidence.get("fasttext_neighbors")
        if isinstance(neighbors, list) and neighbors:
            lines.append("FastText neighbors:")
            for neighbor in neighbors:
                if not isinstance(neighbor, dict):
                    continue
                word = neighbor.get("word")
                similarity = neighbor.get("similarity")
                if word is None:
                    continue
                label = f"{word}"
                if isinstance(similarity, (int, float)):
                    label += f" ({float(similarity):.2f})"
                lines.append(_wrap_text(label, indent=2, bullet=True))

    return "\n".join(lines)


def _wrap_text(text: str, *, indent: int, bullet: bool = False) -> str:
    """Wrap long strings to ~80 columns with optional bullet formatting.

    Consistent wrapping keeps terminal output readable and avoids jagged lines
    when definitions or warnings are verbose.
    """
    prefix = " " * indent
    if bullet:
        prefix = " " * indent + "- "
        subsequent = " " * (indent + 2)
    else:
        subsequent = prefix
    return textwrap.fill(
        text,
        width=80,
        initial_indent=prefix,
        subsequent_indent=subsequent,
    )


def _emit_output(content: str, *, output_path: Optional[Path], newline: bool = True) -> None:
    """Write output to a file or stdout, ensuring a trailing newline.

    Output files intentionally include a trailing newline for POSIX friendliness.
    """
    if output_path:
        suffix = "\n" if newline else ""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + suffix, encoding="utf-8")
    else:
        print(content)


def _emit_error(message: str) -> None:
    """Send an error message to stderr without raising an exception.

    Error output is kept simple so shell users can redirect or capture it
    without parsing structured payloads.
    """
    print(message, file=sys.stderr)


def _resolve_text(args: argparse.Namespace) -> str:
    """Resolve interpret-text input from --text, --input-file, or stdin.

    Interpretation supports both short command-line strings and longer input
    files. If neither is provided, stdin is used as a fallback for piped input.
    """
    if args.text:
        return args.text
    if args.input_file:
        if str(args.input_file) == "-":
            return sys.stdin.read()
        try:
            return args.input_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SystemExit(f"Input file not found: {exc.filename}") from exc
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --input-file, or pipe text via stdin.")


def _dedupe_variants(variants: Optional[List[str]]) -> Optional[List[str]]:
    """Return an ordered, deduplicated list of variants.

    Deduplication prevents repeated DB hits when the user supplies the same
    ``--variant`` flag more than once.
    """
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
    raise SystemExit(main())
