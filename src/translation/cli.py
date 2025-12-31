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
    parser.add_argument(
        "--evidence-mode",
        choices=["all", "clusters-only", "residuals-only"],
        default="all",
        help=(
            "Evidence sources to use when scoring/filtering decompositions "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--weight",
        type=_parse_bool,
        default=True,
        help="Enable weighted scoring (default: true).",
    )
    parser.add_argument(
        "--no-weight",
        dest="weight",
        action="store_false",
        help="Disable weighted scoring.",
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
    parser.add_argument(
        "--fallback-top-n",
        type=int,
        default=5,
        help=(
            "When hard filters remove all candidates, keep the top-N "
            "fallback decompositions that meet the minimum coverage ratio "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include diagnostic details such as FastText metadata and reasoning.",
    )
    parser.add_argument(
        "--trace-filters",
        action="store_true",
        help="Include per-decomposition filter traces in verbose diagnostics.",
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
    fallback_top_n = (
        max(1, int(args.fallback_top_n))
        if args.fallback_top_n is not None
        else 5
    )

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
                    fallback_top_n=fallback_top_n,
                    evidence_mode=_resolve_evidence_mode(args.evidence_mode),
                    weight_enabled=bool(args.weight),
                )
                outputs.append(
                    _build_output_payload(result, variant=variant, verbose=args.verbose)
                )
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
        rendered = _format_text_report(
            payload,
            verbose=args.verbose,
            trace_filters=args.trace_filters,
        )
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


def _resolve_evidence_mode(
    raw_mode: str,
) -> SingleWordTranslationService.EvidenceMode:
    try:
        return SingleWordTranslationService.EvidenceMode(raw_mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported evidence mode: {raw_mode}") from exc


def _parse_bool(value: str) -> bool:
    raw = value.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


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
        keys = (
            "direct_clusters",
            "residual_semantics",
            "morph_hypotheses",
            "attested_definitions",
        )
        return any(evidence.get(key, 0) for key in keys)

    if all(not has_evidence(report) for report in outputs):
        return 1
    return 0


def _build_output_payload(
    result: dict[str, object], *, variant: str, verbose: bool = False
) -> dict[str, object]:
    """Normalize translation results into a CLI-friendly output payload.

    This step intentionally reshapes internal service output into a stable,
    documented schema so downstream consumers (text renderer, JSON output, tests)
    can rely on consistent keys.
    """
    evidence_raw = result.get("evidence")
    evidence: dict[str, object] = evidence_raw if isinstance(evidence_raw, dict) else {}
    senses: List[dict[str, object]] = []
    payload: dict[str, object] = {
        "word": result.get("word"),
        "variant": variant,
        "variants_queried": result.get("variants_queried", []),
        "strategy": result.get("strategy"),
        "evidence_mode": result.get("evidence_mode"),
        "weighting_enabled": result.get("weighting_enabled"),
        "timestamp": result.get("timestamp"),
        "llm_enabled": result.get("llm_enabled"),
        "llm_mode": result.get("llm_mode"),
        "senses": senses,
        "evidence": evidence,
        "fallback_morphs": result.get("fallback_morphs", []),
    }

    candidates_raw = result.get("candidates")
    candidates = candidates_raw if isinstance(candidates_raw, list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        morphs_raw = candidate.get("morphs")
        meanings_raw = candidate.get("meanings")
        warnings_raw = candidate.get("warnings")
        senses.append(
            {
                "rank": candidate.get("rank"),
                "variant": variant,
                "morphs": list(morphs_raw) if isinstance(morphs_raw, (list, tuple)) else [],
                "score": candidate.get("score"),
                "breakdown": candidate.get("breakdown"),
                "meanings": list(meanings_raw) if isinstance(meanings_raw, (list, tuple)) else [],
                "synthesized_definition": candidate.get("synthesized_definition"),
                "concatenated_meanings": candidate.get("concatenated_meanings"),
                "confidence": candidate.get("confidence"),
                "warnings": list(warnings_raw) if isinstance(warnings_raw, (list, tuple)) else [],
            }
        )

    _apply_residual_only_adjustment(payload)
    if _no_direct_evidence(evidence):
        payload[
            "message"
        ] = "No direct evidence found. Showing FastText neighbors as heuristic."

    if verbose:
        payload["diagnostics"] = result.get("diagnostics", {})

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

    Direct evidence includes clusters, residual semantics, accepted morph
    hypotheses, or attested glossary definitions. Without them we fall back to
    FastText neighbors.
    """
    keys = (
        "direct_clusters",
        "residual_semantics",
        "morph_hypotheses",
        "attested_definitions",
        "dictionary_morphs",
    )
    return not any(evidence.get(key, 0) for key in keys)


def _render_json(
    payload: dict[str, object] | List[dict[str, object]], *, pretty: bool
) -> str:
    """Render output payloads as JSON, optionally pretty-printed.

    JSON output is the canonical machine-readable format for this CLI, so we
    keep serialization centralized to avoid formatting drift.
    """
    if pretty:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False)


def _format_text_report(
    payload: dict[str, object] | List[dict[str, object]],
    *,
    verbose: bool = False,
    trace_filters: bool = False,
) -> str:
    """Render translation output in a human-readable, wrapped text format.

    This format is designed for terminal inspection: short labels, 80-column
    wrapping, and clear grouping by candidate rank.
    """
    if isinstance(payload, list):
        blocks = [
            _format_variant_report(
                item,
                verbose=verbose,
                trace_filters=trace_filters,
            )
            for item in payload
        ]
        return "\n\n".join(blocks)
    return _format_variant_report(payload, verbose=verbose, trace_filters=trace_filters)


def _format_variant_report(
    payload: dict[str, object],
    *,
    verbose: bool = False,
    trace_filters: bool = False,
) -> str:
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
    evidence_mode = payload.get("evidence_mode")
    weighting_enabled = payload.get("weighting_enabled")
    llm_mode_label = llm_mode if llm_mode else "n/a"

    lines.append(f"Word: {word}")
    lines.append(f"Variant: {variant}")
    lines.append(f"Strategy: {strategy}")
    if isinstance(evidence_mode, str) and evidence_mode:
        lines.append(f"Evidence mode: {evidence_mode}")
    if isinstance(weighting_enabled, bool):
        lines.append(f"Weighted scoring: {weighting_enabled}")
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
        attestations = evidence.get("attested_definitions", 0)
        dictionary_morphs = evidence.get("dictionary_morphs", 0)
        lines.append(
            "Evidence: "
            f"clusters={clusters}, residuals={residuals}, hypotheses={hypotheses}, "
            f"attested={attestations}, dictionary={dictionary_morphs}"
        )

    senses = payload.get("senses")
    if isinstance(senses, list) and senses:
        for sense in senses:
            if not isinstance(sense, dict):
                continue
            rank = sense.get("rank")
            morphs_raw = sense.get("morphs")
            morphs_list = morphs_raw if isinstance(morphs_raw, list) else []
            morphs_str = " + ".join(str(m) for m in morphs_list)
            lines.append(f"\nRank {rank}: {morphs_str}")
            score = sense.get("score")
            if isinstance(score, (int, float)):
                lines.append(f"Score: {score:.2f}")

            provenance_note = sense.get("provenance_note")
            if isinstance(provenance_note, str) and provenance_note:
                lines.append(_wrap_text(f"Note: {provenance_note}", indent=0))

            breakdown_raw = sense.get("breakdown")
            breakdown: dict[str, object] = (
                breakdown_raw if isinstance(breakdown_raw, dict) else {}
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
            if verbose:
                reasoning = sense.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    lines.append(_wrap_text(f"Reasoning: {reasoning}", indent=0))
    else:
        lines.append("No candidate decompositions found.")
        fallback = payload.get("fallback_morphs")
        if isinstance(fallback, list) and fallback:
            lines.append("Fallback morph hints:")
            for hint in fallback:
                if not isinstance(hint, dict):
                    continue
                morph = hint.get("morph")
                if not morph:
                    continue
                definition = hint.get("definition") or "unknown"
                coverage = hint.get("coverage_ratio")
                similarity = hint.get("fasttext_similarity")
                label = f"{morph}: {definition}"
                details: List[str] = []
                if isinstance(coverage, (int, float)):
                    details.append(f"coverage {float(coverage):.2f}")
                if isinstance(similarity, (int, float)):
                    details.append(f"fasttext {float(similarity):.2f}")
                if details:
                    label += f" ({', '.join(details)})"
                lines.append(_wrap_text(label, indent=2, bullet=True))

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

    if verbose:
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict) and diagnostics:
            lines.append("Diagnostics:")
            fasttext = diagnostics.get("fasttext")
            if isinstance(fasttext, dict):
                model_path = fasttext.get("model_path")
                loaded = fasttext.get("loaded")
                vocab_size = fasttext.get("vocab_size")
                if model_path:
                    lines.append(_wrap_text(f"FastText model path: {model_path}", indent=2))
                if isinstance(loaded, bool):
                    lines.append(_wrap_text(f"FastText loaded: {loaded}", indent=2))
                if isinstance(vocab_size, int):
                    lines.append(_wrap_text(f"FastText vocab size: {vocab_size}", indent=2))
                vocab_sample = fasttext.get("vocab_sample")
                if isinstance(vocab_sample, list) and vocab_sample:
                    sample = ", ".join(str(item) for item in vocab_sample)
                    lines.append(
                        _wrap_text(f"FastText vocab sample: {sample}", indent=2)
                    )
            repository = diagnostics.get("repository")
            if isinstance(repository, dict):
                variants_available = repository.get("variants_available")
                if isinstance(variants_available, list) and variants_available:
                    available = ", ".join(str(item) for item in variants_available)
                    lines.append(
                        _wrap_text(f"Available variants: {available}", indent=2)
                    )
                variant_paths = repository.get("variant_paths")
                if isinstance(variant_paths, dict) and variant_paths:
                    lines.append(_wrap_text("Variant paths:", indent=2))
                    for variant_key, info in variant_paths.items():
                        if not isinstance(info, dict):
                            continue
                        path = info.get("path")
                        exists = info.get("exists")
                        path_info = f"{variant_key}: {path}"
                        if isinstance(exists, bool):
                            path_info += f" (exists={exists})"
                        lines.append(_wrap_text(path_info, indent=4))
            word_lookup = diagnostics.get("word_lookup")
            if isinstance(word_lookup, dict):
                lookup_word = word_lookup.get("word")
                lookup_variants = word_lookup.get("variants")
                counts = word_lookup.get("counts")
                if lookup_word:
                    lines.append(
                        _wrap_text(f"Evidence lookup word: {lookup_word}", indent=2)
                    )
                if isinstance(lookup_variants, list) and lookup_variants:
                    variant_list = ", ".join(str(item) for item in lookup_variants)
                    lines.append(
                        _wrap_text(
                            f"Evidence lookup variants: {variant_list}", indent=2
                        )
                    )
                if isinstance(counts, dict):
                    lines.append(_wrap_text("Evidence lookup counts:", indent=2))
                    for label, per_variant in counts.items():
                        if not isinstance(per_variant, dict):
                            continue
                        parts: List[str] = []
                        for variant_key, count in per_variant.items():
                            if count is None:
                                parts.append(f"{variant_key}=n/a")
                            else:
                                parts.append(f"{variant_key}={count}")
                        if parts:
                            lines.append(
                                _wrap_text(
                                    f"{label}: " + ", ".join(parts), indent=4
                                )
                            )
            parse_count = diagnostics.get("parse_count")
            if isinstance(parse_count, int):
                lines.append(
                    _wrap_text(f"Segmentation parses: {parse_count}", indent=2)
                )
            decomp_count = diagnostics.get("decomposition_count")
            if isinstance(decomp_count, int):
                lines.append(
                    _wrap_text(f"Decompositions built: {decomp_count}", indent=2)
                )
            extra_keys = diagnostics.get("extra_ngram_keys")
            extra_entries = diagnostics.get("extra_ngram_entries")
            if isinstance(extra_keys, int) and isinstance(extra_entries, int):
                lines.append(
                    _wrap_text(
                        f"Evidence ngrams: {extra_keys} keys ({extra_entries} entries)",
                        indent=2,
                    )
                )
            dict_keys = diagnostics.get("dictionary_ngram_keys")
            dict_entries = diagnostics.get("dictionary_ngram_entries")
            if isinstance(dict_keys, int) and isinstance(dict_entries, int):
                lines.append(
                    _wrap_text(
                        f"Dictionary ngrams: {dict_keys} keys ({dict_entries} entries)",
                        indent=2,
                    )
                )
            fallback_used = diagnostics.get("fallback_used")
            if isinstance(fallback_used, bool):
                lines.append(
                    _wrap_text(f"Dictionary fallback used: {fallback_used}", indent=2)
                )
            fallback_morphs = diagnostics.get("fallback_morphs")
            if isinstance(fallback_morphs, list) and fallback_morphs:
                sample = ", ".join(str(item) for item in fallback_morphs)
                lines.append(_wrap_text(f"Fallback morphs: {sample}", indent=2))
            substring_support = diagnostics.get("substring_support")
            if isinstance(substring_support, list) and substring_support:
                sample = ", ".join(str(item) for item in substring_support)
                lines.append(
                    _wrap_text(f"Substring support: {sample}", indent=2)
                )
            decomposition = diagnostics.get("decomposition")
            filtered_count = None
            if isinstance(decomposition, dict):
                generated = decomposition.get("generated")
                filtered = decomposition.get("filtered")
                selected = decomposition.get("selected")
                parts: List[str] = []
                if isinstance(generated, int):
                    parts.append(f"generated={generated}")
                if isinstance(filtered, int):
                    parts.append(f"filtered={filtered}")
                if isinstance(selected, int):
                    parts.append(f"selected={selected}")
                if parts:
                    lines.append(
                        _wrap_text(
                            "Decomposition pipeline: " + ", ".join(parts),
                            indent=2,
                        )
                    )
                if isinstance(filtered, int):
                    filtered_count = filtered
            hard_filters = diagnostics.get("hard_filters")
            if isinstance(hard_filters, dict):
                stage1 = hard_filters.get("stage1_dropped")
                stage2 = hard_filters.get("stage2_dropped")
                stage3 = hard_filters.get("stage3_dropped")
                dropped_parts: List[str] = []
                if isinstance(stage1, int):
                    dropped_parts.append(f"filter1={stage1}")
                if isinstance(stage2, int):
                    dropped_parts.append(f"filter2={stage2}")
                if isinstance(stage3, int):
                    dropped_parts.append(f"filter3={stage3}")
                if dropped_parts:
                    lines.append(
                        _wrap_text(
                            "Hard filter drops: " + ", ".join(dropped_parts),
                            indent=2,
                        )
                    )
                min_residual = hard_filters.get("min_residual_ratio")
                if isinstance(min_residual, (int, float)):
                    lines.append(
                        _wrap_text(
                            f"Hard filter min residual ratio: {float(min_residual):.3f}",
                            indent=2,
                        )
                    )
                max_attestation = hard_filters.get("max_attestation_score")
                if isinstance(max_attestation, int):
                    lines.append(
                        _wrap_text(
                            f"Hard filter max attestation score: {max_attestation}",
                            indent=2,
                        )
                    )
                unsupported = hard_filters.get("unsupported_morphs")
                if isinstance(unsupported, list) and unsupported:
                    sample = ", ".join(
                        f"{item.get('morph')} ({item.get('count')})"
                        for item in unsupported
                        if isinstance(item, dict)
                        and item.get("morph") is not None
                        and item.get("count") is not None
                    )
                    if sample:
                        lines.append(
                            _wrap_text(
                                f"Unsupported morphs (top): {sample}",
                                indent=2,
                            )
                        )
                dictionary_supported = hard_filters.get("dictionary_supported_morphs")
                if isinstance(dictionary_supported, list) and dictionary_supported:
                    sample = ", ".join(
                        f"{item.get('morph')} ({item.get('count')})"
                        for item in dictionary_supported
                        if isinstance(item, dict)
                        and item.get("morph") is not None
                        and item.get("count") is not None
                    )
                    if sample:
                        lines.append(
                            _wrap_text(
                                f"Dictionary-supported morphs: {sample}",
                                indent=2,
                            )
                        )
                attested_supported = hard_filters.get("attested_supported_morphs")
                if isinstance(attested_supported, list) and attested_supported:
                    sample = ", ".join(
                        f"{item.get('morph')} ({item.get('count')})"
                        for item in attested_supported
                        if isinstance(item, dict)
                        and item.get("morph") is not None
                        and item.get("count") is not None
                    )
                    if sample:
                        lines.append(
                            _wrap_text(
                                f"Attested-supported morphs: {sample}",
                                indent=2,
                            )
                        )
                traces = hard_filters.get("filter_traces")
                should_trace = trace_filters or (
                    isinstance(filtered_count, int) and filtered_count == 0
                )
                if (
                    should_trace
                    and isinstance(traces, list)
                    and traces
                ):
                    lines.append(_wrap_text("Filter traces:", indent=2))
                    for trace in traces:
                        if not isinstance(trace, dict):
                            continue
                        morphs_raw = trace.get("morphs")
                        morphs = (
                            " + ".join(str(m) for m in morphs_raw)
                            if isinstance(morphs_raw, list)
                            else ""
                        )
                        if morphs:
                            lines.append(_wrap_text(f"Decomposition: {morphs}", indent=4))
                        missing = trace.get("missing_morphs")
                        if isinstance(missing, list):
                            missing_label = ", ".join(str(m) for m in missing) or "none"
                            lines.append(
                                _wrap_text(f"Missing morphs: {missing_label}", indent=6)
                            )
                        support = trace.get("morph_support")
                        if isinstance(support, dict) and support:
                            pairs = ", ".join(
                                f"{key}={value}"
                                for key, value in support.items()
                            )
                            if pairs:
                                lines.append(
                                    _wrap_text(f"Support: {pairs}", indent=6)
                                )
                        residual = trace.get("residual_ratio")
                        if isinstance(residual, (int, float)):
                            lines.append(
                                _wrap_text(
                                    f"Residual ratio: {float(residual):.3f}",
                                    indent=6,
                                )
                            )
                        attestation = trace.get("attestation_score")
                        if isinstance(attestation, int):
                            lines.append(
                                _wrap_text(
                                    f"Attestation score: {attestation}",
                                    indent=6,
                                )
                            )
                        segments = trace.get("segments")
                        if isinstance(segments, list) and segments:
                            lines.append(_wrap_text("Segments:", indent=6))
                            for segment in segments:
                                if not isinstance(segment, dict):
                                    continue
                                ngram = segment.get("ngram")
                                canonical = segment.get("canonical")
                                label = " / ".join(
                                    part
                                    for part in [
                                        str(ngram) if ngram else "",
                                        str(canonical) if canonical else "",
                                    ]
                                    if part
                                )
                                if label:
                                    lines.append(_wrap_text(label, indent=8, bullet=True))

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
