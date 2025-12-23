from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from enochian_lm.common.config import get_config_paths

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate a single Enochian word using stored insights.",
    )
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
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
    top_k = max(1, int(args.top_k)) if args.top_k is not None else 3

    try:
        from .service import SingleWordTranslationService

        with SingleWordTranslationService.from_config(variants=variants) as service:
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
    normalized = (word or "").strip().upper()
    if not normalized:
        raise ValueError("Word must be a non-empty string.")
    if not normalized.isalpha():
        raise ValueError("Word must contain only alphabetic characters (A-Z).")
    return normalized


def _resolve_variants(variant: str) -> List[str]:
    if variant == "both":
        return ["solo", "debate"]
    return [variant]


def _missing_db_paths(variants: Sequence[str]) -> List[Path]:
    paths = get_config_paths()
    missing: List[Path] = []
    for variant in variants:
        path = paths.get(variant)
        if path and not path.exists():
            missing.append(path)
    return missing


def _determine_exit_code(outputs: Sequence[dict[str, object]]) -> int:
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


def _build_output_payload(result: dict[str, object], *, variant: str) -> dict[str, object]:
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    payload: dict[str, object] = {
        "word": result.get("word"),
        "variant": variant,
        "variants_queried": result.get("variants_queried", []),
        "strategy": result.get("strategy"),
        "timestamp": result.get("timestamp"),
        "llm_enabled": result.get("llm_enabled"),
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
    keys = ("direct_clusters", "residual_semantics", "morph_hypotheses")
    return not any(evidence.get(key, 0) for key in keys)


def _render_json(payload: dict[str, object] | List[dict[str, object]], *, pretty: bool) -> str:
    json_kwargs = {"ensure_ascii": False}
    if pretty:
        json_kwargs["indent"] = 2
    return json.dumps(payload, **json_kwargs)


def _format_text_report(payload: dict[str, object] | List[dict[str, object]]) -> str:
    if isinstance(payload, list):
        blocks = [_format_variant_report(item) for item in payload]
        return "\n\n".join(blocks)
    return _format_variant_report(payload)


def _format_variant_report(payload: dict[str, object]) -> str:
    lines: List[str] = []
    word = payload.get("word", "")
    variant = payload.get("variant", "")
    strategy = payload.get("strategy", "")
    llm_enabled = payload.get("llm_enabled", False)

    lines.append(f"Word: {word}")
    lines.append(f"Variant: {variant}")
    lines.append(f"Strategy: {strategy}")
    lines.append(f"LLM enabled: {llm_enabled}")

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

            breakdown = sense.get("breakdown") if isinstance(sense.get("breakdown"), dict) else {}
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
                lines.append(_wrap_text(f"Concatenated meanings: {concatenated}", indent=0))

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


def _emit_output(content: str, *, output_path: Optional[Path]) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


def _emit_error(message: str) -> None:
    print(message, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
