"""Utilities for aggregating morpheme coverage diagnostics.

The MorphemeCandidateFinder now returns a per-word breakdown that
identifies the portions of a token that are *explained* by known
morphemes versus the *residual* characters that remain unmatched.

This module transforms a collection of those breakdowns into aggregate
scores and prompt-friendly summaries that downstream agents can use to
prioritise unresolved morphology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class ResidualDetail:
    """Lightweight container for per-word residual diagnostics."""

    word: str
    normalized: str
    definition: str
    coverage_ratio: float
    residual_ratio: float
    uncovered: List[str]
    avg_confidence: float | None
    low_conf_segments: List[str]

    @classmethod
    def from_breakdown(cls, payload: dict[str, Any]) -> "ResidualDetail":
        word = str(payload.get("word") or payload.get("normalized") or "").strip()
        normalized = str(payload.get("normalized") or word).strip()
        breakdown = payload.get("breakdown") or {}
        coverage_ratio = float(breakdown.get("coverage_ratio") or 0.0)
        residual_ratio = float(breakdown.get("residual_ratio") or (1.0 - coverage_ratio))
        segments = breakdown.get("segments") or []
        uncovered_entries = breakdown.get("uncovered") or []
        uncovered = [
            str(seg.get("text", "")).strip()
            for seg in uncovered_entries
            if seg.get("text")
        ]
        confidences = [
            seg.get("semantic_confidence")
            for seg in segments
            if seg.get("semantic_confidence") is not None
        ]
        avg_conf = (
            sum(float(c) for c in confidences) / len(confidences)
            if confidences
            else None
        )
        low_conf_segments = [
            f"{str(seg.get('text', '')).strip()}@{float(seg.get('semantic_confidence', 0.0)):.2f}"
            for seg in segments
            if seg.get("semantic_confidence") is not None
            and float(seg.get("semantic_confidence", 0.0)) < 0.5
            and str(seg.get("text", "")).strip()
        ]

        return cls(
            word=word or normalized,
            normalized=normalized,
            definition=str(payload.get("definition") or "").strip(),
            coverage_ratio=max(0.0, min(1.0, coverage_ratio)),
            residual_ratio=max(0.0, min(1.0, residual_ratio)),
            uncovered=uncovered,
            avg_confidence=avg_conf,
            low_conf_segments=low_conf_segments,
        )


def summarize_residuals(
    root: str,
    analyses: Iterable[dict[str, Any]],
    *,
    max_focus: int = 5,
) -> dict[str, Any]:
    """Aggregate residual coverage diagnostics across a cluster of words.

    Parameters
    ----------
    root:
        The candidate root under investigation. Used to contextualise the
        textual prompts.
    analyses:
        Iterable of dictionaries. Each element must provide ``normalized``
        (lower-cased canonical form), ``word`` (for display), optional
        ``definition``, and a ``breakdown`` payload produced by
        :meth:`MorphemeCandidateFinder.find_candidates`.
    max_focus:
        Limit the number of per-word focus lines that appear in the
        generated prompt.

    Returns
    -------
    dict with the following keys:
        ``explained_ratio``: float average of explained characters.
        ``residual_ratio``: complement of the explained ratio.
        ``word_details``: list of :class:`ResidualDetail` instances as dicts.
        ``focus_prompt``: text emphasising unresolved residue.
        ``headline``: compact comma-separated summary of the largest residues.
    """

    details: list[ResidualDetail] = []
    total_chars = 0
    total_explained = 0.0

    for item in analyses:
        detail = ResidualDetail.from_breakdown(item)
        total_chars += len(detail.normalized)
        total_explained += detail.coverage_ratio * len(detail.normalized)
        details.append(detail)

    explained_ratio = (
        total_explained / total_chars if total_chars > 0 else 0.0
    )
    residual_ratio = max(0.0, 1.0 - explained_ratio)

    # Order words by the magnitude of their residue (then by confidence).
    sorted_details = sorted(
        details,
        key=lambda d: (d.residual_ratio, -(d.avg_confidence or 0.0)),
        reverse=True,
    )

    focus_lines: list[str] = []
    for detail in sorted_details:
        residues = [f"'{frag.upper()}'." for frag in detail.uncovered if frag]
        low_conf = detail.low_conf_segments
        parts: list[str] = []
        if residues:
            residue_text = ", ".join(residues)
            parts.append(f"residue {residue_text}")
        if low_conf:
            low_conf_text = ", ".join(low_conf)
            parts.append(f"low-confidence {low_conf_text}")
        if not parts:
            continue
        focus_lines.append(f"{detail.word.upper()}: {'; '.join(parts)}")
        if len(focus_lines) >= max_focus:
            break

    if focus_lines:
        focus_prompt = (
            f"Prioritise explaining unresolved morphology relative to {root.upper()}:\n"
            + "\n".join(f"- {line}" for line in focus_lines)
        )
    else:
        focus_prompt = (
            f"No major residual fragments detected relative to {root.upper()};"
            " confirm cohesion but expect minimal unexplained morphology."
        )

    headline = ", ".join(
        f"{detail.word}:{detail.residual_ratio:.2f}" for detail in sorted_details[:3]
    )

    return {
        "explained_ratio": explained_ratio,
        "residual_ratio": residual_ratio,
        "word_details": [detail.__dict__ for detail in details],
        "focus_prompt": focus_prompt,
        "headline": headline,
    }
