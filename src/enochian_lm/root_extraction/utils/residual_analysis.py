# residual_analysis.py — tighter, more numeric summaries

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter
from typing import Any, Iterable, List, Dict, Tuple


@dataclass
class ResidualDetail:
    word: str
    normalized: str
    definition: str
    coverage_ratio: float
    residual_ratio: float
    uncovered: List[str]            # raw fragments
    avg_confidence: float | None
    low_conf_segments: List[str]    # "frag@0.42"

    @property
    def length(self) -> int:
        return len(self.normalized or "")

    @classmethod
    def from_breakdown(cls, payload: dict[str, Any]) -> "ResidualDetail":
        word = str(payload.get("word") or payload.get("normalized") or "").strip()
        normalized = str(payload.get("normalized") or word).strip()
        breakdown = payload.get("breakdown") or {}
        cov = float(breakdown.get("coverage_ratio") or 0.0)
        res = float(breakdown.get("residual_ratio") or (1.0 - cov))
        segs = breakdown.get("segments") or []
        un = breakdown.get("uncovered") or []

        uncovered = []
        for u in un:
            # support both {"text": "..."} and plain strings
            txt = (u.get("text") if isinstance(u, dict) else u) or ""
            txt = str(txt).strip()
            if txt:
                uncovered.append(txt)

        confs = [
            s.get("semantic_confidence")
            for s in segs
            if s.get("semantic_confidence") is not None
        ]
        avg_conf = (sum(float(c) for c in confs) / len(confs)) if confs else None

        low_conf = [
            f"{(s.get('text') or '').strip()}@{float(s.get('semantic_confidence', 0.0)):.2f}"
            for s in segs
            if s.get("semantic_confidence") is not None
            and float(s.get("semantic_confidence", 0.0)) < 0.5
            and str(s.get("text") or "").strip()
        ]

        return cls(
            word=word or normalized,
            normalized=normalized,
            definition=str(payload.get("definition") or "").strip(),
            coverage_ratio=max(0.0, min(1.0, cov)),
            residual_ratio=max(0.0, min(1.0, res)),
            uncovered=uncovered,
            avg_confidence=avg_conf,
            low_conf_segments=low_conf,
        )


def _top_residue_fragments(details: list[ResidualDetail], k: int = 6) -> list[tuple[str, int, int]]:
    """
    Return top-K uncovered fragments as (frag, freq, total_len).
    """
    c: Counter[str] = Counter()
    total_len: Dict[str, int] = {}
    for d in details:
        for frag in d.uncovered:
            c[frag] += 1
            total_len[frag] = total_len.get(frag, 0) + len(frag)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], total_len[kv[0]]), reverse=True)
    return [(frag, freq, total_len[frag]) for frag, freq in ranked[:k]]


def summarize_residuals(
    root: str,
    analyses: Iterable[dict[str, Any]],
    *,
    max_focus: int = 5,
) -> dict[str, Any]:
    """
    Aggregate residuals with numeric context AND a machine-readable guidance blob.
    """

    details: list[ResidualDetail] = []
    total_chars = 0
    total_explained = 0.0

    for item in analyses:
        d = ResidualDetail.from_breakdown(item)
        total_chars += d.length
        total_explained += d.coverage_ratio * d.length
        details.append(d)

    explained_ratio = (total_explained / total_chars) if total_chars else 0.0
    residual_ratio = max(0.0, 1.0 - explained_ratio)

    # Sort by raw residual heaviness, then by lower confidence
    sorted_details = sorted(
        details,
        key=lambda d: (d.residual_ratio, -(d.avg_confidence or 0.0)),
        reverse=True,
    )

    # Per-word focus lines with numeric breadcrumbs
    focus_lines: list[str] = []
    for d in sorted_details:
        parts: list[str] = []
        if d.uncovered:
            residues = ", ".join([f"'{frag.upper()}'" for frag in d.uncovered])
            parts.append(f"residue={residues}")
        if d.low_conf_segments:
            parts.append(f"low_conf={', '.join(d.low_conf_segments)}")
        if not parts:
            continue
        focus_lines.append(
            f"{d.word.upper()} (len={d.length}, cov={d.coverage_ratio:.2f}, res={d.residual_ratio:.2f}): "
            + "; ".join(parts)
        )
        if len(focus_lines) >= max_focus:
            break

    # Fragment-level histogram (top unresolved pieces)
    top_frags = _top_residue_fragments(details, k=6)

    # Human-facing prompt
    header = (
        f"ROOT={root.upper()} | N={len(details)} | avg_len={ (total_chars/len(details)) if details else 0:.1f} | "
        f"avg_cov={explained_ratio:.2f} | avg_res={residual_ratio:.2f}\n"
        "Focus first on fragments that reduce residual the most."
    )
    if focus_lines:
        focus_prompt = header + "\n\nPer-word hot spots:\n" + "\n".join(f"- {ln}" for ln in focus_lines)
    else:
        focus_prompt = header + "\n\nNo prominent residuals; confirm cohesion."

    # Compact headline (unchanged, but now clearer)
    headline = ", ".join(
        f"{d.word}:{d.residual_ratio:.2f}" for d in sorted_details[:3]
    )

    # Machine-readable guidance block the model can “use as data”
    residual_guidance_json = {
        "root": root.upper(),
        "stats": {
            "n_words": len(details),
            "avg_len": round((total_chars / len(details)) if details else 0.0, 2),
            "avg_coverage": round(explained_ratio, 3),
            "avg_residual": round(residual_ratio, 3),
        },
        "top_fragments": [
            {"frag": f.upper(), "freq": freq, "total_len": tlen}
            for (f, freq, tlen) in top_frags
        ],
        "words": [
            {
                "word": d.word.upper(),
                "len": d.length,
                "coverage": round(d.coverage_ratio, 3),
                "residual": round(d.residual_ratio, 3),
                "uncovered": [u.upper() for u in d.uncovered],
                "low_conf": d.low_conf_segments,
            }
            for d in sorted_details[: max_focus * 2]  # a little deeper than the human block
        ],
    }

    return {
        "explained_ratio": explained_ratio,
        "residual_ratio": residual_ratio,
        "word_details": [asdict(d) for d in details],
        "focus_prompt": focus_prompt,
        "headline": headline,
        "residual_guidance_json": residual_guidance_json,
    }
