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


def exclude_root_segments(
    breakdown: dict[str, Any] | None, root_norm: str, target: str
) -> dict[str, Any]:
    """
    Drop segments that echo the candidate root AND mark the root's position as uncovered.

    This prevents the root n-gram from vacuously explaining 100% of a word's
    meaning during residual analysis. The returned breakdown recomputes
    coverage/residual ratios and uncovered spans after stripping the root.

    Key behavior for residual evaluation:
    - Segments matching the candidate root are removed
    - Positions where the candidate root appears as a substring are marked as NOT covered
      (even if covered by a larger segment like NAZPSAD when evaluating PSAD)
    """

    if not breakdown:
        return {
            "segments": [],
            "uncovered": [],
            "coverage_ratio": 0.0,
            "residual_ratio": 0.0,
        }

    normalized_root = str(root_norm or "").strip().lower()
    target_text = str(target or "").lower()
    total_len = len(target_text)
    if not normalized_root or total_len == 0:
        return breakdown

    # Find all positions where the candidate root appears in the target
    # These positions should be marked as "uncovered" (the residual we're evaluating)
    root_positions: set[int] = set()
    root_len = len(normalized_root)
    for start_pos in range(total_len - root_len + 1):
        if target_text[start_pos:start_pos + root_len] == normalized_root:
            for i in range(start_pos, start_pos + root_len):
                root_positions.add(i)

    kept_segments: list[dict[str, Any]] = []
    coverage_mask = [False] * total_len

    for segment in breakdown.get("segments") or []:
        canonical = str(segment.get("canonical", "")).strip().lower()
        ngram = str(segment.get("ngram", "")).strip().lower()
        seg_text = str(segment.get("text", "")).strip().lower()

        # Skip segments that match the candidate root
        if canonical == normalized_root or ngram == normalized_root or seg_text == normalized_root:
            continue

        span = segment.get("span") or [segment.get("start", 0), segment.get("end", 0)]
        start = int(span[0] or 0)
        end = int(span[1] or start)
        start = max(0, min(total_len, start))
        end = max(start, min(total_len, end))

        # Only mark coverage for positions NOT occupied by the candidate root
        for idx in range(start, end):
            if idx not in root_positions:
                coverage_mask[idx] = True

        kept_segment = dict(segment)
        kept_segment["span"] = [start, end]
        kept_segments.append(kept_segment)

    # Build uncovered spans (positions not covered by kept segments, excluding root positions)
    uncovered: list[dict[str, Any]] = []
    idx = 0
    while idx < total_len:
        if coverage_mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < total_len and not coverage_mask[idx]:
            idx += 1
        uncovered.append({"span": [start, idx], "text": target_text[start:idx]})

    coverage_ratio = sum(1 for flag in coverage_mask if flag) / total_len
    residual_ratio = max(0.0, 1.0 - coverage_ratio)

    updated = dict(breakdown)
    updated["segments"] = kept_segments
    updated["uncovered"] = uncovered
    updated["coverage_ratio"] = coverage_ratio
    updated["residual_ratio"] = residual_ratio
    return updated

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

    # Check if the root itself appears as an uncovered fragment (residual candidate scenario)
    root_is_residual = False
    root_lower = root.lower()
    for d in details:
        for frag in d.uncovered:
            if frag.lower() == root_lower or root_lower in frag.lower():
                root_is_residual = True
                break
        if root_is_residual:
            break

    # Human-facing prompt
    header = (
        f"ROOT={root.upper()} | N={len(details)} | avg_len={ (total_chars/len(details)) if details else 0:.1f} | "
        f"avg_cov={explained_ratio:.2f} | avg_res={residual_ratio:.2f}\n"
    )

    if root_is_residual:
        # We ARE the residual - this is a residual candidate evaluation
        header += f"{root.upper()} is the uncovered fragment; evaluate its compositional contribution to the host word(s)."
    else:
        header += "Focus first on fragments that reduce residual the most."

    if focus_lines:
        focus_prompt = header + "\n\nPer-word hot spots:\n" + "\n".join(f"- {ln}" for ln in focus_lines)
    elif root_is_residual:
        # No other residuals besides the root we're evaluating
        focus_prompt = header + f"\n\n{root.upper()} is the primary uncovered fragment in these words."
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
