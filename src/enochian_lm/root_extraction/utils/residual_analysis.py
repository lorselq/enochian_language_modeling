# residual_analysis.py — tighter, more numeric summaries

from __future__ import annotations
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, asdict


def compute_word_break_subtractions(host_word: str, root: str) -> list[dict[str, object]]:
    """Compute explicit subtraction candidates for ``HOST - ROOT = RESIDUAL``.

    The comparison is case-insensitive, but returned values preserve the caller's
    original spelling for ``host_word`` and ``root``.

    Returns one record per match in left-to-right order.

    Each record includes:
    - ``residual``: concatenated residual text after removing this occurrence
    - ``residuals``: positional residual artifacts for this occurrence
    - ``remove_all_occurrences``: ambiguity-aware option removing every occurrence
      of ``root`` in ``host_word`` (useful for repeated-root cases like ANAZNAZ)
    """

    host_text = str(host_word or "")
    root_text = str(root or "")
    host_norm = host_text.lower()
    root_norm = root_text.lower()

    if not host_norm or not root_norm or len(root_norm) > len(host_norm):
        return []

    spans: list[tuple[int, int]] = []
    search_start = 0
    root_len = len(root_norm)
    while True:
        idx = host_norm.find(root_norm, search_start)
        if idx == -1:
            break
        spans.append((idx, idx + root_len))
        search_start = idx + 1

    if not spans:
        return []

    def _build_artifacts(removed_spans: list[tuple[int, int]]) -> list[dict[str, object]]:
        artifacts: list[dict[str, object]] = []
        cursor = 0
        for rem_start, rem_end in sorted(removed_spans):
            if cursor < rem_start:
                artifacts.append(
                    {
                        "artifact": host_text[cursor:rem_start],
                        "start": cursor,
                        "end": rem_start,
                    }
                )
            cursor = max(cursor, rem_end)
        if cursor < len(host_text):
            artifacts.append(
                {
                    "artifact": host_text[cursor:],
                    "start": cursor,
                    "end": len(host_text),
                }
            )
        return artifacts

    remove_all_artifacts = _build_artifacts(spans)
    remove_all_payload = {
        "residual": "".join(str(a.get("artifact", "")) for a in remove_all_artifacts),
        "residuals": remove_all_artifacts,
        "removed_spans": [[s, e] for s, e in spans],
    }

    matches: list[dict[str, object]] = []
    total = len(spans)
    for index, (start, end) in enumerate(spans):
        residual_artifacts = _build_artifacts([(start, end)])
        residual = "".join(str(fragment.get("artifact", "")) for fragment in residual_artifacts)
        matches.append(
            {
                "host_word": host_text,
                "root": root_text,
                "residual": residual,
                "residuals": residual_artifacts,
                "start": start,
                "end": end,
                "occurrence_index": index,
                "total_occurrences": total,
                "occurrence_spans": [[s, e] for s, e in spans],
                "remove_all_occurrences": remove_all_payload,
            }
        )

    return matches


def build_subtraction_evidence(
    subtraction: dict[str, object],
    *,
    donor_source: str = "host_subtraction",
    recursion_depth: int = 0,
    termination_reason: str | None = None,
) -> dict[str, object]:
    """Normalize a subtraction record into a machine-readable evidence payload.

    This helper centralizes how subtraction traces are represented across the
    residual pipeline so downstream engines and analytics can consume a stable
    schema.
    """

    host_word = str(subtraction.get("host_word", "")).strip().upper()
    root = str(subtraction.get("root", "")).strip().upper()
    residual = str(subtraction.get("residual", "")).strip().upper()

    residual_artifacts_raw = subtraction.get("residuals") or []
    residual_artifacts: list[dict[str, object]] = []
    for artifact in residual_artifacts_raw:
        if not isinstance(artifact, dict):
            continue
        art_text = str(artifact.get("artifact", "")).strip().upper()
        if not art_text:
            continue
        residual_artifacts.append(
            {
                "artifact": art_text,
                "start": int(artifact.get("start", 0) or 0),
                "end": int(artifact.get("end", 0) or 0),
            }
        )

    payload = {
        "host_word": host_word,
        "root": root,
        "residual": residual,
        "start": int(subtraction.get("start", 0) or 0),
        "end": int(subtraction.get("end", 0) or 0),
        "equation": f"{host_word} - {root} = {residual}",
        "remaining_artifacts": residual_artifacts,
        "donor_source": donor_source,
        "recursion_depth": max(0, int(recursion_depth)),
        "termination_reason": termination_reason or "residual_extracted",
    }

    # Optional semantic anchors for prompt construction.
    for optional_key in ("host_definition", "donor_gloss", "donor_definition"):
        value = subtraction.get(optional_key)
        if value not in (None, ""):
            payload[optional_key] = value

    if "occurrence_index" in subtraction:
        payload["occurrence_index"] = int(subtraction.get("occurrence_index", 0) or 0)
    if "total_occurrences" in subtraction:
        payload["total_occurrences"] = int(subtraction.get("total_occurrences", 0) or 0)

    remove_all = subtraction.get("remove_all_occurrences")
    if isinstance(remove_all, dict):
        payload["remove_all_occurrences"] = {
            "residual": str(remove_all.get("residual", "")).strip().upper(),
            "residuals": [
                {
                    "artifact": str(item.get("artifact", "")).strip().upper(),
                    "start": int(item.get("start", 0) or 0),
                    "end": int(item.get("end", 0) or 0),
                }
                for item in (remove_all.get("residuals") or [])
                if isinstance(item, dict) and str(item.get("artifact", "")).strip()
            ],
            "removed_spans": [
                [int(span[0]), int(span[1])]
                for span in (remove_all.get("removed_spans") or [])
                if isinstance(span, (list, tuple)) and len(span) == 2
            ],
        }

    return payload


def prioritize_donor_candidates(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Order donor candidates by hierarchy: dictionary first, then SQLite, then others.

    Within the same source tier, prefer longer donor strings (largest subtractable
    known root first), then preserve lexical order for stability.
    """

    source_rank = {"dictionary": 0, "sqlite": 1}

    def _key(item: dict[str, object]) -> tuple[int, int, str]:
        source = str(item.get("source", "")).strip().lower()
        donor = str(item.get("donor", "")).strip().upper()
        return (source_rank.get(source, 2), -len(donor), donor)

    return sorted(candidates or [], key=_key)




def canonicalize_subtraction_text(value: object) -> str:
    """Normalize subtraction-related text for stable dedupe keys/equations."""

    return " ".join(str(value or "").strip().upper().split())


def canonicalize_subtraction_tuple(row: dict[str, object]) -> tuple[str, str, str]:
    """Return canonical ``(host_word, root, residual)`` tuple for subtraction rows."""

    return (
        canonicalize_subtraction_text(row.get("host_word")),
        canonicalize_subtraction_text(row.get("root")),
        canonicalize_subtraction_text(row.get("residual")),
    )


def canonicalize_subtraction_equation(row: dict[str, object]) -> str:
    """Build canonical ``HOST - ROOT = RESIDUAL`` equation string for a row."""

    host_word, row_root, residual = canonicalize_subtraction_tuple(row)
    return f"{host_word} - {row_root} = {residual}".strip()


def dedupe_residual_word_breaks(
    word_breaks: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Deduplicate rows by ``(host_word, root, residual)`` using source precedence."""

    def _source_rank(value: object) -> int:
        source = canonicalize_subtraction_text(value).lower()
        if source == "host_subtraction":
            return 0
        if source == "dictionary":
            return 1
        if source == "sqlite":
            return 2
        return 3

    deduped_rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in word_breaks or []:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        host_word, row_root, residual = canonicalize_subtraction_tuple(row_copy)
        if not host_word or not row_root:
            continue
        row_copy["host_word"] = host_word
        row_copy["root"] = row_root
        row_copy["residual"] = residual
        row_copy["equation"] = canonicalize_subtraction_equation(row_copy)

        key = (host_word, row_root, residual)
        preferred = deduped_rows_by_key.get(key)
        if preferred is None or _source_rank(row_copy.get("donor_source")) < _source_rank(
            preferred.get("donor_source")
        ):
            deduped_rows_by_key[key] = row_copy

    return list(deduped_rows_by_key.values())
def build_residual_guidance_payload(
    *,
    root: str,
    word_breaks: list[dict[str, object]] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a normalized residual-guidance payload for downstream engines.

    Ensures the integration contract always includes structured word-break rows
    and explicit subtraction equations.
    """

    normalized_root = canonicalize_subtraction_text(root)
    deduped_rows = dedupe_residual_word_breaks(word_breaks)

    equations = [
        canonicalize_subtraction_equation(row)
        for row in deduped_rows
        if any(canonicalize_subtraction_tuple(row))
    ]

    payload: dict[str, object] = {
        "root": normalized_root,
        "word_breaks": deduped_rows,
        "subtraction_equations": equations,
    }

    if isinstance(extra, dict):
        payload.update(extra)

    return payload


@dataclass
class ResidualDetail:
    word: str
    normalized: str
    definition: str
    coverage_ratio: float
    residual_ratio: float
    uncovered: list[str]            # raw fragments
    avg_confidence: float | None
    low_conf_segments: list[str]    # "frag@0.42"

    @property
    def length(self) -> int:
        return len(self.normalized or "")

    @classmethod
    def from_breakdown(cls, payload: dict[str, object]) -> "ResidualDetail":
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
    breakdown: dict[str, object] | None, root_norm: str, target: str
) -> dict[str, object]:
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

    kept_segments: list[dict[str, object]] = []
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
    uncovered: list[dict[str, object]] = []
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
    total_len: dict[str, int] = {}
    for d in details:
        for frag in d.uncovered:
            c[frag] += 1
            total_len[frag] = total_len.get(frag, 0) + len(frag)
    ranked = sorted(c.items(), key=lambda kv: (kv[1], total_len[kv[0]]), reverse=True)
    return [(frag, freq, total_len[frag]) for frag, freq in ranked[:k]]


def summarize_residuals(
    root: str,
    analyses: Iterable[dict[str, object]],
    *,
    max_focus: int = 5,
) -> dict[str, object]:
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
