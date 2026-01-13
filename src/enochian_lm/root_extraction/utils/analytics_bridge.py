"""Bridges analytic priors from ``enochian_lm`` into the translation pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from enochian_lm.common.sqlite_bootstrap import sqlite3


def _casefold(value: str | None) -> str:
    return value.casefold() if isinstance(value, str) else ""


def _normalize_token(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""


@dataclass
class PairwiseDelta:
    partner: str
    delta_root_given_partner: float
    delta_partner_given_root: float
    n_tokens: int
    cluster_freq: int

    @property
    def dominance(self) -> float:
        return self.delta_partner_given_root - self.delta_root_given_partner


@dataclass
class CollocationStat:
    partner: str
    shared: int
    root_total: int
    partner_total: int
    pmi: float | None
    llr: float | None
    asym_root_minus_partner: float | None

    @property
    def partner_excess(self) -> int:
        return max(self.partner_total - self.shared, 0)

    @property
    def root_excess(self) -> int:
        return max(self.root_total - self.shared, 0)


@dataclass
class ResidualClusterInfo:
    fragment: str
    cluster_id: int
    similarity: float
    cluster_size: int
    freq: int


@dataclass
class KnownRootEvidence:
    root: str
    definition: str | None
    semantic_core: str | None
    confidence_score: float | None
    attachment_profile: str | None
    cluster_freq: int


@dataclass
class ResidualOccurrence:
    fragment: str
    occurrences: int
    avg_residual_ratio: float | None
    avg_coverage_ratio: float | None
    freq: int


def _fetch_known_root_evidence(
    conn: sqlite3.Connection,
    partners: list[str],
    partner_counts: Counter[str] | None,
) -> list[KnownRootEvidence]:
    if not partners:
        return []

    partner_counts = partner_counts or Counter()
    cf_map = {_casefold(partner): partner for partner in partners if _normalize_token(partner)}
    if not cf_map:
        return []

    placeholders = ",".join("?" for _ in cf_map)
    try:
        rows = conn.execute(
            f"""
            SELECT g.root,
                   g.definition,
                   g.semantic_core,
                   g.confidence_score,
                   p.estimated_profile
            FROM root_glosses AS g
            LEFT JOIN root_attachment_profile AS p
              ON p.root = g.root
            WHERE g.root IN ({placeholders})
            """,
            tuple(cf_map.keys()),
        ).fetchall()
    except sqlite3.Error:
        return []

    results: list[KnownRootEvidence] = []
    for row in rows:
        root = _normalize_token(row["root"])
        if not root:
            continue
        results.append(
            KnownRootEvidence(
                root=root,
                definition=row["definition"],
                semantic_core=row["semantic_core"],
                confidence_score=(float(row["confidence_score"]) if row["confidence_score"] is not None else None),
                attachment_profile=row["estimated_profile"],
                cluster_freq=partner_counts.get(_casefold(root), 0),
            )
        )

    return sorted(results, key=lambda item: (item.cluster_freq, item.confidence_score or 0), reverse=True)


def _fetch_residual_occurrences(
    conn: sqlite3.Connection,
    fragments: list[str],
    counts: Counter[str] | None,
) -> list[ResidualOccurrence]:
    fragments = [_normalize_token(frag) for frag in fragments if _normalize_token(frag)]
    if not fragments:
        return []

    cf_map = {_casefold(frag): frag for frag in fragments}
    placeholders = ",".join("?" for _ in cf_map)
    try:
        rows = conn.execute(
            f"""
            SELECT LOWER(residual_span) AS span_cf,
                   COUNT(*) AS occurrences,
                   AVG(residual_ratio) AS avg_residual_ratio,
                   AVG(coverage_ratio) AS avg_coverage_ratio
            FROM residual_details
            WHERE residual_span IS NOT NULL
              AND TRIM(residual_span) <> ''
              AND LOWER(residual_span) IN ({placeholders})
            GROUP BY LOWER(residual_span)
            """,
            tuple(cf_map.keys()),
        ).fetchall()
    except sqlite3.Error:
        return []

    counts = counts or Counter()
    results: list[ResidualOccurrence] = []
    for row in rows:
        span_cf = _normalize_token(row["span_cf"])
        if not span_cf:
            continue
        results.append(
            ResidualOccurrence(
                fragment=cf_map.get(span_cf, span_cf),
                occurrences=int(row["occurrences"] or 0),
                avg_residual_ratio=(
                    float(row["avg_residual_ratio"]) if row["avg_residual_ratio"] is not None else None
                ),
                avg_coverage_ratio=(
                    float(row["avg_coverage_ratio"]) if row["avg_coverage_ratio"] is not None else None
                ),
                freq=counts.get(span_cf, 0),
            )
        )

    return sorted(results, key=lambda item: (item.freq, item.occurrences), reverse=True)


def gather_morph_evidence(
    conn: sqlite3.Connection,
    *,
    root: str,
    partner_counts: Counter[str] | None = None,
    residual_counts: Counter[str] | None = None,
    max_partners: int = 6,
    max_residuals: int = 5,
) -> dict[str, object]:
    """Collect analytic evidence for *root* from the shared SQLite database."""

    root_norm = _normalize_token(root)
    if not root_norm:
        return {"pairwise": [], "collocations": {}, "residual_clusters": [], "summary_lines": [], "focus_lines": []}

    partner_counts = partner_counts or Counter()
    residual_counts = residual_counts or Counter()
    partner_counts_cf = Counter({_casefold(k): v for k, v in partner_counts.items()})
    residual_counts_cf = Counter({_casefold(k): v for k, v in residual_counts.items()})

    known_roots = _fetch_known_root_evidence(
        conn,
        partners=list(partner_counts_cf.keys()),
        partner_counts=partner_counts_cf,
    )
    residual_occurrences = _fetch_residual_occurrences(
        conn,
        fragments=[frag for frag in residual_counts_cf.keys()],
        counts=residual_counts_cf,
    )

    summary_lines: list[str] = []
    focus_lines: list[str] = []

    if known_roots:
        summary_lines.append("Known root anchors:")
        for entry in known_roots[:max_partners]:
            pieces = []
            definition = str(entry.definition or "").strip()
            if definition:
                pieces.append(definition)
            if entry.semantic_core:
                pieces.append(f"core={entry.semantic_core}")
            if entry.attachment_profile:
                pieces.append(f"profile={entry.attachment_profile}")
            if entry.confidence_score is not None:
                pieces.append(f"confidence={entry.confidence_score:.2f}")
            if entry.cluster_freq:
                pieces.append(f"cluster_freq={entry.cluster_freq}")
            summary_lines.append(f"- {entry.root.upper()}: " + "; ".join(pieces))

            if entry.confidence_score is not None and entry.confidence_score >= 0.7:
                focus_lines.append(
                    f"{entry.root.upper()} already has an accepted gloss (confidence={entry.confidence_score:.2f}); "
                    "treat its semantics as a strong anchor when interpreting composites."
                )

    if residual_occurrences:
        summary_lines.append("Residual recurrence hints:")
        for info in residual_occurrences[:max_residuals]:
            details = [f"occurrences={info.occurrences}"]
            if info.avg_residual_ratio is not None:
                details.append(f"avg_residual_ratio={info.avg_residual_ratio:.2f}")
            if info.avg_coverage_ratio is not None:
                details.append(f"avg_coverage_ratio={info.avg_coverage_ratio:.2f}")
            if info.freq:
                details.append(f"cluster_freq={info.freq}")
            summary_lines.append(f"- {info.fragment.upper()}: " + ", ".join(details))
            if info.freq > 0 and info.occurrences >= 2:
                focus_lines.append(
                    f"Residual '{info.fragment.upper()}' recurs in prior residuals (n={info.occurrences}); "
                    "consider aligning its semantics with existing residual traces."
                )

    pairwise_payload: list[dict[str, object]] = []
    colloc_payload: dict[str, dict[str, object]] = {}

    residual_payload = [
        {
            "fragment": info.fragment,
            "occurrences": info.occurrences,
            "avg_residual_ratio": info.avg_residual_ratio,
            "avg_coverage_ratio": info.avg_coverage_ratio,
            "freq": info.freq,
        }
        for info in residual_occurrences
    ]

    known_roots_payload = [
        {
            "root": entry.root,
            "definition": entry.definition,
            "semantic_core": entry.semantic_core,
            "confidence_score": entry.confidence_score,
            "attachment_profile": entry.attachment_profile,
            "cluster_freq": entry.cluster_freq,
        }
        for entry in known_roots
    ]

    return {
        "root": root_norm,
        "pairwise": pairwise_payload,
        "collocations": colloc_payload,
        "residual_clusters": residual_payload,
        "known_roots": known_roots_payload,
        "summary_lines": summary_lines,
        "focus_lines": focus_lines,
    }


__all__ = ["gather_morph_evidence", "PairwiseDelta", "CollocationStat", "ResidualClusterInfo"]
