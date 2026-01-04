"""Bridges analytic priors from ``enochian_lm`` into the translation pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
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


def _fetch_attribution(
    conn: sqlite3.Connection,
    root_cf: str,
    partner_counts: Counter[str] | None,
) -> list[PairwiseDelta]:
    try:
        rows = conn.execute(
            """
            SELECT morph_a, morph_b, delta_a_given_b, delta_b_given_a, n_tokens
            FROM attribution_marginals
            WHERE LOWER(morph_a) = ? OR LOWER(morph_b) = ?
            """,
            (root_cf, root_cf),
        ).fetchall()
    except sqlite3.Error:
        return []

    results: list[PairwiseDelta] = []
    partner_counts = partner_counts or Counter()

    for row in rows:
        morph_a = _normalize_token(row["morph_a"])
        morph_b = _normalize_token(row["morph_b"])
        partner: str
        delta_root_given_partner: float
        delta_partner_given_root: float

        if _casefold(morph_a) == root_cf:
            partner = morph_b
            delta_root_given_partner = float(row["delta_a_given_b"])
            delta_partner_given_root = float(row["delta_b_given_a"])
        else:
            partner = morph_a
            delta_root_given_partner = float(row["delta_b_given_a"])
            delta_partner_given_root = float(row["delta_a_given_b"])

        results.append(
            PairwiseDelta(
                partner=partner,
                delta_root_given_partner=round(delta_root_given_partner, 4),
                delta_partner_given_root=round(delta_partner_given_root, 4),
                n_tokens=int(row["n_tokens"] or 0),
                cluster_freq=partner_counts.get(_casefold(partner), 0),
            )
        )

    return sorted(
        results,
        key=lambda item: (item.cluster_freq, abs(item.dominance), item.n_tokens),
        reverse=True,
    )


def _fetch_collocations(
    conn: sqlite3.Connection,
    root_cf: str,
) -> dict[str, CollocationStat]:
    try:
        rows = conn.execute(
            """
            SELECT morph_left, morph_right, count_ab, count_a, count_b, pmi, llr, asym_dep
            FROM collocation_stats
            WHERE LOWER(morph_left) = ? OR LOWER(morph_right) = ?
            """,
            (root_cf, root_cf),
        ).fetchall()
    except sqlite3.Error:
        return {}

    colloc_by_partner: dict[str, CollocationStat] = {}

    for row in rows:
        left = _normalize_token(row["morph_left"])
        right = _normalize_token(row["morph_right"])
        if _casefold(left) == root_cf:
            partner = right
            shared = int(row["count_ab"] or 0)
            root_total = int(row["count_a"] or 0)
            partner_total = int(row["count_b"] or 0)
            asym = row["asym_dep"]
        else:
            partner = left
            shared = int(row["count_ab"] or 0)
            root_total = int(row["count_b"] or 0)
            partner_total = int(row["count_a"] or 0)
            asym = row["asym_dep"]
            if asym is not None:
                asym = -float(asym)

        colloc_by_partner[_casefold(partner)] = CollocationStat(
            partner=partner,
            shared=shared,
            root_total=root_total,
            partner_total=partner_total,
            pmi=(float(row["pmi"]) if row["pmi"] is not None else None),
            llr=(float(row["llr"]) if row["llr"] is not None else None),
            asym_root_minus_partner=(float(asym) if asym is not None else None),
        )

    return colloc_by_partner


def _fetch_residual_clusters(
    conn: sqlite3.Connection,
    fragments: Iterable[str],
    counts: Counter[str] | None,
) -> list[ResidualClusterInfo]:
    fragments = [_normalize_token(frag) for frag in fragments if _normalize_token(frag)]
    if not fragments:
        return []

    cf_map = {_casefold(frag): frag for frag in fragments}
    placeholders = ",".join("?" for _ in cf_map)
    try:
        membership_rows = conn.execute(
            f"""
            SELECT residual_span, cluster_id, sim_to_centroid
            FROM residual_cluster_membership
            WHERE LOWER(residual_span) IN ({placeholders})
            """,
            tuple(cf_map.keys()),
        ).fetchall()
    except sqlite3.Error:
        return []

    cluster_sizes: dict[int, int] = {}
    try:
        for row in conn.execute(
            "SELECT cluster_id, size FROM residual_clusters"
        ).fetchall():
            cluster_sizes[int(row["cluster_id"])] = int(row["size"] or 0)
    except sqlite3.Error:
        cluster_sizes.clear()

    counts = counts or Counter()
    results: list[ResidualClusterInfo] = []

    for row in membership_rows:
        span = _normalize_token(row["residual_span"])
        cf = _casefold(span)
        results.append(
            ResidualClusterInfo(
                fragment=span,
                cluster_id=int(row["cluster_id"]),
                similarity=float(row["sim_to_centroid"] or 0.0),
                cluster_size=cluster_sizes.get(int(row["cluster_id"]), 0),
                freq=counts.get(cf, 0),
            )
        )

    return sorted(
        results,
        key=lambda item: (item.freq, item.similarity),
        reverse=True,
    )


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

    pairwise = _fetch_attribution(conn, _casefold(root_norm), partner_counts_cf)
    colloc = _fetch_collocations(conn, _casefold(root_norm))
    residuals = _fetch_residual_clusters(
        conn,
        fragments=[frag for frag in residual_counts_cf.keys()],
        counts=residual_counts_cf,
    )

    summary_lines: list[str] = []
    focus_lines: list[str] = []

    if pairwise:
        summary_lines.append("Attribution scoreboard:")
        for entry in pairwise[:max_partners]:
            partner_cf = _casefold(entry.partner)
            colloc_stats = colloc.get(partner_cf)
            pieces = [
                f"Δ({entry.partner}|{root_norm})={entry.delta_partner_given_root:.2f}",
                f"Δ({root_norm}|{entry.partner})={entry.delta_root_given_partner:.2f}",
                f"n={entry.n_tokens}",
            ]
            if entry.cluster_freq:
                pieces.append(f"cluster_freq={entry.cluster_freq}")
            if colloc_stats:
                pieces.append(
                    f"co-occurrence={colloc_stats.shared}/{colloc_stats.root_total}:{colloc_stats.partner_total}"
                )
            summary_lines.append(f"- {entry.partner.upper()}: " + ", ".join(pieces))

            dominance = entry.dominance
            if dominance > 0.05 and entry.delta_partner_given_root >= 0.1:
                msg = (
                    f"{entry.partner.upper()} dominates shared semantics with {root_norm.upper()} "
                    f"(Δ={entry.delta_partner_given_root:.2f} vs {entry.delta_root_given_partner:.2f}; n={entry.n_tokens})"
                )
                if colloc_stats:
                    msg += (
                        f"; {entry.partner.upper()}-only occurrences={colloc_stats.partner_excess}, "
                        f"{root_norm.upper()}-only={colloc_stats.root_excess}"
                    )
                focus_lines.append(msg)
            elif dominance < -0.05 and entry.delta_root_given_partner >= 0.1:
                msg = (
                    f"{root_norm.upper()} contributes more than {entry.partner.upper()} "
                    f"(Δ={entry.delta_root_given_partner:.2f} vs {entry.delta_partner_given_root:.2f}; n={entry.n_tokens})"
                )
                focus_lines.append(msg)

    if residuals:
        summary_lines.append("Residual cluster hints:")
        for info in residuals[:max_residuals]:
            summary_lines.append(
                f"- {info.fragment.upper()} → cluster {info.cluster_id} (size={info.cluster_size}, "
                f"sim={info.similarity:.2f}, freq={info.freq})"
            )
            if info.freq > 0 and info.similarity >= 0.6:
                focus_lines.append(
                    f"Residual '{info.fragment.upper()}' recurs (freq={info.freq}) and clusters with ID {info.cluster_id}; "
                    "consider allocating semantics there."
                )

    pairwise_payload = [
        {
            "partner": entry.partner,
            "delta_root_given_partner": entry.delta_root_given_partner,
            "delta_partner_given_root": entry.delta_partner_given_root,
            "n_tokens": entry.n_tokens,
            "cluster_freq": entry.cluster_freq,
            "dominance": round(entry.dominance, 4),
        }
        for entry in pairwise
    ]

    colloc_payload = {
        key: {
            "partner": stat.partner,
            "shared": stat.shared,
            "root_total": stat.root_total,
            "partner_total": stat.partner_total,
            "pmi": stat.pmi,
            "llr": stat.llr,
            "asym_root_minus_partner": stat.asym_root_minus_partner,
            "partner_excess": stat.partner_excess,
            "root_excess": stat.root_excess,
        }
        for key, stat in colloc.items()
    }

    residual_payload = [
        {
            "fragment": info.fragment,
            "cluster_id": info.cluster_id,
            "similarity": info.similarity,
            "cluster_size": info.cluster_size,
            "freq": info.freq,
        }
        for info in residuals
    ]

    return {
        "root": root_norm,
        "pairwise": pairwise_payload,
        "collocations": colloc_payload,
        "residual_clusters": residual_payload,
        "summary_lines": summary_lines,
        "focus_lines": focus_lines,
    }


__all__ = ["gather_morph_evidence", "PairwiseDelta", "CollocationStat", "ResidualClusterInfo"]
