"""Generate an end-to-end pipeline report for the Enochian LM project."""
from __future__ import annotations

import base64
import io
import json
import logging
import math
from common.sqlite_bootstrap import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover - numpy might be unavailable
    _np = None

try:  # pragma: no cover - optional dependency
    import pandas as _pd
except Exception:  # pragma: no cover - pandas might be unavailable
    _pd = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as _plt
except Exception:  # pragma: no cover - matplotlib might be unavailable
    _plt = None

from ..utils.text import utcnow_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TableData:
    """Container describing table rows with an optional DataFrame representation."""

    rows: list[dict[str, Any]]
    dataframe: "_pd.DataFrame | None"


def _vector_from_json(blob: str) -> list[float]:
    values = json.loads(blob)
    if _np is not None:
        return _np.asarray(values, dtype=float).tolist()
    return [float(x) for x in values]


def _vector_norm(values: Sequence[float]) -> float:
    if _np is not None:
        return float(_np.linalg.norm(_np.asarray(values, dtype=float)))
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if _np is not None:
        arr_a = _np.asarray(a, dtype=float)
        arr_b = _np.asarray(b, dtype=float)
        denom = float(_np.linalg.norm(arr_a) * _np.linalg.norm(arr_b))
        if denom == 0:
            return 0.0
        return float(_np.dot(arr_a, arr_b) / (denom + 1e-9))
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(x) * float(x) for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b + 1e-9)


def _load_table(conn: sqlite3.Connection, table: str) -> TableData:
    """Load *table* from *conn* and return rows with optional DataFrame."""

    try:
        cursor = conn.execute(f"SELECT * FROM {table}")
    except sqlite3.OperationalError as exc:
        logger.warning("Table missing; skipping", extra={"table": table, "error": str(exc)})
        return TableData([], None)

    rows = [dict(row) for row in cursor.fetchall()]
    if _pd is not None and rows:
        dataframe = _pd.DataFrame(rows)
    else:
        dataframe = None
    return TableData(rows, dataframe)


def _load_baseline_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "coverage_ratio_mean": None,
            "residual_ratio_mean": None,
            "avg_confidence": None,
            "token_errors": {},
        }

    coverage_values: list[float] = []
    residual_values: list[float] = []
    confidence_values: list[float] = []
    token_errors: dict[str, float] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed baseline line", extra={"line": line[:40]})
                continue

            coverage = payload.get("coverage_ratio") or payload.get("coverage_ratio_mean")
            if isinstance(coverage, (int, float)):
                coverage_values.append(float(coverage))
            residual = payload.get("residual_ratio") or payload.get("residual_ratio_mean")
            if isinstance(residual, (int, float)):
                residual_values.append(float(residual))
            confidence = payload.get("confidence") or payload.get("avg_confidence")
            if isinstance(confidence, (int, float)):
                confidence_values.append(float(confidence))

            token = payload.get("token") or payload.get("normalized")
            error_value = payload.get("recon_error") or payload.get("residual_ratio")
            if isinstance(token, str) and isinstance(error_value, (int, float)):
                token_errors[token] = float(error_value)

    def _mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    return {
        "coverage_ratio_mean": _mean(coverage_values),
        "residual_ratio_mean": _mean(residual_values),
        "avg_confidence": _mean(confidence_values),
        "token_errors": token_errors,
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(list(row))


def _render_plot(fig: "_plt.Figure | None") -> str:
    if fig is None or _plt is None:
        return ""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    _plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"<img src='data:image/png;base64,{encoded}' alt='plot' />"


def _histogram_plot(values: Sequence[float], title: str, bins: int = 20) -> str:
    if _plt is None or not values:
        return ""
    fig, ax = _plt.subplots(figsize=(6, 4))
    ax.hist(list(values), bins=bins, color="#1f77b4", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _render_plot(fig)


def _scatter_plot(x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str) -> str:
    if _plt is None or not x or not y:
        return ""
    fig, ax = _plt.subplots(figsize=(6, 4))
    ax.scatter(list(x), list(y), s=12, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return _render_plot(fig)


def _heatmap_plot(matrix: list[list[float]], labels: list[str], title: str) -> str:
    if _plt is None or not matrix or not labels:
        return ""
    fig, ax = _plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return _render_plot(fig)


def _table_html(title: str, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    html = [f"<h3>{title}</h3>"]
    html.append("<table class='data'>")
    html.append("<thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead>")
    html.append("<tbody>")
    for row in rows:
        html.append("<tr>" + "".join(f"<td>{_format_cell(value)}</td>" for value in row) + "</tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _prefix(value: str, length: int = 3) -> str:
    return value[:length] if value else ""


def _make_section(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2>{body}</section>"


def _summarize_attribution(attrib_rows: list[dict[str, Any]], out_dir: Path) -> tuple[str, dict[str, Any]]:
    if not attrib_rows:
        return _make_section("Attribution", "<p>No attribution data available.</p>"), {
            "pairs": 0,
            "avg_abs_delta": 0.0,
        }

    sorted_rows = sorted(
        attrib_rows,
        key=lambda row: abs(float(row.get("delta_a_given_b", 0.0))) + abs(float(row.get("delta_b_given_a", 0.0))),
        reverse=True,
    )
    top_pairs = [
        (
            row["morph_a"],
            row["morph_b"],
            float(row.get("delta_a_given_b", 0.0)),
            float(row.get("delta_b_given_a", 0.0)),
        )
        for row in sorted_rows[:20]
    ]

    morph_totals: dict[str, list[float]] = defaultdict(list)
    for row in attrib_rows:
        morph_totals[row["morph_a"]].append(abs(float(row.get("delta_a_given_b", 0.0))))
        morph_totals[row["morph_b"]].append(abs(float(row.get("delta_b_given_a", 0.0))))

    morph_avg = sorted(
        (
            morph,
            _mean(values),
        )
        for morph, values in morph_totals.items()
    )

    abs_values = [abs(float(row.get("delta_a_given_b", 0.0))) for row in attrib_rows]
    abs_values += [abs(float(row.get("delta_b_given_a", 0.0))) for row in attrib_rows]
    avg_abs_delta = _mean(abs_values)

    csv_rows = [
        (
            row["morph_a"],
            row["morph_b"],
            float(row.get("delta_a_given_b", 0.0)),
            float(row.get("delta_b_given_a", 0.0)),
        )
        for row in sorted_rows
    ]
    _write_csv(
        out_dir / "attribution_pairs.csv",
        ("morph_a", "morph_b", "delta_a_given_b", "delta_b_given_a"),
        csv_rows,
    )

    body_parts: list[str] = []
    body_parts.append(
        _table_html(
            "Top 20 morph pairs by |Δ|",
            ("Morph A", "Morph B", "Δ A|B", "Δ B|A"),
            top_pairs,
        )
    )

    if morph_avg:
        morph_avg_sorted = sorted(morph_avg, key=lambda item: item[1], reverse=True)[:20]
        body_parts.append(
            _table_html(
                "Average |Δ| per morph (top 20)",
                ("Morph", "Average |Δ|"),
                morph_avg_sorted,
            )
        )

    section_html = _make_section("Attribution (4A)", "".join(body_parts))
    return section_html, {"pairs": len(attrib_rows), "avg_abs_delta": avg_abs_delta}


def _summarize_collocations(colloc_rows: list[dict[str, Any]], out_dir: Path) -> tuple[str, dict[str, Any]]:
    if not colloc_rows:
        return _make_section("Collocations", "<p>No collocation statistics available.</p>"), {
            "pairs": 0,
            "avg_pmi": 0.0,
            "avg_llr": 0.0,
            "avg_asym": 0.0,
        }

    sorted_by_pmi = sorted(colloc_rows, key=lambda row: float(row.get("pmi", 0.0)), reverse=True)
    sorted_by_llr = sorted(colloc_rows, key=lambda row: float(row.get("llr", 0.0)), reverse=True)
    sorted_by_asym = sorted(
        colloc_rows,
        key=lambda row: abs(float(row.get("asym_dep", 0.0))),
        reverse=True,
    )

    csv_rows = [
        (
            row["morph_left"],
            row["morph_right"],
            int(row.get("count_ab", 0)),
            float(row.get("pmi", 0.0)),
            float(row.get("llr", 0.0)),
            float(row.get("asym_dep", 0.0)),
        )
        for row in colloc_rows
    ]
    _write_csv(
        out_dir / "collocations.csv",
        ("morph_left", "morph_right", "count_ab", "pmi", "llr", "asym_dep"),
        csv_rows,
    )

    body_parts: list[str] = []
    body_parts.append(
        _table_html(
            "Top 10 PMI pairs",
            ("Morph L", "Morph R", "PMI", "LLR", "Asym"),
            [
                (
                    row["morph_left"],
                    row["morph_right"],
                    float(row.get("pmi", 0.0)),
                    float(row.get("llr", 0.0)),
                    float(row.get("asym_dep", 0.0)),
                )
                for row in sorted_by_pmi[:10]
            ],
        )
    )
    body_parts.append(
        _table_html(
            "Top 10 LLR pairs",
            ("Morph L", "Morph R", "PMI", "LLR", "Asym"),
            [
                (
                    row["morph_left"],
                    row["morph_right"],
                    float(row.get("pmi", 0.0)),
                    float(row.get("llr", 0.0)),
                    float(row.get("asym_dep", 0.0)),
                )
                for row in sorted_by_llr[:10]
            ],
        )
    )
    body_parts.append(
        _table_html(
            "Top 10 asymmetric dependency pairs",
            ("Morph L", "Morph R", "PMI", "LLR", "Asym"),
            [
                (
                    row["morph_left"],
                    row["morph_right"],
                    float(row.get("pmi", 0.0)),
                    float(row.get("llr", 0.0)),
                    float(row.get("asym_dep", 0.0)),
                )
                for row in sorted_by_asym[:10]
            ],
        )
    )

    pmi_values = [float(row.get("pmi", 0.0)) for row in colloc_rows]
    llr_values = [float(row.get("llr", 0.0)) for row in colloc_rows]
    scatter_html = _scatter_plot(pmi_values, llr_values, "PMI vs LLR", "PMI", "LLR")
    if scatter_html:
        body_parts.append(scatter_html)

    # Optional heatmap for small sets
    high_pmi = [row for row in colloc_rows if float(row.get("pmi", 0.0)) > 1.0]
    if 1 < len(high_pmi) <= 50:
        labels = sorted({row["morph_left"] for row in high_pmi} | {row["morph_right"] for row in high_pmi})
        label_index = {label: idx for idx, label in enumerate(labels)}
        matrix = [[0.0 for _ in labels] for _ in labels]
        for row in high_pmi:
            i = label_index[row["morph_left"]]
            j = label_index[row["morph_right"]]
            matrix[i][j] = float(row.get("pmi", 0.0))
        heatmap_html = _heatmap_plot(matrix, labels, "PMI heatmap (>1.0)")
        if heatmap_html:
            body_parts.append(heatmap_html)

    avg_pmi = _mean(pmi_values)
    avg_llr = _mean(llr_values)
    avg_asym = _mean([abs(float(row.get("asym_dep", 0.0))) for row in colloc_rows])

    section_html = _make_section("Collocation (4B)", "".join(body_parts))
    return section_html, {
        "pairs": len(colloc_rows),
        "avg_pmi": avg_pmi,
        "avg_llr": avg_llr,
        "avg_asym": avg_asym,
    }


def _summarize_clusters(
    cluster_rows: list[dict[str, Any]],
    membership_rows: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    if not cluster_rows:
        return _make_section("Residual clusters", "<p>No residual clusters available.</p>"), {
            "clusters": 0,
            "mean_sim": 0.0,
        }

    clusters_by_id = {int(row["cluster_id"]): row for row in cluster_rows}
    members_by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in membership_rows:
        cluster_id = int(row.get("cluster_id", 0))
        members_by_cluster[cluster_id].append(row)

    summary_rows: list[tuple[int, int, float, list[str]]] = []
    prefix_map: dict[str, set[int]] = defaultdict(set)
    mean_sims: list[float] = []

    for cluster_id, cluster in sorted(clusters_by_id.items()):
        members = members_by_cluster.get(cluster_id, [])
        sims = [float(member.get("sim_to_centroid", 0.0)) for member in members]
        examples = [member.get("residual_span", "") for member in members[:5]]
        summary_rows.append(
            (
                cluster_id,
                int(cluster.get("size", len(members))),
                _mean(sims),
                examples,
            )
        )
        mean_sims.extend(sims)
        for member in members:
            prefix_map[_prefix(member.get("residual_span", ""))].add(cluster_id)

    cluster_csv_rows = [
        (
            cluster_id,
            json.dumps(json.loads(cluster.get("centroid_json", "[]"))),
            size,
            mean_sim,
        )
        for cluster_id, size, mean_sim, _ in summary_rows
    ]
    _write_csv(
        out_dir / "residual_clusters_summary.csv",
        ("cluster_id", "centroid_json", "size", "mean_similarity"),
        cluster_csv_rows,
    )

    overlaps = [
        (prefix, sorted(list(cluster_ids)))
        for prefix, cluster_ids in prefix_map.items()
        if prefix and len(cluster_ids) > 1
    ]

    body_parts = [
        _table_html(
            "Residual cluster summary",
            ("Cluster", "Size", "Mean sim", "Example morphs"),
            [
                (
                    cluster_id,
                    size,
                    mean_sim,
                    ", ".join(example for example in examples if example),
                )
                for cluster_id, size, mean_sim, examples in summary_rows
            ],
        )
    ]

    if overlaps:
        overlap_rows = [
            (prefix, ", ".join(str(cid) for cid in cluster_ids)) for prefix, cluster_ids in sorted(overlaps)
        ]
        body_parts.append(
            _table_html(
                "Shared morph prefixes across clusters",
                ("Prefix", "Cluster IDs"),
                overlap_rows,
            )
        )

    mean_sim = _mean(mean_sims)
    section_html = _make_section("Residual clustering (4C)", "".join(body_parts))
    return section_html, {"clusters": len(summary_rows), "mean_sim": mean_sim, "details": summary_rows}


def _summarize_factorization(
    morph_rows: list[dict[str, Any]],
    composite_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    membership_rows: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[str, dict[str, Any]]:
    if not morph_rows or not composite_rows:
        return _make_section("Morph factorization", "<p>Factorization data unavailable.</p>"), {
            "tokens": len(composite_rows),
            "morphs": len(morph_rows),
            "dim": 0,
            "mean_error": 0.0,
            "median_error": 0.0,
        }

    recon_errors = [float(row.get("recon_error", 0.0)) for row in composite_rows]
    mean_error = _mean(recon_errors)
    median_error = _median(recon_errors)

    morph_vectors = {
        row["morph"]: _vector_from_json(row.get("vector_json", "[]"))
        for row in morph_rows
    }
    dim = len(next(iter(morph_vectors.values()))) if morph_vectors else 0
    norms = {
        morph: _vector_norm(vector)
        for morph, vector in morph_vectors.items()
    }

    morph_table = sorted(norms.items(), key=lambda item: item[1], reverse=True)[:10]

    hist_html = _histogram_plot(recon_errors, "Reconstruction error distribution")

    centroid_vectors = {
        int(row["cluster_id"]): json.loads(row.get("centroid_json", "[]"))
        for row in cluster_rows
    }
    best_similarity: dict[str, float] = {}
    for row in membership_rows:
        morph = row.get("residual_span")
        sim = float(row.get("sim_to_centroid", 0.0))
        if morph and sim >= best_similarity.get(morph, float("-inf")):
            best_similarity[morph] = sim

    common_morphs = [
        (norms[morph], best_similarity[morph])
        for morph in sorted(norms.keys())
        if morph in best_similarity
    ]
    corr = None
    if common_morphs and hasattr(statistics, "correlation"):
        norm_vals, sim_vals = zip(*common_morphs)
        try:
            corr = statistics.correlation(norm_vals, sim_vals)
        except statistics.StatisticsError:
            corr = None

    body_parts: list[str] = []
    body_parts.append(
        _table_html(
            "Top 10 morphs by L2 norm",
            ("Morph", "L2 norm"),
            morph_table,
        )
    )
    if hist_html:
        body_parts.append(hist_html)

    if corr is not None:
        body_parts.append(f"<p>Correlation between morph norm and best cluster similarity: {corr:.4f}</p>")

    centroid_info = []
    if centroid_vectors and best_similarity:
        for morph, sim in sorted(best_similarity.items(), key=lambda item: item[1], reverse=True)[:10]:
            centroid_id = None
            best_value = sim
            # find centroid id by matching membership rows
            for row in membership_rows:
                if row.get("residual_span") == morph and float(row.get("sim_to_centroid", 0.0)) == sim:
                    centroid_id = int(row.get("cluster_id", 0))
                    break
            centroid_info.append((morph, centroid_id, best_value))
        body_parts.append(
            _table_html(
                "Best morph ↔ cluster alignments",
                ("Morph", "Cluster", "Similarity"),
                centroid_info,
            )
        )

    section_html = _make_section("Morph factorization (4D)", "".join(body_parts))
    summary = {
        "tokens": len(composite_rows),
        "morphs": len(morph_rows),
        "dim": dim,
        "mean_error": mean_error,
        "median_error": median_error,
    }
    if corr is not None:
        summary["norm_cluster_correlation"] = corr
    return section_html, summary


def _summarize_coverage(
    composite_rows: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if not composite_rows:
        return _make_section("Coverage", "<p>No composite reconstruction data available.</p>"), {
            "coverage_ratio_mean": None,
            "residual_ratio_mean": None,
            "avg_confidence": None,
        }

    coverage_values: list[float] = []
    residual_values: list[float] = []
    confidence_values: list[float] = []
    recon_errors: list[float] = []
    token_errors: dict[str, float] = {}

    for row in composite_rows:
        try:
            morphs = json.loads(row.get("used_morphs_json", "[]"))
        except json.JSONDecodeError:
            morphs = []
        recon_error = float(row.get("recon_error", 0.0))
        recon_errors.append(recon_error)
        coverage_values.append(1.0 if morphs else 0.0)
        residual_values.append(recon_error)
        confidence_values.append(1.0 / (1.0 + recon_error))
        token = row.get("token")
        if isinstance(token, str):
            token_errors[token] = recon_error

    mean_cov = _mean(coverage_values)
    mean_residual = _mean(residual_values)
    mean_confidence = _mean(confidence_values)

    baseline_cov = baseline.get("coverage_ratio_mean")
    baseline_residual = baseline.get("residual_ratio_mean")
    baseline_conf = baseline.get("avg_confidence")

    def _format_delta(current: float | None, base: float | None) -> str:
        if current is None:
            return "N/A"
        if base is None:
            return f"{current:.4f}"
        delta = current - base
        arrow = "↑" if delta >= 0 else "↓"
        return f"{current:.4f} ({arrow}{abs(delta):.4f})"

    body_parts = [
        "<ul>",
        f"<li>Coverage ratio mean: {_format_delta(mean_cov, baseline_cov)}</li>",
        f"<li>Residual ratio mean: {_format_delta(mean_residual, baseline_residual)}</li>",
        f"<li>Average confidence: {_format_delta(mean_confidence, baseline_conf)}</li>",
        "</ul>",
    ]

    hist_html = _histogram_plot(recon_errors, "Reconstruction errors (current)")
    if hist_html:
        body_parts.append(hist_html)

    baseline_errors: dict[str, float] = baseline.get("token_errors", {})
    diff_rows = []
    for token, error in token_errors.items():
        base_error = baseline_errors.get(token)
        if base_error is not None:
            diff_rows.append((token, base_error, error, error - base_error))
    diff_rows.sort(key=lambda item: item[3])
    decreases = diff_rows[:10]
    increases = diff_rows[-10:][::-1]

    if decreases:
        body_parts.append(
            _table_html(
                "Top 10 tokens with largest error decrease",
                ("Token", "Baseline", "Current", "Δ"),
                decreases,
            )
        )
    if increases:
        body_parts.append(
            _table_html(
                "Top 10 tokens with largest error increase",
                ("Token", "Baseline", "Current", "Δ"),
                increases,
            )
        )

    section_html = _make_section("Coverage & residuals", "".join(body_parts))
    summary = {
        "coverage_ratio_mean": mean_cov,
        "residual_ratio_mean": mean_residual,
        "avg_confidence": mean_confidence,
    }
    return section_html, summary


def _build_metadata(db_path: Path) -> dict[str, Any]:
    import platform
    import sys

    metadata = {
        "timestamp": utcnow_iso(),
        "db_path": str(db_path),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    norm_config = Path("datasets/processed/norm_v1.yaml")
    if norm_config.exists():
        metadata["norm_config_hash"] = norm_config.read_bytes().hex()[:40]

    git_dir = Path(".git")
    head_ref = git_dir / "HEAD"
    if head_ref.exists():
        ref = head_ref.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = git_dir / ref.split(" ", 1)[1]
            if ref_path.exists():
                metadata["git_commit"] = ref_path.read_text().strip()
        else:
            metadata["git_commit"] = ref

    return metadata


def generate_pipeline_report(
    conn: sqlite3.Connection,
    out_dir: str,
    *,
    db_path: str,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    """Generate the pipeline report in *out_dir* using *conn* data."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    baseline_file = Path(baseline_path) if baseline_path else Path("datasets/processed/baseline_residuals.jsonl")
    baseline_metrics = _load_baseline_metrics(baseline_file)

    attribution = _load_table(conn, "attribution_marginals")
    collocations = _load_table(conn, "collocation_stats")
    clusters = _load_table(conn, "residual_clusters")
    memberships = _load_table(conn, "residual_cluster_membership")
    morph_vectors = _load_table(conn, "morph_semantic_vectors")
    composite_recon = _load_table(conn, "composite_reconstruction")

    sections: list[str] = []
    coverage_html, coverage_summary = _summarize_coverage(composite_recon.rows, baseline_metrics)
    sections.append(coverage_html)

    attrib_html, attrib_summary = _summarize_attribution(attribution.rows, out_path)
    sections.append(attrib_html)

    colloc_html, colloc_summary = _summarize_collocations(collocations.rows, out_path)
    sections.append(colloc_html)

    cluster_html, cluster_summary = _summarize_clusters(clusters.rows, memberships.rows, out_path)
    sections.append(cluster_html)

    factor_html, factor_summary = _summarize_factorization(
        morph_vectors.rows,
        composite_recon.rows,
        clusters.rows,
        memberships.rows,
        out_path,
    )
    sections.append(factor_html)

    metadata = _build_metadata(Path(db_path))

    summary_index = {
        "coverage": coverage_summary,
        "attribution": attrib_summary,
        "collocation": colloc_summary,
        "residual_clusters": cluster_summary,
        "factorization": factor_summary,
        "metadata": metadata,
    }

    _write_json(out_path / "pipeline_index.json", summary_index)

    body_html = "".join(sections)
    full_html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>Enochian LM pipeline report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; background-color: #fafafa; }}
section {{ margin-bottom: 2rem; padding: 1rem; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
h1 {{ text-align: center; }}
table.data {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
table.data th, table.data td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
table.data thead {{ background-color: #f0f0f0; }}
img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto; }}
</style>
</head>
<body>
<h1>Enochian LM pipeline report</h1>
<p>Generated at {metadata['timestamp']}.</p>
{body_html}
</body>
</html>
"""
    _write_text(out_path / "pipeline_report.html", full_html)

    logger.info(
        "Pipeline report generated",
        extra={
            "coverage_delta": coverage_summary.get("coverage_ratio_mean"),
            "residual_delta": coverage_summary.get("residual_ratio_mean"),
            "mean_delta": attrib_summary.get("avg_abs_delta"),
            "clusters": cluster_summary.get("clusters"),
            "mean_error": factor_summary.get("mean_error"),
        },
    )

    return summary_index


__all__ = ["generate_pipeline_report"]
