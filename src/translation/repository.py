from __future__ import annotations

import json
from enochian_lm.common.sqlite_bootstrap import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ResidualDetail:
    normalized: str
    definition: Optional[str]
    coverage_ratio: Optional[float]
    residual_ratio: Optional[float]
    avg_confidence: Optional[float]
    uncovered: List[str] = field(default_factory=list)
    low_confidence: List[str] = field(default_factory=list)


@dataclass
class ClusterRecord:
    variant: str
    cluster_id: int
    run_id: str
    ngram: str
    cluster_index: int
    glossator_def: Optional[str]
    residual_explained: Optional[float]
    residual_ratio: Optional[float]
    residual_headline: Optional[str]
    residual_focus_prompt: Optional[str]
    semantic_coverage: Optional[float]
    cohesion: Optional[float]
    semantic_cohesion: Optional[float]
    best_config: Optional[str]
    residual_details: List[ResidualDetail] = field(default_factory=list)
    raw_definitions: List[Dict[str, Optional[str]]] = field(default_factory=list)


class InsightsRepository:
    """Read access to solo/debate insights databases."""

    def __init__(self, *, solo_path: Optional[Path], debate_path: Optional[Path]):
        paths = {"solo": solo_path, "debate": debate_path}
        self._connections: Dict[str, sqlite3.Connection] = {}
        for variant, path in paths.items():
            if not path:
                continue
            abs_path = Path(path)
            if not abs_path.exists():
                continue
            conn = sqlite3.connect(str(abs_path))
            conn.row_factory = sqlite3.Row
            self._connections[variant] = conn

    @property
    def variants(self) -> List[str]:
        return sorted(self._connections.keys())

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def require_variants(self, expected: Iterable[str]) -> None:
        missing = [v for v in expected if v not in self._connections]
        if missing:
            raise FileNotFoundError(
                "Missing insights database(s): " + ", ".join(missing)
            )

    def fetch_clusters(self, ngram: str, *, variants: Optional[Iterable[str]] = None) -> List[ClusterRecord]:
        ngram_key = ngram.upper()
        selected = list(variants) if variants else self.variants
        clusters: List[ClusterRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            variant_clusters = self._fetch_variant_clusters(conn, ngram_key, variant)
            clusters.extend(variant_clusters)
        return clusters

    def _fetch_variant_clusters(
        self, conn: sqlite3.Connection, ngram: str, variant: str
    ) -> List[ClusterRecord]:
        cursor = conn.execute(
            "SELECT * FROM clusters WHERE ngram = ? ORDER BY cluster_index ASC;",
            (ngram,),
        )
        cluster_rows = cursor.fetchall()
        if not cluster_rows:
            return []

        cluster_map: Dict[int, ClusterRecord] = {}
        for row in cluster_rows:
            row_dict = {key: row[key] for key in row.keys()}
            cluster_id = int(row_dict["cluster_id"])
            record = ClusterRecord(
                variant=variant,
                cluster_id=cluster_id,
                run_id=str(row_dict.get("run_id")),
                ngram=str(row_dict.get("ngram")),
                cluster_index=int(row_dict.get("cluster_index", 0)),
                glossator_def=row_dict.get("glossator_def"),
                residual_explained=_safe_float(row_dict.get("residual_explained")),
                residual_ratio=_safe_float(row_dict.get("residual_ratio")),
                residual_headline=row_dict.get("residual_headline"),
                residual_focus_prompt=row_dict.get("residual_focus_prompt"),
                semantic_coverage=_safe_float(row_dict.get("semantic_coverage")),
                cohesion=_safe_float(row_dict.get("cohesion")),
                semantic_cohesion=_safe_float(row_dict.get("semantic_cohesion")),
                best_config=row_dict.get("best_config"),
            )
            cluster_map[cluster_id] = record

        cluster_ids = list(cluster_map.keys())
        placeholders = ",".join("?" for _ in cluster_ids)

        if cluster_ids:
            detail_rows = conn.execute(
                f"SELECT * FROM residual_details WHERE cluster_id IN ({placeholders}) ORDER BY cluster_id, residual_id;",
                cluster_ids,
            ).fetchall()
            for row in detail_rows:
                row_dict = {key: row[key] for key in row.keys()}
                detail = ResidualDetail(
                    normalized=str(row_dict.get("normalized", "")),
                    definition=row_dict.get("definition"),
                    coverage_ratio=_safe_float(row_dict.get("coverage_ratio")),
                    residual_ratio=_safe_float(row_dict.get("residual_ratio")),
                    avg_confidence=_safe_float(row_dict.get("avg_confidence")),
                    uncovered=_safe_json_array(row_dict.get("uncovered_json")),
                    low_confidence=_safe_json_array(row_dict.get("low_conf_json")),
                )
                cluster_map[int(row_dict["cluster_id"])].residual_details.append(detail)

            raw_rows = conn.execute(
                f"SELECT * FROM raw_defs WHERE cluster_id IN ({placeholders}) ORDER BY cluster_id, def_id;",
                cluster_ids,
            ).fetchall()
            for row in raw_rows:
                row_dict = {key: row[key] for key in row.keys()}
                cluster_map[int(row_dict["cluster_id"])].raw_definitions.append(
                    {
                        "source_word": row_dict.get("source_word"),
                        "variant": row_dict.get("variant"),
                        "definition": row_dict.get("definition"),
                        "enhanced_def": row_dict.get("enhanced_def"),
                        "fasttext": _safe_float(row_dict.get("fasttext")),
                        "similarity": _safe_float(row_dict.get("similarity")),
                        "tier": row_dict.get("tier"),
                    }
                )

        return [cluster_map[cid] for cid in sorted(cluster_map)]


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_json_array(payload: object) -> List[str]:
    if isinstance(payload, (list, tuple)):
        return [str(item) for item in payload]
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return [payload]
        if isinstance(data, list):
            return [str(item) for item in data]
        return [str(data)]
    return []
