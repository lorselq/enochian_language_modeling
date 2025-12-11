from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict, Optional, SupportsFloat, SupportsInt

from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.utils.embeddings import get_fasttext_model


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
class RawDefinition:
    source_word: Optional[str]
    variant: Optional[str]
    definition: Optional[str]
    enhanced_def: Optional[str]
    fasttext: Optional[float]
    similarity: Optional[float]
    tier: Optional[str]
    

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
    raw_definitions: List[RawDefinition] = field(default_factory=list)


@dataclass
class ResidualSemanticRecord:
    variant: str
    run_id: str
    residual: str
    parent_word: str
    group_index: int
    group_size: int
    glossator_def: Optional[str]
    glossator_prompt: Optional[str]
    residual_headline: Optional[str]
    residual_focus_prompt: Optional[str]
    semantic_coverage: Optional[float]
    cohesion: Optional[float]
    semantic_cohesion: Optional[float]
    residual_explained: Optional[float]
    residual_ratio: Optional[float]
    derivational_validity: Optional[float]
    rebuttal_resilience: Optional[float]
    created_at: Optional[str]


@dataclass
class MorphHypothesisRecord:
    variant: str
    hyp_id: int
    morph: str
    source_word: str
    anchor: Optional[str]
    seed_glosses: List[str]
    proposed_gloss: Optional[str]
    rationale: Optional[str]
    delta_cosine: Optional[float]
    residual_before: Optional[float]
    residual_after: Optional[float]
    created_at: Optional[str]


@dataclass
class FasttextNeighbor:
    word: str
    similarity: float


@dataclass
class WordEvidence:
    word: str
    variants_queried: List[str]
    direct_clusters: List[ClusterRecord] = field(default_factory=list)
    residual_semantics: List[ResidualSemanticRecord] = field(default_factory=list)
    morph_hypotheses: List[MorphHypothesisRecord] = field(default_factory=list)
    fasttext_neighbors: List[FasttextNeighbor] = field(default_factory=list)


@dataclass
class AcceptedMorphInfo(TypedDict):
    gloss: Optional[str]
    rationale: Optional[str]
    delta_cosine: Optional[float]
    source_word: Optional[str]
    anchor: Optional[str]


class InsightsRepository:
    """Read access to solo/debate insights databases."""

    def __init__(
        self,
        *,
        solo_path: Optional[Path],
        debate_path: Optional[Path],
        fasttext_model_path: Optional[Path] = None,
    ):
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

        self._fasttext_model_path = fasttext_model_path
        self._fasttext_model = None

    @property
    def variants(self) -> List[str]:
        return sorted(self._connections.keys())

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        self._fasttext_model = None

    def require_variants(self, expected: Iterable[str]) -> None:
        missing = [v for v in expected if v not in self._connections]
        if missing:
            raise FileNotFoundError(
                "Missing insights database(s): " + ", ".join(missing)
            )

    def fetch_clusters(
        self, ngram: str, *, variants: Optional[Iterable[str]] = None
    ) -> List[ClusterRecord]:
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
                    RawDefinition(
                        source_word=row_dict.get("source_word"),
                        variant=row_dict.get("variant"),
                        definition=row_dict.get("definition"),
                        enhanced_def=row_dict.get("enhanced_def"),
                        fasttext=_safe_float(row_dict.get("fasttext")),
                        similarity=_safe_float(row_dict.get("similarity")),
                        tier=row_dict.get("tier"),
                    )
                )

        return [cluster_map[cid] for cid in sorted(cluster_map)]

    def fetch_word_evidence(
        self,
        word: str,
        variants: Optional[Iterable[str]] = None,
        *,
        fasttext_top_k: int = 5,
    ) -> WordEvidence:
        normalized = word.upper()
        active_variants = list(variants) if variants else self.variants
        direct_clusters = self.fetch_clusters(normalized, variants=active_variants)
        residual_semantics = self._fetch_residual_semantics(
            normalized, variants=active_variants
        )
        morph_hypotheses = self._fetch_morph_hypotheses(
            normalized, variants=active_variants
        )

        fasttext_neighbors: List[FasttextNeighbor] = []
        if not (direct_clusters or residual_semantics or morph_hypotheses):
            fasttext_neighbors = self._fasttext_neighbors(
                normalized, top_k=fasttext_top_k
            )

        return WordEvidence(
            word=normalized,
            variants_queried=active_variants,
            direct_clusters=direct_clusters,
            residual_semantics=residual_semantics,
            morph_hypotheses=morph_hypotheses,
            fasttext_neighbors=fasttext_neighbors,
        )

    def fetch_accepted_morphs(self, variant: str) -> Dict[str, AcceptedMorphInfo]:
        """Return accepted morph hypotheses for a single variant.

        For each morph, pick the accepted hypothesis with the highest delta_cosine.
        """

        self.require_variants([variant])
        conn = self._connections[variant]

        rows = conn.execute(
            """
            SELECT morph, proposed_gloss, rationale, delta_cosine, source_word, anchor
            FROM morph_hypotheses
            WHERE accepted = 1
            ORDER BY morph, hyp_id;
            """
        ).fetchall()

        accepted: Dict[str, AcceptedMorphInfo] = {}

        for row in rows:
            data = {key: row[key] for key in row.keys()}
            morph = str(data.get("morph"))

            new_record: AcceptedMorphInfo = {
                "gloss": data.get("proposed_gloss"),
                "rationale": data.get("rationale"),
                "delta_cosine": _safe_float(data.get("delta_cosine")),
                "source_word": data.get("source_word"),
                "anchor": data.get("anchor"),
            }

            if morph not in accepted:
                accepted[morph] = new_record
                continue

            existing_delta = accepted[morph]["delta_cosine"]
            new_delta = new_record["delta_cosine"]

            if existing_delta is None or (
                new_delta is not None and new_delta > existing_delta
            ):
                accepted[morph] = new_record

        return accepted

    def _fetch_residual_semantics(
        self, residual: str, *, variants: Optional[Iterable[str]] = None
    ) -> List[ResidualSemanticRecord]:
        selected = list(variants) if variants else self.variants
        records: List[ResidualSemanticRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            rows = conn.execute(
                "SELECT * FROM root_residual_semantics WHERE residual = ? ORDER BY group_idx;",
                (residual,),
            ).fetchall()
            for row in rows:
                data = {key: row[key] for key in row.keys()}
                records.append(
                    ResidualSemanticRecord(
                        variant=variant,
                        run_id=str(data.get("run_id")),
                        residual=str(data.get("residual")),
                        parent_word=str(data.get("parent_word")),
                        group_index=_safe_int(data.get("group_idx")),
                        group_size=_safe_int(data.get("group_size")),
                        glossator_def=data.get("glossator_def"),
                        glossator_prompt=data.get("glossator_prompt"),
                        residual_headline=data.get("residual_headline"),
                        residual_focus_prompt=data.get("residual_focus_prompt"),
                        semantic_coverage=_safe_float(data.get("semantic_coverage")),
                        cohesion=_safe_float(data.get("cohesion")),
                        semantic_cohesion=_safe_float(data.get("semantic_cohesion")),
                        residual_explained=_safe_float(data.get("residual_explained")),
                        residual_ratio=_safe_float(data.get("residual_ratio")),
                        derivational_validity=_safe_float(
                            data.get("derivational_validity")
                        ),
                        rebuttal_resilience=_safe_float(
                            data.get("rebuttal_resilience")
                        ),
                        created_at=data.get("created_at"),
                    )
                )
        return records

    def _fetch_morph_hypotheses(
        self, morph: str, *, variants: Optional[Iterable[str]] = None
    ) -> List[MorphHypothesisRecord]:
        selected = list(variants) if variants else self.variants
        records: List[MorphHypothesisRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            rows = conn.execute(
                "SELECT * FROM morph_hypotheses WHERE morph = ? AND accepted = 1 ORDER BY hyp_id;",
                (morph,),
            ).fetchall()
            for row in rows:
                data = {key: row[key] for key in row.keys()}
                records.append(
                    MorphHypothesisRecord(
                        variant=variant,
                        hyp_id=_safe_int(data.get("hyp_id")),
                        morph=str(data.get("morph")),
                        source_word=str(data.get("source_word")),
                        anchor=data.get("anchor"),
                        seed_glosses=_safe_json_array(data.get("seed_glosses")),
                        proposed_gloss=data.get("proposed_gloss"),
                        rationale=data.get("rationale"),
                        delta_cosine=_safe_float(data.get("delta_cosine")),
                        residual_before=_safe_float(data.get("residual_before")),
                        residual_after=_safe_float(data.get("residual_after")),
                        created_at=data.get("created_at"),
                    )
                )
        return records

    def _fasttext_neighbors(self, word: str, *, top_k: int) -> List[FasttextNeighbor]:
        model = self._load_fasttext_model()
        if model is None:
            return []
        try:
            neighbors = model.wv.similar_by_word(word, topn=top_k)
        except KeyError:
            return []

        results: List[FasttextNeighbor] = []
        for candidate, similarity in neighbors:
            try:
                sim_value = float(similarity)
            except (TypeError, ValueError):
                continue
            results.append(FasttextNeighbor(word=str(candidate), similarity=sim_value))
        return results

    def _load_fasttext_model(self):
        if self._fasttext_model is not None:
            return self._fasttext_model
        if not self._fasttext_model_path:
            return None
        self._fasttext_model = get_fasttext_model(self._fasttext_model_path)
        return self._fasttext_model

def _safe_float(value: SupportsFloat | str | bytes | bytearray | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: SupportsInt | str | bytes | bytearray | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

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
