from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import SupportsFloat, SupportsInt, TypedDict

from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.utils.embeddings import get_fasttext_model
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord


@dataclass
class ResidualDetail:
    normalized: str
    definition: str | None = None
    coverage_ratio: float | None = None
    residual_ratio: float | None = None
    avg_confidence: float | None = None
    uncovered: list[str] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)


@dataclass
class RawDefinition:
    source_word: str | None
    variant: str | None
    definition: str | None
    enhanced_def: str | None
    fasttext: float | None
    similarity: float | None
    tier: str | None
    cluster_id: int | None = None


@dataclass
class AttestedDefinition:
    variant: str
    source_word: str
    definition: str | None
    cluster_id: int
    root_ngram: str


@dataclass
class DictionaryMorph:
    morph: str
    definition: str | None
    senses: list[str] = field(default_factory=list)


@dataclass
class ClusterRecord:
    variant: str
    cluster_id: int
    run_id: str
    ngram: str
    cluster_index: int
    glossator_def: str | None
    residual_explained: float | None
    residual_ratio: float | None
    residual_headline: str | None
    residual_focus_prompt: str | None
    semantic_coverage: float | None
    cohesion: float | None
    semantic_cohesion: float | None
    best_config: str | None
    residual_details: list[ResidualDetail] = field(default_factory=list)
    raw_definitions: list[RawDefinition] = field(default_factory=list)


@dataclass
class ResidualSemanticRecord:
    variant: str
    run_id: str
    residual: str
    parent_word: str
    group_index: int
    group_size: int
    glossator_def: str | None
    glossator_prompt: str | None
    residual_headline: str | None
    residual_focus_prompt: str | None
    semantic_coverage: float | None
    cohesion: float | None
    semantic_cohesion: float | None
    residual_explained: float | None
    residual_ratio: float | None
    derivational_validity: float | None
    rebuttal_resilience: float | None
    created_at: str | None


@dataclass
class MorphHypothesisRecord:
    variant: str
    hyp_id: int
    morph: str
    source_word: str
    anchor: str | None
    seed_glosses: list[str]
    proposed_gloss: str | None
    rationale: str | None
    delta_cosine: float | None
    residual_before: float | None
    residual_after: float | None
    created_at: str | None


@dataclass
class FasttextNeighbor:
    word: str
    similarity: float


@dataclass
class WordEvidence:
    word: str
    variants_queried: list[str]
    direct_clusters: list[ClusterRecord] = field(default_factory=list)
    residual_semantics: list[ResidualSemanticRecord] = field(default_factory=list)
    morph_hypotheses: list[MorphHypothesisRecord] = field(default_factory=list)
    fasttext_neighbors: list[FasttextNeighbor] = field(default_factory=list)
    attested_definitions: list[AttestedDefinition] = field(default_factory=list)
    dictionary_morphs: dict[str, DictionaryMorph] = field(default_factory=dict)
    definition_counts: dict[str, int] = field(default_factory=dict)
    definition_glosses: dict[str, list[tuple[str, float | None]]] = field(
        default_factory=dict
    )


@dataclass
class AcceptedMorphInfo(TypedDict):
    gloss: str | None
    rationale: str | None
    delta_cosine: float | None
    source_word: str | None
    anchor: str | None


class VariantPathInfo(TypedDict):
    path: str | None
    exists: bool


class PathDiagnostics(TypedDict):
    variants_available: list[str]
    variant_paths: dict[str, VariantPathInfo]


class WordLookupCounts(TypedDict):
    clusters: dict[str, int | None]
    residual_semantics: dict[str, int | None]
    morph_hypotheses: dict[str, int | None]
    attested_definitions: dict[str, int | None]


class WordLookupDiagnostics(TypedDict):
    word: str
    variants: list[str]
    counts: WordLookupCounts


class InsightsRepository:
    """Read access to solo/debate insights databases."""

    def __init__(
        self,
        *,
        solo_path: Path | None,
        debate_path: Path | None,
        fasttext_model_path: Path | None = None,
    ):
        paths = {"solo": solo_path, "debate": debate_path}
        self._paths: dict[str, Path | None] = {
            variant: Path(path) if path else None for variant, path in paths.items()
        }
        self._connections: dict[str, sqlite3.Connection] = {}
        for variant, path in paths.items():
            if path is None:
                continue
            abs_path = Path(path) if isinstance(path, str) else path
            if not abs_path.exists():
                continue
            conn = sqlite3.connect(str(abs_path))
            conn.row_factory = sqlite3.Row
            self._connections[variant] = conn

        self._fasttext_model_path = fasttext_model_path
        self._fasttext_model = None

    @property
    def variants(self) -> list[str]:
        return sorted(self._connections.keys())

    def path_diagnostics(self) -> PathDiagnostics:
        """Return configured variant paths and whether they exist on disk."""
        details: dict[str, VariantPathInfo] = {}
        for variant, path in self._paths.items():
            if path is None:
                details[variant] = {"path": None, "exists": False}
                continue
            details[variant] = {"path": str(path), "exists": path.exists()}
        return {
            "variants_available": self.variants,
            "variant_paths": details,
        }

    def word_lookup_diagnostics(
        self, word: str, *, variants: Iterable[str] | None = None
    ) -> WordLookupDiagnostics:
        """Return per-variant evidence match counts for a word."""
        normalized = word.upper()
        selected = list(variants) if variants else self.variants
        breakdown: WordLookupCounts = {
            "clusters": {},
            "residual_semantics": {},
            "morph_hypotheses": {},
            "attested_definitions": {},
        }
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                for key in breakdown:
                    breakdown[key][variant] = None
                continue
            breakdown["clusters"][variant] = _count_matches(
                conn,
                """SELECT COUNT(*) FROM clusters
                   WHERE TRIM(ngram) COLLATE NOCASE = ?
                     AND action = 'escalate'
                     AND verdict = 'True'""",
                (normalized,),
            )
            breakdown["residual_semantics"][variant] = _count_matches(
                conn,
                "SELECT COUNT(*) FROM root_residual_semantics WHERE TRIM(residual) COLLATE NOCASE = ?",
                (normalized,),
            )
            breakdown["morph_hypotheses"][variant] = _count_matches(
                conn,
                "SELECT COUNT(*) FROM morph_hypotheses WHERE TRIM(morph) COLLATE NOCASE = ? AND accepted = 1",
                (normalized,),
            )
            breakdown["attested_definitions"][variant] = _count_matches(
                conn,
                """
                SELECT COUNT(*)
                FROM raw_defs rd
                JOIN clusters c ON c.cluster_id = rd.cluster_id
                WHERE TRIM(rd.source_word) COLLATE NOCASE = ?
                AND TRIM(COALESCE(rd.definition, '')) <> ''
                """,
                (normalized,),
            )

        return {
            "word": normalized,
            "variants": selected,
            "counts": breakdown,
        }

    def close(self) -> None:
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        self._fasttext_model = None

    def fasttext_diagnostics(self, *, sample_size: int = 5) -> dict[str, object]:
        info: dict[str, object] = {
            "model_path": str(self._fasttext_model_path) if self._fasttext_model_path else None,
            "loaded": False,
            "vocab_sample": [],
            "vocab_size": None,
        }
        if not self._fasttext_model_path:
            return info
        try:
            model = self._load_fasttext_model()
        except Exception as exc:  # pragma: no cover - defensive for CLI diagnostics
            info["error"] = str(exc)
            return info

        if model is None:
            return info

        info["loaded"] = True
        vocab: list[str] = []
        vocab_size: int | None = None
        if hasattr(model, "wv"):
            wv = model.wv
            if hasattr(wv, "index_to_key"):
                vocab = list(wv.index_to_key[:sample_size])
                vocab_size = len(wv.index_to_key)
            elif hasattr(wv, "key_to_index"):
                keys = list(wv.key_to_index.keys())
                vocab = keys[:sample_size]
                vocab_size = len(keys)
        elif hasattr(model, "get_words"):
            words = list(model.get_words())
            vocab = words[:sample_size]
            vocab_size = len(words)

        info["vocab_sample"] = vocab
        info["vocab_size"] = vocab_size
        return info

    def require_variants(self, expected: Iterable[str]) -> None:
        missing = [v for v in expected if v not in self._connections]
        if missing:
            raise FileNotFoundError(
                "Missing insights database(s): " + ", ".join(missing)
            )

    def fetch_clusters(
        self, ngram: str, *, variants: Iterable[str] | None = None
    ) -> list[ClusterRecord]:
        ngram_key = ngram.upper()
        selected = list(variants) if variants else self.variants
        clusters: list[ClusterRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            variant_clusters = self._fetch_variant_clusters(conn, ngram_key, variant)
            clusters.extend(variant_clusters)
        return clusters

    def _fetch_variant_clusters(
        self, conn: sqlite3.Connection, ngram: str, variant: str
    ) -> list[ClusterRecord]:
        # Only fetch accepted clusters (action='escalate' AND verdict='True')
        # Non-accepted clusters are either skipped or rejected definitions
        supports_json = _sqlite_supports_json(conn)
        if supports_json:
            query = """SELECT *, json_valid(glossator_def) AS glossator_valid FROM clusters
               WHERE TRIM(ngram) COLLATE NOCASE = ?
                 AND action = 'escalate'
                 AND verdict = 'True'
                 AND (
                   (json_valid(glossator_def)
                     AND COALESCE(TRIM(json_extract(glossator_def, '$.DEFINITION')), '') <> ''
                     AND COALESCE(LOWER(CAST(json_extract(glossator_def, '$.REJECTED') AS TEXT)), '') NOT IN ('1', 'true', 'yes'))
                   OR NOT json_valid(glossator_def)
                 )
               ORDER BY cluster_index ASC;"""
        else:
            query = """SELECT * FROM clusters
               WHERE TRIM(ngram) COLLATE NOCASE = ?
                 AND action = 'escalate'
                 AND verdict = 'True'
               ORDER BY cluster_index ASC;"""
        cursor = conn.execute(query, (ngram,))
        cluster_rows = cursor.fetchall()
        if cluster_rows:
            if supports_json:
                cluster_rows = [
                    row
                    for row in cluster_rows
                    if row["glossator_valid"]
                    or _glossator_definition_is_accepted(row["glossator_def"])
                ]
            else:
                cluster_rows = [
                    row
                    for row in cluster_rows
                    if _glossator_definition_is_accepted(row["glossator_def"])
                ]
        if not cluster_rows:
            return []

        cluster_map: dict[int, ClusterRecord] = {}
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
                        cluster_id=_safe_int(row_dict.get("cluster_id")),
                    )
                )

        return [cluster_map[cid] for cid in sorted(cluster_map)]

    def fetch_word_evidence(
        self,
        word: str,
        variants: Iterable[str] | None = None,
        *,
        fasttext_top_k: int = 5,
        dictionary_entries: Mapping[str, EntryRecord] | None = None,
        min_n: int | None = None,
        max_n: int | None = None,
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
        attested_definitions = self._fetch_attested_definitions(
            normalized, variants=active_variants
        )
        dictionary_min_n = min_n
        if dictionary_min_n is not None and dictionary_min_n > 1:
            dictionary_min_n = 1
        dictionary_morphs = _dictionary_morphs_for_word(
            normalized,
            dictionary_entries,
            min_n=dictionary_min_n,
            max_n=max_n,
        )

        fasttext_neighbors: list[FasttextNeighbor] = []
        if not (
            direct_clusters
            or residual_semantics
            or morph_hypotheses
            or attested_definitions
            or dictionary_morphs
        ):
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
            attested_definitions=attested_definitions,
            dictionary_morphs=dictionary_morphs,
        )

    def fetch_morph_support(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> tuple[
        list[ClusterRecord],
        list[ResidualSemanticRecord],
        list[MorphHypothesisRecord],
    ]:
        """Fetch evidence records for a collection of morphs."""
        selected = list(variants) if variants else self.variants
        unique = {m.upper() for m in morphs if m}
        clusters: list[ClusterRecord] = []
        residuals: list[ResidualSemanticRecord] = []
        hypotheses: list[MorphHypothesisRecord] = []
        for morph in sorted(unique):
            clusters.extend(self.fetch_clusters(morph, variants=selected))
            residuals.extend(
                self._fetch_residual_semantics(morph, variants=selected)
            )
            hypotheses.extend(
                self._fetch_morph_hypotheses(morph, variants=selected)
            )
        return clusters, residuals, hypotheses

    def fetch_accepted_definition_counts(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> dict[str, int]:
        """Return the count of accepted definitions per morph.

        Accepted definitions are clusters with action='escalate' AND verdict='True'.
        This count is used to determine specificity: fewer definitions = more specific.
        """
        selected = list(variants) if variants else self.variants
        unique = {m.upper() for m in morphs if m}
        counts: dict[str, int] = {}

        for morph in sorted(unique):
            total = 0
            for variant in selected:
                conn = self._connections.get(variant)
                if conn is None:
                    continue
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM clusters
                       WHERE TRIM(ngram) COLLATE NOCASE = ?
                         AND action = 'escalate'
                         AND verdict = 'True'""",
                    (morph,),
                )
                row = cursor.fetchone()
                if row:
                    total += row[0]
            counts[morph] = total

        return counts

    def fetch_accepted_definition_glosses(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
        include_clusters: bool = True,
        include_residuals: bool = True,
    ) -> dict[str, list[tuple[str, float | None]]]:
        """Return glossator definitions and confidence cues for accepted evidence."""
        selected = list(variants) if variants else self.variants
        unique = {m.upper() for m in morphs if m}
        glosses: dict[str, dict[str, float | None]] = {}

        for morph in sorted(unique):
            for variant in selected:
                conn = self._connections.get(variant)
                if conn is None:
                    continue
                if include_clusters:
                    # solo variant doesn't have semantic_cohesion column
                    if variant == "solo":
                        cluster_sql = """SELECT glossator_def, semantic_coverage, cohesion, cohesion
                           FROM clusters
                           WHERE TRIM(ngram) COLLATE NOCASE = ?
                             AND action = 'escalate'
                             AND verdict = 'True'"""
                    else:
                        cluster_sql = """SELECT glossator_def, semantic_coverage, semantic_cohesion, cohesion
                           FROM clusters
                           WHERE TRIM(ngram) COLLATE NOCASE = ?
                             AND action = 'escalate'
                             AND verdict = 'True'"""
                    rows = conn.execute(cluster_sql, (morph,)).fetchall()
                    for row in rows:
                        definition, rejected = _parse_glossator_definition(row[0])
                        if rejected or not definition:
                            continue
                        score = _safe_float(row[1])
                        if score is None:
                            score = _safe_float(row[2])
                        if score is None:
                            score = _safe_float(row[3])
                        normalized = str(definition).strip()
                        gloss_bucket = glosses.setdefault(morph, {})
                        existing = gloss_bucket.get(normalized)
                        if existing is None or (
                            score is not None and score > existing
                        ):
                            gloss_bucket[normalized] = score
                if include_residuals:
                    residual_rows = conn.execute(
                        """SELECT glossator_def, semantic_coverage, semantic_cohesion, cohesion,
                                  derivational_validity, rebuttal_resilience
                           FROM root_residual_semantics
                           WHERE TRIM(residual) COLLATE NOCASE = ?""",
                        (morph,),
                    ).fetchall()
                    for row in residual_rows:
                        definition, rejected = _parse_glossator_definition(row[0])
                        if rejected or not definition:
                            continue
                        score = _safe_float(row[1])
                        if score is None:
                            score = _safe_float(row[2])
                        if score is None:
                            score = _safe_float(row[3])
                        if score is None:
                            score = _safe_float(row[4])
                        if score is None:
                            score = _safe_float(row[5])
                        normalized = str(definition).strip()
                        gloss_bucket = glosses.setdefault(morph, {})
                        existing = gloss_bucket.get(normalized)
                        if existing is None or (
                            score is not None and score > existing
                        ):
                            gloss_bucket[normalized] = score

        output: dict[str, list[tuple[str, float | None]]] = {}
        for morph, entries in glosses.items():
            output[morph] = [(text, score) for text, score in entries.items()]
        return output

    def fetch_accepted_morphs(self, variant: str) -> dict[str, AcceptedMorphInfo]:
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

        accepted: dict[str, AcceptedMorphInfo] = {}

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
        self, residual: str, *, variants: Iterable[str] | None = None
    ) -> list[ResidualSemanticRecord]:
        selected = list(variants) if variants else self.variants
        records: list[ResidualSemanticRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            supports_json = _sqlite_supports_json(conn)
            if supports_json:
                query = """SELECT *, json_valid(glossator_def) AS glossator_valid FROM root_residual_semantics
                    WHERE TRIM(residual) COLLATE NOCASE = ?
                      AND (
                        (json_valid(glossator_def)
                          AND COALESCE(TRIM(json_extract(glossator_def, '$.DEFINITION')), '') <> ''
                          AND COALESCE(LOWER(CAST(json_extract(glossator_def, '$.REJECTED') AS TEXT)), '') NOT IN ('1', 'true', 'yes'))
                        OR NOT json_valid(glossator_def)
                      )
                    ORDER BY group_idx;"""
            else:
                query = """SELECT * FROM root_residual_semantics
                    WHERE TRIM(residual) COLLATE NOCASE = ?
                    ORDER BY group_idx;"""
            rows = conn.execute(query, (residual,)).fetchall()
            if rows:
                if supports_json:
                    rows = [
                        row
                        for row in rows
                        if row["glossator_valid"]
                        or _glossator_definition_is_accepted(row["glossator_def"])
                    ]
                else:
                    rows = [
                        row
                        for row in rows
                        if _glossator_definition_is_accepted(row["glossator_def"])
                    ]
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
        self, morph: str, *, variants: Iterable[str] | None = None
    ) -> list[MorphHypothesisRecord]:
        selected = list(variants) if variants else self.variants
        records: list[MorphHypothesisRecord] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            rows = conn.execute(
                "SELECT * FROM morph_hypotheses WHERE TRIM(morph) COLLATE NOCASE = ? AND accepted = 1 ORDER BY hyp_id;",
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

    def _fetch_attested_definitions(
        self, word: str, *, variants: Iterable[str] | None = None
    ) -> list[AttestedDefinition]:
        selected = list(variants) if variants else self.variants
        records: list[AttestedDefinition] = []
        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            rows = conn.execute(
                """
                SELECT rd.source_word, rd.definition, rd.cluster_id, c.ngram
                FROM raw_defs rd
                JOIN clusters c ON c.cluster_id = rd.cluster_id
                WHERE TRIM(rd.source_word) COLLATE NOCASE = ? AND TRIM(COALESCE(rd.definition, '')) <> ''
                ORDER BY rd.cluster_id, rd.def_id;
                """,
                (word,),
            ).fetchall()
            for row in rows:
                data = {key: row[key] for key in row.keys()}
                records.append(
                    AttestedDefinition(
                        variant=variant,
                        source_word=str(data.get("source_word")),
                        definition=data.get("definition"),
                        cluster_id=_safe_int(data.get("cluster_id")),
                        root_ngram=str(data.get("ngram")),
                    )
                )
        return records

    def _fasttext_neighbors(self, word: str, *, top_k: int) -> list[FasttextNeighbor]:
        model = self._load_fasttext_model()
        if model is None:
            return []
        try:
            neighbors = model.wv.similar_by_word(word, topn=top_k)
        except KeyError:
            return []

        results: list[FasttextNeighbor] = []
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


def _safe_float(value: SupportsFloat | str | bytes | bytearray | None) -> float | None:
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


def _dictionary_morphs_for_word(
    word: str,
    dictionary_entries: Mapping[str, EntryRecord] | None,
    *,
    min_n: int | None = None,
    max_n: int | None = None,
) -> dict[str, DictionaryMorph]:
    if not word or not dictionary_entries:
        return {}

    word_upper = word.upper()
    word_len = len(word_upper)
    if word_len == 0:
        return {}

    min_len = max(1, int(min_n) if min_n is not None else 1)
    max_len = (
        max(min_len, int(max_n)) if max_n is not None else word_len
    )
    max_len = min(max_len, word_len)

    morphs: dict[str, DictionaryMorph] = {}
    for start in range(word_len):
        for end in range(start + min_len, min(word_len, start + max_len) + 1):
            slice_text = word_upper[start:end]
            entry = dictionary_entries.get(slice_text.lower())
            if not entry:
                continue
            if slice_text in morphs:
                continue
            definition = _dictionary_entry_definition(entry)
            senses = _dictionary_entry_senses(entry)
            morphs[slice_text] = DictionaryMorph(
                morph=slice_text,
                definition=definition,
                senses=senses,
            )
    return morphs


def _dictionary_entry_definition(entry: Mapping[str, object]) -> str | None:
    enhanced = entry.get("enhanced_definition")
    if isinstance(enhanced, str) and enhanced.strip():
        return enhanced.strip()
    senses = entry.get("senses")
    if isinstance(senses, list):
        for sense in senses:
            if not isinstance(sense, Mapping):
                continue
            definition = sense.get("definition")
            if isinstance(definition, str) and definition.strip():
                return definition.strip()
    definition = entry.get("definition")
    if isinstance(definition, str) and definition.strip():
        return definition.strip()
    return None


def _dictionary_entry_senses(entry: Mapping[str, object]) -> list[str]:
    senses = entry.get("senses")
    if not isinstance(senses, list):
        return []
    definitions: list[str] = []
    for sense in senses:
        if not isinstance(sense, Mapping):
            continue
        definition = sense.get("definition")
        if isinstance(definition, str) and definition.strip():
            definitions.append(definition.strip())
    return definitions


def _safe_json_array(payload: object) -> list[str]:
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


def _parse_glossator_definition(payload: object) -> tuple[str | None, bool]:
    if payload is None:
        return None, False
    if not isinstance(payload, str):
        payload = str(payload)
    payload = payload.strip()
    if not payload:
        return None, False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return payload, False
    if isinstance(data, Mapping):
        definition = data.get("DEFINITION")
        if definition is None:
            definition = data.get("definition")
        if isinstance(definition, str):
            definition = definition.strip()
        else:
            definition = None
        rejected = data.get("REJECTED")
        if rejected is None:
            rejected = data.get("rejected")
        if isinstance(rejected, str):
            rejected_flag = rejected.strip().lower() in {"1", "true", "yes", "y"}
        else:
            rejected_flag = bool(rejected)
        return definition, rejected_flag
    if isinstance(data, str):
        definition = data.strip()
        return definition or None, False
    return payload, False


def _glossator_definition_is_accepted(payload: object) -> bool:
    definition, rejected = _parse_glossator_definition(payload)
    if rejected:
        return False
    return bool(definition and definition.strip())


def _sqlite_supports_json(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT json_extract('{\"a\": 1}', '$.a');").fetchone()
    except sqlite3.OperationalError:
        return False
    return True


def _count_matches(conn: sqlite3.Connection, query: str, params: tuple[object, ...]) -> int:
    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    if row is None:
        return 0
    value = row[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
