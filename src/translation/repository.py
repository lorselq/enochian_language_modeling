from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import SupportsFloat, SupportsInt, TypedDict

from enochian_lm.common.sqlite_bootstrap import sqlite3
from enochian_lm.root_extraction.utils.embeddings import get_fasttext_model
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord

from .fnp import (
    RootFNPObservation,
    RootFNPProfile,
    aggregate_root_fnp_profile,
)


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
    """Carry dictionary-backed morph metadata into translation decisions.

    Dictionary entries are authoritative anchors for exact known words and a
    useful weak signal for phrase translation. Keeping the definition, senses,
    and coarse POS together allows downstream code to separate definitive
    whole-word anchors from weaker substring hints without reparsing the raw
    dictionary entry.
    """

    morph: str
    definition: str | None
    senses: list[str] = field(default_factory=list)
    part_of_speech: str | None = None


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
    semantic_core: list[str] = field(default_factory=list)
    negative_contrast: list[str] = field(default_factory=list)
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
    semantic_core: list[str] = field(default_factory=list)
    negative_contrast: list[str] = field(default_factory=list)


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
    rejected_morphs: set[str] = field(default_factory=set)


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
        self._cluster_cache: dict[tuple[str, str], tuple[ClusterRecord, ...]] = {}
        self._residual_cache: dict[tuple[str, str], tuple[ResidualSemanticRecord, ...]] = {}
        self._hypothesis_cache: dict[tuple[str, str], tuple[MorphHypothesisRecord, ...]] = {}
        self._attested_cache: dict[tuple[str, str], tuple[AttestedDefinition, ...]] = {}
        self._rejected_morph_cache: dict[tuple[str, str], bool] = {}
        self._accepted_definition_count_cache: dict[tuple[str, str], int] = {}
        self._accepted_definition_gloss_cache: dict[
            tuple[str, str, bool, bool],
            dict[str, float | None],
        ] = {}
        self._root_fnp_observation_cache: dict[
            tuple[str, str],
            tuple[RootFNPObservation, ...],
        ] = {}

    @property
    def variants(self) -> list[str]:
        return sorted(self._connections.keys())

    def llm_logging_target(
        self,
        variant: str,
    ) -> tuple[sqlite3.Connection | None, str | None]:
        """Return the DB handle and a stable run id for LLM response caching.

        Phrase translation now issues user-facing rendering calls through the
        same ``QueryModelTool`` infrastructure used elsewhere in the project.
        Reusing the variant insights database and its most recent run id lets
        those render calls participate in existing prompt-hash caching without
        introducing a second persistence layer.
        """

        conn = self._connections.get(variant)
        if conn is None:
            return None, None
        try:
            row = conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            try:
                row = conn.execute(
                    "SELECT run_id FROM runs ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
            except sqlite3.Error:
                row = None
        if row is None:
            return conn, None
        return conn, str(row["run_id"])

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
        self._cluster_cache.clear()
        self._residual_cache.clear()
        self._hypothesis_cache.clear()
        self._attested_cache.clear()
        self._rejected_morph_cache.clear()
        self._accepted_definition_count_cache.clear()
        self._accepted_definition_gloss_cache.clear()
        self._root_fnp_observation_cache.clear()

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
        self._populate_cluster_cache_for_ngrams([ngram_key], variants=selected)
        clusters: list[ClusterRecord] = []
        for variant in selected:
            cache_key = (variant, ngram_key)
            cached = self._cluster_cache.get(cache_key)
            if cached is None:
                continue
            variant_clusters = [record for record in cached]
            clusters.extend(variant_clusters)
        return clusters

    def _populate_cluster_cache_for_ngrams(
        self,
        ngrams: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill cluster caches for many ngrams at once.

        Phrase translation repeatedly asks for overlapping substrings from the
        same handful of databases. Populating the cache in one query per
        variant removes the per-morph query storm while preserving the
        repository's single-morph public API.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(ngram).upper() for ngram in ngrams if ngram}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                ngram for ngram in unique if (variant, ngram) not in self._cluster_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            rows = conn.execute(
                f"""SELECT * FROM clusters
                    WHERE UPPER(TRIM(ngram)) IN ({placeholders})
                      AND action = 'escalate'
                      AND verdict = 'True'
                    ORDER BY ngram COLLATE NOCASE ASC, cluster_index ASC;""",
                tuple(missing),
            ).fetchall()
            grouped = self._build_cluster_records_by_ngram(
                conn,
                variant,
                rows,
            )
            for ngram in missing:
                self._cluster_cache[(variant, ngram)] = grouped.get(ngram, ())

    def _fetch_variant_clusters(
        self, conn: sqlite3.Connection, ngram: str, variant: str
    ) -> list[ClusterRecord]:
        # Only fetch accepted clusters (action='escalate' AND verdict='True')
        # Non-accepted clusters are either skipped or rejected definitions
        query = """SELECT * FROM clusters
           WHERE TRIM(ngram) COLLATE NOCASE = ?
             AND action = 'escalate'
             AND verdict = 'True'
           ORDER BY cluster_index ASC;"""
        cursor = conn.execute(query, (ngram,))
        grouped = self._build_cluster_records_by_ngram(
            conn,
            variant,
            cursor.fetchall(),
        )
        return list(grouped.get(ngram.upper(), ()))

    def _build_cluster_records_by_ngram(
        self,
        conn: sqlite3.Connection,
        variant: str,
        cluster_rows: Iterable[sqlite3.Row],
    ) -> dict[str, tuple[ClusterRecord, ...]]:
        """Hydrate cluster rows into ngram-grouped records for translation.

        This helper exists so both single-word lookup and phrase prewarming can
        share the same hydration logic. Centralizing the row-to-record mapping
        keeps the cache format stable while allowing phrase translation to load
        many overlapping ngrams in one pass.
        """

        accepted_rows = [
            row
            for row in cluster_rows
            if _glossator_definition_is_accepted(row["glossator_def"])
        ]
        if not accepted_rows:
            return {}

        cluster_map: dict[int, ClusterRecord] = {}
        for row in accepted_rows:
            row_dict = {key: row[key] for key in row.keys()}
            cluster_id = int(row_dict["cluster_id"])
            semantic_core, negative_contrast = _parse_glossator_semantics(
                row_dict.get("glossator_def")
            )
            cluster_map[cluster_id] = ClusterRecord(
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
                semantic_core=semantic_core,
                negative_contrast=negative_contrast,
            )

        cluster_ids = list(cluster_map.keys())
        if cluster_ids:
            placeholders = ",".join("?" for _ in cluster_ids)
            detail_rows = conn.execute(
                f"""SELECT * FROM residual_details
                    WHERE cluster_id IN ({placeholders})
                    ORDER BY cluster_id, residual_id;""",
                tuple(cluster_ids),
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
                f"""SELECT * FROM raw_defs
                    WHERE cluster_id IN ({placeholders})
                    ORDER BY cluster_id, def_id;""",
                tuple(cluster_ids),
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

        grouped: dict[str, list[ClusterRecord]] = {}
        for record in cluster_map.values():
            grouped.setdefault(record.ngram.upper(), []).append(record)
        return {
            ngram: tuple(sorted(records, key=lambda record: record.cluster_index))
            for ngram, records in grouped.items()
        }

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
        rejected_morphs = self.fetch_rejected_morphs(
            [normalized], variants=active_variants
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
            rejected_morphs=rejected_morphs,
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
        self._populate_cluster_cache_for_ngrams(unique, variants=selected)
        self._populate_residual_cache_for_morphs(unique, variants=selected)
        self._populate_hypothesis_cache_for_morphs(unique, variants=selected)
        clusters: list[ClusterRecord] = []
        residuals: list[ResidualSemanticRecord] = []
        hypotheses: list[MorphHypothesisRecord] = []
        for morph in sorted(unique):
            for variant in selected:
                clusters.extend(
                    list(self._cluster_cache.get((variant, morph), ()))
                )
                residuals.extend(
                    list(self._residual_cache.get((variant, morph), ()))
                )
                hypotheses.extend(
                    list(self._hypothesis_cache.get((variant, morph), ()))
                )
        return clusters, residuals, hypotheses

    def fetch_rejected_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> set[str]:
        """Return morphs that are explicitly rejected in residual analysis.

        Negative evidence matters for translation because a tempting split can
        look plausible from raw attestation while a glossator payload has
        already rejected that piece as a valid standalone root. We surface
        these tokens separately so ranking can penalize them without treating
        them as support.
        """
        selected = list(variants) if variants else self.variants
        unique = {m.upper() for m in morphs if m}
        rejected: set[str] = set()
        self._populate_rejected_morph_cache_for_morphs(unique, variants=selected)

        for morph in sorted(unique):
            for variant in selected:
                cache_key = (variant, morph)
                cached = self._rejected_morph_cache.get(cache_key)
                if cached is None:
                    continue
                if cached:
                    rejected.add(morph)
                if morph in rejected:
                    break

        return rejected

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
        self._populate_accepted_definition_count_cache_for_morphs(
            unique,
            variants=selected,
        )

        for morph in sorted(unique):
            total = 0
            for variant in selected:
                cache_key = (variant, morph)
                cached = self._accepted_definition_count_cache.get(cache_key)
                if cached is None:
                    continue
                total += cached
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
        self._populate_accepted_definition_gloss_cache_for_morphs(
            unique,
            variants=selected,
            include_clusters=include_clusters,
            include_residuals=include_residuals,
        )

        for morph in sorted(unique):
            for variant in selected:
                cache_key = (variant, morph, include_clusters, include_residuals)
                cached = self._accepted_definition_gloss_cache.get(cache_key)
                if cached:
                    gloss_bucket = glosses.setdefault(morph, {})
                    for normalized, score in cached.items():
                        existing = gloss_bucket.get(normalized)
                        if existing is None or (
                            score is not None and (existing is None or score > existing)
                        ):
                            gloss_bucket[normalized] = score

        output: dict[str, list[tuple[str, float | None]]] = {}
        for morph, entries in glosses.items():
            output[morph] = [(text, score) for text, score in entries.items()]
        return output

    def fetch_root_fnp_profiles(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> dict[str, RootFNPProfile]:
        """Return deterministic FNP profiles for accepted root-gloss morphs.

        Translation ranking and phrase parsing both need the same root-level
        functional priors. This method batches SQLite reads, reuses the
        observation cache, and aggregates across selected variants without
        changing the underlying evidence/decomposition logic.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        self._populate_root_fnp_observation_cache_for_morphs(
            unique,
            variants=selected,
        )

        profiles: dict[str, RootFNPProfile] = {}
        for morph in sorted(unique):
            observations: list[RootFNPObservation] = []
            for variant in selected:
                observations.extend(
                    self._root_fnp_observation_cache.get((variant, morph), ())
                )
            profile = aggregate_root_fnp_profile(morph, observations)
            if profile is not None:
                profiles[morph] = profile
        return profiles

    def prewarm_root_fnp_profiles(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Populate root FNP caches for a batch of phrase/word morphs.

        Phrase translation already prewarms support and gloss caches for many
        overlapping substrings. Warming FNP observations alongside those caches
        keeps the new structural prior cheap and deterministic.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        self._populate_root_fnp_observation_cache_for_morphs(
            unique,
            variants=selected,
        )

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

    def _populate_residual_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill accepted residual semantics for many morphs.

        Residual lookups are hot in blind decomposition mode because nearly
        every substring participates in support labeling. Fetching them in one
        query per variant keeps phrase translation from paying that cost once
        per morph.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph for morph in unique if (variant, morph) not in self._residual_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            rows = conn.execute(
                f"""SELECT * FROM root_residual_semantics
                    WHERE UPPER(TRIM(residual)) IN ({placeholders})
                    ORDER BY residual COLLATE NOCASE ASC, group_idx ASC;""",
                tuple(missing),
            ).fetchall()
            grouped: dict[str, list[ResidualSemanticRecord]] = {
                morph: [] for morph in missing
            }
            for row in rows:
                if not _glossator_definition_is_accepted(row["glossator_def"]):
                    continue
                data = {key: row[key] for key in row.keys()}
                residual = str(data.get("residual", "")).upper()
                semantic_core, negative_contrast = _parse_glossator_semantics(
                    data.get("glossator_def")
                )
                grouped.setdefault(residual, []).append(
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
                        semantic_core=semantic_core,
                        negative_contrast=negative_contrast,
                    )
                )

            for morph in missing:
                self._residual_cache[(variant, morph)] = tuple(grouped.get(morph, ()))

    def _populate_hypothesis_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill accepted morph hypotheses used during decomposition.

        Hypothesis evidence is a weaker signal than accepted clusters, but it
        still participates in support checks and blind-mode ranking. Loading it
        in bulk lets phrase translation reuse one database scan across many
        overlapping substrings.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph for morph in unique if (variant, morph) not in self._hypothesis_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            rows = conn.execute(
                f"""SELECT * FROM morph_hypotheses
                    WHERE UPPER(TRIM(morph)) IN ({placeholders})
                      AND accepted = 1
                    ORDER BY morph COLLATE NOCASE ASC, hyp_id ASC;""",
                tuple(missing),
            ).fetchall()
            grouped: dict[str, list[MorphHypothesisRecord]] = {
                morph: [] for morph in missing
            }
            for row in rows:
                data = {key: row[key] for key in row.keys()}
                morph = str(data.get("morph", "")).upper()
                grouped.setdefault(morph, []).append(
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

            for morph in missing:
                self._hypothesis_cache[(variant, morph)] = tuple(
                    grouped.get(morph, ())
                )

    def _populate_rejected_morph_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill rejected-morph flags from residual analysis payloads.

        Rejected residuals act as negative evidence during decomposition. This
        helper keeps that signal cheap to consult by collapsing all requested
        morphs for a variant into one residual-table scan.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph
                for morph in unique
                if (variant, morph) not in self._rejected_morph_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            rows = conn.execute(
                f"""SELECT residual, glossator_def
                    FROM root_residual_semantics
                    WHERE UPPER(TRIM(residual)) IN ({placeholders})
                    ORDER BY residual COLLATE NOCASE ASC, group_idx ASC;""",
                tuple(missing),
            ).fetchall()
            grouped: dict[str, bool] = {morph: False for morph in missing}
            for row in rows:
                residual = str(row["residual"]).upper()
                definition, rejected = _parse_glossator_definition(row["glossator_def"])
                if rejected or not definition:
                    grouped[residual] = True

            for morph in missing:
                self._rejected_morph_cache[(variant, morph)] = grouped.get(morph, False)

    def _populate_accepted_definition_count_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill accepted-definition counts used for ambiguity penalties.

        The beam-search scorer consults accepted-definition counts for many
        overlapping substrings. Counting them in bulk keeps ambiguity scoring
        available without turning phrase translation into hundreds of tiny
        SQLite reads.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph
                for morph in unique
                if (variant, morph) not in self._accepted_definition_count_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            rows = conn.execute(
                f"""SELECT ngram, glossator_def
                    FROM clusters
                    WHERE UPPER(TRIM(ngram)) IN ({placeholders})
                      AND action = 'escalate'
                      AND verdict = 'True';""",
                tuple(missing),
            ).fetchall()
            grouped: dict[str, int] = {morph: 0 for morph in missing}
            for row in rows:
                if not _glossator_definition_is_accepted(row["glossator_def"]):
                    continue
                grouped[str(row["ngram"]).upper()] += 1

            for morph in missing:
                self._accepted_definition_count_cache[(variant, morph)] = grouped.get(
                    morph,
                    0,
                )

    def _populate_accepted_definition_gloss_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
        include_clusters: bool = True,
        include_residuals: bool = True,
    ) -> None:
        """Batch-fill accepted gloss buckets used by semantic deduplication.

        Candidate scoring now depends heavily on semantic-core-aware definition
        selection. This helper lets phrase translation reuse one bulk load per
        variant instead of repeatedly asking SQLite for the same gloss payloads
        one morph at a time.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph
                for morph in unique
                if (
                    variant,
                    morph,
                    include_clusters,
                    include_residuals,
                )
                not in self._accepted_definition_gloss_cache
            )
            if not missing:
                continue

            grouped: dict[str, dict[str, float | None]] = {
                morph: {} for morph in missing
            }
            placeholders = ",".join("?" for _ in missing)

            if include_clusters:
                if variant == "solo":
                    cluster_sql = f"""SELECT ngram, glossator_def, semantic_coverage, cohesion, cohesion
                        FROM clusters
                        WHERE UPPER(TRIM(ngram)) IN ({placeholders})
                          AND action = 'escalate'
                          AND verdict = 'True'"""
                else:
                    cluster_sql = f"""SELECT ngram, glossator_def, semantic_coverage, semantic_cohesion, cohesion
                        FROM clusters
                        WHERE UPPER(TRIM(ngram)) IN ({placeholders})
                          AND action = 'escalate'
                          AND verdict = 'True'"""
                rows = conn.execute(cluster_sql, tuple(missing)).fetchall()
                for row in rows:
                    definition, rejected = _parse_glossator_definition(row[1])
                    if rejected or not definition:
                        continue
                    score = _safe_float(row[2])
                    if score is None:
                        score = _safe_float(row[3])
                    if score is None:
                        score = _safe_float(row[4])
                    morph = str(row[0]).upper()
                    normalized = str(definition).strip()
                    existing = grouped[morph].get(normalized)
                    if existing is None or (
                        score is not None and score > existing
                    ):
                        grouped[morph][normalized] = score

            if include_residuals:
                residual_rows = conn.execute(
                    f"""SELECT residual, glossator_def, semantic_coverage, semantic_cohesion, cohesion,
                               derivational_validity, rebuttal_resilience
                        FROM root_residual_semantics
                        WHERE UPPER(TRIM(residual)) IN ({placeholders})""",
                    tuple(missing),
                ).fetchall()
                for row in residual_rows:
                    definition, rejected = _parse_glossator_definition(row[1])
                    if rejected or not definition:
                        continue
                    score = _safe_float(row[2])
                    if score is None:
                        score = _safe_float(row[3])
                    if score is None:
                        score = _safe_float(row[4])
                    if score is None:
                        score = _safe_float(row[5])
                    if score is None:
                        score = _safe_float(row[6])
                    morph = str(row[0]).upper()
                    normalized = str(definition).strip()
                    existing = grouped[morph].get(normalized)
                    if existing is None or (
                        score is not None and score > existing
                    ):
                        grouped[morph][normalized] = score

            for morph in missing:
                self._accepted_definition_gloss_cache[
                    (variant, morph, include_clusters, include_residuals)
                ] = grouped.get(morph, {})

    def _populate_root_fnp_observation_cache_for_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
    ) -> None:
        """Batch-fill accepted root-gloss FNP observations for many morphs.

        FNP is read-only evidence derived from existing SQLite views. This
        cache keeps word and phrase translation from querying `root_glosses`
        repeatedly for overlapping substrings while tolerating older databases
        that do not yet expose the views.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique:
            return

        for variant in selected:
            conn = self._connections.get(variant)
            if conn is None:
                continue
            missing = sorted(
                morph
                for morph in unique
                if (variant, morph) not in self._root_fnp_observation_cache
            )
            if not missing:
                continue

            placeholders = ",".join("?" for _ in missing)
            grouped: dict[str, list[RootFNPObservation]] = {
                morph: [] for morph in missing
            }
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                      g.root,
                      g.definition,
                      g.semantic_core,
                      g.decoding_guide,
                      g.pos_bias_nounness,
                      g.pos_bias_modifier,
                      g.pos_bias_verbness,
                      g.attachment_prefix_likelihood,
                      g.attachment_suffix_likelihood,
                      g.attachment_free_likelihood,
                      g.attachment_productivity,
                      g.confidence_score,
                      g.source_cluster_id,
                      p.estimated_profile,
                      p.observed_prefix_count,
                      p.observed_suffix_count,
                      p.observed_infix_count,
                      p.observed_free_count
                    FROM root_glosses AS g
                    LEFT JOIN root_attachment_profile AS p
                      ON p.root = g.root
                     AND p.source_cluster_id = g.source_cluster_id
                    WHERE UPPER(TRIM(g.root)) IN ({placeholders})
                      AND LOWER(COALESCE(TRIM(g.evaluation), 'accepted')) = 'accepted'
                    ORDER BY g.root COLLATE NOCASE ASC, g.source_cluster_id ASC;
                    """,
                    tuple(missing),
                ).fetchall()
            except Exception:
                for morph in missing:
                    self._root_fnp_observation_cache[(variant, morph)] = ()
                continue

            evidence_examples = _fetch_root_fnp_evidence_examples(
                conn,
                missing,
            )
            for row in rows:
                observation = _root_fnp_observation_from_row(row, variant=variant)
                root = observation.root.upper()
                cluster_id = observation.source_cluster_id
                observation.evidence_examples = list(
                    evidence_examples.get((root, cluster_id), ())
                )
                if not observation.evidence_examples:
                    observation.evidence_examples = list(
                        evidence_examples.get((root, None), ())
                    )
                if root in grouped:
                    grouped[root].append(observation)

            for morph in missing:
                self._root_fnp_observation_cache[(variant, morph)] = tuple(
                    grouped.get(morph, [])
                )

    def _fetch_residual_semantics(
        self, residual: str, *, variants: Iterable[str] | None = None
    ) -> list[ResidualSemanticRecord]:
        selected = list(variants) if variants else self.variants
        self._populate_residual_cache_for_morphs([residual], variants=selected)
        records: list[ResidualSemanticRecord] = []
        for variant in selected:
            cache_key = (variant, residual.upper())
            cached = self._residual_cache.get(cache_key)
            if cached is None:
                continue
            records.extend(list(cached))
        return records

    def _fetch_morph_hypotheses(
        self, morph: str, *, variants: Iterable[str] | None = None
    ) -> list[MorphHypothesisRecord]:
        selected = list(variants) if variants else self.variants
        self._populate_hypothesis_cache_for_morphs([morph], variants=selected)
        records: list[MorphHypothesisRecord] = []
        for variant in selected:
            cache_key = (variant, morph.upper())
            cached = self._hypothesis_cache.get(cache_key)
            if cached is None:
                continue
            records.extend(list(cached))
        return records

    def _fetch_attested_definitions(
        self, word: str, *, variants: Iterable[str] | None = None
    ) -> list[AttestedDefinition]:
        selected = list(variants) if variants else self.variants
        records: list[AttestedDefinition] = []
        for variant in selected:
            cache_key = (variant, word.upper())
            cached = self._attested_cache.get(cache_key)
            if cached is None:
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
                variant_records: list[AttestedDefinition] = []
                for row in rows:
                    data = {key: row[key] for key in row.keys()}
                    variant_records.append(
                        AttestedDefinition(
                            variant=variant,
                            source_word=str(data.get("source_word")),
                            definition=data.get("definition"),
                            cluster_id=_safe_int(data.get("cluster_id")),
                            root_ngram=str(data.get("ngram")),
                        )
                    )
                cached = tuple(variant_records)
                self._attested_cache[cache_key] = cached
            records.extend(list(cached))
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

    def prewarm_translation_morphs(
        self,
        morphs: Iterable[str],
        *,
        variants: Iterable[str] | None = None,
        include_clusters: bool = True,
        include_residuals: bool = True,
    ) -> None:
        """Populate repository caches for a batch of translation morphs.

        Phrase translation repeatedly asks the repository for the same support
        and gloss metadata across many overlapping substrings. Warming those
        caches once per phrase keeps later per-token calls cheap without
        changing the single-word translation contract.
        """

        selected = list(variants) if variants else self.variants
        unique = {str(morph).upper() for morph in morphs if morph}
        if not unique or not selected:
            return
        # Phrase translation already knows it is entering a substring-heavy
        # phase. Warm the internal caches directly here so later token lookups
        # can stay cheap without materializing support records that would be
        # immediately thrown away.
        self._populate_cluster_cache_for_ngrams(unique, variants=selected)
        self._populate_residual_cache_for_morphs(unique, variants=selected)
        self._populate_hypothesis_cache_for_morphs(unique, variants=selected)
        self._populate_rejected_morph_cache_for_morphs(unique, variants=selected)
        self._populate_accepted_definition_count_cache_for_morphs(
            unique,
            variants=selected,
        )
        self._populate_accepted_definition_gloss_cache_for_morphs(
            unique,
            variants=selected,
            include_clusters=include_clusters,
            include_residuals=include_residuals,
        )
        self._populate_root_fnp_observation_cache_for_morphs(
            unique,
            variants=selected,
        )


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


def _root_fnp_observation_from_row(
    row: sqlite3.Row,
    *,
    variant: str,
) -> RootFNPObservation:
    """Convert one `root_glosses` row into a typed FNP observation.

    Repository methods keep SQLite concerns local. The FNP module receives this
    stable dataclass instead of raw rows, which keeps aggregation independent
    from view naming and schema quirks.
    """

    root = str(row["root"] or "").strip().upper()
    return RootFNPObservation(
        root=root,
        variant=variant,
        source_cluster_id=_safe_int(row["source_cluster_id"], default=0)
        if row["source_cluster_id"] is not None
        else None,
        nounness=_safe_float(row["pos_bias_nounness"]),
        modifier=_safe_float(row["pos_bias_modifier"]),
        verbness=_safe_float(row["pos_bias_verbness"]),
        confidence=_safe_float(row["confidence_score"]),
        definition=(
            str(row["definition"]).strip()
            if row["definition"] is not None and str(row["definition"]).strip()
            else None
        ),
        semantic_core_terms=_safe_json_array(row["semantic_core"]),
        decoding_guide=(
            str(row["decoding_guide"]).strip()
            if row["decoding_guide"] is not None
            and str(row["decoding_guide"]).strip()
            else None
        ),
        attachment_prefix_likelihood=_safe_float(row["attachment_prefix_likelihood"]),
        attachment_suffix_likelihood=_safe_float(row["attachment_suffix_likelihood"]),
        attachment_free_likelihood=_safe_float(row["attachment_free_likelihood"]),
        attachment_productivity=_safe_float(row["attachment_productivity"]),
        estimated_attachment_profile=(
            str(row["estimated_profile"]).strip()
            if row["estimated_profile"] is not None
            and str(row["estimated_profile"]).strip()
            else None
        ),
        observed_prefix_count=_safe_int(row["observed_prefix_count"]),
        observed_suffix_count=_safe_int(row["observed_suffix_count"]),
        observed_infix_count=_safe_int(row["observed_infix_count"]),
        observed_free_count=_safe_int(row["observed_free_count"]),
    )


def _fetch_root_fnp_evidence_examples(
    conn: sqlite3.Connection,
    roots: Iterable[str],
) -> dict[tuple[str, int | None], tuple[str, ...]]:
    """Read representative evidence snippets for root FNP debug profiles.

    Root-level FNP should remain auditable back to concrete examples when the
    `root_evidence_examples` view exists, while older fixtures/databases should
    continue to work with an empty example set.
    """

    unique = sorted({str(root).upper() for root in roots if root})
    if not unique:
        return {}
    placeholders = ",".join("?" for _ in unique)
    grouped: dict[tuple[str, int | None], list[str]] = {}
    try:
        rows = conn.execute(
            f"""
            SELECT
              root,
              evidence_word,
              evidence_definition,
              role,
              effect,
              sense_alignment,
              confidence,
              note,
              source_cluster_id
            FROM root_evidence_examples
            WHERE UPPER(TRIM(root)) IN ({placeholders})
            ORDER BY root COLLATE NOCASE ASC, confidence DESC, evidence_word ASC;
            """,
            tuple(unique),
        ).fetchall()
    except Exception:
        return {}

    for row in rows:
        root = str(row["root"] or "").strip().upper()
        if not root:
            continue
        cluster_id = (
            _safe_int(row["source_cluster_id"], default=0)
            if row["source_cluster_id"] is not None
            else None
        )
        snippet = _format_root_fnp_evidence_example(row)
        if not snippet:
            continue
        grouped.setdefault((root, cluster_id), []).append(snippet)
        grouped.setdefault((root, None), []).append(snippet)
    return {key: tuple(dict.fromkeys(values)) for key, values in grouped.items()}


def _format_root_fnp_evidence_example(row: sqlite3.Row) -> str:
    """Compress one root evidence row into a short human-readable snippet."""

    word = str(row["evidence_word"] or "").strip()
    definition = str(row["evidence_definition"] or "").strip()
    role = str(row["role"] or "").strip()
    effect = str(row["effect"] or "").strip()
    alignment = str(row["sense_alignment"] or "").strip()
    note = str(row["note"] or "").strip()
    confidence = _safe_float(row["confidence"])
    headline = " ".join(part for part in (word, definition) if part).strip()
    qualifiers = [part for part in (role, effect, alignment, note) if part]
    if confidence is not None:
        qualifiers.append(f"confidence={confidence:.2f}")
    if not headline and not qualifiers:
        return ""
    if qualifiers:
        return f"{headline} ({'; '.join(qualifiers)})" if headline else "; ".join(qualifiers)
    return headline


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
                part_of_speech=_dictionary_entry_pos(entry),
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


def _dictionary_entry_pos(entry: Mapping[str, object]) -> str | None:
    pos = entry.get("pos")
    if isinstance(pos, str) and pos.strip():
        return pos.strip()
    return None


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
    data = _parse_glossator_json(payload)
    if isinstance(data, Mapping):
        definition = data.get("DEFINITION")
        if definition is None:
            definition = data.get("definition")
        if isinstance(definition, str):
            definition = definition.strip() or None
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


def _parse_glossator_semantics(payload: object) -> tuple[list[str], list[str]]:
    if payload is None:
        return [], []
    if isinstance(payload, Mapping):
        parsed = payload
    else:
        parsed = _parse_glossator_json(str(payload).strip())
    if not isinstance(parsed, Mapping):
        return [], []
    semantic_core = _safe_json_array(
        parsed.get("SEMANTIC_CORE", parsed.get("semantic_core"))
    )
    negative_contrast = _safe_json_array(
        parsed.get("NEGATIVE_CONTRAST", parsed.get("negative_contrast"))
    )
    return semantic_core, negative_contrast


def _glossator_definition_is_accepted(payload: object) -> bool:
    definition, rejected = _parse_glossator_definition(payload)
    if rejected:
        return False
    return bool(definition and definition.strip())


def _parse_glossator_json(payload: str) -> Mapping[str, object] | str | None:
    """Parse nested glossator payloads into a consistent mapping.

    Translation data includes several generations of payload formatting:
    plain JSON objects, JSON wrapped in Markdown fences, and top-level objects
    that stash the real payload under ``RAW_TEXT``. Repository reads need to
    understand all of them so exact-word anchors and rejected residuals are not
    silently lost.
    """
    for attempt in (_load_nested_raw_text, _load_json_from_code_fence, _load_json):
        result = attempt(payload)
        if result is not None:
            return result
    return None


def _load_json(text: str) -> Mapping[str, object] | str | None:
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(data, Mapping):
        return data
    if isinstance(data, str):
        return data
    return None


def _load_json_from_code_fence(text: str) -> Mapping[str, object] | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return None
    parsed = _load_json(match.group(1))
    return parsed if isinstance(parsed, Mapping) else None


def _load_nested_raw_text(text: str) -> Mapping[str, object] | None:
    parsed = _load_json(text)
    if not isinstance(parsed, Mapping):
        return None
    raw_text = parsed.get("RAW_TEXT")
    if not isinstance(raw_text, str):
        return None
    nested = _load_json_from_code_fence(raw_text) or _load_json(raw_text)
    return nested if isinstance(nested, Mapping) else None


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
