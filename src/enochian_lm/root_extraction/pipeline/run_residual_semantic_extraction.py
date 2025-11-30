import logging
import math
import re
import json
from enochian_lm.common.sqlite_bootstrap import sqlite3
import random
import statistics as st
import time
import uuid, json, sys, platform, datetime
from typing import List, Optional, Dict, Any, Tuple, Sequence
from collections import defaultdict, Counter
from enochian_lm.root_extraction.utils.logger import save_log
from enochian_lm.root_extraction.tools.debate_residual_semantic_engine import (
    debate_remainder,
)
from enochian_lm.root_extraction.tools.solo_residual_semantic_engine import (
    solo_analyze_remainder,
)
from enochian_lm.common.config import get_config_paths
from enochian_lm.root_extraction.utils.semantic_search import (
    find_semantically_similar_words,
    compute_cluster_cohesion,
    cluster_definitions,
)
from enochian_lm.root_extraction.utils.candidate_finder import MorphemeCandidateFinder
from enochian_lm.root_extraction.utils.dictionary_loader import load_dictionary
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord
from enochian_lm.root_extraction.utils.embeddings import (
    get_fasttext_model,
    get_sentence_transformer,
    stream_text,
)
from enochian_lm.root_extraction.utils.residual_analysis import (
    exclude_root_segments,
    summarize_residuals,
)
from enochian_lm.root_extraction.utils.remainders import (
    RootRemainder,
    extract_root_remainders,
    persist_root_remainders,
    summarize_root_remainders,
)
from enochian_lm.root_extraction.utils.analytics_bridge import gather_morph_evidence
from enochian_lm.root_extraction.utils.preanalysis import (
    fetch_preanalysis_summary,
    mark_preanalysis_consumed,
    load_trusted_ngrams,
)
from enochian_lm.analysis.utils.sql import ensure_analysis_tables

logger = logging.getLogger(__name__)

CANDIDATE_TOP_K = 5
MIN_MULTI_SEGMENTS = 2

GOLD = "\033[38;5;178m"
GREEN = "\033[38;5;120m"
YELLOW = "\033[38;5;190m"
BLUE = "\033[38;5;153m"
PINK = "\033[38;5;213m"
RESET = "\033[0m"


class RemainderExtractionCrew:
    def __init__(self, style, use_remote):
        paths = get_config_paths()
        self.dictionary_path = paths["dictionary"]
        self.model_path = paths["model_output"]
        self.subst_map_path = paths["substitution_map"]
        self.output_path = paths["root_word_insights"]
        self.ngram_path = paths["ngram_index"]
        self.new_definitions_path = paths[style]

        # Load everything
        self.entries: list[EntryRecord] = self.load_entries()
        self.ngram_db = sqlite3.connect(paths["ngram_index"])
        self.new_definitions_db = sqlite3.connect(paths[style])
        self._prepare_db(self.new_definitions_db)
        ensure_analysis_tables(self.new_definitions_db)
        self._init_queue_table()
        self._ngram_inventory: list[tuple[str, int]] = []
        self._ngram_df: dict[str, int] = {}
        self._refresh_ngram_inventory()
        self.subst_map = self.load_subst_map()
        self.fasttext = get_fasttext_model(self.model_path)
        self.sentence_model_name = "paraphrase-MiniLM-L6-v2"
        self.sentence_model = get_sentence_transformer(self.sentence_model_name)
        self.trusted_ngrams = set(load_trusted_ngrams())
        self._trusted_max_len = max(
            (len(token) for token in self.trusted_ngrams), default=0
        )
        self.completed_roots, self.incomplete_roots = self._load_cluster_progress()
        self.completed_roots |= self._load_root_level_skips()
        self._breakdown_cache: dict[str, dict | None] = {}
        self._cycle_map: dict[str, int] = {}
        self.candidate_finder = MorphemeCandidateFinder(
            ngram_db_path=paths["ngram_index"],
            fasttext_model_path=self.model_path,
            dictionary_entries=self.entries,
            min_candidate_cos_sim=0.15,
            min_overlap_ratio=0.1,
            max_candidates=CANDIDATE_TOP_K,
            multi_segment_bonus=0.25,
        )

        self._upgrade_run_schema()
        self.run_id = self._begin_run(engine=style)
        self.use_remote = use_remote

    def _record_preanalysis_consumed(self, ngram: str) -> None:
        try:
            mark_preanalysis_consumed(
                self.new_definitions_db,
                root=ngram,
                run_id=self.run_id,
            )
        except sqlite3.Error:
            pass

    def _prepare_db(self, conn: sqlite3.Connection) -> None:
        """Set pragmatic SQLite settings for long runs."""
        conn.row_factory = sqlite3.Row
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA mmap_size=268435456;")  # 256MB

    def _begin_run(self, *, engine: str) -> str:
        """Create a row in `runs` and store the id on `self.run_id`."""
        run_id = uuid.uuid4().hex
        # Try to recover a readable embedder name
        embedder = self.sentence_model_name
        current_cycle = self._get_current_cycle()

        env_json = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sentence_transformers": self.sentence_model_name,
            "fasttext_model_path": str(getattr(self, "model_path", "")),
        }

        with self.new_definitions_db:
            try:
                self.new_definitions_db.execute(
                    """
                    INSERT INTO runs (run_id, run_name, engine, embedder, env_json, queue_cycle)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        f"{engine}-{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}Z",
                        engine,  # 'debate' or 'solo'
                        embedder,
                        json.dumps(env_json),
                        current_cycle,
                    ),
                )
            except sqlite3.Error:
                self.new_definitions_db.execute(
                    """
                    INSERT INTO runs (run_id, run_name, engine, embedder, env_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        f"{engine}-{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}Z",
                        engine,  # 'debate' or 'solo'
                        embedder,
                        json.dumps(env_json),
                    ),
                )
        self.run_id = run_id
        return run_id

    def _dynamic_coh_floor(self, n: int | None, gram_len: int | None) -> float:
        """
        Lenient cohesion floor:
        n ≤ 3  → 0.30   (down from 0.40)
        n ≤ 5  → 0.27
        n ≤ 10 → 0.23
        n > 10 → 0.18
        N-gram adjustment:
        gram_len ≤ 2  → -0.05
        gram_len ≥ 5  → +0.05
        """
        if not n or n <= 0:
            base = 0.23
        elif n <= 3:
            base = 0.30
        elif n <= 5:
            base = 0.27
        elif n <= 10:
            base = 0.23
        else:
            base = 0.18
        if gram_len is not None:
            if gram_len <= 2:
                base -= 0.05
            elif gram_len >= 5:
                base += 0.05
        return max(0.05, min(0.95, base))

    def _get_source_label(
        self, sem_entry: Optional[Dict[str, Any]], index_entry: Optional[Dict[str, Any]]
    ) -> str:
        if sem_entry and index_entry:
            return "both"
        if sem_entry:
            return "semantic"
        if index_entry:
            return "index"
        return "other"

    def _insert_skip(
        self,
        ngram,
        *,
        reason_code: str,
        sem_count: int | None = None,
        idx_count: int | None = None,
        overlap_count: int | None = None,
        cluster_index: int | None = None,
    ) -> None:
        """Write a skip record. All *_count fields may be NULL."""
        root = ngram[0] if isinstance(ngram, (list, tuple)) else ngram
        with self.new_definitions_db:
            self.new_definitions_db.execute(
                """
                INSERT INTO skips (
                run_id, ngram, cluster_index, reason_code, sem_count, idx_count, overlap_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    root.upper(),
                    cluster_index,
                    reason_code,
                    sem_count,
                    idx_count,
                    overlap_count,
                ),
            )

    def _normalize_citation(self, c):
        """Return (location, context) from a variety of citation shapes."""
        if c is None:
            return (None, None)
        # dict-like
        if isinstance(c, dict):
            loc = c.get("location") or c.get("folio") or c.get("page") or c.get("loc")
            ctx = c.get("context") or c.get("snippet") or c.get("text")
            return (
                str(loc) if loc is not None else None,
                str(ctx) if ctx is not None else None,
            )
        # tuple/list
        if isinstance(c, (list, tuple)):
            if len(c) == 2:
                return (
                    str(c[0]) if c[0] is not None else None,
                    str(c[1]) if c[1] is not None else None,
                )
            if len(c) == 1:
                return (str(c[0]) if c[0] is not None else None, None)
        # plain string
        if isinstance(c, str):
            return (c, None)
        # fallback
        return (None, None)

    def load_entries(self) -> List[EntryRecord]:
        return load_dictionary(str(self.dictionary_path))

    def load_subst_map(self):
        with open(self.subst_map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_ngrams(self, word, min_n=1, max_n=4):
        ngrams = set()
        word = word.lower()
        for n in range(min_n, max_n + 1):
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n]
                if not re.search(r"\d", ngram):
                    ngrams.add(ngram)
        return ngrams

    def _normalize_root(self, value: str) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _get_field_value(item, field: str, default=""):
        if isinstance(item, dict):
            return item.get(field, default)
        return getattr(item, field, default)

    def _init_queue_table(self) -> None:
        cursor = self.new_definitions_db.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ngram_queue (
                ngram TEXT PRIMARY KEY,
                queue_index INTEGER NOT NULL,
                cycles_completed INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self.new_definitions_db.commit()

    def _upgrade_run_schema(self) -> None:
        """Add newer optional columns without dropping user data."""
        cursor = self.new_definitions_db.cursor()

        # Inspect table schema
        cursor.execute("PRAGMA table_info(runs)")
        cols = {row[1] for row in cursor.fetchall()}  # row[1] = column name

        if "queue_cycle" not in cols:
            cursor.execute("ALTER TABLE runs ADD COLUMN queue_cycle INTEGER DEFAULT 0")

        self.new_definitions_db.commit()

    def _refresh_ngram_inventory(self) -> None:
        inventory = [
            (self._normalize_root(ngram), int(df or 0))
            for ngram, df in self.stream_ngrams_from_sqlite(min_freq=1)
        ]
        self._ngram_inventory = inventory
        self._ngram_df = {ngram: df for ngram, df in inventory}
        self._sync_queue_with_inventory()

    def _sync_queue_with_inventory(self) -> None:
        cursor = self.new_definitions_db.cursor()
        rows = cursor.execute("SELECT ngram, queue_index FROM ngram_queue").fetchall()
        seen = {
            self._normalize_root(row[0]): int(row[1]) for row in rows if row and row[0]
        }
        inserts = []
        for idx, (ngram, _) in enumerate(self._ngram_inventory):
            if not ngram or ngram in seen:
                continue
            inserts.append((ngram, idx))
        if inserts:
            cursor.executemany(
                """
                INSERT INTO ngram_queue (ngram, queue_index, cycles_completed)
                VALUES (?, ?, 0)
                """,
                inserts,
            )
            self.new_definitions_db.commit()

    def _load_queue_order(self) -> list[tuple[str, int, int]]:
        cursor = self.new_definitions_db.cursor()
        rows = cursor.execute(
            "SELECT ngram, queue_index, cycles_completed FROM ngram_queue "
            "ORDER BY cycles_completed ASC, queue_index ASC"
        ).fetchall()
        ordered: list[tuple[str, int, int]] = []
        for row in rows:
            norm = self._normalize_root(row[0])
            if not norm:
                continue
            ordered.append((norm, self._ngram_df.get(norm, 0), int(row[2] or 0)))
        return ordered

    def _load_skipped_queue(
        self, *, reason_code: str | None = None
    ) -> list[tuple[str, int, int]]:
        """Return skipped roots aligned to queue order for reprocessing."""

        cursor = self.new_definitions_db.cursor()
        clauses = ["cluster_index IS NULL"]
        params: list[str] = []
        if reason_code:
            clauses.append("reason_code = ?")
            params.append(reason_code)

        where_clause = " AND ".join(clauses)

        rows = cursor.execute(
            f"""
            SELECT s.ngram,
                   COALESCE(q.queue_index, 9223372036854775807) AS queue_index,
                   COALESCE(q.cycles_completed, 0) AS cycles_completed
            FROM skips s
            LEFT JOIN ngram_queue q ON lower(q.ngram) = lower(s.ngram)
            WHERE {where_clause}
            GROUP BY s.ngram
            ORDER BY queue_index ASC, s.ngram ASC
            """,
            params,
        ).fetchall()

        ordered: list[tuple[str, int, int]] = []
        seen: set[str] = set()
        for row in rows:
            norm = self._normalize_root(row[0])
            if not norm or norm in seen:
                continue
            seen.add(norm)
            ordered.append((norm, self._ngram_df.get(norm, 0), int(row[2] or 0)))
        return ordered

    def _get_current_cycle(self) -> int:
        cursor = self.new_definitions_db.cursor()
        row = cursor.execute("SELECT MIN(cycles_completed) FROM ngram_queue").fetchone()
        if not row or row[0] is None:
            return 0
        return int(row[0])

    def _get_cycles_completed(self, root_norm: str) -> int:
        if not root_norm:
            return 0
        if root_norm in self._cycle_map:
            return self._cycle_map[root_norm]
        cursor = self.new_definitions_db.cursor()
        try:
            row = cursor.execute(
                "SELECT cycles_completed FROM ngram_queue WHERE ngram = ?",
                (root_norm,),
            ).fetchone()
        except sqlite3.Error:
            return 0
        value = int(row[0]) if row and row[0] is not None else 0
        self._cycle_map[root_norm] = value
        return value

    def _advance_queue(self, root_norm: str) -> None:
        if not root_norm:
            return
        cursor = self.new_definitions_db.cursor()
        cursor.execute(
            "UPDATE ngram_queue SET cycles_completed = cycles_completed + 1 WHERE ngram = ?",
            (root_norm,),
        )
        if cursor.rowcount == 0:
            cursor.execute(
                """
                INSERT INTO ngram_queue (ngram, queue_index, cycles_completed)
                VALUES (?, ?, 1)
                """,
                (root_norm, len(self._ngram_inventory)),
            )
        self.new_definitions_db.commit()
        self._cycle_map[root_norm] = self._get_cycles_completed(root_norm)

    def _load_cluster_progress(self) -> tuple[set[str], set[str]]:
        cursor = self.new_definitions_db.cursor()
        status: dict[str, dict[str, bool]] = {}
        query = """
            SELECT ngram,
                   run_id,
                   MAX(COALESCE(cluster_size, 0)) AS cluster_size,
                   COUNT(*) AS cluster_count
            FROM clusters
            GROUP BY ngram, run_id
        """
        try:
            rows = cursor.execute(query).fetchall()
        except sqlite3.OperationalError:
            try:
                fallback = cursor.execute(
                    "SELECT DISTINCT ngram FROM clusters"
                ).fetchall()
            except sqlite3.Error:
                return set(), set()
            completed = {self._normalize_root(row[0]) for row in fallback if row[0]}
            return completed, set()

        for row in rows:
            ngram = self._normalize_root(row[0])
            if not ngram:
                continue
            entry = status.setdefault(ngram, {"complete": False, "incomplete": False})
            total = int(row[2] or 0)
            recorded = int(row[3] or 0)
            if total > 0 and recorded < total:
                entry["incomplete"] = True
            else:
                entry["complete"] = True

        completed = {n for n, flags in status.items() if flags.get("complete")}
        incomplete = {
            n
            for n, flags in status.items()
            if flags.get("incomplete") and n not in completed
        }
        return completed, incomplete

    def _load_root_level_skips(self) -> set[str]:
        cursor = self.new_definitions_db.cursor()
        try:
            rows = cursor.execute(
                "SELECT DISTINCT ngram FROM skips WHERE cluster_index IS NULL"
            ).fetchall()
        except sqlite3.Error:
            return set()
        return {self._normalize_root(row[0]) for row in rows if row and row[0]}

    def _is_root_processed(
        self, root: str, *, current_cycle: int | None = None
    ) -> bool:
        norm = self._normalize_root(root)
        if not norm:
            return False

        cycle = self._get_cycles_completed(norm)
        if current_cycle is None:
            current_cycle = self._get_current_cycle()
        return cycle > current_cycle

    def _is_root_incomplete(self, root: str) -> bool:
        return self._normalize_root(root) in self.incomplete_roots

    def _mark_root_complete(self, root: str) -> None:
        norm = self._normalize_root(root)
        if norm:
            self.completed_roots.add(norm)
            self.incomplete_roots.discard(norm)
            self._advance_queue(norm)

    @staticmethod
    def _round_vector(values: Sequence[float], places: int = 6) -> list[float]:
        factor = 10**places
        return [math.floor(float(val) * factor + 0.5) / factor for val in values]

    @staticmethod
    def _vector_norm(values: Sequence[float]) -> float:
        return math.sqrt(sum(float(v) * float(v) for v in values))

    def _persist_root_remainders(self, rows: list[RootRemainder]) -> None:
        """Persist deterministic root remainder spans for the current run."""

        if not rows:
            return

        persist_root_remainders(self.new_definitions_db, rows=rows)

    def _persist_composite_reconstruction(
        self, composites: list[dict[str, object]]
    ) -> None:
        if not composites:
            return

        ensure_analysis_tables(self.new_definitions_db)

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        composite_rows: list[tuple[str, str | None, str, float, str, str, str]] = []
        morph_rows: dict[str, tuple[str, float, str]] = {}

        for item in composites:
            norm = self._normalize_root(item.get("normalized"))
            if not norm:
                continue

            breakdown = item.get("breakdown") or {}
            vector = self.fasttext.get_word_vector(norm)
            morphs: list[str] = []
            for seg in breakdown.get("segments") or []:
                canonical = str(seg.get("canonical") or "").strip()
                if canonical and canonical not in morphs:
                    morphs.append(canonical)
                if canonical and canonical not in morph_rows:
                    morph_vector = self.fasttext.get_word_vector(canonical)
                    morph_rows[canonical] = (
                        json.dumps(self._round_vector(morph_vector)),
                        round(self._vector_norm(morph_vector), 4),
                        timestamp,
                    )

            gold_gloss = item.get("definition") or None
            composite_rows.append(
                (
                    norm,
                    (
                        gold_gloss
                        if isinstance(gold_gloss, str) and gold_gloss.strip()
                        else None
                    ),
                    json.dumps(self._round_vector(vector)),
                    round(float(breakdown.get("residual_ratio") or 0.0), 4),
                    json.dumps(morphs),
                    "fasttext",
                    timestamp,
                )
            )

        if not composite_rows:
            return

        tokens = [(row[0],) for row in composite_rows]

        with self.new_definitions_db:
            self.new_definitions_db.executemany(
                "DELETE FROM composite_reconstruction WHERE token = ?",
                tokens,
            )
            self.new_definitions_db.executemany(
                """
                INSERT INTO composite_reconstruction (
                  token, gold_gloss, pred_vector_json, recon_error, used_morphs_json, vector_source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                composite_rows,
            )

            if morph_rows:
                self.new_definitions_db.executemany(
                    """
                    INSERT INTO morph_semantic_vectors (morph, vector_json, l2_norm, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(morph) DO UPDATE SET
                        vector_json=excluded.vector_json,
                        l2_norm=excluded.l2_norm,
                        updated_at=excluded.updated_at
                    """,
                    [
                        (morph, vector_json, l2_norm, ts)
                        for morph, (vector_json, l2_norm, ts) in morph_rows.items()
                    ],
                )

        logger.info(
            "Persisted composite reconstructions and morph vectors",
            extra={"composites": len(composite_rows), "morphs": len(morph_rows)},
        )

    def _get_candidate_breakdown(self, word: str) -> dict | None:
        norm = self._normalize_root(word)
        if not norm:
            return None
        if norm in self._breakdown_cache:
            return self._breakdown_cache[norm]
        candidates = self.candidate_finder.find_candidates(norm, top_k=CANDIDATE_TOP_K)
        breakdown = None
        multi = []
        singles = []
        for cand in candidates:
            bd = cand.get("breakdown") if isinstance(cand, dict) else None
            if not bd:
                continue
            segments = bd.get("segments") if isinstance(bd, dict) else []
            if segments and len(segments) >= MIN_MULTI_SEGMENTS:
                multi.append(bd)
            else:
                singles.append(bd)
        if multi:
            breakdown = multi[0]
        elif singles:
            breakdown = singles[0]
        self._breakdown_cache[norm] = breakdown
        return breakdown

    def _has_residual_fragments(self, breakdown: dict | None, root_norm: str) -> bool:
        if not breakdown:
            return False
        segments = breakdown.get("segments") or []
        if not segments:
            return False
        uncovered = breakdown.get("uncovered") or []
        for frag in uncovered:
            text = frag.get("text") if isinstance(frag, dict) else frag
            text = str(text or "").strip()
            if text and text.lower() != root_norm:
                return True
        return False

    def _cluster_supports_residual_override(self, cluster, root_token: str) -> bool:
        root_norm = self._normalize_root(root_token)
        if not root_norm:
            return False
        for entry in cluster:
            norm_value = self._normalize_root(
                self._get_field_value(entry, "normalized", "")
            )
            if not norm_value or norm_value == root_norm:
                continue
            breakdown = self._get_candidate_breakdown(norm_value)
            if self._has_residual_fragments(breakdown, root_norm):
                return True
        return False

    def _fetch_ngram_contexts(self, ngram: str) -> list[str]:
        cursor = self.ngram_db.cursor()
        try:
            rows = cursor.execute(
                "SELECT DISTINCT canonical FROM ngram_membership WHERE ngram = ?",
                (str(ngram).lower(),),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]).strip() for row in rows if row and row[0]]

    def _is_text_covered_by_trusted(self, text: str) -> bool:
        value = str(text or "").strip().upper()
        if not value:
            return True
        if not self.trusted_ngrams:
            return False
        length = len(value)
        max_len = self._trusted_max_len or length
        dp = [False] * (length + 1)
        dp[0] = True
        for i in range(1, length + 1):
            start = max(0, i - max_len)
            for j in range(start, i):
                if dp[j] and value[j:i] in self.trusted_ngrams:
                    dp[i] = True
                    break
        return dp[length]

    def _can_isolate_ngram_in_word(self, ngram: str, word: str) -> bool:
        target = str(ngram or "").upper()
        haystack = str(word or "").upper()
        if not target or not haystack:
            return False
        start = 0
        while start <= len(haystack):
            idx = haystack.find(target, start)
            if idx == -1:
                break
            prefix = haystack[:idx]
            suffix = haystack[idx + len(target) :]
            if self._is_text_covered_by_trusted(
                prefix
            ) and self._is_text_covered_by_trusted(suffix):
                return True
            start = idx + 1
        return False

    def _should_skip_single_occurrence(self, ngram: str, df_count: int) -> bool:
        if df_count != 1:
            return False
        contexts = self._fetch_ngram_contexts(ngram)
        if not contexts or not self.trusted_ngrams:
            return False
        for word in contexts:
            if self._can_isolate_ngram_in_word(ngram, word):
                return False
        return True

    def _extract_evaluation(self, s: str) -> str | None:
        # 1) Try strict JSON first
        try:
            obj = json.loads(s)
            return obj.get("EVALUATION") or obj.get("evaluation")
        except json.JSONDecodeError:
            pass

        # 2) Fallback: pull it directly from text (case-insensitive)
        m = re.search(r'"(?i:evaluation)"\s*:\s*"([^"]+)"', s)
        return m.group(1) if m else None

    def stream_ngrams_from_sqlite(self, min_freq=2):
        cursor = self.ngram_db.cursor()
        # DF is the count of canonicals in membership; TF is in ngrams table.
        query = """
            SELECT n.ngram,
                COALESCE(COUNT(m.canonical), 0) AS df_count
            FROM ngrams AS n
            LEFT JOIN ngram_membership AS m
                ON m.ngram = n.ngram
            GROUP BY n.ngram
            HAVING df_count >= ?
            ORDER BY LENGTH(n.ngram) DESC, df_count DESC
        """
        for ngram, df in cursor.execute(query, (min_freq,)):
            yield ngram, df

    def is_ngram_in_variants(self, ngram, canonical):
        cursor = self.ngram_db.cursor()
        cursor.execute(
            "SELECT 1 FROM ngram_membership WHERE ngram = ? AND canonical = ? LIMIT 1",
            (ngram.lower(), canonical.lower()),
        )
        return cursor.fetchone() is not None

    def get_matching_variant(self, ngram: str, canonical: str) -> str | None:
        """
        With the new schema, membership stores only (ngram, canonical).
        Return canonical if the pair exists; otherwise None.
        """
        cursor = self.ngram_db.cursor()
        cursor.execute(
            """
            SELECT canonical
            FROM ngram_membership
            WHERE ngram = ?
            AND canonical = ?
            LIMIT 1
            """,
            (ngram.lower(), canonical.lower()),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def dedupe_by_normalized(self, entries: list[EntryRecord]):
        seen = set()
        unique = []
        for entry in entries:
            key = entry.get("canonical")
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return unique

    def _build_stats_summary(
        self,
        root: str,
        *,
        cluster_size: int,
        cohesion_score: float,
        semantic_hits: int,
        semantic_coverage: float,
        sem_count: int,
        idx_count: int,
        overlap_count: int,
        residual_report: dict | None = None,
    ) -> str:
        """
        Compact, human-readable summary passed to debate/solo engines.
        Keep it short; these strings are used inside prompts.
        """
        summary = (
            f"root={root.upper()} | "
            f"cluster_size={cluster_size} | "
            f"cohesion={cohesion_score:.3f} | "
            f"semantic_hits={semantic_hits} | "
            f"coverage={semantic_coverage:.3f} | "
            f"sem={sem_count} | idx={idx_count} | overlap_count={overlap_count}"
        )
        if residual_report:
            explained = residual_report.get("explained_ratio")
            residual = residual_report.get("residual_ratio")
            headline = residual_report.get("headline")
            if explained is not None:
                summary += f" | explained={float(explained):.2f}"
            if residual is not None:
                summary += f" | residual={float(residual):.2f}"
            if headline:
                summary += f" | residues={headline}"
        return summary

    def evaluate_ngram(
        self,
        ngram: str,
        cluster: List[Dict[str, Any]],
        cohesion_score: float,
        semantic_hits: int,
        semantic_coverage: float,
        sem_count: int,
        idx_count: int,
        overlap_count: int,
        stats_summary,
        stream_callback=None,
        style: str = "debate",
    ) -> Dict[str, Any]:
        """
        Wraps either the debate engine or the solo engine, and standardizes keys.
        """

        def _get_field(item, field, default=""):
            # Unified accessor for Entry objects and dicts.
            if isinstance(item, dict):
                return item.get(field, default)
            else:
                return getattr(item, field, default)

        trimmed_cluster = []
        anchor_entry = None
        ngram_lower = ngram.lower()
        root_entry = None

        for c in cluster:
            if not _get_field(c, "definition", ""):
                continue

            citations = _get_field(c, "key_citations", "")
            if citations and isinstance(citations[0], dict):
                contexts = [
                    _get_field(cite, "context", "").strip()
                    for cite in citations
                    if _get_field(cite, "context", "")
                ]
            else:
                contexts = citations

            definition = (
                f"{_get_field(c, 'definition', '')} [{' / '.join(contexts[:2])}]"
                if contexts
                else _get_field(c, "definition", "")
            )

            entry = {
                "word": _get_field(c, "word", "").upper(),
                "definition": definition,
                "enhanced_definition": _get_field(c, "enhanced_definition", ""),
                "normalized": _get_field(c, "normalized", "").lower(),
                "fasttext": _get_field(c, "fasttext", "0.0"),
                "semantic": _get_field(c, "semantic", "0.0"),
                "score": _get_field(c, "score", "0.0"),
                "tier": _get_field(c, "tier", "Untiered"),
                "priority": _get_field(c, "priority", "0"),
                "citations": _get_field(c, "citations", ""),
                "source": _get_field(c, "source", "unknown"),
            }

            # Check if this is the literal ngram word
            if _get_field(entry, "normalized", "") == ngram_lower:
                root_entry = c  # We can directly use the Entry object
                anchor_entry = {
                    "word": f"{_get_field(c, 'word', '')} ⭐️",
                    "definition": _get_field(c, "enhanced_definition", ""),
                    "normalized": _get_field(c, "normalized", "").lower(),
                }
            else:
                trimmed_cluster.append(entry)

        # If anchor exists, put it at the top
        if anchor_entry:
            trimmed_cluster.insert(0, anchor_entry)

        # Check for exact-match anchor word in the cluster
        if root_entry and _get_field(root_entry, "definition", ""):
            root_callout = (
                f"📌 Note: The proposed root '{ngram.upper()}' is itself a defined word: "
                f"**{_get_field(root_entry, 'definition', '')}**\n"
                "This may indicate its role as a base morpheme from which related forms are derived.\n"
            )
        else:
            root_callout = ""

        tiered_groups = defaultdict(list)
        for c in cluster:
            tier = _get_field(c, "tier", "Untiered")
            tiered_groups[tier].append(c)
        for tier, group in tiered_groups.items():
            group.sort(
                key=lambda x: (
                    -x.get("priority", 0),
                    -x.get("cluster_similarity", 0),
                    -x.get("score", 0),
                )
            )
            seen = set()
            unique = []
            for entry in group:
                key = _get_field(entry, "normalized", "").lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(entry)
            tiered_groups[tier] = unique
        # Cohesion & coverage already computed as `cohesion_score` and `semantic_coverage`
        semantic_hits = sum(
            1 for c in cluster if _get_field(c, "source", "") in ("semantic", "both")
        )

        residual_inputs = []
        segment_counter: Counter[str] = Counter()
        residual_counter: Counter[str] = Counter()
        remainder_rows: list[RootRemainder] = []
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        seen_norms = set()
        for original in cluster:
            norm_form = _get_field(original, "normalized", "").lower()
            if not norm_form or norm_form in seen_norms:
                continue
            seen_norms.add(norm_form)
            display_word = _get_field(original, "word", norm_form)
            breakdown = self._get_candidate_breakdown(norm_form)
            if not breakdown:
                uncovered = (
                    [{"span": [0, len(norm_form)], "text": norm_form}]
                    if norm_form
                    else []
                )
                breakdown = {
                    "segments": [],
                    "uncovered": uncovered,
                    "coverage_ratio": 0.0,
                    "residual_ratio": 1.0 if uncovered else 0.0,
                }
            else:
                breakdown = exclude_root_segments(
                    breakdown,
                    root_norm=ngram_lower,
                    target=norm_form,
                )
            if breakdown:
                for seg in breakdown.get("segments", []):
                    canonical = str(seg.get("canonical", "")).strip()
                    if canonical and canonical.lower() != ngram_lower:
                        segment_counter[canonical] += 1
                for uncovered_entry in breakdown.get("uncovered", []):
                    if isinstance(uncovered_entry, dict):
                        frag = str(uncovered_entry.get("text", "")).strip()
                    else:
                        frag = str(uncovered_entry).strip()
                    if frag and frag.lower() != ngram_lower:
                        residual_counter[frag] += 1
            residual_inputs.append(
                {
                    "word": display_word,
                    "normalized": norm_form,
                    "definition": _get_field(original, "definition", ""),
                    "breakdown": breakdown,
                }
            )

            for remainder in extract_root_remainders(ngram_lower, norm_form):
                remainder_rows.append(
                    RootRemainder(
                        run_id=self.run_id,
                        root=ngram_lower,
                        word=str(display_word or norm_form),
                        normalized=norm_form,
                        remainder=remainder["remainder"],
                        kind=remainder["kind"],
                        span_start=int(remainder["span_start"]),
                        span_end=int(remainder["span_end"]),
                        created_at=now_ts,
                    )
                )

        residual_report = summarize_residuals(
            root=ngram_lower,
            analyses=residual_inputs,
        )

        self._persist_root_remainders(remainder_rows)
        remainder_summary = summarize_root_remainders(
            self.new_definitions_db, run_id=self.run_id, root=ngram_lower
        )
        if isinstance(residual_report, dict):
            guidance_json = residual_report.get("residual_guidance_json") or {}
            guidance_json["remainders"] = remainder_summary
            residual_report["residual_guidance_json"] = guidance_json

        analytics_summary = gather_morph_evidence(
            self.new_definitions_db,
            root=ngram_lower,
            partner_counts=segment_counter,
            residual_counts=residual_counter,
        )

        self._persist_composite_reconstruction(residual_inputs)

        fallback_summary = fetch_preanalysis_summary(
            self.new_definitions_db,
            root=ngram_lower,
        )
        if fallback_summary:
            analytics_summary = analytics_summary or {}
            merged_summary = list(analytics_summary.get("summary_lines") or [])
            for line in fallback_summary.get("summary_lines", []):
                if line not in merged_summary:
                    merged_summary.append(line)
            analytics_summary["summary_lines"] = merged_summary

            merged_focus = list(analytics_summary.get("focus_lines") or [])
            for line in fallback_summary.get("focus_lines", []):
                if line not in merged_focus:
                    merged_focus.append(line)
            analytics_summary["focus_lines"] = merged_focus

        # Summarize stats for agents with residual diagnostics
        coverage_pct = round(semantic_coverage * 100, 1)
        stats_lines = [
            f"{root_callout}The proposed root '{ngram.upper()}' has:",
            f"- Cohesion Score: {cohesion_score} (semantic similarity among definitions; from 0.0 to 1.0, with higher being better)",
            f"- Semantic Coverage: {coverage_pct}% ({semantic_hits}/{len(cluster)} words match semantically)",
            f"- Candidate Count: {len(cluster)}",
        ]
        focus_prompt = residual_report.get("focus_prompt") if residual_report else ""
        if focus_prompt:
            stats_lines.append("")
            stats_lines.append("Morphological diagnostics:")
            stats_lines.append(focus_prompt)

        analytics_summary = analytics_summary or {}
        analytics_lines = analytics_summary.get("summary_lines") or []
        if analytics_lines:
            stats_lines.append("")
            stats_lines.append("Analytics priors:")
            stats_lines.extend(analytics_lines)

        analytics_focus = analytics_summary.get("focus_lines") or []
        if analytics_focus:
            focus_block = "Attribution highlights:\n" + "\n".join(
                f"- {line}" for line in analytics_focus
            )
            if focus_prompt:
                focus_prompt = f"{focus_prompt}\n\n{focus_block}"
            else:
                focus_prompt = focus_block

        compact_summary = self._build_stats_summary(
            ngram,  # or ngram if iterating strings
            cluster_size=len(cluster),
            cohesion_score=cohesion_score,
            semantic_hits=semantic_hits,
            semantic_coverage=semantic_coverage,
            sem_count=sem_count,
            idx_count=idx_count,
            overlap_count=overlap_count,
            residual_report=residual_report,
        )

        stats_summary = (
            "\n".join(stats_lines) + f"\n\nCompact metrics: {compact_summary}"
        )

        the_result = {"raw_output": {}}
        # NOTE: debate is presently inaccessible!
        if style == "debate":
            the_result = debate_remainder(
                root=ngram.upper(),
                candidates=trimmed_cluster,
                stats_summary=stats_summary,
                stream_callback=stream_callback,
                root_entry=None,  # can be None; debate engine will handle
                use_remote=self.use_remote,
                residual_prompt=focus_prompt,
                query_db=self.new_definitions_db,
                query_run_id=self.run_id,
            )
        else:
            the_result = solo_analyze_remainder(
                root=ngram.upper(),
                candidates=trimmed_cluster,
                stats_summary=stats_summary,
                stream_callback=stream_callback,
                root_entry=None,  # can be None; debate engine will handle
                use_remote=self.use_remote,
                residual_prompt=focus_prompt,
                residual_guidance=(
                    residual_report.get("residual_guidance_json")
                    if residual_report
                    else None
                ),
                query_db=self.new_definitions_db,
                query_run_id=self.run_id,
            )

        if isinstance(the_result, dict):
            the_result.setdefault("residual_report", residual_report)
            the_result.setdefault("analytics_summary", analytics_summary)

        # Normalize expected keys (be defensive)
        model = (
            the_result.get("Model")
            or the_result.get("Glossator_Model")
            or the_result.get("raw_output", {}).get("Model")
            or the_result.get("raw_output", {}).get("Glossator_Model")
            or ""
        )
        normalized = {
            "Glossator": the_result.get("Glossator", "") or "",
            "Model": model,
            "Glossator_Prompt": the_result.get("Glossator_Prompt", ""),
            "Adjudicator": the_result.get("Adjudicator", ""),
            "Adjudicator_Prompt": the_result.get("Adjudicator_Prompt", ""),
            "Archivist": the_result.get("Archivist", "") or "",
            "raw_output": the_result,
        }
        if isinstance(residual_report, dict):
            residual_report.setdefault("analytics_summary", analytics_summary)
        normalized["residual_report"] = residual_report
        normalized["analytics_summary"] = analytics_summary
        return normalized

    def process_ngrams(
        self,
        max_words=None,
        stream_callback=None,
        single_ngram=None,
        style="debate",
        min_semantic_similarity: float = 0.60,
        process_remainders: bool = False,
        skipped_reason_code: str | None = None,
    ):
        # TEMPORARY; delete when debate_remainder is come online!
        if style != "solo":
            raise ValueError(
                "run_remainder_extraction is intended for style='solo' only."
            )

        def _get_field(item, field, default=""):
            return self._get_field_value(item, field, default)

        if single_ngram:
            self._refresh_ngram_inventory()
            stream = self._load_queue_order()
            self._cycle_map = {ng: cyc for ng, _, cyc in stream}
            current_cycle = self._get_current_cycle()
            pending = [(single_ngram.upper(), 1)]
            cycle_msg = f"🔍 Focusing on single root {single_ngram.upper()}...\n\n"
        else:

            skip_roots = self._load_root_level_skips()
            pending = [
                (root, df)
                for root, df in self._ngram_inventory
                if self._normalize_root(root) in skip_roots
            ]
            cycle_msg = "📎 Running remainder extraction on *skipped* roots only.\n\n"
            incomplete_roots = {
                item[0] for item in pending if self._is_root_incomplete(item[0])
            }
            ngrams = [item for item in pending if item[0] in incomplete_roots] + [
                item for item in pending if item[0] not in incomplete_roots
            ]
            stream_text(cycle_msg)

        output = []
        seen_words = 0

        if style == "debate":
            stream_text("🪄 Initializing semantic tribunal...\n\n")
        elif style == "solo":
            stream_text("⏰ Waking up the Enochiana scholar...\n\n")
        time.sleep(0.25)

        for count, ngram in enumerate(ngrams):
            root_token, df_count = ngram
            if self._is_root_processed(
                root_token,
                current_cycle=current_cycle if "current_cycle" in locals() else None,
            ):
                continue

            semantic_candidates = find_semantically_similar_words(
                ft_model=self.fasttext,
                sentence_model=self.sentence_model,
                entries=self.entries,
                target_word=root_token,  # or just ngram if you’re iterating strings
                subst_map=self.subst_map,  # <-- required now
                min_similarity=min_semantic_similarity,
            )

            index_candidates = self.candidate_finder.get_all_ngram_candidates(
                root_token
            )

            stream_text(
                f"[{GREEN}✓{RESET}] Beginning examination of root-word candidate {GOLD}{root_token.upper()}{RESET}.\n",
                delay=0.005,
            )

            clusters_result = cluster_definitions(
                semantic_candidates, self.sentence_model
            )
            clusters = clusters_result["clusters"]
            total_clusters = len(clusters)
            # dedupe any clusters that are identical up to ordering
            unique, seen_sigs = [], set()
            for cl in clusters:
                # build a signature: sorted tuple of each member’s normalized form
                sig = tuple(sorted(c["normalized"] for c in cl))
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    unique.append(cl)
            count_deduped = len(clusters) - len(unique)
            if count_deduped > 1:
                deduped_clusters_text = (
                    f"[⚠️] Skipping {PINK}{count_deduped}{RESET} duplicate cluster{'s' if count_deduped > 1 else ''}"
                    f" for '{GOLD}{root_token.upper()}{RESET}'; no sense in evaluating if it's the same set of words.\n\n"
                )
                stream_text(deduped_clusters_text)

            clusters = unique
            best_config = clusters_result["config"]

            if len(clusters) == 0:
                follow_phrase = ""
            elif len(clusters) < 10:
                follow_phrase = "Should be fairly quick, all considered!\n\n"
            else:
                follow_phrase = "This could take a while if we're being honest...\n\n"
            if len(clusters) > 0:
                full_phrase = (
                    f"Beginning the evaluation of {PINK}{len(clusters)}{RESET} group{'s' if len(clusters) > 1 else ''}.\n\n"
                    + follow_phrase
                )
            else:
                full_phrase = f"Looks like this one ain't happening. 🤷‍♀️\n\n"
            stream_text(full_phrase)
            time.sleep(0.25)

            for cluster_id, cluster in enumerate(clusters):
                sem_norms = {
                    _get_field(c, "normalized", "")
                    for c in cluster
                    if _get_field(c, "normalized", "")
                }
                index_norms = {
                    _get_field(c, "normalized", "")
                    for c in index_candidates
                    if _get_field(c, "normalized", "")
                }
                overlap = sem_norms & index_norms

                idx_formatted_list = [
                    f"{BLUE}{norm.upper()}{RESET}" for norm in list(index_norms)[:5]
                ]
                random.shuffle(idx_formatted_list)
                idx_formatted_list = (
                    f"And some words/variants containing the ngram (up to 5): "
                    + ", ".join(idx_formatted_list)
                )
                idx_formatted_list += "..." if len(index_norms) > 5 else ""
                stream_text(idx_formatted_list)
                time.sleep(0.25)
                print()

                merged_cluster = []

                merged_items = list(cluster) + index_candidates

                seen_norms = set()
                for word in merged_items:
                    norm = _get_field(word, "normalized", "").lower()
                    if not norm or norm in seen_norms:
                        continue
                    seen_norms.add(norm)

                    sem_entry = next(
                        (c for c in cluster if _get_field(c, "normalized", "") == norm),
                        None,
                    )
                    index_entry = next(
                        (
                            c
                            for c in index_candidates
                            if _get_field(c, "normalized", "") == norm
                        ),
                        None,
                    )

                    if not self.is_ngram_in_variants(root_token, norm):
                        continue

                    if isinstance(norm, str) and norm:
                        variant_used = self.get_matching_variant(root_token, norm)
                    else:
                        variant_used = None

                    def safe_list(ls):
                        return ls if isinstance(ls, list) else []

                    sem_cits = (
                        safe_list(sem_entry.get("citations")) if sem_entry else []
                    )
                    idx_cits = (
                        safe_list(index_entry.get("citations")) if index_entry else []
                    )

                    citations = []

                    for cit in sem_cits + idx_cits:
                        if isinstance(cit, dict):
                            loc = cit.get("location") or None
                            ctx = (cit.get("context") or "").strip() or None
                            if loc or ctx:
                                citations.append({"location": loc, "context": ctx})

                    dict_entry = next(
                        (
                            e
                            for e in self.entries
                            if e.get("normalized") == norm or e.get("canonical") == norm
                        ),
                        None,
                    )
                    if not dict_entry:
                        definition = _get_field(word, "definition", "")
                        enhanced = _get_field(word, "enhanced_definition", "")
                    else:
                        senses = dict_entry.get("senses") or []
                        definition = "; ".join(
                            s.get("definition", "")
                            for s in senses
                            if s.get("definition")
                        )
                        cits_for_enh = (
                            "Possible uses: "
                            + ", ".join([f"`{cit}`" for cit in citations])
                            + "."
                        )
                        enhanced = (
                            f"{definition}." + cits_for_enh
                            if citations and len(citations) > 0
                            else ""
                        )

                    merged_cluster.append(
                        {
                            "word": (
                                f"{variant_used.upper()}"
                                if variant_used
                                and variant_used.lower()
                                != norm.lower()  # .lower() for word might be redundant but oh well
                                else f"{norm.upper()}"
                            ),
                            "normalized": norm.upper(),
                            "definition": definition,
                            "enhanced_definition": enhanced,
                            "fasttext": float(
                                _get_field(index_entry, "fasttext", "0.0")
                            ),
                            "semantic": float(
                                _get_field(index_entry, "semantic", "0.0")
                            ),
                            "score": float(_get_field(index_entry, "score", "0.0")),
                            "tier": (
                                index_entry.get("tier", "Untiered")
                                if index_entry
                                else "Untiered"
                            ),
                            "priority": (
                                index_entry.get("priority", 0) if index_entry else 0
                            ),
                            "levenshtein": (
                                index_entry.get("levenshtein", 99)
                                if index_entry
                                else 99
                            ),
                            "source": self._get_source_label(sem_entry, index_entry),
                            "citations": list(citations),
                        }
                    )

                stream_text(
                    f"\n[→] Beginning {GOLD}{root_token.upper()}{RESET} via analysis of cluster #{cluster_id + 1} (of {len(clusters)}).\n"
                )
                time.sleep(0.25)

                definitions = []
                for c in cluster:
                    # pull out the raw definition text, whether c is an Entry or a dict
                    if hasattr(c, "definition"):
                        def_text = c.definition
                    else:
                        def_text = c.get("definition", "")

                    # only keep non-empty definitions
                    if def_text:
                        definitions.append(def_text)
                cohesion_score = compute_cluster_cohesion(
                    definitions, self.sentence_model
                )

                semantic_hits = sum(
                    1 for c in merged_cluster if c["source"] in ("semantic", "both")
                )
                semantic_coverage = (
                    round(semantic_hits / len(merged_cluster), 3)
                    if merged_cluster
                    else 0.0
                )

                sem_count = sum(
                    1
                    for c in merged_cluster
                    if _get_field(c, "source", "") in ("semantic", "both")
                )

                idx_count = sum(
                    1
                    for c in merged_cluster
                    if _get_field(c, "source", "") in ("index", "both")
                )

                overlap_count = sum(
                    1 for c in cluster if _get_field(c, "source", "") == "both"
                )

                coh_floor = self._dynamic_coh_floor(
                    len(cluster), len(root_token) if root_token else None
                )
                # Treat each host word as its own pseudo-cluster of size 1
                singleton_results = []
                for single in cluster:
                    single_candidates = [single]
                    single_stats_summary = self._build_stats_summary(
                        root_token,
                        cluster_size=1,
                        cohesion_score=cohesion_score,
                        semantic_hits=semantic_hits,
                        semantic_coverage=semantic_coverage,
                        sem_count=sem_count,
                        idx_count=idx_count,
                        overlap_count=overlap_count,
                        residual_report=None,
                    )

                    single_result = self.evaluate_ngram(
                        ngram=root_token,
                        cluster=single_candidates,
                        cohesion_score=cohesion_score,
                        semantic_hits=semantic_hits,
                        semantic_coverage=semantic_coverage,
                        sem_count=sem_count,
                        idx_count=idx_count,
                        overlap_count=overlap_count,
                        stats_summary=single_stats_summary,
                        stream_callback=stream_callback,
                        style=style,
                    )

                    singleton_results.append(
                        {
                            "root": root_token,
                            "cluster_index": cluster_id,
                            "word": single.get("canonical") or single.get("word"),
                            "llm_result": single_result,
                        }
                    )

                    evaluated = single_result
                    evaluated["overlap_count"] = len(overlap)
                    evaluated["cluster_size"] = len(merged_cluster)

                    output.append(evaluated)

                    # 1) Save the log
                    if style == "debate":
                        save_log(
                            evaluated["Archivist"]
                            or evaluated["raw_output"].get("Archivist"),
                            label=root_token,
                            cluster_number=str(cluster_id + 1),
                            cluster_total=str(len(clusters)),
                            accepted=len(evaluated["Glossator"]) > 0,
                            style=style,
                        )
                    elif style == "solo":
                        txt = self._extract_evaluation(
                            evaluated["Glossator"].strip().lower()
                        )
                        print(f"\n\n[Debug] Just so you know: {txt}\n\n")
                        verdict = "accept" in txt.lower() if txt else False
                        save_log(
                            evaluated["Archivist"]
                            or evaluated["raw_output"].get("Archivist"),
                            label=root_token,
                            cluster_number=str(cluster_id + 1),
                            cluster_total=str(len(clusters)),
                            accepted=verdict,
                            style=style,
                        )

                    def _to_text(x, sep="\n\n========\n\n"):
                        if x is None:
                            return None
                        if isinstance(x, str):
                            return x
                        if isinstance(x, (list, tuple)):
                            # If it's a list of strings, join; otherwise JSON.
                            if all(isinstance(i, str) for i in x):
                                return sep.join(x)
                            return json.dumps(x, ensure_ascii=False)
                        if isinstance(x, dict):
                            return json.dumps(x, ensure_ascii=False)
                        # numbers / other simple types
                        return str(x)

                    cursor = self.new_definitions_db.cursor()

                    def _safe_float(value):
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return None

                    residual_report = evaluated.get("residual_report") or {}
                    residual_explained = _safe_float(residual_report.get("explained_ratio"))
                    residual_ratio = _safe_float(residual_report.get("residual_ratio"))

                    # Per-word residual diagnostics (used for both headline and prompt)
                    residual_word_details = (
                        residual_report.get("word_details") or []
                        if isinstance(residual_report, dict)
                        else []
                    )

                    # --- 1) Build a clearer residual_headline ---------------------------------
                    # Example target:
                    #   "Top residuals: GEBOFAL:1.00, ZEBOG:1.00"
                    residual_headline = None
                    if residual_word_details:
                        # sort by highest residual_ratio first
                        sorted_details = sorted(
                            residual_word_details,
                            key=lambda d: _safe_float(d.get("residual_ratio")) or 0.0,
                            reverse=True,
                        )
                        bits: list[str] = []
                        for d in sorted_details[:3]:
                            w = str(d.get("word") or d.get("normalized") or "").strip()
                            r = _safe_float(d.get("residual_ratio"))
                            if w and r is not None:
                                bits.append(f"{w}:{r:.2f}")
                        if bits:
                            residual_headline = "Top residuals: " + ", ".join(bits)

                    # --- 2) Build an explicit, LLM-friendly residual_focus_prompt --------------
                    residual_focus_prompt = None
                    if residual_word_details:
                        cov_vals = [
                            _safe_float(d.get("coverage_ratio"))
                            for d in residual_word_details
                            if d.get("coverage_ratio") is not None
                        ]
                        res_vals = [
                            _safe_float(d.get("residual_ratio"))
                            for d in residual_word_details
                            if d.get("residual_ratio") is not None
                        ]

                        # simple averages over words in this cluster
                        avg_cov = (
                            sum(v for v in cov_vals if v is not None) / len(cov_vals)
                            if cov_vals
                            else None
                        )
                        avg_res = (
                            sum(v for v in res_vals if v is not None) / len(res_vals)
                            if res_vals
                            else None
                        )

                        n_words = len(residual_word_details)

                        # Human / LLM readable explanation
                        pieces: list[str] = [
                            f"ROOT={root_token.upper()}",
                            f"N={n_words}",
                        ]
                        if avg_cov is not None:
                            pieces.append(f"avg_cov={avg_cov:.2f}")
                        if avg_res is not None:
                            pieces.append(f"avg_res={avg_res:.2f}")
                        if residual_explained is not None:
                            pieces.append(f"explained={residual_explained:.2f}")
                        if residual_ratio is not None:
                            pieces.append(f"residual={residual_ratio:.2f}")
                        if residual_headline:
                            pieces.append(residual_headline)

                        summary_line = " | ".join(pieces)

                        # Short instruction that explains what these metrics are and how to use them
                        metric_explainer = (
                            "Here, N is the number of candidate words in this cluster. "
                            "avg_cov is the average fraction of each word that is morphologically covered "
                            "by the proposed root (0.0–1.0). "
                            "avg_res is the average fraction of each word that remains morphologically "
                            "unexplained by the root (0.0–1.0). "
                            "'explained' is a cluster-level score for how much of the residual morphology "
                            "is accounted for by the root, and 'residual' is the remaining unexplained "
                            "portion (roughly 1.0 - explained). "
                        )

                        usage_hint = (
                            "Use these values when deciding whether to accept the root: higher avg_cov and "
                            "explained, and lower residual, mean the root morphologically explains more of "
                            "the cluster. If residual is close to 1.0 and avg_cov is near 0.0, treat the "
                            "listed words as unexplained residuals and be skeptical of the root."
                        )

                        residual_focus_prompt = (
                            summary_line + ". " + metric_explainer + usage_hint
                        )

                        if isinstance(residual_report, dict):
                            residual_report["focus_prompt"] = residual_focus_prompt

                    # 2) insert residual and llm records into sqlite
                    residual_rows = []
                    group_idx = cluster_id
                    group_size = len(residual_word_details) if residual_word_details else 0

                    # Reuse model and glossator fields we already computed for the cluster
                    model_text = _to_text(evaluated["Model"]) or _to_text(
                        evaluated["raw_output"].get("Model")
                    )
                    glossator_prompt_text = _to_text(
                        evaluated["Glossator_Prompt"]
                    ) or _to_text(evaluated["raw_output"].get("Glossator_Prompt"))
                    glossator_def_text = _to_text(evaluated["Glossator"]) or _to_text(
                        evaluated["raw_output"].get("Glossator")
                    )

                    # These may be populated later if you teach the LLM to emit them explicitly
                    derivational_validity = _safe_float(
                        evaluated.get("raw_output", {}).get("Derivational_Validity")
                    )
                    rebuttal_resilience = _safe_float(
                        evaluated.get("raw_output", {}).get("Rebuttal_Resilience")
                    )

                    if residual_word_details:
                        for detail in residual_word_details:
                            if not isinstance(detail, dict):
                                continue

                            parent_word = str(
                                detail.get("word") or detail.get("normalized") or ""
                            ).strip()
                            if not parent_word:
                                continue

                            uncovered = detail.get("uncovered") or []
                            if not isinstance(uncovered, list):
                                uncovered = [uncovered] if uncovered else []

                            # If we somehow have no uncovered spans, treat the whole parent word as residual
                            if not uncovered:
                                uncovered = [parent_word]

                            for frag in uncovered:
                                if isinstance(frag, dict):
                                    residual_text = str(frag.get("text") or "").strip()
                                else:
                                    residual_text = str(frag or "").strip()

                                if not residual_text:
                                    continue

                                if style == "debate":
                                    residual_rows.append(
                                        (
                                            self.run_id,  # run_id
                                            residual_text,  # residual
                                            parent_word,  # parent_word
                                            group_idx,  # group_idx
                                            group_size,  # group_size
                                            len(sem_norms),  # sem_count
                                            len(index_norms),  # idx_count
                                            evaluated["overlap_count"],  # overlap_count
                                            model_text,  # model
                                            _to_text(
                                                evaluated["raw_output"].get("Proposal")
                                            ),
                                            _to_text(
                                                evaluated["raw_output"].get("Critique")
                                            ),
                                            _to_text(
                                                evaluated["raw_output"].get(
                                                    "Initial_Defense"
                                                )
                                            ),
                                            _to_text(
                                                evaluated["raw_output"].get("Adjudicator")
                                            ),
                                            _to_text(
                                                evaluated["raw_output"].get("Skeptic")
                                            ),
                                            _to_text(
                                                evaluated["raw_output"].get("Linguist")
                                            ),
                                            glossator_prompt_text,  # glossator_prompt
                                            glossator_def_text,  # glossator_def
                                            derivational_validity,  # derivational_validity
                                            rebuttal_resilience,  # rebuttal_resilience
                                            cohesion_score,  # semantic_cohesion
                                            cohesion_score,  # cohesion (you can swap if you later distinguish)
                                            semantic_coverage,  # semantic_coverage
                                            residual_explained,  # residual_explained
                                            residual_ratio,  # residual_ratio
                                            residual_headline,  # residual_headline
                                            residual_focus_prompt,  # residual_focus_prompt
                                        )
                                    )
                                else:
                                    # SOLO style: no full tribunal rounds, just glossator-centric
                                    residual_rows.append(
                                        (
                                            self.run_id,  # run_id
                                            residual_text,  # residual
                                            parent_word,  # parent_word
                                            group_idx,  # group_idx
                                            group_size,  # group_size
                                            len(sem_norms),  # sem_count
                                            len(index_norms),  # idx_count
                                            evaluated["overlap_count"],  # overlap_count
                                            model_text,  # model
                                            glossator_prompt_text,  # glossator_prompt
                                            glossator_def_text,  # glossator_def
                                            derivational_validity,  # derivational_validity
                                            rebuttal_resilience,  # rebuttal_resilience
                                            cohesion_score,  # semantic_cohesion
                                            cohesion_score,  # cohesion
                                            semantic_coverage,  # semantic_coverage
                                            residual_explained,  # residual_explained
                                            residual_ratio,  # residual_ratio
                                            residual_headline,  # residual_headline
                                            residual_focus_prompt,  # residual_focus_prompt
                                        )
                                    )

                    if residual_rows:
                        if style == "debate":
                            cursor.executemany(
                                """
                                INSERT INTO root_residual_semantics (
                                    run_id,
                                    residual,
                                    parent_word,
                                    group_idx,
                                    group_size,
                                    sem_count,
                                    idx_count,
                                    overlap_count,
                                    model,
                                    proposal,
                                    critique,
                                    defense,
                                    adjudicator_rounds,
                                    skeptic_rounds,
                                    linguist_rounds,
                                    glossator_prompt,
                                    glossator_def,
                                    derivational_validity,
                                    rebuttal_resilience,
                                    semantic_cohesion,
                                    cohesion,
                                    semantic_coverage,
                                    residual_explained,
                                    residual_ratio,
                                    residual_headline,
                                    residual_focus_prompt
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                residual_rows,
                            )
                        else:
                            cursor.executemany(
                                """
                                INSERT INTO root_residual_semantics (
                                    run_id,
                                    residual,
                                    parent_word,
                                    group_idx,
                                    group_size,
                                    sem_count,
                                    idx_count,
                                    overlap_count,
                                    model,
                                    glossator_prompt,
                                    glossator_def,
                                    derivational_validity,
                                    rebuttal_resilience,
                                    semantic_cohesion,
                                    cohesion,
                                    semantic_coverage,
                                    residual_explained,
                                    residual_ratio,
                                    residual_headline,
                                    residual_focus_prompt
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                residual_rows,
                            )

                    cluster_rowid = cursor.lastrowid

                    if residual_word_details:
                        detail_rows = []
                        for detail in residual_word_details:
                            if not isinstance(detail, dict):
                                continue
                            residual_span = str(detail.get("word") or "").strip()
                            normalized_word = str(
                                detail.get("normalized") or residual_span
                            ).strip()
                            if not (residual_span or normalized_word):
                                continue
                            uncovered = detail.get("uncovered") or []
                            if not isinstance(uncovered, list):
                                uncovered = [str(uncovered)] if uncovered else []
                            low_conf = detail.get("low_conf_segments") or []
                            if not isinstance(low_conf, list):
                                low_conf = [str(low_conf)] if low_conf else []

                            detail_rows.append(
                                (
                                    cluster_rowid,
                                    residual_span or normalized_word,
                                    normalized_word,
                                    str(detail.get("definition") or ""),
                                    _safe_float(detail.get("coverage_ratio")),
                                    _safe_float(detail.get("residual_ratio")),
                                    _safe_float(detail.get("avg_confidence")),
                                    json.dumps(uncovered, ensure_ascii=False),
                                    json.dumps(low_conf, ensure_ascii=False),
                                )
                            )

                        if detail_rows:
                            cursor.executemany(
                                """
                                INSERT INTO residual_details (
                                    cluster_id,
                                    residual_span,
                                    normalized,
                                    definition,
                                    coverage_ratio,
                                    residual_ratio,
                                    avg_confidence,
                                    uncovered_json,
                                    low_conf_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                detail_rows,
                            )

                    # 3) Insert each of the merged defs into `raw_defs`
                    for entry in merged_cluster:
                        cursor.execute(
                            """
                            INSERT INTO raw_defs (
                                cluster_id,
                                source_word,
                                variant,
                                definition,
                                enhanced_def,
                                fasttext,
                                similarity,
                                tier
                            ) VALUES (?,?,?,?,?,?,?,?)
                            """,
                            (
                                cluster_rowid,
                                entry["normalized"].upper(),  # canonical appearance
                                (
                                    entry["word"].upper()
                                    if entry["normalized"].upper() == entry["word"].upper()
                                    else ""
                                ),  # variant if applicable
                                entry.get("definition", "") or "",
                                entry.get("enhanced_definition", "") or "",
                                float(entry.get("fasttext", 0.0)),
                                float(
                                    max(
                                        entry.get("semantic", 0.0),
                                        entry.get("score", 0.0),
                                    )
                                ),
                                entry.get("tier", "[no tiering given]"),
                            ),
                        )

                        raw_def_id = cursor.lastrowid

                        # Insert this entry's citations → citations(def_id, location, context)
                        rows = []
                        for c in entry.get("citations") or []:
                            loc = c.get("location") if isinstance(c, dict) else None
                            ctx = c.get("context") if isinstance(c, dict) else None
                            if loc or ctx:
                                rows.append((raw_def_id, loc, ctx))
                        if rows:
                            cursor.executemany(
                                "INSERT INTO citations (def_id, location, context) VALUES (?,?,?)",
                                rows,
                            )

                        # 4) commit once per cluster-member
                        self.new_definitions_db.commit()

                        cluster_rowid = cursor.lastrowid

            self._mark_root_complete(root_token)
            self._record_preanalysis_consumed(root_token)
            seen_words += 1

            # consolidate definitions into synth_defs
            # MAYBE IMPLEMENT LATER AS PART OF A CLEANUP PROCESS ONCE DB IS FINISHED??
            # consolidate_ngram_senses(
            #     self.new_definitions_db, root_token, sentence_model=self.sentence_model
            # )

            if max_words and seen_words >= max_words:
                break

        stream_text("🎊 Residual ngrams complete! \n")
        time.sleep(0.25)

        if self.ngram_db:
            self.ngram_db.close()
        if self.new_definitions_db:
            self.new_definitions_db.close()

    def save_results(self, new_data):
        # Load existing results if they exist
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Combine existing and new
        combined_data = {}
        for item in existing_data + new_data:
            combined_data[item["root"]] = item

        # Save all back to file
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(list(combined_data.values()), f, indent=2)
