from enochian_translation_team.utils import sqlite_bootstrap  # noqa: F401
import re
import json
import sqlite3
import random
import statistics as st
import time
import uuid, json, sys, platform, datetime
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from gensim.models import FastText
from sentence_transformers import SentenceTransformer
from enochian_translation_team.utils.logger import save_log
from enochian_translation_team.tools.debate_engine import debate_ngram, safe_output
from enochian_translation_team.tools.solo_analysis_engine import (
    solo_agent_ngram_analysis,
)
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.semantic_search import (
    find_semantically_similar_words,
    compute_cluster_cohesion,
    cluster_definitions,
)
from enochian_translation_team.utils.candidate_finder import MorphemeCandidateFinder
from enochian_translation_team.utils.def_reducer import consolidate_ngram_senses
from enochian_translation_team.utils.dictionary_loader import load_dictionary, Entry

GOLD = "\033[38;5;178m"
GREEN = "\033[38;5;120m"
YELLOW = "\033[38;5;190m"
BLUE = "\033[38;5;153m"
PINK = "\033[38;5;213m"
RESET = "\033[0m"


def stream_text(text: str, delay: float = 0.001):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            # if you really need to interrupt, break cleanly
            break


class RootExtractionCrew:
    def __init__(self, style, use_remote):
        paths = get_config_paths()
        self.dictionary_path = paths["dictionary"]
        self.model_path = paths["model_output"]
        self.subst_map_path = paths["substitution_map"]
        self.output_path = paths["root_word_insights"]
        self.ngram_path = paths["ngram_index"]
        self.processed_ngrams_path = paths[f"{style}_processed"]
        self.new_definitions_path = paths[style]

        # Load everything
        self.entries = self.load_entries()
        self.ngram_db = sqlite3.connect(paths["ngram_index"])
        self.new_definitions_db = sqlite3.connect(paths[style])
        self._prepare_db(self.new_definitions_db)
        self.subst_map = self.load_subst_map()
        self.fasttext = FastText.load(str(self.model_path))
        self.sentence_model_name = "paraphrase-MiniLM-L6-v2"
        self.sentence_model = SentenceTransformer(self.sentence_model_name)
        self.processed_ngrams = self.load_processed_ngrams()
        self.candidate_finder = MorphemeCandidateFinder(
            ngram_db_path=paths["ngram_index"],
            fasttext_model_path=self.model_path,
            dictionary_entries=self.entries,
        )

        self.run_id = self._begin_run(engine=style)
        self.use_remote = use_remote

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

        env_json = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "sentence_transformers": self.sentence_model_name,
            "fasttext_model_path": str(getattr(self, "model_path", "")),
        }

        with self.new_definitions_db:
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

    def load_processed_ngrams(self):
        try:
            with open(self.processed_ngrams_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
        except json.decoder.JSONDecodeError:
            return set()

    def save_processed_ngrams(self):
        try:
            with open(self.processed_ngrams_path, "r", encoding="utf-8") as f:
                existing = set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            existing = set()

        combined = existing | self.processed_ngrams

        ordered = sorted(combined, key=lambda s: (-len(s), s))

        with open(self.processed_ngrams_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2)

    def load_entries(self) -> List[Entry]:
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
                if ngram not in self.processed_ngrams and not re.search(r"\d", ngram):
                    ngrams.add(ngram)
        return ngrams

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
        query = """
            SELECT
            ngram,
            df_count
            FROM ngrams
            GROUP BY ngram
            HAVING df_count >= ?
            ORDER BY
            LENGTH(ngram) DESC,
            df_count     DESC
        """
        for ngram, df in cursor.execute(query, (min_freq,)):
            yield ngram, df

    def is_ngram_in_variants(self, ngram, canonical):
        cursor = self.ngram_db.cursor()
        cursor.execute(
            "SELECT 1 FROM ngrams WHERE ngram = ? AND canonical = ? LIMIT 1",
            (ngram, canonical),
        )
        return cursor.fetchone() is not None

    def get_matching_variant(self, ngram: str, canonical: str) -> str | None:
        """
        Look up in the ngrams table which canonical word
        corresponds to this ngram+canonical pair. Since we've
        removed the 'variant' column, we now return the 'canonical'.
        """
        cursor = self.ngram_db.cursor()
        cursor.execute(
            """
            SELECT canonical
              FROM ngrams
             WHERE ngram = ?
               AND canonical = ?
             LIMIT 1
            """,
            (ngram, canonical),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def dedupe_by_normalized(self, entries):
        seen = set()
        unique = []
        for entry in entries:
            key = entry.canonical
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return unique

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
    ) -> str:
        """
        Compact, human-readable summary passed to debate/solo engines.
        Keep it short; these strings are used inside prompts.
        """
        return (
            f"root={root.upper()} | "
            f"cluster_size={cluster_size} | "
            f"cohesion={cohesion_score:.3f} | "
            f"semantic_hits={semantic_hits} | "
            f"coverage={semantic_coverage:.3f} | "
            f"sem={sem_count} | idx={idx_count} | overlap_count={overlap_count}"
        )

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

    def prejudge_cluster(
        self, stats: dict, rules: dict | None = None, *, force_model: bool = False
    ) -> dict:
        """
        Pruning-only triage (more lenient):
        - Skips ONLY 'extra-bad' clusters.
        - Requires TWO independent red flags for most skips.
        - Small clusters (n≤3) are NEVER pruned on dp==0 alone.
        """
        rules = rules or {}
        n = int(stats.get("n", 0) or 0)
        coh = stats.get("coh", stats.get("cohesion", None))
        coh = float(coh) if coh is not None else None
        dp = int(stats.get("deriv_patterns", 0) or 0)
        inc = float(stats.get("incompatible", 0.0) or 0.0)
        scat = float(stats.get("scatter", 0.0) or 0.0)
        gram_len = stats.get("gram_len", None)
        gram_len = int(gram_len) if gram_len is not None else None

        # More tolerant thresholds (override via rules if you like)
        rej_scat = float(rules.get("reject_scatter", 0.45))  # was 0.30
        rej_inc = float(rules.get("reject_incompatible", 0.45))  # was 0.30
        mild_scat = float(rules.get("mild_scatter", 0.35))
        mild_inc = float(rules.get("mild_incompatible", 0.20))
        coh_margin = float(
            rules.get("coh_margin", 0.10)
        )  # how far below floor counts as "very low"

        reasons, tags = [], []
        floor = self._dynamic_coh_floor(n=n, gram_len=gram_len)

        # 1) EXTREME pathologies → immediate skip
        extreme = False
        if scat >= rej_scat:
            reasons.append(f"scatter≥{rej_scat:.2f}")
            tags.append("SCATTER_HIGH")
            extreme = True
        if inc >= rej_inc:
            reasons.append(f"incompatible≥{rej_inc:.2f}")
            tags.append("INCOMPAT_HIGH")
            extreme = True
        if extreme and not force_model:
            return {
                "action": "skip",
                "needs_llm": False,
                "reason": "; ".join(reasons),
                "tags": tags,
            }

        # 2) TWO-STRIKE rule for non-extreme cases
        strikes = 0
        details = []

        if coh is not None and coh < (floor - coh_margin):
            strikes += 1
            details.append(f"cohesion<{floor - coh_margin:.2f}")
            tags.append("LOW_COH_STRONG")

        if dp == 0 and n >= 5:  # dp=0 only matters with ample evidence
            strikes += 1
            details.append("dp=0 with n≥5")
            tags.append("NO_DERIV_AMPLY")

        if inc >= mild_inc:
            strikes += 1
            details.append(f"incompatible≥{mild_inc:.2f}")
            tags.append("INCOMPAT_MILD")

        if scat >= mild_scat:
            strikes += 1
            details.append(f"scatter≥{mild_scat:.2f}")
            tags.append("SCATTER_MILD")

        # Small clusters (n≤3): do NOT prune on dp=0; require BOTH low cohesion (strong) AND (inc or scat mild)
        if n <= 3:
            if (coh is not None and coh < (floor - coh_margin)) and (
                inc >= mild_inc or scat >= mild_scat
            ):
                reasons.append(
                    "small_n two-strike (very low coh + conflict/dispersion)"
                )
            # else: never skip—let the model decide
        else:
            if strikes >= 2:
                reasons.append("two-strike prune: " + ", ".join(details))

        if reasons and not force_model:
            return {
                "action": "skip",
                "needs_llm": False,
                "reason": "; ".join(reasons),
                "tags": tags,
            }

        # 3) Everything else → evaluate with LLM
        return {
            "action": "escalate",
            "needs_llm": True,
            "reason": "lenient triage → model should evaluate",
            "tags": tags or ["TRIAGE"],
        }

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

        # Summarize stats for agents
        stats_summary = (
            f"{root_callout}"
            f"The proposed root '{ngram.upper()}' has:\n"
            f"- Cohesion Score: {cohesion_score} (semantic similarity among definitions; from 0.0 to 1.0, with higher being better)\n"
            f"- Semantic Coverage: {round(semantic_coverage * 100, 1)}% ({semantic_hits}/{len(cluster)} words match semantically)\n"
            f"- Candidate Count: {len(cluster)}"
        )

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

        stats_summary = self._build_stats_summary(
            ngram,  # or ngram if iterating strings
            cluster_size=len(cluster),
            cohesion_score=cohesion_score,
            semantic_hits=semantic_hits,
            semantic_coverage=semantic_coverage,
            sem_count=sem_count,
            idx_count=idx_count,
            overlap_count=overlap_count,
        )

        the_result = {"raw_output": {}}
        if style == "debate":
            the_result = debate_ngram(
                root=ngram.upper(),
                candidates=trimmed_cluster,
                stats_summary=stats_summary,
                stream_callback=stream_callback,
                root_entry=None,  # can be None; debate engine will handle
                use_remote=self.use_remote,
                # blind_evaluation=True,
                # debate_lite=True,
            )
        else:
            the_result = solo_agent_ngram_analysis(
                root=ngram.upper(),
                candidates=trimmed_cluster,
                stats_summary=stats_summary,
                stream_callback=stream_callback,
                root_entry=None,
                use_remote=self.use_remote,
            )

        # Normalize expected keys (be defensive)
        normalized = {
            "Glossator": the_result.get("Glossator", "") or "",
            "Model": the_result.get("Model", ""),
            "Glossator_Prompt": the_result.get("Glossator_Prompt", ""),
            "Adjudicator": the_result.get("Adjudicator", ""),
            "Adjudicator_Prompt": the_result.get("Adjudicator_Prompt", ""),
            "Archivist": the_result.get("Archivist", "") or "",
            "raw_output": the_result,
        }
        return normalized

    def process_ngrams(
        self,
        max_words=None,
        stream_callback=None,
        single_ngram=None,
        style="debate",
        min_semantic_similarity: float = 0.60,
    ):
        def _get_field(item, field, default=""):
            # Unified accessor for Entry objects and dicts.
            if isinstance(item, dict):
                return item.get(field, default)
            else:
                return getattr(item, field, default)

        if single_ngram:
            ngrams = [(single_ngram, 9001)]  # Use a fake count
            max_words = 999
            if single_ngram in self.processed_ngrams:
                stream_text(
                    f"[Warning] The ngram {GOLD}{single_ngram.upper()}{RESET} has already been processed, so our work is already done.\n"
                )
                time.sleep(0.25)
        else:
            ngrams = sorted(
                self.stream_ngrams_from_sqlite(min_freq=2),
                key=lambda x: (len(x[0]), -x[1], x[0]),
            )

        output = []
        seen_words = 0

        if style == "debate":
            stream_text("🪄 Initializing semantic tribunal...\n\n")
        elif style == "solo":
            stream_text("⏰ Waking up the Enochiana scholar...\n\n")
        time.sleep(0.25)

        for count, ngram in enumerate(ngrams):
            if ngram[0] in self.processed_ngrams:
                continue

            semantic_candidates = find_semantically_similar_words(
                ft_model=self.fasttext,
                sentence_model=self.sentence_model,
                entries=self.entries,
                target_word=ngram[0],  # or just ngram if you’re iterating strings
                subst_map=self.subst_map,  # <-- required now
                min_similarity=min_semantic_similarity,
            )

            index_candidates = self.candidate_finder.get_all_ngram_candidates(ngram[0])

            if not semantic_candidates or len(semantic_candidates) < 2:
                self.processed_ngrams.add(ngram[0])
                self.save_processed_ngrams()
                continue
            else:
                stream_text(
                    f"[{GREEN}✓{RESET}] Beginning examination of root-word candidate {GOLD}{ngram[0].upper()}{RESET}.\n",
                    delay=0.005,
                )

            clusters_result = cluster_definitions(
                semantic_candidates, self.sentence_model
            )
            clusters = clusters_result["clusters"]
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
                    f" for '{GOLD}{ngram[0].upper()}{RESET}'; no sense in evaluating if it's the same set of words.\n\n"
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
                    f"Beginning the evaluation of {PINK}{len(clusters)}{RESET} cluster{'s' if len(clusters) > 1 else ''}.\n\n"
                    + follow_phrase
                )
            else:
                full_phrase = f"All possible definitions are too far apart to meaningfully cluster! Oh well...\n\n"
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

                stream_text(
                    f"Now examining cluster #{PINK}{cluster_id + 1}{RESET} of {PINK}{len(clusters)}{RESET} by finding the intersection between "
                    f"the {GREEN}{len(sem_norms)}{RESET} semantic candidates and the {BLUE}{len(index_norms)}{RESET} the words "
                    f"(and their extended variations) that contain '{GOLD}{ngram[0].upper()}{RESET}' as one of its components.\n"
                )
                time.sleep(0.25)

                sem_formatted_list = [
                    f"{GREEN}{norm.upper()}{RESET}" for norm in list(sem_norms)[:5]
                ]
                sem_formatted_list = (
                    f"Some words from the semantic candidates (up to 5): "
                    + ", ".join(sem_formatted_list)
                )
                sem_formatted_list += "..." if len(sem_norms) > 5 else ""
                stream_text(sem_formatted_list)
                time.sleep(0.25)
                print()

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

                stream_text(
                    "... Now, the important question: is there sufficient overlap between the two groups?\n"
                )
                time.sleep(0.25)
                stream_text(
                    f"{YELLOW}Yes! There is!{RESET}"
                    if len(overlap) >= 2
                    else f"{PINK}Either only one shared item or none whatsoever.{RESET} 😢"
                    + "\n"
                )
                time.sleep(0.25)

                if not overlap or len(overlap) < 2:
                    self.processed_ngrams.add(ngram[0])
                    self.save_processed_ngrams()
                    stream_text(
                        f"We have skipped cluster #{cluster_id + 1} for {GOLD}{ngram[0].upper()}{RESET} because not enough words (or their variants) in the cluster contained the ngram while also meaning similar things.\n\n"
                    )
                    self._insert_skip(
                        ngram[0].upper(),
                        reason_code="OVERLAP_LESS_THAN_TWO",
                        cluster_index=cluster_id + 1,
                    )
                    time.sleep(0.25)
                    continue
                else:
                    print("")

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

                    if (
                        not sem_entry
                        and index_entry
                        and norm.lower() != ngram[0].lower()
                    ):
                        continue

                    if not self.is_ngram_in_variants(ngram[0], norm):
                        continue

                    if isinstance(norm, str) and norm:
                        variant_used = self.get_matching_variant(ngram[0], norm)
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
                    #     c["context"].strip()
                    #     for c in sem_cits + idx_cits
                    #     if c["context"]
                    # ]
                    for cit in sem_cits + idx_cits:
                        if isinstance(cit, dict):
                            loc = cit.get("location") or None
                            ctx = (cit.get("context") or "").strip() or None
                            if loc or ctx:
                                citations.append({"location": loc, "context": ctx})

                    if sem_entry:
                        definition = sem_entry["definition"]
                        enhanced = sem_entry["enhanced_definition"]
                    else:
                        dict_entry = next(
                            (e for e in self.entries if e.normalized == norm), None
                        )
                        if not dict_entry:
                            definition = _get_field(word, "definition", "")
                            enhanced = _get_field(word, "enhanced_definition", "")
                        else:
                            definition = "; ".join(
                                s.definition for s in dict_entry.senses or []
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
                            "fasttext": float(_get_field(sem_entry, "fasttext", "0.0")),
                            "semantic": float(_get_field(sem_entry, "semantic", "0.0")),
                            "score": float(_get_field(sem_entry, "score", "0.0")),
                            "tier": (
                                sem_entry.get("tier", "Untiered")
                                if sem_entry
                                else "Untiered"
                            ),
                            "priority": (
                                sem_entry.get("priority", 0) if sem_entry else 0
                            ),
                            "levenshtein": (
                                sem_entry.get("levenshtein", 99) if sem_entry else 99
                            ),
                            "source": self._get_source_label(sem_entry, index_entry),
                            "citations": list(citations),
                        }
                    )

                if len(merged_cluster) < 2:
                    stream_text(
                        f"[⚠️] Merged cluster too small for {GOLD}{ngram[0].upper()}{RESET}. Skipping.\n"
                    )
                    time.sleep(0.25)
                    self._insert_skip(
                        ngram[0].upper(),
                        reason_code="MERGED_CLUSTER_LESS_THAN_TWO_MEMBERS",
                        cluster_index=cluster_id + 1,
                    )
                    continue

                stream_text(
                    f"\n[→] Beginning {GOLD}{ngram[0].upper()}{RESET} via analysis of cluster #{cluster_id + 1} (of {len(clusters)}).\n"
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
                clustering_meta_json = json.dumps(best_config)
                stats_summary = self._build_stats_summary(
                    root="asdf",
                    cluster_size=len(merged_cluster),
                    cohesion_score=cohesion_score,
                    semantic_hits=semantic_hits,
                    semantic_coverage=semantic_coverage,
                    sem_count=sem_count,
                    idx_count=idx_count,
                    overlap_count=overlap_count,
                )

                prefix_count = sum(
                    e["normalized"].startswith(ngram[0].lower()) for e in merged_cluster
                )
                suffix_count = sum(
                    e["normalized"].endswith(ngram[0].lower()) for e in merged_cluster
                )
                deriv_patterns = int(prefix_count > 0) + int(suffix_count > 0)

                sims = [c.get("cluster_similarity", 0.0) for c in merged_cluster]

                stats = {
                    "n": len(merged_cluster),  # cluster size
                    "coh": cohesion_score,  # 0..1
                    "semantic_hits": semantic_hits,  # raw count
                    "semantic_coverage": semantic_coverage,  # 0..1
                    "gram_len": len(ngram[0]),  # char length of the n-gram
                    # Optional signals derived from entries:
                    "scatter": st.pstdev(sims) if len(sims) > 1 else 0.0,
                    "deriv_patterns": deriv_patterns,
                }

                prevaluate = self.prejudge_cluster(stats)

                if prevaluate["needs_llm"]:
                    evaluated = self.evaluate_ngram(
                        ngram[0],
                        cluster=merged_cluster,
                        cohesion_score=cohesion_score,
                        semantic_hits=semantic_hits,
                        semantic_coverage=semantic_coverage,
                        stream_callback=stream_callback,
                        stats_summary=stats_summary,
                        sem_count=sem_count,
                        idx_count=idx_count,
                        overlap_count=overlap_count,
                        style=style,
                    )
                    evaluated["overlap_count"] = len(overlap)
                    evaluated["cluster_size"] = len(merged_cluster)

                    output.append(evaluated)

                    # 1) Save the log
                    if style == "debate":
                        save_log(
                            evaluated["Archivist"]
                            or evaluated["raw_output"].get("Archivist"),
                            label=ngram[0],
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
                            label=ngram[0],
                            cluster_number=str(cluster_id + 1),
                            cluster_total=str(len(clusters)),
                            accepted=verdict,
                            style=style,
                        )
                    #                self.save_results(output)

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

                    # 2) insert cluster and llm records into sqlite
                    xstr = lambda s: str(s).lower() or ""
                    if style == "debate":
                        cursor.execute(
                            """
                            INSERT INTO clusters (
                                run_id,
                                ngram,
                                cluster_index,
                                sem_count,
                                idx_count,
                                overlap_count,
                                prevaluation,
                                reason,
                                model,
                                proposal,
                                critique,
                                defense,
                                adjudicator_rounds,
                                skeptic_rounds,
                                linguist_rounds,
                                glossator_prompt,
                                glossator_def,
                                verdict,
                                semantic_cohesion,
                                semantic_coverage,
                                best_config
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                self.run_id,
                                ngram[0].upper(),
                                cluster_id,
                                len(sem_norms),
                                len(index_norms),
                                evaluated["overlap_count"],
                                _to_text(prevaluate["action"]),
                                _to_text(prevaluate["reason"]),
                                # proposal
                                _to_text(evaluated["Proposal"])
                                or _to_text(
                                    evaluated["raw_output"].get("Proposal")
                                ),
                                # critique
                                _to_text(evaluated["Critique"])
                                or _to_text(
                                    evaluated["raw_output"].get("Critique")
                                ),
                                # defense
                                _to_text(evaluated["Initial_Defense"])
                                or _to_text(
                                    evaluated["raw_output"].get("Initial_Defense")
                                ),
                                # adjudicator_rounds
                                _to_text(evaluated["Adjudicator"])
                                or _to_text(
                                    evaluated["raw_output"].get("Adjudicator")
                                ),
                                # skeptic_rounds
                                _to_text(evaluated["Skeptic"])
                                or _to_text(
                                    evaluated["raw_output"].get("Skeptic")
                                ),
                                # linguist_rounds
                                _to_text(evaluated["Linguist"])
                                or _to_text(
                                    evaluated["raw_output"].get("Linguist")
                                ),
                                # gloss model
                                _to_text(evaluated["Model"])
                                or _to_text(
                                    evaluated["raw_output"].get("Model")
                                ),
                                # gloss prompt
                                _to_text(evaluated["Glossator_Prompt"])
                                or _to_text(
                                    evaluated["raw_output"].get("Glossator_Prompt")
                                ),
                                # glossator def
                                _to_text(evaluated["Glossator"])
                                or _to_text(evaluated["raw_output"].get("Glossator")),
                                # verdict
                                str(
                                    "accepted" in xstr(_to_text(evaluated["Glossator"]))
                                    or "accepted"
                                    in xstr(
                                        _to_text(
                                            evaluated["raw_output"].get("Glossator")
                                        )
                                    )
                                ),
                                cohesion_score,
                                semantic_coverage,
                                clustering_meta_json,
                            ),
                        )
                    elif style == "solo":
                        cursor.execute(
                            """
                            INSERT INTO clusters (
                                run_id,
                                ngram,
                                cluster_index,
                                sem_count,
                                idx_count,
                                overlap_count,
                                prevaluation,
                                reason,
                                model,
                                glossator_prompt,
                                glossator_def,
                                verdict,
                                cohesion,
                                semantic_coverage,
                                best_config
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                self.run_id,
                                ngram[0].upper(),
                                cluster_id,
                                len(sem_norms),
                                len(index_norms),
                                evaluated["overlap_count"],
                                _to_text(prevaluate["action"]),
                                _to_text(prevaluate["reason"]),
                                _to_text(evaluated["Glossator_Prompt"])
                                or _to_text(
                                    evaluated["raw_output"].get("Glossator_Prompt")
                                ),
                                _to_text(evaluated["Model"])
                                or _to_text(
                                    evaluated["raw_output"].get("Model")
                                ),
                                _to_text(evaluated["Glossator"])
                                or _to_text(evaluated["raw_output"].get("Glossator")),
                                cohesion_score,
                                semantic_coverage,
                                clustering_meta_json,
                            ),
                        )

                    cluster_rowid = cursor.lastrowid

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
                                    if entry["normalized"].upper()
                                    == entry["word"].upper()
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
                else:
                    # 1) notify about the skip
                    stream_text(
                        f"❗\t{PINK}A sudden plot twist!{RESET} 😮\n"
                        "The pre-screening denies the cluster from even being evaluated!\n"
                    )
                    stream_text(
                        f"These are the {BLUE}reasons{RESET}...\n"
                        if len(prevaluate["reason"].split("; ")) > 1
                        else f"This is the {BLUE}reason{RESET}...\n"
                    )
                    stream_text(prevaluate["reason"] + "\n\n")

                    # 2) insert record of skip into sqlite
                    cursor = self.new_definitions_db.cursor()

                    cursor.execute(
                        """
                        INSERT INTO clusters (
                            run_id,
                            ngram,
                            cluster_index,
                            sem_count,
                            idx_count,
                            overlap_count,
                            action,
                            reason,
                            cohesion,
                            semantic_coverage,
                            best_config
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.run_id,
                            ngram[0].upper(),
                            cluster_id,
                            sem_count,
                            idx_count,
                            overlap_count,
                            prevaluate["action"],
                            prevaluate["reason"],
                            cohesion_score,
                            semantic_coverage,
                            clustering_meta_json,
                        ),
                    )

                    cluster_rowid = cursor.lastrowid

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
                                    if entry["normalized"].upper()
                                    == entry["word"].upper()
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

            self.processed_ngrams.add(ngram[0])
            self.save_processed_ngrams()
            seen_words += 1

            # consolidate definitions into synth_defs
            # MAYBE IMPLEMENT LATER AS PART OF A CLEANUP PROCESS ONCE DB IS FINISHED??
            # consolidate_ngram_senses(
            #     self.new_definitions_db, ngram[0], sentence_model=self.sentence_model
            # )

            if max_words and seen_words >= max_words:
                break

        stream_text("🎊 Clusters complete! \n")
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
