import re
import json
import sqlite3
from itertools import chain
from tqdm import tqdm
from collections import Counter, defaultdict
from gensim.models import FastText
from sentence_transformers import SentenceTransformer
from enochian_translation_team.utils.logger import save_log
from enochian_translation_team.tools.debate_engine import debate_ngram, safe_output
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.semantic_search import (
    find_semantically_similar_words,
    compute_cluster_cohesion,
    cluster_definitions,
)
from enochian_translation_team.utils.candidate_finder import MorphemeCandidateFinder
from enochian_translation_team.utils.build_ngram_index import build_and_save_ngram_index


class RootExtractionCrew:
    def __init__(self):
        paths = get_config_paths()
        self.dictionary_path = paths["dictionary"]
        self.model_path = paths["model_output"]
        self.subst_map_path = paths["substitution_map"]
        self.output_path = paths["root_word_insights"]
        self.ngram_path = paths["ngram_index"]
        self.processed_ngrams_path = paths["processed_ngrams"]
        self.new_definitions_path = paths["new_definitions"]

        # Load everything
        self.entries = self.load_entries()
        self.ngram_db = sqlite3.connect(paths["ngram_index"])
        self.subst_map = self.load_subst_map()
        self.fasttext = FastText.load(str(self.model_path))
        self.sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.candidate_finder = MorphemeCandidateFinder(
            ngram_db_path=paths["ngram_index"],
            fasttext_model_path=self.model_path,
            dictionary_entries=self.entries,
        )

        # Keep track of processed roots
        self.processed_ngrams = self.load_processed_ngrams()

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

        with open(self.processed_ngrams_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(combined)), f, indent=2)

    def load_entries(self):
        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        entries = []
        for e in raw_entries:
            if (
                e.get("normalized")
                and e.get("definition")
                and e.get("canon_word") is True
            ):
                entries.append(e)
        return entries

    def load_subst_map(self):
        with open(self.subst_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        subst_map = {}
        for k, v in raw.items():
            subs = [
                alt["value"]
                for alt in v["alternates"]
                if alt["direction"] in ["to", "both"]
            ]
            subst_map[k] = subs if subs else [k]
        return subst_map

    def extract_ngrams(self, word, min_n=1, max_n=4):
        ngrams = set()
        word = word.lower()
        for n in range(min_n, max_n + 1):
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n]
                if ngram not in self.processed_ngrams and not re.search(r"\d", ngram):
                    ngrams.add(ngram)
        return ngrams

    def stream_ngrams_from_sqlite(self, min_freq=2):
        cursor = self.ngram_db.cursor()
        query = """
            SELECT ngram, COUNT(*) as frequency
            FROM ngrams
            GROUP BY ngram
            HAVING frequency >= ?
            ORDER BY frequency DESC
        """
        for row in cursor.execute(query, (min_freq,)):
            yield row[0], row[1]

    def is_ngram_in_variants(self, ngram, canonical):
        cursor = self.ngram_db.cursor()
        cursor.execute(
            "SELECT 1 FROM ngrams WHERE ngram = ? AND canonical = ? LIMIT 1",
            (ngram, canonical),
        )
        return cursor.fetchone() is not None

    def get_matching_variant(self, ngram, canonical):
        cursor = self.ngram_db.cursor()
        cursor.execute(
            "SELECT variant FROM ngrams WHERE ngram = ? AND canonical = ? LIMIT 1",
            (ngram, canonical),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def dedupe_by_normalized(self, entries):
        seen = set()
        unique = []
        for entry in entries:
            key = entry["normalized"]
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return unique

    def _get_source_label(self, sem, idx):
        if sem and idx:
            return "both"
        elif sem:
            return "semantic"
        elif idx:
            return "index"
        return "unknown"

    def evaluate_ngram(self, ngram, cluster, stream_callback=None, cluster_id=None):
        print(
            f"[Debug] Evaluating cluster with length of {len(cluster)} for {ngram}..."
        )

        cluster_note = f"[Cluster {cluster_id}]" if cluster_id is not None else ""
        definitions = [
            c["definition"]
            for c in tqdm(
                cluster, desc=f"{cluster_note} Evaluating cluster for {ngram.upper()}"
            )
            if c["definition"]
        ]
        cohesion_score = compute_cluster_cohesion(definitions, self.sentence_model)
        semantic_hits = sum(1 for c in cluster if c["source"] in ("semantic", "both"))
        semantic_coverage = round(semantic_hits / len(cluster), 3) if cluster else 0.0

        # Build the full trimmed cluster and find anchor if it exists
        trimmed_cluster = []
        anchor_entry = None
        ngram_lower = ngram.lower()
        root_entry = None

        for c in cluster:
            if not c.get("definition"):
                continue

            citations = c.get("citations", [])
            if citations and isinstance(citations[0], dict):
                contexts = [
                    cite.get("context", "").strip()
                    for cite in citations
                    if cite.get("context")
                ]
            else:
                contexts = citations

            definition = (
                f"{c.get('definition', '')} [{' / '.join(contexts[:2])}]"
                if contexts
                else c.get("definition", "")
            )

            entry = {
                "word": c.get("word", "").upper(),
                "definition": definition,
                "normalized": c.get("normalized", "").lower(),
                "fasttext": c.get("fasttext", 0.0),
                "semantic": c.get("semantic", 0.0),
                "score": c.get("score", 0.0),
                "tier": c.get("tier", "Untiered"),
                "priority": c.get("priority", 0),
                "citations": c.get("citations", []),
                "source": c.get("source", "unknown"),
            }

            # Check if this is the literal ngram word
            if (
                entry.get("normalized", "")
                == ngram_lower
                # or entry.get("word", "").lower() == ngram_lower    # temporarily suspended from action
            ):
                root_entry = c.copy()  # <- This preserves all fields
                anchor_entry = {
                    "word": f"{c.get('word', '')} ⭐️ (root form)",
                    "definition": definition,
                    "normalized": c.get("normalized", "").lower(),
                }
            else:
                trimmed_cluster.append(entry)

        # If anchor exists, put it at the top
        if anchor_entry:
            trimmed_cluster.insert(0, anchor_entry)

        # Check for exact-match anchor word in the cluster
        if root_entry and root_entry.get("definition"):
            root_callout = (
                f"📌 Note: The proposed root '{ngram.upper()}' is itself a defined word: "
                f"**{root_entry.get('definition')}**\n"
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
            tier = c.get("tier", "Untiered")
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
                key = entry.get("normalized", "").lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(entry)
            tiered_groups[tier] = unique

        # Let the agents have their nerd war
        debate_result = debate_ngram(
            root=ngram,
            candidates=trimmed_cluster,
            stats_summary=stats_summary,
            stream_callback=stream_callback,
            root_entry=root_entry,
        )

        if isinstance(debate_result, dict):
            raw = debate_result.get("raw_output", {})
        elif hasattr(debate_result, "raw_output"):
            raw = safe_output(debate_result)
        else:
            raw = {}

        result = {
            "root": ngram,
            "cohesion": str(cohesion_score),
            "semantic_coverage": str(semantic_coverage),
            "tiered_candidates": dict(tiered_groups),
            "summary": raw.get("summary", "[No summary]"),
        }

        # Include role-specific responses
        for role in ["Linguist", "Skeptic", "Adjudicator", "Glossator", "Archivist"]:
            if role in raw:
                result[role] = raw[role]

        return result


    def run_with_streaming(self, max_words=None, stream_callback=None, single_ngram=None):
        build_and_save_ngram_index()
        if single_ngram:
            ngram_generator = [(single_ngram, 418)]  # Use a fake count
            max_words = 999
            if single_ngram in self.processed_ngrams:
                print(
                    f"[Warning] The ngram {single_ngram.upper()} has already been processed, so our work is already done."
                )
        else:
            ngram_generator = self.stream_ngrams_from_sqlite(min_freq=2)

        output = []
        seen_words = 0

        print("🪄 Initializing semantic tribunal...\n")
        for ngram, count in ngram_generator:
            if ngram in self.processed_ngrams:
                print(
                    f"[Skipping] Root candidate '{ngram}' has already been processed; moving on..."
                )
                continue

            print(
                f"[✓] Processing root candidate: '{ngram.upper()}' (appears in {count} forms)"
            )

            semantic_candidates = find_semantically_similar_words(
                ft_model=self.fasttext,
                sentence_model=self.sentence_model,
                entries=self.entries,
                target_word=ngram,
                subst_map=self.subst_map,
            )

            index_candidates = self.candidate_finder.get_candidates(
                ngram=ngram,
                top_n=20,
                similarity_threshold=0.35,
                include_context=True,
            )

            if not semantic_candidates or len(semantic_candidates) < 2:
                print(f"[⚠️] Too few semantic candidates for '{ngram}'. Skipping.")
                continue

            clusters = cluster_definitions(semantic_candidates, self.sentence_model)

            for cluster_id, cluster in enumerate(clusters):
                sem_norms = {c["normalized"] for c in cluster}
                index_norms = {c["normalized"] for c in index_candidates}
                overlap = sem_norms & index_norms

                if not overlap or len(overlap) < 2:
                    print(
                        f"[Skipped] Potential cluster for '{ngram}' did not meet overlap threshold."
                    )
                    continue

                merged_cluster = []

                for word in {c["normalized"] for c in cluster + index_candidates}:
                    sem_entry = next((c for c in cluster if c["normalized"] == word), None)
                    index_entry = next(
                        (c for c in index_candidates if c["normalized"] == word), None
                    )

                    if not sem_entry and index_entry:
                        continue

                    if not self.is_ngram_in_variants(ngram, word):
                        continue

                    entry_word = sem_entry.get("word") if sem_entry else word
                    variant_used = self.get_matching_variant(ngram, word)

                    merged_cluster.append(
                        {
                            "word": (
                                f"{entry_word.upper()} (via: {variant_used})"
                                if variant_used and variant_used != word
                                else entry_word.upper()
                            ),
                            "normalized": word,
                            "definition": (
                                sem_entry.get("definition")
                                if sem_entry
                                else index_entry.get(
                                    "definition", "[Error] no definition provided semantic entry"
                                )
                                if index_entry
                                else "[Error] no definition provided for semantic entry or index entry"
                            ),
                            "fasttext": (
                                float(sem_entry.get("fasttext", 0.0)) if sem_entry else 0.0
                            ),
                            "semantic": (
                                float(sem_entry.get("semantic", 0.0)) if sem_entry else 0.0
                            ),
                            "score": (
                                float(sem_entry.get("score", 0.0)) if sem_entry else 0.0
                            ),
                            "tier": (
                                sem_entry.get("tier", "Untiered")
                                if sem_entry
                                else "Untiered"
                            ),
                            "priority": sem_entry.get("priority", 0) if sem_entry else 0,
                            "levenshtein": (
                                sem_entry.get("levenshtein", 99) if sem_entry else 99
                            ),
                            "source": self._get_source_label(sem_entry, index_entry),
                            "citations": list(
                                {
                                    c.get("context", "").strip()
                                    for c in (
                                        sem_entry.get("citations", []) if sem_entry else []
                                    )
                                    + (
                                        index_entry.get("citations", [])
                                        if index_entry
                                        else []
                                    )
                                    if c.get("context")
                                }
                            ),
                        }
                    )

                if len(merged_cluster) < 2:
                    print(f"[⚠️] Merged cluster too small for '{ngram}'. Skipping.")
                    continue

                print(
                    f"[→] Evaluating cluster {cluster_id + 1}/{len(clusters)} for '{ngram.upper()}'..."
                )

                evaluated = self.evaluate_ngram(
                    ngram,
                    merged_cluster,
                    stream_callback=stream_callback,
                    cluster_id=cluster_id,
                )
                evaluated["overlap_count"] = len(overlap)
                evaluated["cluster_size"] = len(merged_cluster)

                output.append(evaluated)

                self.save_results(output)
                self.processed_ngrams.add(ngram)
                self.save_processed_ngrams()

                if "Archivist" in evaluated:
                    save_log(
                        [("Archivist", evaluated["Archivist"])],
                        label=f"{ngram}_cluster{cluster_id}",
                    )

                if "Glossator" in evaluated:
                    with open(self.new_definitions_path, "a", encoding="utf-8") as f:
                        source_words = ", ".join(c["word"] for c in merged_cluster)
                        f.write(
                            f"{evaluated['Glossator'].strip()} NOTE: this was inferred from the following words: {source_words}.\n\n"
                        )

                if stream_callback:
                    for role in ["Linguist", "Skeptic", "Adjudicator", "Glossator"]:
                        if role in evaluated:
                            stream_callback(role, evaluated[role])

            seen_words += 1
            if max_words and seen_words >= max_words:
                break

        if self.ngram_db:
            self.ngram_db.close()

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
