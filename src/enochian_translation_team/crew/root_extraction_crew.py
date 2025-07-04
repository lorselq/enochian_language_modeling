import re
import sys
import json
import sqlite3
import random
import time
from typing import List
from collections import defaultdict
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
from enochian_translation_team.utils.dictionary_loader import load_dictionary, Entry


def stream_text(text: str, delay: float = 0.0125):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            # if you really need to interrupt, break cleanly
            break


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
        self.new_definitions_db = sqlite3.connect(paths["new_definitions"])
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

    def _get_source_label(self, sem, idx):
        if sem and idx:
            return "both"
        elif sem:
            return "semantic"
        elif idx:
            return "index"
        return "unknown"

    def evaluate_ngram(
        self,
        ngram: str,
        cluster,
        cohesion_score,
        semantic_hits,
        semantic_coverage,
        stream_callback=None,
    ):
        def _get_field(item, field, default=""):
            """
            Unified accessor for Entry objects and dicts.
            """
            if isinstance(item, dict):
                return item.get(field, default)
            else:
                return getattr(item, field, default)

        # Build the full trimmed cluster and find anchor if it exists
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
        for role in [
            "Linguist",
            "Skeptic",
            "Defense",
            "Rebuttal",
            "Adjudicator",
            "Glossator",
            "Glossator_Prompt",
            "Glossator_Model",
            "Archivist",
            "summary",
        ]:
            if role in raw:
                result[role] = raw[role]

        return result

    def run_with_streaming(
        self, max_words=None, stream_callback=None, single_ngram=None
    ):
        GOLD = "\033[38;5;178m"
        GREEN = "\033[38;5;120m"
        YELLOW = "\033[38;5;190m"
        BLUE = "\033[38;5;153m"
        PINK = "\033[38;5;213m"
        RESET = "\033[0m"

        def _get_field(item, field, default=""):
            # Unified accessor for Entry objects and dicts.
            if isinstance(item, dict):
                return item.get(field, default)
            else:
                return getattr(item, field, default)

        build_and_save_ngram_index()
        if single_ngram:
            ngrams = [(single_ngram, 9001)]  # Use a fake count
            max_words = 999
            if single_ngram in self.processed_ngrams:
                stream_text(
                    f"[Warning] The ngram {GOLD}{single_ngram.upper()}{RESET} has already been processed, so our work is already done.\n"
                )
                time.sleep(1)
        else:
            ngrams = sorted(
                self.stream_ngrams_from_sqlite(min_freq=2),
                key=lambda x: (-len(x[0]), -x[1], x[0]),
            )

        output = []
        seen_words = 0

        stream_text("🪄 Initializing semantic tribunal...\n\n")
        time.sleep(0.5)

        for count, ngram in enumerate(ngrams):
            if ngram[0] in self.processed_ngrams:
                continue

            semantic_candidates = find_semantically_similar_words(
                ft_model=self.fasttext,
                sentence_model=self.sentence_model,
                entries=self.entries,
                target_word=ngram[0],
                subst_map=self.subst_map,
            )

            index_candidates = self.candidate_finder.get_all_ngram_candidates(ngram[0])

            if not semantic_candidates or len(semantic_candidates) < 2:
                self.processed_ngrams.add(ngram[0])
                self.save_processed_ngrams()
                continue
            else:
                stream_text(
                    f"[✓] Beginning examination of root-word candidate {GOLD}{ngram[0].upper()}{RESET}.\n",
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
                    f"for '{GOLD}{ngram[0].upper()}{RESET}'; no sense in evaluating if it's the same set of words."
                )
                stream_text(deduped_clusters_text)
            clusters = unique
            best_config = clusters_result["config"]
            if len(clusters) == 0:
                follow_phrase = ""
            elif len(clusters) < 10:
                follow_phrase = "Should be fairly quick, all considered!\n"
            else:
                follow_phrase = "This could take a while if we're being honest...\n"
            stream_text(
                f"Beginning the evaluation of {PINK}{len(clusters)}{RESET} cluster{'s' if len(clusters) > 1 else ''}. "
                if len(clusters) > 0
                else f"All possible definitions are too far apart to meaningfully cluster! Oh well..."
                + follow_phrase
                + "\n"
            )
            time.sleep(1)

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
                time.sleep(0.7)

                sem_formatted_list = [
                    f"{GREEN}{norm.upper()}{RESET}" for norm in list(sem_norms)[:5]
                ]
                sem_formatted_list = (
                    f"Some words from the semantic candidates (up to 5): "
                    + ", ".join(sem_formatted_list)
                )
                sem_formatted_list += "..." if len(sem_norms) > 5 else ""
                stream_text(sem_formatted_list)
                time.sleep(0.3)
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
                time.sleep(1.2)
                print()

                stream_text(
                    "... Now, the important question: is there sufficient overlap between the two groups?\n"
                )
                time.sleep(1.5)
                stream_text(
                    f"{YELLOW}Yes! There is!{RESET}"
                    if len(overlap) >= 2
                    else f"{PINK}Either only one shared item or none whatsoever.{RESET} 😢"
                    + "\n"
                )
                time.sleep(1)

                if not overlap or len(overlap) < 2:
                    self.processed_ngrams.add(ngram[0])
                    self.save_processed_ngrams()
                    stream_text(
                        f"We have skipped cluster #{cluster_id + 1} for {GOLD}{ngram[0].upper()}{RESET} because not enough words (or their variants) in the cluster contained the ngram while also meaning similar things.\n\n"
                    )
                    time.sleep(2)
                    continue
                else:
                    print("")

                merged_cluster = []

                merged_items = list(cluster) + index_candidates

                for word in merged_items:
                    norm = _get_field(word, "normalized", "").lower()
                    if not norm:
                        continue

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

                    citations = [
                        c["context"].strip()
                        for c in sem_cits + idx_cits
                        if c["context"]
                    ]

                    if sem_entry:
                        definition = sem_entry["definition"]
                        enhanced = sem_entry["enhanced_definition"]
                    else:
                        dict_entry = next(
                            e for e in self.entries if e.normalized == norm
                        )
                        definition = "; ".join(
                            s.definition for s in dict_entry.senses or []
                        )
                        cits_for_enh = (
                            "Possible uses: "
                            + ", ".join([f"`{c}`" for c in citations])
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
                                f"{norm.upper()}"
                                if variant_used and variant_used != word
                                else f"{norm.upper()}"
                            ),
                            "normalized": norm,
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
                    time.sleep(0.7)
                    continue

                stream_text(
                    f"\n[→] Beginning {GOLD}{ngram[0].upper()}{RESET} via analysis of cluster #{cluster_id + 1} (of {len(clusters)}).\n"
                )
                time.sleep(0.5)

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

                evaluated = self.evaluate_ngram(
                    ngram[0],
                    cluster=merged_cluster,
                    cohesion_score=cohesion_score,
                    semantic_hits=semantic_hits,
                    semantic_coverage=semantic_coverage,
                    stream_callback=stream_callback,
                )
                evaluated["overlap_count"] = len(overlap)
                evaluated["cluster_size"] = len(merged_cluster)

                output.append(evaluated)

                save_log(
                    evaluated["Archivist"] or evaluated["raw_output"].get("Archivist"),
                    label=ngram[0],
                    cluster_number=str(cluster_id + 1),
                    cluster_total=str(len(clusters)),
                    accepted=len(evaluated["Glossator"]) > 0,
                )
                #                self.save_results(output)

                if "Glossator" in evaluated:
                    cursor = self.new_definitions_db.cursor()

                    # cluster-level fields
                    cluster_idx = cluster_id + 1
                    cluster_count = len(clusters)
                    prompt = evaluated["Glossator_Prompt"]
                    model = evaluated["Glossator_Model"]
                    sem_count = len(sem_norms)
                    idx_count = len(index_norms)
                    overlap_count = len(overlap)
                    definition = evaluated["Glossator"].strip()
                    clustering_meta = best_config

                    # 2) Insert into `clusters`
                    cursor.execute(
                        """
                    INSERT INTO clusters (
                        ngram,
                        cluster_index,
                        count_clusters,
                        sem_count,
                        idx_count,
                        overlap_count,
                        clustering_meta,
                        model_used,
                        prompt_given,
                        glossator_def
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                        (
                            ngram[0].upper(),  # or ngram.lower(), up to you
                            cluster_idx,  # human‐friendly 1-based index
                            cluster_count,
                            sem_count,
                            idx_count,
                            overlap_count,
                            clustering_meta,
                            model,
                            prompt,
                            definition,
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
                            definition,
                            enhanced_def,
                            citations,
                            fasttext,
                            similarity,
                            tier
                        ) VALUES (?,?,?,?,?,?,?,?)
                        """,
                            (
                                cluster_rowid,
                                entry["normalized"],  # the word form
                                entry["definition"],
                                entry["enhanced_definition"],
                                json.dumps(entry.get("citations", [])),
                                entry.get("fasttext", 0.0),
                                entry.get("semantic", 0.0),
                                entry.get("tier", ""),
                            ),
                        )

                    # 4) commit once per cluster
                    self.new_definitions_db.commit()
            self.processed_ngrams.add(ngram[0])
            self.save_processed_ngrams()
            seen_words += 1

            if max_words and seen_words >= max_words:
                break

        stream_text("🎊 Clusters complete! \n")
        time.sleep(2)

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
