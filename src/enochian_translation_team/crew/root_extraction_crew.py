import json
from collections import Counter, defaultdict
from gensim.models import FastText
from sentence_transformers import SentenceTransformer
from enochian_translation_team.utils.logger import save_log
from enochian_translation_team.tools.debate_engine import debate_ngram, safe_output
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.semantic_search import (
    find_semantically_similar_words,
    compute_cluster_cohesion,
)
from enochian_translation_team.utils.candidate_finder import MorphemeCandidateFinder
from enochian_translation_team.utils.build_ngram_index import (
    build_and_save_ngram_index,
    load_ngrams,
)


class RootExtractionCrew:
    def __init__(self):
        paths = get_config_paths()
        self.dictionary_path = paths["dictionary"]
        self.model_path = paths["model_output"]
        self.subst_map_path = paths["substitution_map"]
        self.output_path = paths["root_word_insights"]
        self.ngram_path = paths["ngram_index"]
        self.processed_ngrams_path = paths["processed_ngrams"]

        # Reprocess ngrams
        build_and_save_ngram_index()

        # Load everything
        self.entries = self.load_entries()
        self.subst_map = self.load_subst_map()
        self.ngrams = load_ngrams(self.ngram_path)
        self.fasttext = FastText.load(str(self.model_path))
        self.sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.candidate_finder = MorphemeCandidateFinder(
            ngram_path=self.ngram_path,
            fasttext_model_path=self.model_path,
            dictionary_entries=self.entries,
        )

        # Keep track of processed roots
        self.processed_ngrams = set()

    def load_processed_ngrams(self):
        try:
            with open(self.processed_ngrams_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except FileNotFoundError:
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
            return [
                e
                for e in json.load(f)
                if e.get("normalized") and e.get("definition") and e.get("canon_word")
            ]

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
                if ngram not in self.processed_ngrams:
                    ngrams.add(ngram)
        return ngrams

    def get_ngram_frequencies(self, min_n=1, max_n=4):
        counter = Counter()
        for entry in self.entries:
            word = entry["normalized"].lower()
            for n in range(min_n, max_n + 1):
                for i in range(len(word) - n + 1):
                    ngram = word[i : i + n]
                    counter[ngram] += 1
        return counter

    def _get_source_label(self, sem, idx):
        if sem and idx:
            return "both"
        elif sem:
            return "semantic"
        elif idx:
            return "index"
        return "unknown"

    def evaluate_ngram(self, ngram, cluster, stream_callback=None):
        definitions = [c["definition"] for c in cluster if c["definition"]]
        cohesion_score = compute_cluster_cohesion(definitions, self.sent_model)
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
                "word": c.get("word", ""),
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
                entry.get("normalized", "") == ngram_lower
                or entry.get("word", "").lower() == ngram_lower
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
                f"📌 Note: The proposed root '{ngram}' is itself a defined word: "
                f"**{root_entry.get('definition')}**\n"
                "This may indicate its role as a base morpheme from which related forms are derived.\n"
            )
        else:
            root_callout = ""

        # Summarize stats for agents
        stats_summary = (
            f"{root_callout}"
            f"The proposed root '{ngram}' has:\n"
            f"- Cohesion Score: {cohesion_score} (semantic similarity among definitions)\n"
            f"- Semantic Coverage: {semantic_coverage} ({semantic_hits}/{len(cluster)} words match semantically)\n"
            f"- Candidate Count: {len(cluster)}"
        )

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

        tiered_groups = defaultdict(list)
        for c in cluster:
            tier = c.get("tier", "Untiered")
            tiered_groups[tier].append(c)
            for c in cluster:
                tier = c.get("tier", "Untiered")
                tiered_groups[tier].append(c)
            for group in tiered_groups.values():
                group.sort(key=lambda x: (-x.get("score", 0), -x.get("priority", 0)))

        result = {
            "root": ngram,
            "cohesion": str(cohesion_score),
            "semantic_coverage": str(semantic_coverage),
            "tiered_candidates": dict(tiered_groups),  # Make it JSON-serializable
            "summary": raw.get("summary", "[No summary]"),
        }

        # Include role-specific responses
        for role in ["Linguist", "Skeptic", "Adjudicator", "Archivist"]:
            if role in raw:
                result[role] = raw[role]

        return result

    def run_with_streaming(self, max_words=None, stream_callback=None):
        output = []
        seen_words = 0
        ngram_counts = self.get_ngram_frequencies()

        for entry in self.entries:
            word = entry["normalized"]
            ngrams = self.extract_ngrams(word)

            for ngram in ngrams:
                if ngram in self.processed_ngrams:
                    continue
                if ngram_counts[ngram] <= 1:
                    print(f"[Skipped] '{ngram}' only appears in 1 word. Ignoring it.")
                    continue
                self.processed_ngrams.add(ngram)

                semantic_candidates = find_semantically_similar_words(
                    ft_model=self.fasttext,
                    sent_model=self.sent_model,
                    entries=self.entries,
                    target_word=ngram,
                    subst_map=self.subst_map,
                    topn=20,
                )

                index_candidates = self.candidate_finder.get_candidates(
                    ngram=ngram,
                    top_n=20,
                    similarity_threshold=0.35,
                    include_context=True,
                )

                sem_norms = {c["normalized"] for c in semantic_candidates}
                index_norms = {c["normalized"] for c in index_candidates}
                overlap = sem_norms & index_norms

                if len(overlap) < 2:
                    continue

                merged_cluster = []
                merged_words = []
                seen = set()

                # Start with semantic candidates — already sorted by priority/score
                for c in semantic_candidates:
                    norm = c["normalized"]
                    if norm not in seen:
                        merged_words.append(norm)
                        seen.add(norm)

                # Add index-only extras at the end
                for c in index_candidates:
                    norm = c["normalized"]
                    if norm not in seen:
                        merged_words.append(norm)
                        seen.add(norm)

                for word in merged_words:
                    sem_entry = next(
                        (c for c in semantic_candidates if c["normalized"] == word),
                        None,
                    )
                    index_entry = next(
                        (c for c in index_candidates if c["normalized"] == word), None
                    )
                    if not sem_entry and index_entry:
                        continue

                    merged_cluster.append(
                        {
                            "word": (
                                sem_entry.get("word")
                                if sem_entry
                                else (
                                    index_entry.get("word", word)
                                    if index_entry
                                    else word
                                )
                            ),
                            "normalized": word,
                            "definition": (
                                sem_entry.get("definition")
                                if sem_entry
                                else (
                                    index_entry.get(
                                        "definition", "ERROR: no definition provided"
                                    )
                                    if index_entry
                                    else "ERROR: no definition provided"
                                )
                            ),
                            "fasttext": (
                                float(sem_entry.get("fasttext", 0.0))
                                if sem_entry
                                else 0.0
                            ),
                            "semantic": (
                                float(sem_entry.get("semantic", 0.0))
                                if sem_entry
                                else 0.0
                            ),
                            "score": (
                                float(sem_entry.get("score", 0.0)) if sem_entry else 0.0
                            ),
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
                            "citations": list(
                                {
                                    c.get("context", "").strip()
                                    for c in (
                                        sem_entry.get("citations", [])
                                        if sem_entry
                                        else []
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

                evaluated = self.evaluate_ngram(ngram, merged_cluster)
                evaluated["overlap_count"] = len(overlap)

                output.append(evaluated)

                self.save_results(output)
                self.save_processed_ngrams()
                if "Archivist" in evaluated:
                    save_log([("Archivist", evaluated["Archivist"])], label=ngram)

                # Stream out each agent’s statement
                if stream_callback:
                    for role in ["Linguist", "Skeptic", "Adjudicator", "Glossator"]:
                        if role in evaluated:
                            stream_callback(role, evaluated[role])
                    # Show only the TLDR, labeled clearly
                    # if "summary" in evaluated:
                    #     stream_callback("📜 Summary", f"{evaluated['summary']}\n")

            seen_words += 1
            if max_words and seen_words >= max_words:
                break

    def run(self, max_words=None, min_cluster_size=2):
        output = []
        seen_words = 0

        for entry in self.entries:
            word = entry["normalized"]
            ngrams = self.extract_ngrams(word)

            for ngram in ngrams:
                # Mark ngram as processed
                if ngram in self.processed_ngrams:
                    continue
                self.processed_ngrams.add(ngram)

                # Step 1: Get candidates from both paths
                semantic_candidates = find_semantically_similar_words(
                    ft_model=self.fasttext,
                    sent_model=self.sent_model,
                    entries=self.entries,
                    target_word=ngram,
                    subst_map=self.subst_map,
                    topn=20,
                )

                index_candidates = self.candidate_finder.get_candidates(
                    ngram=ngram,
                    top_n=20,
                    similarity_threshold=0.35,
                    include_context=True,
                )

                # Get normalized names for easy comparison
                sem_norms = {c["normalized"] for c in semantic_candidates}
                index_norms = {c["normalized"] for c in index_candidates}
                overlap = sem_norms & index_norms

                if len(overlap) < min_cluster_size:
                    continue

                # Step 2: Merge info and score
                merged_cluster = []
                dropped_count = 0
                for word in sem_norms | index_norms:
                    sem_entry = next(
                        (c for c in semantic_candidates if c["normalized"] == word),
                        None,
                    )
                    index_entry = next(
                        (c for c in index_candidates if c["normalized"] == word), None
                    )
                    # Coarse filter: skip if it's index-only—which means it is probably noise
                    if not sem_entry and index_entry:
                        dropped_count += 1
                        continue

                    merged_cluster.append(
                        {
                            "word": (
                                sem_entry.get("word")
                                if sem_entry
                                else (
                                    index_entry.get("word", word)
                                    if index_entry
                                    else word
                                )
                            ),
                            "normalized": word,
                            "definition": (
                                sem_entry.get("definition")
                                if sem_entry
                                else (
                                    index_entry.get(
                                        "definition", "[ERROR: no definition provided]"
                                    )
                                    if index_entry
                                    else "[ERROR: no definition provided]"
                                )
                            ),
                            "fasttext": (
                                float(sem_entry.get("fasttext", 0.0))
                                if sem_entry
                                else 0.0
                            ),
                            "semantic": (
                                float(sem_entry.get("semantic", 0.0))
                                if sem_entry
                                else 0.0
                            ),
                            "score": (
                                float(sem_entry.get("score", 0.0)) if sem_entry else 0.0
                            ),
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
                            "citations": list(
                                {
                                    c.get("context", "").strip()
                                    for c in (
                                        sem_entry.get("citations", [])
                                        if sem_entry
                                        else []
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
                evaluated = self.evaluate_ngram(ngram, merged_cluster)
                evaluated["overlap_count"] = len(overlap)
                evaluated["dropped_index_only"] = dropped_count
                output.append(evaluated)

            seen_words += 1
            if max_words and seen_words >= max_words:
                break

        output.sort(key=lambda x: (x["cohesion"], x["overlap_count"]), reverse=True)
        self.save_results(output)
        self.save_processed_ngrams()

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
