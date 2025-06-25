import sqlite3
import math
import json
import numpy as np
import logging
from pathlib import Path
from functools import lru_cache
from gensim.models import FastText
from gensim.utils import simple_preprocess
from rapidfuzz import process as rf_process, fuzz
from enochian_translation_team.utils.dictionary_loader import Entry

# --- Logging setup ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


class MorphemeCandidateFinder:
    """
    Finds and ranks candidate morphemes/entries for a target string using:
      - TF–IDF scores from an n-gram SQLite index
      - FastText semantic similarity
      - RapidFuzz edit-distance fuzzy matching
      - Beam-search segmentation
      - Composite scoring and explainability
    """

    def __init__(
        self,
        ngram_db_path: Path,
        fasttext_model_path: Path,
        dictionary_entries: list[Entry],
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        similarity_threshold: float = 0.4,
        edit_threshold: int = 70,
        prune_threshold: float = 0.0, # above 0.0 will prune words not meeting semantic threshold; for character-cares only, set to 0.0
        min_n: int = 2,
        max_n: int = 6,
        beam_width: int = 5,
    ):
        # Connect to ngram SQLite index
        self.conn = sqlite3.connect(str(ngram_db_path))
        self.cursor = self.conn.cursor()

        # Load FastText model
        self.fasttext_model = FastText.load(str(fasttext_model_path))

        # Build dictionary: canonical -> Entry
        self.dictionary = {e.canonical: e for e in dictionary_entries}
        self.known_words = set(self.dictionary.keys())
        self.total_docs = len(self.known_words)

        # Scoring and thresholds
        self.weights = weights  # (alpha, beta, gamma)
        self.similarity_threshold = similarity_threshold
        self.edit_threshold = edit_threshold
        self.prune_threshold = prune_threshold
        self.min_n = min_n
        self.max_n = max_n
        self.beam_width = beam_width

        # Load the ngram → [(canonical, tf, df), ...] map
        self._load_ngram_index()
        logger.info(
            f"Initialized MorphemeCandidateFinder with {self.total_docs} entries"
        )

    def _load_ngram_index(self):
        """Populate self.ngram_index from the SQLite table."""
        self.ngram_index: dict[str, list[tuple[str, int, int]]] = {}
        query = "SELECT ngram, canonical, tf_count, df_count FROM ngrams"
        for ngram, canon, tf_count, df_count in self.cursor.execute(query):
            self.ngram_index.setdefault(ngram, []).append((canon, tf_count, df_count))

    @lru_cache(maxsize=512)
    def get_exact_matches(self, ngram: str) -> list[str]:
        """Return all canonicals containing the exact ngram."""
        return [canon for canon, _, _ in self.ngram_index.get(ngram, [])]

    @lru_cache(maxsize=512)
    def get_fasttext_matches(self, ngram: str, top_n: int = 10) -> list[str]:
        """Return semantically similar canonicals via FastText."""
        results = []
        try:
            for word, score in self.fasttext_model.wv.similar_by_word(
                ngram, topn=top_n
            ):
                if word in self.known_words and score >= self.similarity_threshold:
                    results.append(word)
        except KeyError:
            pass
        return results

    @lru_cache(maxsize=512)
    def get_edit_distance_matches(self, ngram: str, top_n: int = 10) -> list[str]:
        """Return near-matches by edit distance ratio."""
        choices = list(self.known_words)
        matches = rf_process.extract(ngram, choices, scorer=fuzz.ratio, limit=top_n)
        return [w for w, score, _ in matches if score >= self.edit_threshold]

    def segment_target(
        self, target: str
    ) -> list[tuple[list[str], float, dict[str, float]]]:
        """
        Beam-search segmentation over target string.
        Returns list of (path, cumulative_tfidf, {ngram: tfidf}).
        """
        beams = [(0, [], 0.0, {})]
        tgt = target.lower()
        final = []

        while beams:
            new_beams = []
            for pos, path, score, ngram_scores in beams:
                if pos >= len(tgt):
                    final.append((path, score, ngram_scores))
                    continue
                # try next n-grams
                for n in range(self.min_n, min(self.max_n, len(tgt) - pos) + 1):
                    ng = tgt[pos : pos + n]
                    for canon, tf, df in self.ngram_index.get(ng, []):
                        idf = math.log(self.total_docs / (df + 1))
                        tfidf = tf * idf
                        new_path = path + [canon]
                        new_score = score + tfidf
                        new_scores = {**ngram_scores, ng: tfidf}
                        new_beams.append((pos + n, new_path, new_score, new_scores))
            # keep top-K beams
            beams = sorted(new_beams, key=lambda b: b[2], reverse=True)[
                : self.beam_width
            ]
        return final

    def score_parse(
        self, path: list[str], ngram_scores: dict[str, float], target: str
    ) -> dict:
        """
        Compute composite score and return explanation.
        Composite = α·sum(tfidf) + β·cos_sim + γ·confidence
        """
        # 1. TFIDF component
        tfidf_total = sum(ngram_scores.values())

        # 2. Semantic similarity
        tokens = simple_preprocess(target, deacc=True, min_len=1)
        if tokens:
            tgt_vec = np.mean(
                [
                    self.fasttext_model.wv[t]
                    for t in tokens
                    if t in self.fasttext_model.wv
                ],
                axis=0,
            )
        else:
            tgt_vec = np.zeros(self.fasttext_model.vector_size)
        cand_vecs = [
            self.fasttext_model.wv[w] for w in path if w in self.fasttext_model.wv
        ]
        cand_vec = np.mean(cand_vecs, axis=0) if cand_vecs else np.zeros_like(tgt_vec)
        cos_sim = float(
            np.dot(tgt_vec, cand_vec)
            / (np.linalg.norm(tgt_vec) * np.linalg.norm(cand_vec) + 1e-9)
        )

        # 3. Confidence weight
        entry = self.dictionary.get(path[-1])
        if entry and entry.senses:
            conf = max(s.confidence for s in entry.senses)
        elif entry:
            conf = max((a.confidence for a in entry.alternates), default=1.0)
        else:
            conf = 1.0

        alpha, beta, gamma = self.weights
        composite = alpha * tfidf_total + beta * cos_sim + gamma * conf

        return {
            "path": path,
            "tfidf": tfidf_total,
            "cos_sim": cos_sim,
            "confidence": conf,
            "composite": composite,
            "explanation": {
                "ngram_scores": ngram_scores,
                "weights": {"tfidf": alpha, "cos_sim": beta, "confidence": gamma},
            },
        }

    def find_candidates(self, target: str, top_k: int = 5) -> list[dict]:
        """
        Full pipeline: segment, score, ensure normalized, prune by cos_sim, and return top-K.
        Guarantees we never lose valid index hits simply because their 'normalized' was missing.
        """
        # 1) Segment the target into possible ngram paths
        parses = self.segment_target(target)
        print(
            f"[Debug][find_candidates] parses for '{target}': {parses[:3]}"
        )  # peek first 3

        # 2) Score each path and force a normalized value
        scored = []
        for path, _, ngram_scores in parses:
            c = self.score_parse(path, ngram_scores, target)
            # Ensure every candidate has a 'normalized'
            if not c.get("normalized"):
                # Take the last element of the segmentation path as the candidate token
                c["normalized"] = path[-1].lower()
            scored.append(c)

        print(
            "[Debug][find_candidates] scored (normalized, cos_sim):",
            [(c["normalized"], c["cos_sim"]) for c in scored[:5]],
        )

        # 3) Prune low-semantic matches (skip for very short targets)
        if len(target) <= 2:
            filtered = scored
        else:
            filtered = [c for c in scored if c["cos_sim"] >= self.prune_threshold]

        # 4) Guarantee at least the root itself
        root_norm = target.lower()
        fallback = {
            "word": target.upper(),
            "normalized": root_norm,
            "cos_sim": 1.0,
            "composite": 1.0,
        }

        if not filtered:
            print(
                f"[Debug][find_candidates] no survivors, falling back to root '{root_norm}'"
            )
            filtered = [fallback]
        else:
            # Prepend the root if it's not already in the list
            if all(c["normalized"] != root_norm for c in filtered):
                filtered.insert(0, fallback)

        # 5) Sort by composite score and trim to top_k
        candidates = sorted(filtered, key=lambda c: c["composite"], reverse=True)[
            :top_k
        ]
        print(
            "[Debug][find_candidates] final candidates:",
            [c["normalized"] for c in candidates],
        )
        return candidates
    
    def get_all_ngram_candidates(self, target: str) -> list[dict]:
        """
        Return every canonical whose normalized form (or variant)
        contains the exact ngram substrings (via your SQLite ngrams table).
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT canonical FROM ngrams WHERE ngram = ?",
            (target.lower(),)
        )
        rows = cursor.fetchall()
        return [
            {"word": canon.upper(), "normalized": canon}
            for (canon,) in rows
        ]

    def close(self):
        self.conn.close()
