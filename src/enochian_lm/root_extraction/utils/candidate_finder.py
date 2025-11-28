from enochian_lm.common.sqlite_bootstrap import sqlite3
import math
import json
import numpy as np
import logging
from typing import Sequence
from pathlib import Path
from functools import lru_cache
from gensim.utils import simple_preprocess
from rapidfuzz import process as rf_process, fuzz
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord
from enochian_lm.root_extraction.utils.embeddings import get_fasttext_model

DEFAULT_MIN_CANDIDATE_COS_SIM = 0.15
DEFAULT_MIN_OVERLAP_RATIO = 0.1
DEFAULT_MAX_CANDIDATES = 15
DEFAULT_MULTI_SEGMENT_BONUS = 0.25

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s", level=logging.WARNING
)
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
        dictionary_entries: Sequence[EntryRecord],
        *,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        similarity_threshold: float = 0.4,
        edit_threshold: int = 70,
        prune_threshold: float = 0.0,
        min_n: int = 2,
        max_n: int = 7,
        beam_width: int = 5,
        min_candidate_cos_sim: float = DEFAULT_MIN_CANDIDATE_COS_SIM,
        min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        multi_segment_bonus: float = DEFAULT_MULTI_SEGMENT_BONUS,
    ):
        # Connect to ngram SQLite index
        self.conn = sqlite3.connect(str(ngram_db_path))
        self.cursor = self.conn.cursor()

        # Load FastText model
        self.fasttext_model = get_fasttext_model(fasttext_model_path)

        # Build dictionary: canonical -> EntryRecord mapping
        self.dictionary = {
            entry["canonical"]: entry
            for entry in dictionary_entries
            if entry.get("canonical")
        }
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
        self.min_candidate_cos_sim = min_candidate_cos_sim
        self.min_overlap_ratio = min_overlap_ratio
        self.max_candidates = max_candidates
        self.multi_segment_bonus = multi_segment_bonus

        # Load the ngram → [(canonical, tf, df), ...] map
        self._load_ngram_index()
        logger.info(
            f"Initialized MorphemeCandidateFinder with {self.total_docs} entries"
        )
        self.stats = {"tokens": 0, "multi_candidates": 0, "filtered": 0, "kept": 0}

    def _load_ngram_index(self):
        """Populate self.ngram_index using new schema:
        - TF from ngrams.total_occurrences
        - DF = count of distinct canonicals in ngram_membership
        - canonicals from ngram_membership
        """
        self.ngram_index = {}

        # 1) TF per ngram
        tf_map: dict[str, int] = {}
        for ng, tf in self.cursor.execute(
            "SELECT ngram, total_occurrences FROM ngrams"
        ):
            tf_map[ng] = int(tf or 0)

        # 2) DF per ngram
        df_map: dict[str, int] = {}
        for ng, df in self.cursor.execute(
            "SELECT ngram, COUNT(*) AS df FROM ngram_membership GROUP BY ngram"
        ):
            df_map[ng] = int(df or 0)

        # 3) Canonical membership
        for ng, canon in self.cursor.execute(
            "SELECT ngram, canonical FROM ngram_membership"
        ):
            tf = tf_map.get(ng, 0)
            df = df_map.get(ng, 0)
            self.ngram_index.setdefault(ng, []).append((canon, tf, df))

    @lru_cache(maxsize=512)
    def get_exact_matches(self, ngram: str) -> list[str]:
        """Return all canonicals containing the exact ngram."""
        return [canon for canon, _, _ in self.ngram_index.get(ngram, [])]

    @lru_cache(maxsize=512)
    def get_fasttext_matches(self, ngram: str, top_n: int = 50) -> list[str]:
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
    ) -> list[
        tuple[list[str], float, dict[str, float], list[dict[str, float | int | str]]]
    ]:
        """
        Beam-search segmentation over target string.
        Returns list of (path, cumulative_tfidf, {ngram: tfidf}, coverage_segments).

        Each coverage segment captures the positional slice that a canonical word
        explains inside ``target``. We retain positional metadata so downstream
        diagnostics can distinguish between explained and unexplained residue.
        """
        beams: list[
            tuple[
                int,
                list[str],
                float,
                dict[str, float],
                list[dict[str, float | int | str]],
            ]
        ] = [(0, [], 0.0, {}, [])]
        tgt = target.lower()
        final = []

        while beams:
            new_beams = []
            for pos, path, score, ngram_scores, coverage in beams:
                if pos >= len(tgt):
                    final.append((path, score, ngram_scores, coverage))
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
                        new_segment = {
                            "start": pos,
                            "end": pos + n,
                            "ngram": ng,
                            "canonical": canon,
                            "tfidf": tfidf,
                        }
                        new_beams.append(
                            (
                                pos + n,
                                new_path,
                                new_score,
                                new_scores,
                                coverage + [new_segment],
                            )
                        )
            # keep top-K beams
            beams = sorted(new_beams, key=lambda b: b[2], reverse=True)[
                : self.beam_width
            ]
        return final

    def score_parse(
        self,
        path: list[str],
        ngram_scores: dict[str, float],
        target: str,
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
        senses = entry.get("senses") if entry else None
        if senses:
            conf = max((s.get("confidence", 0.0) for s in senses), default=1.0)
        elif entry:
            conf = max(
                (a.get("confidence", 1.0) for a in entry.get("alternates", [])),
                default=1.0,
            )
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

    def _segment_confidence(self, canonical: str) -> float:
        """
        Heuristic per-segment semantic confidence sourced from the dictionary.

        We surface the highest confidence score available for the entry to give
        downstream diagnostics a sense of how trustworthy each matched segment is.
        """

        entry = self.dictionary.get(canonical)
        if not entry:
            return 0.0
        senses = entry.get("senses")
        if senses:
            return max((s.get("confidence", 0.0) for s in senses), default=0.0)
        alternates = entry.get("alternates")
        if alternates:
            return max((a.get("confidence", 0.0) for a in alternates), default=0.0)
        return 0.0

    def _build_breakdown(
        self, target: str, coverage: list[dict[str, float | int | str]]
    ) -> dict:
        """Compute explained vs residual spans for a given segmentation path."""

        if not target:
            return {
                "segments": [],
                "uncovered": [],
                "coverage_ratio": 0.0,
                "residual_ratio": 0.0,
            }

        # If the only coverage is a self-match over the full target, treat it as
        # uncovered so residuals reflect unmet decomposition rather than
        # vacuously explaining the word with itself.
        if len(coverage) == 1:
            seg = coverage[0]
            canonical = str(seg.get("canonical", ""))
            start = int(seg.get("start", 0))
            end = int(seg.get("end", len(target)))
            if (
                canonical
                and canonical.lower() == target.lower()
                and start <= 0
                and end >= len(target)
            ):
                return {
                    "segments": [],
                    "uncovered": [{"span": [0, len(target)], "text": target}],
                    "coverage_ratio": 0.0,
                    "residual_ratio": 1.0,
                }

        mask = [False] * len(target)
        explained_segments: list[dict[str, float | int | str]] = []
        for seg in coverage:
            start = int(seg.get("start", 0))
            end = int(seg.get("end", start))
            start = max(0, min(len(target), start))
            end = max(start, min(len(target), end))
            for idx in range(start, end):
                mask[idx] = True
            explained_segments.append(
                {
                    "span": [start, end],
                    "text": target[start:end],
                    "ngram": str(seg.get("ngram", "")),
                    "canonical": str(seg.get("canonical", "")),
                    "tfidf": float(seg.get("tfidf", 0.0)),
                    "semantic_confidence": self._segment_confidence(
                        str(seg.get("canonical", ""))
                    ),
                }
            )

        uncovered: list[dict[str, object]] = []
        idx = 0
        while idx < len(mask):
            if mask[idx]:
                idx += 1
                continue
            start = idx
            while idx < len(mask) and not mask[idx]:
                idx += 1
            uncovered.append({"span": [start, idx], "text": target[start:idx]})

        covered_chars = sum(1 for flag in mask if flag)
        coverage_ratio = covered_chars / len(mask) if mask else 0.0
        residual_ratio = max(0.0, 1.0 - coverage_ratio)

        return {
            "segments": explained_segments,
            "uncovered": uncovered,
            "coverage_ratio": coverage_ratio,
            "residual_ratio": residual_ratio,
        }

    def _passes_overlap(self, target: str, breakdown: dict) -> bool:
        """Heuristic coverage check to keep loosest possible overlaps configurable."""

        if not target:
            return False
        if self.min_overlap_ratio <= 0:
            return True
        coverage_ratio = float(breakdown.get("coverage_ratio") or 0.0)
        return coverage_ratio >= self.min_overlap_ratio

    def find_candidates(
        self,
        target: str,
        *,
        top_k: int | None = None,
        min_cos_sim: float | None = None,
    ) -> list[dict]:
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
        for path, _, ngram_scores, coverage in parses:
            c = self.score_parse(path, ngram_scores, target)
            # Ensure every candidate has a 'normalized'
            if not c.get("normalized"):
                # Take the last element of the segmentation path as the candidate token
                c["normalized"] = path[-1].lower()
            c["target"] = target
            c["target_length"] = len(target)
            c["breakdown"] = self._build_breakdown(target, coverage)
            scored.append(c)

        print(
            "[Debug][find_candidates] scored (normalized, cos_sim):",
            [(c["normalized"], c["cos_sim"]) for c in scored[:5]],
        )

        # 3) Prune low-semantic matches (skip for very short targets)
        cos_cutoff = max(self.prune_threshold, self.min_candidate_cos_sim)
        if min_cos_sim is not None:
            cos_cutoff = max(cos_cutoff, float(min_cos_sim))
        if len(target) <= 2:
            filtered = scored
        else:
            filtered = [c for c in scored if c["cos_sim"] >= cos_cutoff]

        # 4) Guarantee at least the root itself
        root_norm = target.lower()
        fallback = {
            "word": target.upper(),
            "normalized": root_norm,
            "cos_sim": 1.0,
            "composite": 1.0,
            "target": target,
            "target_length": len(target),
            "breakdown": self._build_breakdown(target, []),
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

        # 5) Apply coverage-based overlap filter
        overlap_filtered = []
        dropped_overlap = 0
        for cand in filtered:
            bd = cand.get("breakdown") or {}
            if self._passes_overlap(target, bd):
                overlap_filtered.append(cand)
            else:
                dropped_overlap += 1

        if dropped_overlap:
            logger.info(
                "Dropped candidates below overlap threshold",
                extra={"target": target, "dropped": dropped_overlap},
            )
        filtered = overlap_filtered or filtered

        # 6) Sort by composite score (with optional bonus for multi-segment parses)
        def _score_for_sort(cand: dict) -> float:
            segments = cand.get("breakdown", {}).get("segments") or []
            bonus = self.multi_segment_bonus if len(segments) >= 2 else 0.0
            return cand.get("composite", 0.0) + bonus

        top_limit = top_k if top_k is not None else self.max_candidates
        candidates = sorted(filtered, key=_score_for_sort, reverse=True)[:top_limit]
        print(
            "[Debug][find_candidates] final candidates:",
            [c["normalized"] for c in candidates],
        )

        # --- stats & debug logging ---
        self.stats["tokens"] += 1
        if len(candidates) > 1:
            self.stats["multi_candidates"] += 1
        self.stats["kept"] += len(candidates)
        # treat `scored` as the "raw" set before pruning+top_k
        self.stats["filtered"] += max(0, len(scored) - len(candidates))

        logger.debug(
            "[Finder] %s: %d kept / %d raw (sim≥%.3f, overlap≥%.3f)",
            target,
            len(candidates),
            len(scored),
            self.min_candidate_cos_sim,
            self.min_overlap_ratio,
        )

        return candidates

    def get_stats(self): 
        return self.stats

    def get_all_ngram_candidates(self, target: str) -> list[dict]:
        """
        Return every canonical whose normalized form contains the exact ngram
        via the ngram_membership table.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT canonical FROM ngram_membership WHERE ngram = ?",
            (target.lower(),),
        )
        rows = cursor.fetchall()
        return [{"word": canon.upper(), "normalized": canon} for (canon,) in rows]

    def close(self):
        self.conn.close()
