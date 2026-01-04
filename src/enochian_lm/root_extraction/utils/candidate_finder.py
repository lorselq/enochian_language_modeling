from __future__ import annotations

from enochian_lm.common.sqlite_bootstrap import sqlite3
import math
import json
import numpy as np
import logging
from collections.abc import Sequence
from typing import TypedDict
from pathlib import Path
from functools import lru_cache
from gensim.utils import simple_preprocess
from rapidfuzz import process as rf_process, fuzz
from enochian_lm.root_extraction.utils.types_lexicon import EntryRecord
from enochian_lm.root_extraction.utils.embeddings import (
    cluster_definitions,
    get_fasttext_model,
)

DEFAULT_MIN_CANDIDATE_COS_SIM = 0.15
DEFAULT_MIN_OVERLAP_RATIO = 0.1
DEFAULT_MAX_CANDIDATES = 15
DEFAULT_MULTI_SEGMENT_BONUS = 0.25
# Increased length bonus to favor longer, more specific morphs (NAZ > NA)
# 0.75 per extra char: NAZ (3 chars) gets 1.5 bonus, NA (2 chars) gets 0.75
DEFAULT_LENGTH_BONUS = 0.75
DEFAULT_SEGMENT_PENALTY = 0.35
DEFAULT_SINGLE_CHAR_PENALTY = 2.5
DEFAULT_EDGE_BONUS = 0.6
# Ambiguity penalty: penalizes high-TF morphs (more common = more ambiguous)
# This helps specific morphs (NAZ with 1 definition) beat ambiguous ones (NA with 28)
# Scale of 0.5 means: log1p(32) * 0.5 = 1.75 penalty for highly frequent morphs
DEFAULT_AMBIGUITY_PENALTY_SCALE = 0.5
# All Enochian letters are allowed as singletons; penalties vary by letter
DEFAULT_ALLOWED_SINGLETONS = set("abcdefghiklmnopqrstuxyz")
# Per-letter singleton penalties: L and I get reduced penalties (common morphemes)
DEFAULT_SINGLETON_PENALTIES: dict[str, float] = {
    "l": 0.25,  # Very common morpheme
    "i": 0.75,  # Common morpheme
    # All other letters use DEFAULT_SINGLE_CHAR_PENALTY (2.5)
}

# --- Logging setup ---
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s", level=logging.WARNING
)
logger = logging.getLogger(__name__)


class CoverageSegment(TypedDict):
    start: int
    end: int
    ngram: str
    canonical: str
    tfidf: float
    raw_tfidf: float
    length_bonus: float
    edge_bonus: float
    segment_penalty: float
    singleton_penalty: float
    ambiguity_penalty: float
    specificity_penalty: float


class SegmentScore(TypedDict):
    ngram: str
    score: float
    start: int
    end: int


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
        beam_width: int = 15,
        min_candidate_cos_sim: float = DEFAULT_MIN_CANDIDATE_COS_SIM,
        min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        multi_segment_bonus: float = DEFAULT_MULTI_SEGMENT_BONUS,
        length_bonus: float = DEFAULT_LENGTH_BONUS,
        segment_penalty: float = DEFAULT_SEGMENT_PENALTY,
        single_char_penalty: float = DEFAULT_SINGLE_CHAR_PENALTY,
        edge_bonus: float = DEFAULT_EDGE_BONUS,
        allowed_singletons: set[str] | None = None,
        singleton_penalties: dict[str, float] | None = None,
        ambiguity_penalty_scale: float = DEFAULT_AMBIGUITY_PENALTY_SCALE,
        n_best: int | None = None,
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
        self.known_words_lower = {word.lower() for word in self.known_words}
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
        self.length_bonus = length_bonus
        self.segment_penalty = segment_penalty
        self.single_char_penalty = single_char_penalty
        self.edge_bonus = edge_bonus
        self.allowed_singletons = (
            {token.lower() for token in allowed_singletons}
            if allowed_singletons
            else set(DEFAULT_ALLOWED_SINGLETONS)
        )
        self.singleton_penalties = (
            {k.lower(): v for k, v in singleton_penalties.items()}
            if singleton_penalties
            else dict(DEFAULT_SINGLETON_PENALTIES)
        )
        self.ambiguity_penalty_scale = ambiguity_penalty_scale
        self.n_best = n_best

        # Load the ngram → [(canonical, tf, df), ...] map
        self._load_ngram_index()
        logger.info(
            f"Initialized MorphemeCandidateFinder with {self.total_docs} entries"
        )
        self.stats = {"tokens": 0, "multi_candidates": 0, "filtered": 0, "kept": 0}

    def _score_with_bonus(self, composite: float, breakdown: dict | None) -> float:
        """Return composite score with optional multi-segment bonus."""

        segments = []
        if isinstance(breakdown, dict):
            segments = breakdown.get("segments") or []
        bonus = self.multi_segment_bonus if len(segments) >= 2 else 0.0
        return composite + bonus

    def _load_ngram_index(self):
        """Populate self.ngram_index using new schema:
        - TF from ngrams.total_occurrences
        - DF = count of distinct canonicals in ngram_membership
        - canonicals from ngram_membership
        """
        self.ngram_index: dict[str, list[tuple[str, int, int]]] = {}

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
        self,
        target: str,
        *,
        extra_ngrams: dict[str, list[tuple[str, int, int]]] | None = None,
        min_n: int | None = None,
        n_best: int | None = None,
        definition_counts: dict[str, int] | None = None,
        definition_glosses: dict[str, list[tuple[str, float | None]]] | None = None,
        definition_cluster_threshold: float = 0.8,
    ) -> list[
        tuple[
            list[str],
            float,
            list["SegmentScore"],
            list["CoverageSegment"],
        ]
    ]:
        """
        Beam-search segmentation over target string.
        Returns list of (path, cumulative_tfidf, ngram_scores, coverage_segments).

        Each coverage segment captures the positional slice that a canonical word
        explains inside ``target``. We retain positional metadata so downstream
        diagnostics can distinguish between explained and unexplained residue.
        """
        beams: list[
            tuple[
                int,
                list[str],
                float,
                list["SegmentScore"],
                list["CoverageSegment"],
            ]
        ] = [(0, [], 0.0, [], [])]
        tgt = target.lower()
        final = []
        min_len = min_n if min_n is not None else self.min_n

        cluster_count_cache: dict[str, int] = {}

        def _cluster_count(ngram: str) -> int:
            ngram_key = ngram.upper()
            if ngram_key in cluster_count_cache:
                return cluster_count_cache[ngram_key]
            glosses = definition_glosses.get(ngram_key) if definition_glosses else None
            if not glosses:
                cluster_count_cache[ngram_key] = 0
                return 0
            texts = [gloss for gloss, _score in glosses if gloss]
            scores = [score for gloss, score in glosses if gloss]
            if not texts:
                cluster_count_cache[ngram_key] = 0
                return 0
            clusters = cluster_definitions(
                texts,
                similarity_threshold=definition_cluster_threshold,
                scores=scores,
            )
            cluster_count_cache[ngram_key] = len(clusters)
            return cluster_count_cache[ngram_key]

        while beams:
            new_beams = []
            for pos, path, score, ngram_scores, coverage in beams:
                if pos >= len(tgt):
                    final.append((path, score, ngram_scores, coverage))
                    continue
                # try next n-grams
                for n in range(min_len, min(self.max_n, len(tgt) - pos) + 1):
                    ng = tgt[pos : pos + n]
                    entries = self.ngram_index.get(ng, [])
                    if extra_ngrams:
                        entries = entries + extra_ngrams.get(ng, [])
                    if definition_glosses:
                        cluster_count = _cluster_count(ng)
                        if cluster_count > 0 and len(entries) > cluster_count:
                            entries = sorted(entries, key=lambda item: item[0])[
                                :cluster_count
                            ]
                    for canon, tf, df in entries:
                        total_docs = max(1, self.total_docs)
                        idf = math.log(total_docs / (df + 1))
                        tf_weight = math.log1p(tf)
                        raw_tfidf = tf_weight * idf
                        length_bonus = self.length_bonus * max(0, n - 1)
                        edge_bonus = 0.0
                        if pos == 0:
                            edge_bonus += self.edge_bonus
                        if pos + n == len(tgt):
                            edge_bonus += self.edge_bonus
                        # Singleton penalty: 0 for multi-char, per-letter for singletons
                        if n > 1:
                            singleton_penalty = 0.0
                        elif ng in self.allowed_singletons:
                            # Use per-letter penalty if defined, else default
                            singleton_penalty = self.singleton_penalties.get(
                                ng, self.single_char_penalty
                            )
                        else:
                            singleton_penalty = self.single_char_penalty
                        segment_penalty = self.segment_penalty
                        # Ambiguity penalty: penalize high-TF morphs (more common = more ambiguous)
                        # This helps specific morphs (NAZ) beat ambiguous ones (NA)
                        ambiguity_penalty = self.ambiguity_penalty_scale * tf_weight
                        # Specificity penalty: penalize morphs with many accepted definitions
                        # Fewer definitions = more specific = better
                        # NAZ (1 def) vs NA (28 defs): log1p(1)*0.5=0.35 vs log1p(28)*0.5=1.68
                        specificity_penalty = 0.0
                        if definition_counts:
                            def_count = definition_counts.get(ng.upper(), 0)
                            if def_count > 0:
                                specificity_penalty = 0.5 * math.log1p(def_count)
                        tfidf = (
                            raw_tfidf
                            + length_bonus
                            + edge_bonus
                            - segment_penalty
                            - singleton_penalty
                            - ambiguity_penalty
                            - specificity_penalty
                        )
                        new_path = path + [canon]
                        new_score = score + tfidf
                        new_scores: list[SegmentScore] = ngram_scores + [
                            {
                                "ngram": ng,
                                "score": tfidf,
                                "start": pos,
                                "end": pos + n,
                            }
                        ]
                        new_segment: CoverageSegment = {
                            "start": pos,
                            "end": pos + n,
                            "ngram": ng,
                            "canonical": canon,
                            "tfidf": tfidf,
                            "raw_tfidf": raw_tfidf,
                            "length_bonus": length_bonus,
                            "edge_bonus": edge_bonus,
                            "segment_penalty": segment_penalty,
                            "singleton_penalty": singleton_penalty,
                            "ambiguity_penalty": ambiguity_penalty,
                            "specificity_penalty": specificity_penalty,
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
        final_sorted = sorted(final, key=lambda item: item[1], reverse=True)
        limit = n_best if n_best is not None else self.n_best
        if limit is None or limit <= 0:
            return final_sorted
        return final_sorted[:limit]

    def score_parse(
        self,
        path: list[str],
        ngram_scores: list["SegmentScore"] | dict[str, float],
        target: str,
        coverage: list["CoverageSegment"] | None = None,
    ) -> dict:
        """
        Compute composite score and return explanation.
        Composite = α·sum(tfidf) + β·cos_sim + γ·confidence
        """
        # 1. TFIDF component
        if coverage:
            tfidf_total = sum(float(seg.get("tfidf", 0.0)) for seg in coverage)
        elif isinstance(ngram_scores, dict):
            tfidf_total = sum(float(value) for value in ngram_scores.values())
        else:
            tfidf_total = sum(float(seg.get("score", 0.0)) for seg in ngram_scores)

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
        confs: list[float] = []
        weights: list[float] = []
        if coverage:
            for seg in coverage:
                canonical = str(seg.get("canonical", ""))
                confs.append(self._segment_confidence(canonical))
                start = int(seg.get("start", 0))
                end = int(seg.get("end", start))
                weights.append(max(1.0, float(max(0, end - start))))
        else:
            for canon in path:
                confs.append(self._segment_confidence(canon))
                weights.append(1.0)
        total_weight = sum(weights) if weights else 0.0
        if confs and total_weight > 0:
            conf = float(sum(c * w for c, w in zip(confs, weights)) / total_weight)
        else:
            conf = 0.0

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
        self, target: str, coverage: list[CoverageSegment]
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
                and target.lower() not in self.known_words_lower
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
            c = self.score_parse(path, ngram_scores, target, coverage=coverage)
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
            return self._score_with_bonus(cand.get("composite", 0.0), cand.get("breakdown"))

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

    def best_segmentation(
        self,
        target: str,
        *,
        min_segments: int = 1,
        min_cos_sim: float | None = None,
    ) -> dict | None:
        """Return the highest-scoring segmentation for a token.

        This is root-independent and applies the same composite + bonus scoring
        as ``find_candidates`` while allowing a caller to require a minimum
        number of morph segments.
        """

        if not target:
            return None

        cos_cutoff = max(self.prune_threshold, self.min_candidate_cos_sim)
        if min_cos_sim is not None:
            cos_cutoff = max(cos_cutoff, float(min_cos_sim))

        parses = self.segment_target(target)
        scored: list[dict] = []
        for path, _, ngram_scores, coverage in parses:
            composite = self.score_parse(path, ngram_scores, target, coverage=coverage)
            breakdown = self._build_breakdown(target, coverage)
            if not self._passes_overlap(target, breakdown):
                continue
            cos_sim = float(composite.get("cos_sim", 0.0))
            if len(target) > 2 and cos_sim < cos_cutoff:
                continue
            scored.append({**composite, "breakdown": breakdown})

        if not scored:
            return None

        scored = [s for s in scored if len(s.get("breakdown", {}).get("segments", [])) >= min_segments]
        if not scored:
            return None

        def _score(entry: dict) -> float:
            return self._score_with_bonus(
                float(entry.get("composite", 0.0)), entry.get("breakdown")
            )

        return max(scored, key=_score)

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
