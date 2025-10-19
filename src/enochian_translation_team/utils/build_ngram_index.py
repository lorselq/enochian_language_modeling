from enochian_translation_team.utils import sqlite_bootstrap  # noqa: F401
import os
import json
import sqlite3
import hashlib
import logging
import marisa_trie
from collections import defaultdict
from typing import List, Dict, Tuple


from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.variant_utils import (
    normalize_word,
    apply_sequence_compressions,
    generate_variants,
)

# --- Logging setup ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Helpers for computing DF and TF ---
def build_ngram_tf_df(
    entries: List[Dict],
    min_n: int,
    max_n: int,
    subst_map: Dict,
    compression_rules: Dict,
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Returns:
      ngram_tf: {ngram: {entry_key: count, ...}, ...}
      ngram_df: {ngram: document_frequency, ...}
    """
    ngram_tf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in entries:
        canon = entry.get("normalized", "").lower()
        if not canon or not entry.get("definition"):
            continue

        norm = normalize_word(canon, subst_map=subst_map)
        norm = apply_sequence_compressions(norm, compression_rules=compression_rules)
        variants = generate_variants(norm, subst_map=subst_map, return_subst_meta=True)

        seen_ngrams = set()
        for variant, _, _ in variants:
            for n in range(min_n, max_n + 1):
                for i in range(len(variant) - n + 1):
                    ngram = variant[i : i + n]
                    # TF: count occurrences in this entry
                    ngram_tf[ngram][canon] += 1
                    seen_ngrams.add(ngram)
        # Note: DF will count unique presence per entry via seen_ngrams below

    # Build DF from TF
    ngram_df = {ng: len(canon_dict) for ng, canon_dict in ngram_tf.items()}
    return ngram_tf, ngram_df


# --- PMI-based filtering (stub) ---
def filter_by_pmi(
    ngram_tf: Dict[str, Dict[str, int]],
    ngram_df: Dict[str, int],
    total_docs: int,
    pmi_thresh: float = 2.0,
) -> List[str]:
    """
    Compute simple PMI for bigrams and trigrams and filter by threshold.
    This is a placeholder: you may want to refine with full joint probabilities.
    """
    filtered = []
    # Only consider ngrams of length >= 2
    for ng, df in ngram_df.items():
        if len(ng) < 2:
            filtered.append(ng)
            continue
        # very rough PMI: log(df / (sum of dfs of subparts)^2 * total_docs)
        # split into two halves
        mid = len(ng) // 2 or 1
        left, right = ng[:mid], ng[mid:]
        df_left = ngram_df.get(left, 1)
        df_right = ngram_df.get(right, 1)
        import math

        pmi = math.log((df * total_docs) / (df_left * df_right + 1e-9) + 1e-9)
        if pmi >= pmi_thresh:
            filtered.append(ng)
    return filtered


# --- Main building & saving ---
def build_and_save_ngram_index(
    min_n: int = 1, max_n: int = 6, pmi_threshold: float = 2.0
):
    paths = get_config_paths()
    dict_path = paths["dictionary"]
    ngram_db_path = paths["ngram_index"]

    # Check dictionary hash to skip rebuild
    # dict_bytes = open(dict_path, "rb").read()
    # dict_hash = hashlib.md5(dict_bytes).hexdigest()
    # meta_path = str(ngram_db_path) + ".meta.json"
    # if os.path.exists(ngram_db_path) and os.path.exists(meta_path):
    #     meta = json.loads(open(meta_path).read())
    #     if meta.get("dict_hash") == dict_hash:
    #         logger.info("N-gram index up-to-date (cache hit)")
    #         return

    # Load raw entries
    entries = json.loads(open(dict_path, "r", encoding="utf-8").read())
    subst_map = json.loads(
        open(paths["substitution_map"], "r", encoding="utf-8").read()
    )
    compression_rules = json.loads(
        open(paths["sequence_compressions"], "r", encoding="utf-8").read()
    )

    # Build TF and DF
    logger.info("Building ngram TF and DF...")
    ngram_tf, ngram_df = build_ngram_tf_df(
        entries, min_n, max_n, subst_map, compression_rules
    )

    total_docs = len(
        {
            entry.get("normalized", "").lower()
            for entry in entries
            if entry.get("normalized")
        }
    )

    # Filter by PMI to get high-salience substrings
    logger.info("Filtering ngrams by PMI threshold...")
    salient_ngrams = set(ngram_tf.keys())

    # Build marisa trie of salient ngrams
    logger.info("Building Marisa Trie for salient ngrams...")
    trie = marisa_trie.Trie(sorted(salient_ngrams))
    trie.save(str(ngram_db_path) + ".trie")

    # Save into SQLite: include Term Frequency (TF) and Document Frequency (DF)
    # TF is the number of times an ngram occurs across all the words
    # DF is the number of distinct entries it shows up in
    logger.info("Saving ngram index to SQLite...")
    conn = sqlite3.connect(ngram_db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS ngrams")
    cur.execute(
        "CREATE TABLE ngrams (ngram TEXT, canonical TEXT, tf_count INTEGER, df_count INTEGER)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ngram ON ngrams (ngram)")

    for ngram, canon_dict in ngram_tf.items():
        if ngram not in salient_ngrams:
            continue
        df_count = ngram_df.get(ngram, 0)
        for canon, tf_count in canon_dict.items():
            cur.execute(
                "INSERT INTO ngrams (ngram, canonical, tf_count, df_count) VALUES (?, ?, ?, ?)",
                (ngram, canon, tf_count, df_count),
            )
    conn.commit()
    conn.close()

    # Save metadata
    # json.dump(
    #     {
    #         "dict_hash": dict_hash,
    #         "min_n": min_n,
    #         "max_n": max_n,
    #         "pmi_threshold": pmi_threshold,
    #     },
    #     open(meta_path, "w"),
    #     indent=2,
    # )
    # logger.info(f"N-gram index built and saved to {ngram_db_path}")


def main():
    build_and_save_ngram_index()


if __name__ == "__main__":
    main()
