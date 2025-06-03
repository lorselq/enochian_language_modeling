from tqdm import tqdm
import sqlite3
import json
from collections import defaultdict
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.variant_utils import (
    normalize_word,
    apply_sequence_compressions,
    generate_variants,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ngrams(path):
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            index[obj["ngram"]] = obj["entries"]
    return index


def build_ngram_index(entries, min_n=1, max_n=6):
    paths = get_config_paths()
    subst_map = load_json(paths["substitution_map"])
    compression_rules = load_json(paths["sequence_compressions"])

    index = defaultdict(list)
    seen = set()

    for entry in tqdm(entries, "Building out ngrams..."):
        if not entry.get("canon_word") or not entry.get("definition"):
            continue

        canon = entry["normalized"].lower()
        norm = normalize_word(canon, subst_map=subst_map)
        norm = apply_sequence_compressions(norm, compression_rules=compression_rules)
        variants = generate_variants(norm, subst_map=subst_map, return_subst_meta=True)

        for variant, num_subs, letter_names in variants:
            variant_key = f"{variant}|{canon}"
            if variant_key in seen:
                continue
            seen.add(variant_key)

            for n in range(min_n, max_n + 1):
                for i in range(len(variant) - n + 1):
                    ngram = variant[i : i + n]
                    index[ngram].append(
                        {
                            "variant": variant,
                            "canonical": canon,
                            "num_subs": num_subs,
                            "letter_names": (
                                ",".join(letter_names) if letter_names else None
                            ),
                        }
                    )
    return index


def build_and_save_ngram_index():
    paths = get_config_paths()
    conn = sqlite3.connect(paths["ngram_index"])
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS ngrams")

    cursor.execute(
        """
        CREATE TABLE ngrams (
            ngram TEXT,
            variant TEXT,
            canonical TEXT,
            subst_count INTEGER,
            letter_names TEXT
        )
    """
    )

    print("[+] Loading dictionary...")
    entries = load_json(paths["dictionary"])
    index = build_ngram_index(entries)
    for ngram, entries in tqdm(
        index.items(), "[+] Inserting ngram data into SQLite..."
    ):
        for e in entries:
            cursor.execute(
                "INSERT INTO ngrams (ngram, variant, canonical, subst_count, letter_names) VALUES (?, ?, ?, ?, ?)",
                (
                    ngram,
                    e["variant"],
                    e["canonical"],
                    e.get("subst_count", 0),
                    e.get("letter_names"),
                ),
            )
    print("[+] Creating index on the sqlite table 'ngram'...")
    cursor.execute("CREATE INDEX idx_ngram ON ngrams (ngram)")
    conn.commit()
    conn.close()
    print(f"[✓] SQLite N-gram database created at {paths['ngram_index']}")


def main():
    build_and_save_ngram_index()


if __name__ == "__main__":
    main()
