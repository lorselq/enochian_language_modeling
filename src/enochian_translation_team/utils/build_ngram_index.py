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


def build_ngram_index(entries, min_n=1, max_n=6):
    paths = get_config_paths()
    index = defaultdict(set)
    for entry in entries:
        if not entry.get("canon_word") or not entry.get("definition"):
            continue
        canon = entry["normalized"].lower()
        norm = normalize_word(canon, subst_map=load_json(paths["substitution_map"]))
        norm = apply_sequence_compressions(
            norm, compression_rules=load_json(paths["sequence_compressions"])
        )
        variants = generate_variants(
            norm, subst_map=load_json(paths["substitution_map"])
        )
        for variant in variants:
            for n in range(min_n, max_n + 1):
                for i in range(len(variant) - n + 1):
                    ngram = variant[i : i + n]
                    index[ngram].add({"variant": variant, "canonical": canon})
    return {k: sorted(v) for k, v in index.items()}


def save_index(index, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ngrams(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_and_save_ngram_index():
    paths = get_config_paths()
    print("[+] Loading dictionary...")
    entries = load_words(paths["dictionary"])
    index = build_ngram_index(entries)
    save_index(index, paths["ngram_index"])
    print(f"[✓] N-gram index saved to {paths['ngram_index']}")


def main():
    build_and_save_ngram_index()


if __name__ == "__main__":
    main()
