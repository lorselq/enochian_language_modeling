import json
from collections import defaultdict
from enochian_translation_team.utils.config import get_config_paths

def build_ngram_index(entries, min_n=1, max_n=4):
    index = defaultdict(set)
    for entry in entries:
        if not entry.get("canon_word") or not entry.get("definition"):
            continue
        normalized = entry["normalized"].lower()
        for n in range(min_n, max_n + 1):
            for i in range(len(normalized) - n + 1):
                ngram = normalized[i:i+n]
                index[ngram].add(normalized)
    return {k: sorted(v) for k, v in index.items()}

def save_index(index, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def load_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    paths = get_config_paths()
    print("[+] Loading dictionary...")
    entries = load_words(paths["dictionary"])
    index = build_ngram_index(entries)
    save_index(index, paths["ngram_index"])
    print(f"[✓] N-gram index saved to {paths['ngram_index']}")

if __name__ == "__main__":
    main()