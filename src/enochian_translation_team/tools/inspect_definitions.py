from typing import List
import sys
import json
from enochian_translation_team.utils.dictionary_loader import load_dictionary, EntryLike
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import util
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.embeddings import (
    get_fasttext_model,
    get_sentence_transformer,
)

# === 0. Establish substitution map for later reference ===
def load_substitution_map():
    path = get_config_paths()["substitution_map"]
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    subst_map = {}
    for k, v in raw.items():
        subs = []
        for alt in v["alternates"]:
            if alt["direction"] in ["to", "both"]:
                subs.append(alt["value"])
        subst_map[k] = subs if subs else [k]
    return subst_map

# === 1. Load Dictionary ===
def load_entries() -> List[Entry]:
    path = get_config_paths()["dictionary"]
    return load_dictionary(str(path))

# === 2. Load Model ===
def load_fasttext():
    path = get_config_paths()["model_output"]
    return get_fasttext_model(path)

def load_sentence_model():
    return get_sentence_transformer("all-MiniLM-L6-v2")

# === 3. Normalize Word (Y = I, lowercase, etc) ===
def normalize_form(word):
    return word.lower()

# === 4. Generate Variants (just like training)
from itertools import product, combinations

def generate_variants(word, subst_map, max_subs=2):
    word = word.lower()
    variants = set()
    variants.add(word)

    for positions in combinations(range(len(word)), max_subs):
        for replacements in product(*[subst_map.get(word[i], [word[i]]) for i in positions]):
            temp = list(word)
            for idx, sub in zip(positions, replacements):
                temp[idx] = sub
            variant = "".join(temp)
            variants.add(variant)
    return list(variants)

# === 5. Semantic Similarity for Definitions ===
def definition_similarity(def1, def2, sentence_model):
    if not def1 or not def2:
        return 0.0
    emb1 = sentence_model.encode(def1, convert_to_tensor=True)
    emb2 = sentence_model.encode(def2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

# === 6. Find Similar Words ===
def find_semantically_similar_words(ft_model, sent_model, entries, target_word, subst_map, topn=11):
    fasttext_weight = 0.53
    definition_weight = 0.47
    
    normalized_query = normalize_form(target_word)
    variants = generate_variants(normalized_query, subst_map)

    target_entry = next((e for e in entries if normalize_form(e["normalized"]) == normalized_query), None)
    if not target_entry:
        print(f"[!] Word '{target_word}' not found in dictionary.")
        return []

    results = []
    for entry in entries:
        cand_norm = normalize_form(entry["normalized"])
        if cand_norm == normalized_query:
            continue

        # FastText similarity (max across variants)
        ft_score = 0.0
        if cand_norm in ft_model.wv:
            ft_score = max(
                (ft_model.wv.similarity(v, cand_norm) for v in variants if v in ft_model.wv),
                default=0.0
            )

        # Semantic similarity (definitions)
        def_score = definition_similarity(
            target_entry.get("definition", ""),
            entry.get("definition", ""),
            sent_model
        )

        # Combined weighted score
        final_score = (fasttext_weight * ft_score) + (definition_weight * def_score)

        results.append({
            "word": entry["word"],
            "normalized": entry["normalized"],
            "definition": entry.get("definition", ""),
            "fasttext": round(ft_score, 3),
            "semantic": round(def_score, 3),
            "score": round(final_score, 3),
            "levenshtein": levenshtein_distance(normalized_query, cand_norm),
            "citations": entry.get("key_citations", [])
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topn]

# === 7. CLI Entry Point ===
def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python inspect_model.py <word>")
        sys.exit(1)

    query = sys.argv[1]
    print(f"[+] Loading models and dictionary...")
    entries = load_entries()
    ft_model = load_fasttext()
    sent_model = load_sentence_model()
    subst_map = load_substitution_map()

    print(f"[+] Querying semantically similar roots for: '{query}'\n")
    results = find_semantically_similar_words(
        ft_model=ft_model,
        sent_model=sent_model,
        entries=entries,
        target_word=query,
        subst_map=subst_map
    )
    
    for r in results:
        print(f"\n🔹 {r['word']} (score: {r['score']} | FT: {r['fasttext']} | Sem: {r['semantic']} | Lev: {r['levenshtein']})")
        print(f"   ↳ {r['definition']}")
        for c in r["citations"]:
            print(f"     📜 {c['context']}")

    if not results:
        print("[!] No similar words found.")

if __name__ == "__main__":
    main()
