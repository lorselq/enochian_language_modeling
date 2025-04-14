import sys
import json
from gensim.models import FastText
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
from enochian_translation_team.utils.config import get_config_paths

# === 0. Establish substitution map for later reference ===
def establish_subst_map():
    return {
        "a": ["a", "un"],
        "b": ["b", "be", "bi", "pa", "pah"],
        "c": ["c", "t", "veh", "ve"],
        "d": ["d", "de", "di"],
        "e": ["e", "graph", "graf"],
        "f": ["f", "ef", "or"],
        "g": ["g", "ge", "ged"],
        "h": ["h", "na", "nahath"],
        "i": ["i", "y", "j", "gon"],
        "l": ["l", "el", "ur"],
        "m": ["m", "em", "tal"],
        "n": ["n", "en", "drux", "drun", "dru"],
        "o": ["o", "med"],
        "p": ["p", "mal", "mals"],
        "q": ["q", "qu", "qua", "ger"],
        "s": ["s", "es", "fam"],
        "t": ["t", "c", "gisg"],
        "u": ["u", "v", "w", "van", "uan"],
        "v": ["v", "u", "w", "van", "uan"],
        "x": ["x", "ex", "pal"],
        "z": ["z", "zod", "ceph", "cef"]
    }

# === 1. Load Dictionary ===
def load_entries():
    path = get_config_paths()["dictionary"]
    with open(path, "r", encoding="utf-8") as f:
        return [e for e in json.load(f) if e.get("normalized")]

# === 2. Load Model ===
def load_fasttext():
    path = get_config_paths()["model_output"]
    return FastText.load(str(path))

def load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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
def find_semantically_similar_words(ft_model, sent_model, entries, target_word, topn=10):
    fasttext_weight = 0.6
    definition_weight = 0.4
    
    subst_map = establish_subst_map()
    normalized_query = normalize_form(target_word)
    variants = generate_variants(normalized_query, subst_map, max_subs=2)

    target_entry = next((e for e in entries if normalize_form(e["normalized"]) == normalized_query), None)
    if not target_entry:
        print(f"[!] Word '{target_word}' not found in dictionary.")
        return []

    results = []
    for entry in entries:
        cand_norm = normalize_form(entry["normalized"])
        if cand_norm == normalized_query:
            continue

        # Check if candidate word is a variant match
        variant_score = max((ft_model.wv.similarity(v, cand_norm)
                             for v in variants if v in ft_model.wv and cand_norm in ft_model.wv), default=0.0)

        # Semantic definition similarity
        def_score = definition_similarity(target_entry.get("definition", ""), entry.get("definition", ""), sent_model)

        # Combined score
        final_score = (fasttext_weight * variant_score) + (definition_weight * def_score)

        results.append({
            "word": entry["word"],
            "normalized": entry["normalized"],
            "definition": entry.get("definition", ""),
            "fasttext": round(variant_score, 3),
            "semantic": round(def_score, 3),
            "score": round(final_score, 3),
            "levenshtein": levenshtein_distance(normalized_query, cand_norm)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topn]

# === 7. CLI Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: poetry run python inspect_model.py <word>")
        sys.exit(1)

    query = sys.argv[1]
    print(f"[+] Loading models and dictionary...")
    entries = load_entries()
    ft_model = load_fasttext()
    sent_model = load_sentence_model()

    print(f"[+] Querying semantically similar roots for: '{query}'\n")
    results = find_semantically_similar_words(ft_model, sent_model, entries, query)

    for r in results:
        print(f"  - {r['normalized']} (score: {r['score']}, fast: {r['fasttext']}, sem: {r['semantic']}, lev: {r['levenshtein']}): {r['definition']}")

    if not results:
        print("[!] No similar words found.")
