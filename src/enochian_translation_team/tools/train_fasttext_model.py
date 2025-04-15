import json
import os
from gensim.models import FastText
from itertools import product, combinations
from enochian_translation_team.utils.config import get_config_paths

# === 0. Load Substitution and Compression Maps ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === 1. Load JSON Words ===
def load_enochian_words(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry for entry in data if entry.get("normalized")]

# === 2. Normalize Words Using Substitution Map ===
def normalize_word(word, subst_map):
    normalized = ""
    for char in word:
        if char in subst_map and any(
            alt["direction"] in ["from", "both"] for alt in subst_map[char]["alternates"]
        ):
            normalized += subst_map[char]["canonical"]
        else:
            normalized += char
    return normalized

# === 3. Apply Sequence Compressions ===
def apply_sequence_compressions(word, compression_rules):
    for rule in compression_rules:
        if rule["direction"] in ["from", "both"] and rule["from"] in word:
            word = word.replace(rule["from"], rule["to"])
    return word

# === 4. Generate Variants ===
def generate_variants(word, subst_map, max_subs=2):
    word = word.lower()
    variants = set()
    variants.add(word)

    # Build simple substitution lookup
    sub_dict = {
        k: [alt["value"] for alt in v["alternates"] if alt["direction"] in ["to", "both"]]
        for k, v in subst_map.items()
    }

    for positions in combinations(range(len(word)), max_subs):
        replacement_options = []
        for i in positions:
            char = word[i]
            replacement_options.append(sub_dict.get(char, [char]))

        for replacements in product(*replacement_options):
            temp = list(word)
            for idx, sub in zip(positions, replacements):
                temp[idx] = sub
            variant = "".join(temp)
            variants.add(variant)
    return list(variants)

# === 5. Prepare Training Data ===
def prepare_training_data(entries, subst_map, compression_rules):
    all_variants = []
    for entry in entries:
        base = entry["normalized"].lower()
        norm = normalize_word(base, subst_map)
        norm = apply_sequence_compressions(norm, compression_rules)
        variants = generate_variants(norm, subst_map, max_subs=2)
        ngram_lists = [
            [variant[i : i + 3] for i in range(len(variant) - 2)]
            for variant in variants
            if len(variant) >= 3
        ]
        all_variants.extend(ngram_lists)
    return all_variants

# === 6. Train FastText ===
def train_fasttext_model(sentences):
    model = FastText(vector_size=50, window=3, min_count=1, workers=2, sg=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100)
    return model

# === 7. Save & Load Model ===
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(str(path))

def load_model(path):
    return FastText.load(path)

# === 8. Main ===
if __name__ == "__main__":
    paths = get_config_paths()

    print("[+] Loading corpus...")
    entries = load_enochian_words(paths["dictionary"])
    print(f"[+] Loaded {len(entries)} usable words.")

    print("[+] Loading substitution and compression rules...")
    subst_map = load_json(paths["substitution_map"])
    compression_rules = load_json(paths["sequence_compressions"])

    print("[+] Preparing training data...")
    training_data = prepare_training_data(entries, subst_map, compression_rules)
    print(f"[+] Generated {len(training_data)} variant n-gram samples.")

    print("[+] Training FastText model...")
    model = train_fasttext_model(training_data)

    print(f"[+] Saving model to {paths['model_output']}...")
    save_model(model, paths["model_output"])

    print("[✓] Done. Use inspect_model.py to examine your weird beautiful word-children.")