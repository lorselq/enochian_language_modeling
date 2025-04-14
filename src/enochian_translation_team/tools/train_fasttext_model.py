import json
import os
from gensim.models import FastText
from itertools import product, combinations
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

# === 1. Load JSON Words ===
def load_enochian_words(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry for entry in data if entry.get("normalized")]  # skip blank normalized entries

# === 2. Variant Generator ===
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

# === 3. Prepare Training Data ===
def prepare_training_data(entries):
    subst_map = establish_subst_map()
    all_variants = []
    for entry in entries:
        base = entry["normalized"].lower()
        variants = generate_variants(base, subst_map, max_subs=2)
        ngram_lists = [[variant[i:i+3] for i in range(len(variant)-2)] for variant in variants if len(variant) >= 3]
        all_variants.extend(ngram_lists)
    return all_variants

# === 3. Train FastText ===
def train_fasttext_model(sentences):
    model = FastText(vector_size=50, window=3, min_count=1, workers=2, sg=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100)
    return model

# === 4. Save & Load Model ===
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(str(path))

def load_model(path):
    return FastText.load(path)

# === 6. Main ===
if __name__ == "__main__":
    paths = get_config_paths()

    print("[+] Loading corpus...")
    entries = load_enochian_words(paths["dictionary"])
    print(f"[+] Loaded {len(entries)} usable words.")

    print("[+] Preparing training data...")
    training_data = prepare_training_data(entries)
    print(f"[+] Generated {len(training_data)} variant n-gram samples.")

    print("[+] Training FastText model...")
    model = train_fasttext_model(training_data)

    print(f"[+] Saving model to {paths['model_output']}...")
    save_model(model, paths["model_output"])

    print("[✓] Done. Insights generated. Now avail yourself of inspect_model.py to behold and wonder at what we've wrought.")
