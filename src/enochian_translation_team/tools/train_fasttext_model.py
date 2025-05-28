import json
import os
from gensim.models import FastText
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.variant_utils import (
    normalize_word,
    apply_sequence_compressions,
    generate_variants,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_enochian_words(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry for entry in data if entry.get("normalized")]


def prepare_training_data(entries, subst_map, compression_rules):
    all_variants = []
    for entry in entries:
        base = entry["normalized"].lower()
        norm = normalize_word(base, subst_map)
        norm = apply_sequence_compressions(norm, compression_rules)
        variants = generate_variants(norm, subst_map, max_subs=3)
        ngram_lists = [
            [variant[i : i + 3] for i in range(len(variant) - 2)]
            for variant in variants
            if len(variant) >= 3
        ]
        all_variants.extend(ngram_lists)
    return all_variants


def train_fasttext_model(sentences):
    model = FastText(vector_size=50, window=3, min_count=1, workers=2, sg=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=100)
    return model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(str(path))


def load_model(path):
    return FastText.load(path)


def main():
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

    print(
        "[✓] Done. Use inspect_model.py to examine your weird beautiful word-children."
    )


if __name__ == "__main__":
    main()
