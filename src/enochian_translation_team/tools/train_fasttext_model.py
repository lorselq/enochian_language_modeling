import os
import json
import hashlib
import logging
import random
import numpy as np
from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.dictionary_loader import load_dictionary, Entry

# --- Setup logging ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Data classes for hyperparameters ---
@dataclass
class FastTextParams:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    sg: int = 1
    min_n: int = 3
    max_n: int = 6
    epochs: int = 30
    alpha: float = 0.05
    seed: int = 42


# --- Corpus & caching helpers ---
def hash_entries(entries: List[Entry]) -> str:
    # deterministic JSON of the entries (canonical + alternates + senses)
    simplified = [
        {
            "canonical": e.canonical,
            "alts": sorted([a.value for a in e.alternates]),
            "senses": sorted([s.definition for s in (e.senses or [])]),
        }
        for e in entries
    ]
    return hashlib.md5(json.dumps(simplified, sort_keys=True).encode()).hexdigest()


# --- Sentence generator ---
def load_sentences(entries: List[Entry]) -> List[List[str]]:
    sentences: list[list[str]] = []
    for e in entries:
        # context window: canonical + all senses + all alternates
        parts = [e.canonical]
        parts += [s.definition for s in (e.senses or [])]
        parts += [a.value for a in e.alternates]
        # join into synthetic sentence
        sent = " ".join(parts)
        # tokenize using gensim's simple_preprocess (lowercase, remove punctuation/accents)
        tokens = simple_preprocess(sent, deacc=True, min_len=1)
        if tokens:
            sentences.append(tokens)
    return sentences


# --- Training & caching ---


def train_fasttext_model(
    entries: List[Entry], out_path: Path, params: FastTextParams
) -> Word2Vec:
    # reproducibility
    random.seed(params.seed)
    np.random.seed(params.seed)

    # corpus hashing for cache
    corpus_hash = hash_entries(entries)
    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("hash") == corpus_hash and meta.get("params") == asdict(params):
            logger.info("Loading existing FastText model (cache hit)")
            return FastText.load(str(out_path))

    # prepare data
    sentences = load_sentences(entries)
    logger.info(f"Corpus: {len(sentences)} sentences, vocab sample: {sentences[:2]}")

    # build & train
    model = FastText(
        vector_size=params.vector_size,
        window=params.window,
        min_count=params.min_count,
        sg=params.sg,
        min_n=params.min_n,
        max_n=params.max_n,
        alpha=params.alpha,
        seed=params.seed,
        workers=os.cpu_count() or 1,
    )
    model.build_vocab(sentences)
    logger.info(f"Vocab size: {len(model.wv)}")

    for epoch in range(params.epochs):
        model.train(sentences, total_examples=len(sentences), epochs=1)
        logger.info(f"Epoch {epoch+1}/{params.epochs} complete")

    # save model + metadata
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    meta = {"hash": corpus_hash, "params": asdict(params)}
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Model and metadata saved to {out_path}")

    return model


# --- Main script ---


def main():
    paths = get_config_paths()
    dict_path = Path(paths["dictionary"])
    model_path = Path(paths.get("model_output", "fasttext.bin"))

    logger.info("Loading dictionary entries...")
    entries = load_dictionary(str(dict_path))
    logger.info(f"Loaded {len(entries)} entries")

    params = FastTextParams()
    logger.info(f"Training with params: {params}")

    model = train_fasttext_model(entries, model_path, params)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
