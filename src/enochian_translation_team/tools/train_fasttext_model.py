import os
import json
import hashlib
import logging
import random
import numpy as np
from typing import List, Optional, Sequence
from pathlib import Path
from dataclasses import dataclass, asdict
from gensim.models import FastText, Word2Vec
from gensim.utils import simple_preprocess
from enochian_translation_team.utils.config import get_config_paths
from enochian_translation_team.utils.dictionary_loader import (
    load_dictionary,
    load_dictionary_v2,
)
from enochian_translation_team.utils.types_lexicon import EntryRecord, AltRecord, SenseRecord

# --- Setup logging ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Data classes ---
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
def hash_entries(entries: Sequence[EntryRecord]) -> str:
    """Stable hash for caching—works for both legacy and v2-adapted entries."""
    import hashlib, json as _json
    normalized = [
        {
            "canonical": e.canonical,
            "alts": sorted([a.value for a in (getattr(e, "alternates", []) or [])]),
            "senses": sorted([s.definition for s in (getattr(e, "senses", []) or [])]),
        }
        for e in entries
    ]
    payload = _json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Sentence generator ---
def load_sentences(entries: Sequence[EntryRecord]) -> List[List[str]]:
    """
    Turn entries into short 'sentences' for FastText by concatenating
    canonical, alternates, and all definitions as tokens.
    """
    sents: List[List[str]] = []
    for e in entries:
        parts: List[str] = []
        if getattr(e, "canonical", None):
            parts.append(e.canonical)
        parts += [a.value for a in (getattr(e, "alternates", []) or [])]
        parts += [s.definition for s in (getattr(e, "senses", []) or [])]
        parts = [p for p in (parts or []) if p]
        if parts:
            sents.append(parts)
    return sents


# --- Training & caching ---


def train_fasttext_model(
    entries: Sequence[EntryRecord], out_path: Path, params: FastTextParams
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
    entries = None
    try:
        # If your legacy loader still works with v1 files, this will succeed.
        entries = load_dictionary(str(dict_path))
        logger.info("Loaded dictionary via legacy loader (v1).")
    except Exception as ex:
        logger.info(f"Legacy loader failed ({ex}); using v2 adapter.")
        entries = load_dictionary_v2(str(dict_path))

    logger.info(f"Loaded {len(entries)} entries")
    sentences = load_sentences(entries)
    logger.info(f"Prepared {len(sentences)} training sentences")

    params = FastTextParams()
    logger.info(f"Training with params: {params}")

    model = train_fasttext_model(entries, model_path, params)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
