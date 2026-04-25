from __future__ import annotations

import sys
import types
from pathlib import Path
import importlib

import numpy as np


# Keep import-time dependencies lightweight for unit-level batching checks.
gensim_module = sys.modules.setdefault("gensim", types.ModuleType("gensim"))
gensim_models = sys.modules.setdefault("gensim.models", types.ModuleType("gensim.models"))


class _DummyFastText:  # pragma: no cover - import shim only
    @classmethod
    def load(cls, _path: str) -> "_DummyFastText":
        return cls()


gensim_models.FastText = _DummyFastText  # type: ignore[attr-defined]
gensim_module.models = gensim_models  # type: ignore[attr-defined]

sentence_module = sys.modules.setdefault(
    "sentence_transformers",
    types.ModuleType("sentence_transformers"),
)
sentence_module.util = getattr(  # type: ignore[attr-defined]
    sentence_module,
    "util",
    types.SimpleNamespace(cos_sim=lambda *_args, **_kwargs: [[1.0]]),
)


class _DummySentenceTransformer:  # pragma: no cover - import shim only
    def __init__(self, *args, **kwargs) -> None:
        pass


sentence_module.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_EMBEDDINGS_MODULE = "enochian_lm.root_extraction.utils.embeddings"
if (
    _EMBEDDINGS_MODULE in sys.modules
    and not hasattr(sys.modules[_EMBEDDINGS_MODULE], "_SENTENCE_EMBED_CACHE")
):
    sys.modules.pop(_EMBEDDINGS_MODULE)

embeddings_module = importlib.import_module(_EMBEDDINGS_MODULE)


class _BatchEmbedder:
    """Record encode calls so batching behavior can be asserted directly."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def encode(self, texts, normalize_embeddings: bool = True):  # noqa: ANN001
        self.calls.append(texts)
        if isinstance(texts, list):
            return np.array(
                [[float(index + 1), 0.0] for index in range(len(texts))],
                dtype=float,
            )
        return np.array([1.0, 0.0], dtype=float)


def test_cluster_definitions_batches_uncached_definition_encoding() -> None:
    """Encode cache misses in one batch call instead of per-definition calls."""

    embeddings_module._SENTENCE_EMBED_CACHE.clear()
    embedder = _BatchEmbedder()
    definitions = ["alpha", "beta", "gamma"]

    clusters = embeddings_module.cluster_definitions(
        definitions,
        model=embedder,
        similarity_threshold=0.99,
    )

    assert clusters
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == definitions
    assert ("paraphrase-MiniLM-L6-v2", "alpha") in embeddings_module._SENTENCE_EMBED_CACHE
    assert ("paraphrase-MiniLM-L6-v2", "beta") in embeddings_module._SENTENCE_EMBED_CACHE
    assert ("paraphrase-MiniLM-L6-v2", "gamma") in embeddings_module._SENTENCE_EMBED_CACHE

    embedder.calls.clear()
    embeddings_module.cluster_definitions(
        definitions,
        model=embedder,
        similarity_threshold=0.99,
    )
    assert embedder.calls == []
