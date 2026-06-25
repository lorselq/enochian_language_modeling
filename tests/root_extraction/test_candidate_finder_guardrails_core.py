from __future__ import annotations

import importlib
import pathlib
import random
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _TinyArray(list):
    """Provide the minimal ndarray behavior shared root tests expect.

    Why: candidate-finder guardrail tests stub numpy when optional scientific
    dependencies are unavailable, and later tests may reuse that module.
    How: support only simple numeric vector/matrix operations used by the test
    suite: shape, ndim, copy, addition, division, and dot inputs.
    Responsibility: keep the lightweight import shim from polluting later tests
    with a completely empty numpy module.
    """

    @property
    def ndim(self) -> int:
        return 2 if self and isinstance(self[0], (list, _TinyArray)) else 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self.ndim == 2:
            return (len(self), len(self[0]) if self else 0)
        return (len(self),)

    def copy(self) -> "_TinyArray":
        return _tiny_array([item.copy() if isinstance(item, _TinyArray) else item for item in self])

    def __truediv__(self, value: float) -> "_TinyArray":
        return _tiny_array([float(item) / float(value) for item in self])

    def __add__(self, other: object) -> "_TinyArray":
        return _tiny_array([float(left) + float(right) for left, right in zip(self, other, strict=False)])


def _tiny_array(values: object, dtype: object | None = None) -> _TinyArray:
    """Build a minimal numpy.array stand-in for numeric test vectors.

    Why: batching tests use numpy.array to construct deterministic embeddings.
    How: recursively wraps list-like values in _TinyArray and casts scalars to
    float when requested by the caller.
    Responsibility: emulate only the tiny vector behavior required here.
    """

    del dtype
    if isinstance(values, _TinyArray):
        return values
    if isinstance(values, (list, tuple)):
        return _TinyArray([_tiny_array(item) if isinstance(item, (list, tuple)) else float(item) for item in values])
    return _TinyArray([float(values)])


def _tiny_dot(left: object, right: object) -> float:
    """Compute a dot product for the tiny numpy shim.

    Why: clustering code scores vectors with np.dot.
    How: coerce both inputs to simple iterables of floats and multiply pairs.
    Responsibility: keep existing tests independent of a real numpy install.
    """

    return sum(float(a) * float(b) for a, b in zip(left, right, strict=False))


def _install_tiny_numpy_stub() -> None:
    """Install a small numpy stub only when numpy is otherwise absent.

    Why: an empty numpy module breaks later root-extraction tests that expect
    array and isscalar.
    How: register the minimum API surface consumed by current unit tests.
    Responsibility: avoid cross-test pollution from optional dependency stubs.
    """

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.ndarray = _TinyArray
    numpy_mod.bool_ = bool
    numpy_mod.array = _tiny_array
    numpy_mod.dot = _tiny_dot
    numpy_mod.isscalar = lambda value: isinstance(value, (str, bytes, int, float, complex, bool))
    sys.modules["numpy"] = numpy_mod


def _import_candidate_finder_with_lightweight_stubs():
    """Import candidate_finder without optional heavy deps.

    What: inject tiny module stubs for optional runtime deps used at import time.
    Why: guardrail tests must always run in CI even when scientific/NLP deps are absent.
    Big picture: protects scoring invariants from being silently skipped.
    """

    if "numpy" not in sys.modules:
        _install_tiny_numpy_stub()

    if "gensim" not in sys.modules:
        gensim_mod = types.ModuleType("gensim")
        gensim_utils = types.ModuleType("gensim.utils")
        gensim_utils.simple_preprocess = lambda text: str(text or "").split()
        gensim_mod.utils = gensim_utils
        sys.modules["gensim"] = gensim_mod
        sys.modules["gensim.utils"] = gensim_utils

    if "rapidfuzz" not in sys.modules:
        rapidfuzz_mod = types.ModuleType("rapidfuzz")
        rapidfuzz_mod.process = types.SimpleNamespace(extract=lambda *_a, **_k: [])
        rapidfuzz_mod.fuzz = types.SimpleNamespace(ratio=lambda *_a, **_k: 0)
        sys.modules["rapidfuzz"] = rapidfuzz_mod

    if "enochian_lm.root_extraction.utils.types_lexicon" not in sys.modules:
        lex_mod = types.ModuleType("enochian_lm.root_extraction.utils.types_lexicon")
        lex_mod.AltRecord = dict
        lex_mod.EntryRecord = dict
        lex_mod.SenseRecord = dict
        sys.modules["enochian_lm.root_extraction.utils.types_lexicon"] = lex_mod

    if "enochian_lm.root_extraction.utils.embeddings" not in sys.modules:
        emb_mod = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
        emb_mod.cluster_definitions = lambda *_a, **_k: []
        emb_mod.cluster_definition_counts = lambda *_a, **_k: {}
        emb_mod.get_fasttext_model = lambda *_a, **_k: object()
        emb_mod.get_sentence_transformer = lambda *_a, **_k: object()
        emb_mod.get_sentence_transformer_if_available = (
            lambda *_a, **_k: object()
        )
        emb_mod.select_definitions = lambda definitions, max_words=300: definitions
        emb_mod.stream_text = lambda *_a, **_k: None
        sys.modules["enochian_lm.root_extraction.utils.embeddings"] = emb_mod

    mod = importlib.import_module("enochian_lm.root_extraction.utils.candidate_finder")
    return mod.MorphemeCandidateFinder


MorphemeCandidateFinder = _import_candidate_finder_with_lightweight_stubs()


def _finder_with_bonus(bonus: float = 0.3) -> MorphemeCandidateFinder:
    finder = MorphemeCandidateFinder.__new__(MorphemeCandidateFinder)
    finder.multi_segment_bonus = bonus
    finder.last_candidate_diagnostics = []
    return finder


def test_redundant_split_does_not_outscore_less_redundant_split() -> None:
    finder = _finder_with_bonus(0.3)
    base = {
        "segments": [{"canonical": "NAZ"}, {"canonical": "PSAD"}],
        "coverage_ratio": 1.0,
        "residual_ratio": 0.0,
    }
    redundant = {
        "segments": [{"canonical": "NAZ"}, {"canonical": "NAZ"}, {"canonical": "PSAD"}],
        "coverage_ratio": 1.0,
        "residual_ratio": 0.0,
    }

    base_score = finder._score_with_bonus(1.0, base)
    redundant_score = finder._score_with_bonus(1.0, redundant)

    assert redundant_score <= base_score


def test_bonus_not_applied_without_full_coverage() -> None:
    finder = _finder_with_bonus(0.3)
    partial = {
        "segments": [{"canonical": "NAZ"}, {"canonical": "PS"}],
        "coverage_ratio": 0.75,
        "residual_ratio": 0.25,
    }

    assert finder._score_with_bonus(1.0, partial) == 1.0


def test_piece_count_score_corr_stays_near_zero_when_independent() -> None:
    """Sanity-check correlation helper behavior on deterministic synthetic rows."""

    finder = _finder_with_bonus(0.3)
    rng = random.Random(7)
    finder.last_candidate_diagnostics = [
        {
            "piece_count": rng.randint(1, 6),
            "unknown_piece_count": rng.randint(0, 4),
            "final_score": rng.random(),
        }
        for _ in range(300)
    ]

    summary = finder.summarize_last_run_diagnostics()
    assert abs(float(summary["piece_count_score_corr"])) < 0.2
