from __future__ import annotations

import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lightweight stubs for heavy optional deps.
if "crewai" not in sys.modules:
    crewai_stub = types.ModuleType("crewai")

    class _Task:
        def __init__(self, description: str = "", expected_output: str = ""):
            self.description = description
            self.expected_output = expected_output

    class _Agent:
        def __init__(self, *args, **kwargs):
            pass

    class _Crew:
        def __init__(self, *args, **kwargs):
            pass

    crewai_stub.Task = _Task
    crewai_stub.Agent = _Agent
    crewai_stub.Crew = _Crew
    sys.modules["crewai"] = crewai_stub

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")
    st_stub.util = types.SimpleNamespace(cos_sim=lambda *_a, **_k: [[1.0]])
    sys.modules["sentence_transformers"] = st_stub

if "enochian_lm.root_extraction.tools.query_model_tool" not in sys.modules:
    tool_stub = types.ModuleType("enochian_lm.root_extraction.tools.query_model_tool")

    class _QueryModelTool:
        def __init__(self, *args, **kwargs):
            pass

    tool_stub.QueryModelTool = _QueryModelTool
    sys.modules["enochian_lm.root_extraction.tools.query_model_tool"] = tool_stub

if "enochian_lm.root_extraction.utils.types_lexicon" not in sys.modules:
    lex_stub = types.ModuleType("enochian_lm.root_extraction.utils.types_lexicon")
    lex_stub.EntryRecord = dict
    sys.modules["enochian_lm.root_extraction.utils.types_lexicon"] = lex_stub

embeddings_stub = sys.modules.get("enochian_lm.root_extraction.utils.embeddings")
if embeddings_stub is None:
    embeddings_stub = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
    sys.modules["enochian_lm.root_extraction.utils.embeddings"] = embeddings_stub

embeddings_stub.get_fasttext_model = getattr(
    embeddings_stub, "get_fasttext_model", lambda *a, **k: object()
)
embeddings_stub.get_sentence_transformer = getattr(
    embeddings_stub, "get_sentence_transformer", lambda *a, **k: object()
)
embeddings_stub.select_definitions = getattr(
    embeddings_stub, "select_definitions", lambda defs, max_words=300: defs
)
embeddings_stub.stream_text = getattr(
    embeddings_stub, "stream_text", lambda *a, **k: None
)

# Other subtraction tests install lightweight module stubs during collection.
# Drop those placeholders here so this file exercises the real formatter helpers.
sys.modules.pop("enochian_lm.root_extraction.tools.debate_semantic_subtraction_engine", None)
sys.modules.pop("enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine", None)

from enochian_lm.root_extraction.tools.debate_semantic_subtraction_engine import (  # noqa: E402
    _format_subtraction_guidance_compact as debate_compact,
)
from enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine import (  # noqa: E402
    _format_subtraction_guidance_compact as solo_compact,
)


def _sample_guidance() -> dict:
    return {
        "word_breaks": [
            {"host_word": "NAZPSAD", "root": "NAZ", "residual": "PSAD"},
            {"host_word": "NAZPSAD", "root": "NAZ", "residual": "PSAD"},  # duplicate triple
            {
                "host_word": "ANAZNAZ",
                "root": "NAZ",
                "residual": "A",
                "remove_all": True,
            },
            {"host_word": "ZANAZ", "root": "NAZ", "residual": "ZA"},
            {"host_word": "ANANAZ", "root": "NAZ", "residual": "ANA"},
            {"host_word": "QNAZ", "root": "NAZ", "residual": "Q"},
        ]
    }


def _assert_common_compact_rules(rendered: str) -> None:
    assert "Compact semantic-subtraction guidance" in rendered
    assert rendered.count("NAZPSAD - NAZ = PSAD") == 1
    assert "ANAZNAZ - NAZ = A (remove-all option)" in rendered
    # default cap is 5 unique equations
    assert len([line for line in rendered.splitlines() if line.startswith("-")]) == 5
    # no duplicate triple rendering format should appear
    assert "host=" not in rendered


def test_solo_compact_guidance_dedupes_and_limits_equations():
    rendered = solo_compact("NAZ", _sample_guidance())
    _assert_common_compact_rules(rendered)


def test_debate_compact_guidance_dedupes_and_limits_equations():
    rendered = debate_compact("NAZ", _sample_guidance())
    _assert_common_compact_rules(rendered)
