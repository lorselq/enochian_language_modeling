from __future__ import annotations

import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _Task:
    """Provide the tiny Task surface needed to render engine prompts.

    Why: prompt regression tests should not depend on CrewAI being installed.
    How: the engines only read description and expected_output in this path.
    Responsibility: keep imports lightweight while exercising real prompt text.
    """

    def __init__(self, description: str = "", expected_output: str = ""):
        self.description = description
        self.expected_output = expected_output


class _CapturingQueryModelTool:
    """Capture rendered prompts while avoiding real LLM calls.

    Why: this suite validates prompt wording, not model behavior.
    How: engine calls are allowed to proceed until _run, where this fake stores
    the prompt and returns deterministic JSON.
    Responsibility: make solo prompt rendering testable without network access.
    """

    calls: list[dict[str, str]] = []

    def __init__(self, *args, **kwargs):
        self.system_prompt = str(kwargs.get("system_prompt", ""))

    def attach_logging(self, *_args, **_kwargs) -> None:
        """Ignore logging attachment in prompt-only tests.

        Why: production tools can attach SQLite logging, but these tests use no
        run database.
        How: accept the call and do nothing.
        Responsibility: preserve the engine call contract for lightweight tests.
        """

    def _run(self, prompt: str, **_kwargs) -> dict[str, str]:
        """Return deterministic JSON after recording the prompt.

        Why: solo engines expect a model-shaped response after prompt assembly.
        How: store the prompt and return a minimal accepted JSON payload.
        Responsibility: stop execution before any external model call.
        """

        self.calls.append({"system_prompt": self.system_prompt, "prompt": prompt})
        return {
            "response_text": '{"ROOT":"IO","EVALUATION":"accepted","DEFINITION":"manifested distress"}',
            "gloss_model": "test-model",
        }


def _install_lightweight_stubs() -> None:
    """Install import stubs for optional prompt-engine dependencies.

    Why: the prompt engines import CrewAI, embeddings, and model tooling at
    module import time.
    How: register minimal modules before importing the engine modules.
    Responsibility: isolate prompt tests from heavyweight runtime packages.
    """

    crewai_stub = types.ModuleType("crewai")
    crewai_stub.Task = _Task
    crewai_stub.Agent = type("_Agent", (), {"__init__": lambda self, *a, **k: None})
    crewai_stub.Crew = type("_Crew", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["crewai"] = crewai_stub

    st_stub = types.ModuleType("sentence_transformers")
    st_stub.util = types.SimpleNamespace(cos_sim=lambda *_a, **_k: [[1.0]])
    sys.modules["sentence_transformers"] = st_stub

    tool_stub = types.ModuleType("enochian_lm.root_extraction.tools.query_model_tool")
    tool_stub.QueryModelTool = _CapturingQueryModelTool
    sys.modules["enochian_lm.root_extraction.tools.query_model_tool"] = tool_stub

    lex_stub = types.ModuleType("enochian_lm.root_extraction.utils.types_lexicon")
    lex_stub.AltRecord = dict
    lex_stub.EntryRecord = dict
    lex_stub.SenseRecord = dict
    sys.modules["enochian_lm.root_extraction.utils.types_lexicon"] = lex_stub

    embeddings_stub = types.ModuleType("enochian_lm.root_extraction.utils.embeddings")
    embeddings_stub.get_sentence_transformer = lambda *a, **k: object()
    embeddings_stub.select_definitions = lambda defs, max_words=300: defs
    embeddings_stub.stream_text = lambda *a, **k: None
    embeddings_stub.get_fasttext_model = lambda *a, **k: object()
    embeddings_stub.get_sentence_transformer_if_available = lambda *a, **k: object()
    embeddings_stub.cluster_definitions = lambda *a, **k: []
    embeddings_stub.cluster_definition_counts = lambda *a, **k: {}
    sys.modules["enochian_lm.root_extraction.utils.embeddings"] = embeddings_stub

    for name in (
        "enochian_lm.root_extraction.tools.solo_analysis_engine",
        "enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine",
    ):
        sys.modules.pop(name, None)


_install_lightweight_stubs()

from enochian_lm.root_extraction.tools.solo_analysis_engine import (  # noqa: E402
    solo_agent_ngram_analysis,
)
from enochian_lm.root_extraction.tools.solo_semantic_subtraction_engine import (  # noqa: E402
    solo_semantic_subtraction,
)

TOOLS_ROOT = SRC_ROOT / "enochian_lm" / "root_extraction" / "tools"
PROMPT_FILES = [
    TOOLS_ROOT / "solo_analysis_engine.py",
    TOOLS_ROOT / "solo_semantic_subtraction_engine.py",
    TOOLS_ROOT / "debate_engine.py",
    TOOLS_ROOT / "debate_semantic_subtraction_engine.py",
]


def _ohio_candidate() -> dict[str, object]:
    """Return an OHIO candidate that exposes the valence-laundering failure.

    Why: the original prompt tended to flatten OHIO = woe into generic state
    language.
    How: provide the same kind of dictionary/citation evidence seen in the DB.
    Responsibility: make rendered prompts carry a forceful hardship example.
    """

    return {
        "word": "OHIO",
        "normalized": "ohio",
        "definition": "woe",
        "enhanced_definition": "woe. usage: `woe, woe...yea, woe be to the earth.`",
        "fasttext": 0.91,
        "semantic": 0.84,
        "tier": "Very strong connection",
    }


def _assert_opinionated_prompt(prompt: str) -> None:
    """Assert that rendered prompt text preserves concrete evidence valence.

    Why: the regression is prompt wording that over-abstracts difficult glosses.
    How: check for the new style directive and field-level anti-laundering text.
    Responsibility: keep future prompt edits from reintroducing bland glosses.
    """

    assert "OPINIONATED GLOSS STYLE" in prompt
    assert "preserve negative, painful, judgmental, affective, or hardship valence" in prompt
    assert "woe-manifestation" in prompt
    assert 'bare phrases like "state of being"' in prompt
    assert "concrete effect of IO on OHIO" in prompt
    assert "valence-bearing lemmas" in prompt
    assert "no negatives" not in prompt
    assert "non-negative definition" not in prompt


def test_solo_residual_prompt_preserves_hardship_valence() -> None:
    """Render the residual solo prompt and verify the new gloss policy.

    Why: residual solo analysis is the path most likely to analyze OHIO-style
    leftover fragments.
    How: run the real prompt assembly with fake model tooling and inspect the
    persisted Glossator_Prompt.
    Responsibility: ensure the prompt asks for concrete, opinionated glosses.
    """

    _CapturingQueryModelTool.calls.clear()
    result = solo_semantic_subtraction(
        root="IO",
        candidates=[_ohio_candidate()],
        stats_summary="OHIO appears as woe with strong semantic support.",
        use_remote=False,
    )

    _assert_opinionated_prompt(str(result["Glossator_Prompt"]))


def test_classic_solo_prompt_preserves_hardship_valence() -> None:
    """Render the classic solo prompt and verify matching gloss policy.

    Why: classic solo and residual solo should not diverge in definition style.
    How: run the real classic solo prompt assembly with fake model tooling.
    Responsibility: keep non-residual solo glosses concrete and evidence-led.
    """

    _CapturingQueryModelTool.calls.clear()
    result = solo_agent_ngram_analysis(
        root="IO",
        candidates=[_ohio_candidate()],
        stats_summary="OHIO appears as woe with strong semantic support.",
        use_remote=False,
    )

    _assert_opinionated_prompt(str(result["Glossator_Prompt"]))


def test_debate_prompt_files_include_opinionated_gloss_policy() -> None:
    """Verify active debate prompt files carry the same direct directive.

    Why: debate prompts are expensive to render fully, but the source strings are
    the prompt contract the engines pass to the model.
    How: read the two debate engine files and assert the direct policy appears.
    Responsibility: ensure solo and debate modes stay stylistically aligned.
    """

    for file_name in ("debate_engine.py", "debate_semantic_subtraction_engine.py"):
        source = (TOOLS_ROOT / file_name).read_text()
        assert "OPINIONATED GLOSS STYLE" in source
        assert "OHIO = woe" in source
        assert "woe-manifestation" in source
        assert "valence-bearing lemmas" in source


def test_active_prompt_files_do_not_use_old_negative_suppression() -> None:
    """Guard active prompt files against the old negative-suppression wording.

    Why: phrases like "no negatives" caused evidence-backed hardship meanings to
    be softened away.
    How: scan only active prompt files, excluding archived scratch material.
    Responsibility: catch future regressions in prompt wording.
    """

    for prompt_file in PROMPT_FILES:
        source = prompt_file.read_text()
        assert "no negatives" not in source
        assert "non-negative definition" not in source
