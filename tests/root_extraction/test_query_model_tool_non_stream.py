"""Regression coverage for QueryModelTool transport selection.

These tests protect the phrase-translation fix for remote bundle rendering.
That path now prefers a plain completion over streaming because the bundle
request only needs one short JSON blob, and streaming could sit for minutes
waiting on the first visible content chunk.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_query_model_tool_module():
    """Load the real query-model module even if other tests install shims.

    The large translation regression file installs lightweight dependency
    shims into ``sys.modules`` during import. Loading the module directly from
    disk under a private test-only name keeps this regression focused on the
    actual implementation we want to verify.
    """

    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "enochian_lm"
        / "root_extraction"
        / "tools"
        / "query_model_tool.py"
    )
    spec = importlib.util.spec_from_file_location(
        "query_model_tool_non_stream_actual",
        module_path,
    )
    assert spec is not None and spec.loader is not None

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = object  # type: ignore[attr-defined]
    sys.modules["openai"] = openai_module

    crewai_module = sys.modules.setdefault("crewai", types.ModuleType("crewai"))
    crewai_tools_module = types.ModuleType("crewai.tools")

    class _BaseTool:
        def __init__(self, *args, **kwargs) -> None:
            self.name = kwargs.get("name", "Query LLM")
            self.description = kwargs.get("description", "")

    crewai_tools_module.BaseTool = _BaseTool  # type: ignore[attr-defined]
    crewai_module.tools = crewai_tools_module  # type: ignore[attr-defined]
    sys.modules["crewai.tools"] = crewai_tools_module

    pydantic_module = sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))
    pydantic_module.PrivateAttr = lambda default=None, **kwargs: default  # type: ignore[attr-defined]

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_model_tool_uses_non_stream_completion_when_requested(
    monkeypatch,
) -> None:
    """Use a plain completion when callers opt out of streaming.

    Phrase bundle rendering only needs the final JSON payload, not token-by-
    token output. This regression proves the tool sends ``stream=False``,
    returns the final content, and reports non-stream progress states instead
    of hanging on ``waiting for first token``.
    """

    calls: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(dict(kwargs))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"answer":"ok"}')
                    )
                ]
            )

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    query_model_tool_module = _load_query_model_tool_module()
    monkeypatch.setattr(query_model_tool_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("LOCAL_OPENAI_API_BASE", "http://localhost:1234")
    monkeypatch.setenv("LOCAL_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "test-model")

    tool = query_model_tool_module.QueryModelTool(
        system_prompt="Return JSON only.",
        use_remote=False,
        progress_style="silent",
        stream_response=False,
    )

    response = tool._run(
        prompt="Say ok.",
        progress_callback=lambda event: events.append(dict(event)),
    )

    assert response["response_text"] == '{"answer":"ok"}'
    assert calls[0]["stream"] is False
    assert any(
        event.get("state") == "waiting for non-stream response" for event in events
    )
    assert not any(event.get("state") == "waiting for first token" for event in events)
