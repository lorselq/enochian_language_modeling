"""Regression coverage for QueryModelTool transport selection.

These tests protect the non-stream completion path used by short JSON-only
jobs. They intentionally load the real module under a private name so other
test files can keep their lightweight dependency stubs without hiding the
transport logic we want to exercise here.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_query_model_tool_module():
    """Load the real query-model module with tiny import-time stubs.

    What: import the production module directly from disk.
    Why: other translation tests intentionally register lightweight shims in
    ``sys.modules`` and this regression needs the real transport code instead.
    Big picture: protects the phrase-render transport contract from being
    masked by collection-time test scaffolding elsewhere in the suite.
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

    openai_module = sys.modules.setdefault("openai", types.ModuleType("openai"))
    openai_module.OpenAI = getattr(openai_module, "OpenAI", object)  # type: ignore[attr-defined]

    crewai_module = sys.modules.setdefault("crewai", types.ModuleType("crewai"))
    crewai_tools_module = types.ModuleType("crewai.tools")

    class _BaseTool:
        """Provide the tiny BaseTool surface QueryModelTool needs in tests."""

        def __init__(self, *args, **kwargs) -> None:
            self.name = kwargs.get("name", "Query LLM")
            self.description = kwargs.get("description", "")

    crewai_tools_module.BaseTool = _BaseTool  # type: ignore[attr-defined]
    crewai_module.tools = crewai_tools_module  # type: ignore[attr-defined]
    sys.modules["crewai.tools"] = crewai_tools_module

    pydantic_module = sys.modules.setdefault("pydantic", types.ModuleType("pydantic"))
    pydantic_module.PrivateAttr = lambda default=None, **_kwargs: default  # type: ignore[attr-defined]

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_query_model_tool_uses_non_stream_completion_when_requested(
    monkeypatch,
) -> None:
    """Use a plain completion when local callers explicitly opt out of streaming.

    Phrase bundle rendering only needs the final JSON payload, not token-by-
    token output. This regression proves the tool sends ``stream=False``,
    returns the final content, and avoids the first-token wait states used by
    the streaming transport.
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
    assert any(event.get("state") == "waiting for full response" for event in events)
    assert not any(event.get("state") == "waiting for first token" for event in events)


def test_query_model_tool_non_stream_returns_full_response(monkeypatch) -> None:
    """Allow remote phrase rendering to skip first-token waits with one call.

    Phrase bundle rendering uses ``stream_response=False`` so short JSON
    payloads do not sit around waiting for the first streamed content delta.
    This regression proves the transport sends ``stream=False`` and returns the
    full message body without trying to iterate a streaming object.
    """

    client_inits: list[dict[str, object]] = []
    request_calls: list[dict[str, object]] = []
    progress_events: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            client_inits.append(dict(kwargs))

            def _create(**request_kwargs):
                request_calls.append(dict(request_kwargs))
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"lay_translation":"plain english"}'
                            )
                        )
                    ]
                )

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_create)
            )

    query_model_tool_module = _load_query_model_tool_module()
    monkeypatch.setattr(query_model_tool_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("REMOTE_OPENAI_API_BASE", "https://example.test")
    monkeypatch.setenv("REMOTE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REMOTE_MODEL_NAME", "test-model")

    tool = query_model_tool_module.QueryModelTool(
        system_prompt="You are terse.",
        use_remote=True,
        progress_style="silent",
        local_fallback_enabled=False,
        stream_response=False,
    )
    tool._db = None
    tool._run_id = None
    tool._progress_callback = None
    tool._current_attempt = 0
    tool._current_source = "remote"
    tool._heartbeat_stop = None
    tool._heartbeat_thread = None

    result = tool._run(
        prompt="translate this",
        progress_callback=lambda event: progress_events.append(dict(event)),
    )

    assert result["response_text"] == '{"lay_translation":"plain english"}'
    assert client_inits
    assert request_calls[0]["stream"] is False
    assert request_calls[0]["model"] == "test-model"
    assert [event["state"] for event in progress_events] == [
        "queued remote request",
        "connecting",
        "queued request with provider",
        "waiting for full response",
        "received full response",
        "response complete",
    ]
