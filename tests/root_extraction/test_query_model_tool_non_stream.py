<<<<<<< HEAD
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
=======
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "crewai.tools" not in sys.modules:
    crewai_stub = types.ModuleType("crewai")
    crewai_tools_stub = types.ModuleType("crewai.tools")

    class _BaseTool:
        """Provide the tiny BaseTool surface QueryModelTool needs in tests.

        The production class inherits from CrewAI's tool base, but this
        regression only needs constructor storage so the real transport logic
        can be exercised without pulling in the whole optional dependency.
        """

        def __init__(self, *args, **kwargs) -> None:
            self.name = kwargs.get("name")
            self.description = kwargs.get("description")

    crewai_tools_stub.BaseTool = _BaseTool
    crewai_stub.tools = crewai_tools_stub
    sys.modules["crewai"] = crewai_stub
    sys.modules["crewai.tools"] = crewai_tools_stub

from enochian_lm.root_extraction.tools import query_model_tool as query_model_tool_module

if not hasattr(query_model_tool_module, "OpenAI"):
    real_module_path = (
        SRC_ROOT
>>>>>>> d242e9c4572b7962c27025dc35b57e204bca26b6
        / "enochian_lm"
        / "root_extraction"
        / "tools"
        / "query_model_tool.py"
    )
    spec = importlib.util.spec_from_file_location(
<<<<<<< HEAD
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
=======
        "real_query_model_tool_for_tests",
        real_module_path,
    )
    assert spec is not None and spec.loader is not None
    real_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_module)
    query_model_tool_module = real_module


def test_query_model_tool_non_stream_returns_full_response(monkeypatch) -> None:
    """Allow phrase rendering to skip first-token waits with a true non-stream call.

    Phrase bundle rendering uses `stream_response=False` so short JSON payloads
    do not sit around waiting for the first streamed content delta. This
    regression exercises the real transport class and proves it accepts that
    mode, sends `stream=False` to the provider, and returns the full message
    body without trying to iterate a streaming object.
    """

    client_inits: list[dict[str, object]] = []
    request_calls: list[dict[str, object]] = []
    progress_events: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            client_inits.append(dict(kwargs))

            def _create(**request_kwargs):
                request_calls.append(dict(request_kwargs))
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content='{"lay_translation":"plain english"}'
                            )
                        )
                    ]
                )

            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=_create)
            )

    monkeypatch.setattr(query_model_tool_module, "OpenAI", _FakeOpenAI, raising=False)
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
>>>>>>> d242e9c4572b7962c27025dc35b57e204bca26b6
