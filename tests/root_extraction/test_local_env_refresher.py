"""Regression tests for LM Studio local-model env synchronization.

The local env refresher is now responsible for keeping `LOCAL_MODEL_NAME`
aligned with LM Studio's loaded-model state. These tests lock the selection
rules and ensure sync remains non-destructive under network failures.
"""

from __future__ import annotations

from pathlib import Path

from enochian_lm.root_extraction.utils import local_env_refresher


def _write_env(path: Path, content: str) -> None:
    """Create deterministic dotenv fixtures for sync tests."""
    path.write_text(content, encoding="utf-8")


def test_sync_local_model_name_keeps_current_when_loaded(monkeypatch, tmp_path: Path) -> None:
    """Preserve explicit user model selection when LM Studio still has it loaded.

    Keeping the current loaded model avoids churn in `.env_local` and prevents
    unnecessary rewrites when the configured model is already valid.
    """
    env_path = tmp_path / ".env_local"
    _write_env(
        env_path,
        "LOCAL_OPENAI_API_BASE=http://127.0.0.1:1234/v1\n"
        "LOCAL_MODEL_NAME=current-model\n"
        "PYTHONPATH=src\n",
    )

    monkeypatch.setattr(
        local_env_refresher,
        "_fetch_json",
        lambda url, timeout_seconds: {
            "data": [
                {"id": "current-model", "type": "llm", "state": "loaded"},
                {"id": "other-model", "type": "llm", "state": "loaded"},
            ]
        }
        if url.endswith("/api/v0/models")
        else None,
    )
    load_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        local_env_refresher,
        "load_dotenv",
        lambda path, override=True: load_calls.append((str(path), bool(override))) or True,
    )

    result = local_env_refresher.sync_local_model_name(env_path=str(env_path))

    assert result["ok"] is True
    assert result["updated"] is False
    assert result["selected_model"] == "current-model"
    assert result["source"] == "current_loaded_model"
    assert "LOCAL_MODEL_NAME=current-model\n" in env_path.read_text(encoding="utf-8")
    assert load_calls == [(str(env_path), True)]


def test_sync_local_model_name_replaces_stale_with_first_loaded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Switch stale model ids to the first loaded LLM from LM Studio metadata.

    This is the auto-heal path for identifiers like `openai/local-model` that
    are not valid runtime model ids in LM Studio.
    """
    env_path = tmp_path / ".env_local"
    _write_env(
        env_path,
        "LOCAL_OPENAI_API_BASE=http://127.0.0.1:1234/v1\n"
        "LOCAL_MODEL_NAME=openai/local-model\n"
        "PYTHONPATH=src\n",
    )

    monkeypatch.setattr(
        local_env_refresher,
        "_fetch_json",
        lambda url, timeout_seconds: {
            "data": [
                {"id": "loaded-a", "type": "llm", "state": "loaded"},
                {"id": "loaded-b", "type": "llm", "state": "loaded"},
            ]
        }
        if url.endswith("/api/v0/models")
        else None,
    )
    monkeypatch.setattr(local_env_refresher, "load_dotenv", lambda *args, **kwargs: True)

    result = local_env_refresher.sync_local_model_name(env_path=str(env_path))
    rewritten = env_path.read_text(encoding="utf-8")

    assert result["ok"] is True
    assert result["updated"] is True
    assert result["selected_model"] == "loaded-a"
    assert result["source"] == "first_loaded_llm"
    assert "LOCAL_MODEL_NAME=loaded-a\n" in rewritten
    assert "LOCAL_MODEL_NAME=openai/local-model\n" not in rewritten
    assert "PYTHONPATH=src\n" in rewritten


def test_sync_local_model_name_falls_back_to_v1_models_when_native_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Use OpenAI-compatible model listing when native model metadata is missing.

    Some LM Studio setups may not expose `/api/v0/models`; fallback to
    `/v1/models` keeps automatic model discovery functional.
    """
    env_path = tmp_path / ".env_local"
    _write_env(
        env_path,
        "LOCAL_OPENAI_API_BASE=http://127.0.0.1:1234/v1\n"
        "PYTHONPATH=src\n",
    )

    def _fake_fetch(url: str, _timeout: float):
        if url.endswith("/api/v0/models"):
            return None
        if url.endswith("/v1/models"):
            return {"data": [{"id": "fallback-model"}]}
        return None

    monkeypatch.setattr(local_env_refresher, "_fetch_json", _fake_fetch)
    monkeypatch.setattr(local_env_refresher, "load_dotenv", lambda *args, **kwargs: True)

    result = local_env_refresher.sync_local_model_name(env_path=str(env_path))

    assert result["ok"] is True
    assert result["updated"] is True
    assert result["selected_model"] == "fallback-model"
    assert result["source"] == "first_openai_model"
    assert "LOCAL_MODEL_NAME=fallback-model\n" in env_path.read_text(encoding="utf-8")


def test_sync_local_model_name_keeps_file_stable_when_endpoints_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Return non-fatal status without clobbering `.env_local` on API failures.

    Startup sync should never erase existing env content if LM Studio is
    offline; this regression ensures write operations are skipped in that case.
    """
    env_path = tmp_path / ".env_local"
    original = (
        "LOCAL_OPENAI_API_BASE=http://127.0.0.1:1234/v1\n"
        "LOCAL_MODEL_NAME=existing-model\n"
        "PYTHONPATH=src\n"
    )
    _write_env(env_path, original)

    monkeypatch.setattr(local_env_refresher, "_fetch_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(local_env_refresher, "load_dotenv", lambda *args, **kwargs: True)

    result = local_env_refresher.sync_local_model_name(env_path=str(env_path))

    assert result["ok"] is True
    assert result["updated"] is False
    assert result["selected_model"] == "existing-model"
    assert result["source"] == "kept_existing_model"
    assert env_path.read_text(encoding="utf-8") == original
