from __future__ import annotations

import json
import os
import re
import subprocess
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv, find_dotenv


def get_windows_host_ip():
    try:
        result = subprocess.run(["ip", "route"], stdout=subprocess.PIPE, check=True)
        output = result.stdout.decode()
        match = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            print("[WARN] No default gateway IP found.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to parse routing table: {e}")
        return None


def _resolve_local_env_path() -> str:
    """Resolve the canonical `.env_local` path used by local LLM workflows.

    Both translation and extraction entrypoints need a shared source of truth
    for where local model settings are persisted. Centralizing path resolution
    here keeps model sync behavior consistent across those entrypoints.
    """
    explicit_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../.env_local")
    )
    if os.path.exists(explicit_path):
        return explicit_path
    return find_dotenv(".env_local")


def _parse_env_lines(env_path: str) -> list[str]:
    """Read the local env file into editable line records.

    The sync routine updates only one key while preserving all other settings
    and comments. Operating on full lines keeps the file stable for humans.
    """
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            return handle.readlines()
    except FileNotFoundError:
        return []


def _extract_env_value(lines: list[str], key: str) -> str | None:
    """Extract a KEY=VALUE entry from dotenv-style lines."""
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(prefix):
            return stripped[len(prefix) :]
    return None


def _upsert_env_value(lines: list[str], key: str, value: str) -> list[str]:
    """Replace or append a dotenv key while preserving unrelated content."""
    prefix = f"{key}="
    replacement = f"{key}={value}\n"
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            updated.append(replacement)
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(replacement)
    return updated


def _lm_studio_root_from_api_base(api_base: str) -> str:
    """Convert LOCAL_OPENAI_API_BASE into an LM Studio host root URL."""
    normalized = (api_base or "").strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized.rstrip("/")


def _fetch_json(url: str, timeout_seconds: float) -> dict[str, object] | None:
    """Fetch JSON data from LM Studio endpoints with safe failure handling.

    Auto-sync must never break translation startup. Returning ``None`` on
    network/protocol failures allows callers to keep existing config values.
    """
    if not url:
        return None
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except (OSError, URLError, ValueError):
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_loaded_llm_ids(payload: dict[str, object] | None) -> list[str]:
    """Return loaded LLM IDs from LM Studio native metadata payloads."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    loaded_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        model_type = str(item.get("type", "")).lower()
        state = str(item.get("state", "")).lower()
        if isinstance(model_id, str) and model_id and model_type == "llm" and state == "loaded":
            loaded_ids.append(model_id)
    return loaded_ids


def _extract_openai_model_ids(payload: dict[str, object] | None) -> list[str]:
    """Return OpenAI-compatible model IDs from `/v1/models` payloads."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)
    return model_ids


def sync_local_model_name(
    *,
    env_path: str | None = None,
    timeout_seconds: float = 3.0,
) -> dict[str, object]:
    """Synchronize LOCAL_MODEL_NAME with the currently loaded LM Studio model.

    Translation and extraction frequently load `.env_local`, and stale model ids
    cause avoidable runtime failures. This helper keeps the selected model
    aligned with LM Studio state while remaining non-fatal when LM Studio is
    offline or unreachable.
    """
    resolved_env_path = env_path or _resolve_local_env_path()
    if not resolved_env_path:
        return {
            "ok": False,
            "updated": False,
            "reason": "env_local_not_found",
        }

    lines = _parse_env_lines(resolved_env_path)
    current_model = (_extract_env_value(lines, "LOCAL_MODEL_NAME") or "").strip()
    api_base = (
        _extract_env_value(lines, "LOCAL_OPENAI_API_BASE")
        or os.getenv("LOCAL_OPENAI_API_BASE", "")
    ).strip()
    lm_studio_root = _lm_studio_root_from_api_base(api_base)
    if not lm_studio_root:
        return {
            "ok": False,
            "updated": False,
            "reason": "missing_local_api_base",
        }

    native_payload = _fetch_json(f"{lm_studio_root}/api/v0/models", timeout_seconds)
    loaded_llms = _extract_loaded_llm_ids(native_payload)

    selected_model = ""
    source = ""
    if current_model and current_model in loaded_llms:
        selected_model = current_model
        source = "current_loaded_model"
    elif loaded_llms:
        selected_model = loaded_llms[0]
        source = "first_loaded_llm"
    else:
        openai_payload = _fetch_json(f"{lm_studio_root}/v1/models", timeout_seconds)
        openai_model_ids = _extract_openai_model_ids(openai_payload)
        if not current_model and openai_model_ids:
            selected_model = openai_model_ids[0]
            source = "first_openai_model"
        elif current_model:
            selected_model = current_model
            source = "kept_existing_model"

    if not selected_model:
        return {
            "ok": False,
            "updated": False,
            "reason": "no_model_available",
            "current_model": current_model,
        }

    has_model_entry = _extract_env_value(lines, "LOCAL_MODEL_NAME") is not None
    should_write = (selected_model != current_model) or not has_model_entry
    if should_write:
        updated_lines = _upsert_env_value(lines, "LOCAL_MODEL_NAME", selected_model)
        try:
            with open(resolved_env_path, "w", encoding="utf-8") as handle:
                handle.writelines(updated_lines)
        except OSError as exc:
            return {
                "ok": False,
                "updated": False,
                "reason": f"write_failed: {exc}",
                "current_model": current_model,
                "selected_model": selected_model,
                "source": source,
            }

    load_dotenv(resolved_env_path, override=True)
    return {
        "ok": True,
        "updated": should_write,
        "reason": "synced" if should_write else "already_current",
        "current_model": current_model,
        "selected_model": selected_model,
        "source": source,
        "env_path": resolved_env_path,
    }


def refresh_local_env(local=False) -> bool:
    """
    If local=True, regenerate .local_env with current Windows host IP.
    Then load .local_env into os.environ. Return True on success.
    """
    env_path = _resolve_local_env_path()
    if not env_path:
        explicit_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../.env_local")
        )
        env_path = explicit_path

    if local:
        ip = get_windows_host_ip()
        if not ip:
            print("[FAIL] Could not detect Windows host IP.")
            return False

        base_url = f"http://{ip}:1234/v1"
        existing_lines = _parse_env_lines(env_path)
        existing_model = (_extract_env_value(existing_lines, "LOCAL_MODEL_NAME") or "").strip()
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("LOCAL_OPENAI_API_KEY=sk-local-testing-lol\n")
                f.write(f"LOCAL_OPENAI_API_BASE={base_url}\n")
                if existing_model:
                    f.write(f"LOCAL_MODEL_NAME={existing_model}\n")
                f.write("PYTHONPATH=src\n")
        except Exception as e:
            print(f"[ERROR] Writing .local_env failed: {e}")
            return False

    # Now load it
    if not os.path.exists(env_path):
        print(f"[ERROR] Could not find local env file at {env_path}.")
        return False

    # load_dotenv will read KEY=VALUE lines into os.environ
    loaded = load_dotenv(env_path, override=True)
    if not loaded:
        print(f"[ERROR] Failed to load environment from {env_path}.")
        return False

    return True
