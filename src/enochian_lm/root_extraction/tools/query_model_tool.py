from __future__ import annotations

from enochian_lm.common.sqlite_bootstrap import sqlite3
import os
import sys
import logging
import httpx
import random
import threading
import time
from yaspin import yaspin, Spinner
from yaspin.spinners import Spinners
from tenacity import retry, stop_after_attempt, wait_exponential, RetryCallState
from collections.abc import Callable
from typing import ClassVar, Literal
from openai import OpenAI
from crewai.tools import BaseTool
from pydantic import PrivateAttr
from enochian_lm.root_extraction.utils.llm_jobs import (
    make_prompt_hash, llm_job_try_cache, llm_job_start, llm_job_finish
)
# Silence those INFO logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.api_requestor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

GRAY = "\033[38;5;250m"
WHITE = "\033[97m"
GREEN = "\033[32m"
PINK = "\033[38;5;213m"
YELLOW = "\033[38;5;190m"
RESET = "\033[0m"


class QueryModelTool(BaseTool):
    MAX_ATTEMPTS: int = 10
    REMOTE_ATTEMPTS: ClassVar[int] = 5
    HEARTBEAT_INTERVAL_SECONDS: ClassVar[float] = 5.0
    default_name: ClassVar[str] = "Query LLM"
    default_description: ClassVar[str] = (
        "Sends a prompt to the LLM for linguistic analysis."
    )
    system_prompt: str = (
        "You are a research linguist specializing in rare, obscure, dead constructed-languages."
    )
    gloss_model: str = ""
    progress_style: Literal["verbose", "compact", "silent"] = "verbose"
    # private attribute (not a field)
    _use_remote: bool = PrivateAttr(default=True)
    _db: sqlite3.Connection | None = PrivateAttr(default=None)
    _run_id: str | None = PrivateAttr(default=None)
    _progress_callback: Callable[[dict[str, object]], None] | None = PrivateAttr(default=None)
    _current_attempt: int = PrivateAttr(default=0)
    _current_source: str = PrivateAttr(default="remote")
    _heartbeat_stop: threading.Event | None = PrivateAttr(default=None)
    _heartbeat_thread: threading.Thread | None = PrivateAttr(default=None)
    _remote_attempts: int = PrivateAttr(default=REMOTE_ATTEMPTS)
    _read_timeout_seconds: float = PrivateAttr(default=120.0)
    _local_fallback_enabled: bool = PrivateAttr(default=True)
    _stream_response: bool = PrivateAttr(default=True)

    def __init__(
        self,
        *,
        system_prompt: str,
        name: str | None = None,
        description: str | None = None,
        use_remote: bool = True,
        progress_style: Literal["verbose", "compact", "silent"] = "verbose",
        remote_attempts: int | None = None,
        read_timeout_seconds: float | None = None,
        local_fallback_enabled: bool = True,
        stream_response: bool = True,
    ):
        super().__init__(
            name=name or self.default_name,
            description=description or self.default_description,
        )
        # now Pydantic knows system_prompt exists and is a string
        self.system_prompt = system_prompt
        self._use_remote = use_remote
        self.progress_style = progress_style
        self._remote_attempts = max(1, int(remote_attempts or self.REMOTE_ATTEMPTS))
        self._read_timeout_seconds = max(5.0, float(read_timeout_seconds or 120.0))
        self._local_fallback_enabled = bool(local_fallback_enabled)
        self._stream_response = bool(stream_response)

    @staticmethod
    def _progress_style_from_retry_state(retry_state: RetryCallState) -> str:
        tool = retry_state.args[0] if retry_state.args else None
        if isinstance(tool, QueryModelTool):
            return tool.progress_style
        return "verbose"

    @staticmethod
    def _tool_from_retry_state(retry_state: RetryCallState) -> "QueryModelTool | None":
        """Recover the tool instance from Tenacity retry callbacks.

        Tenacity invokes ``before`` and ``before_sleep`` callbacks outside the
        normal method body, so structured progress updates need a small helper
        to reach back into the active ``QueryModelTool`` instance.
        """

        tool = retry_state.args[0] if retry_state.args else None
        return tool if isinstance(tool, QueryModelTool) else None

    @staticmethod
    def _progress_callback_from_retry_state(
        retry_state: RetryCallState,
    ) -> Callable[[dict[str, object]], None] | None:
        """Return the per-call progress callback attached to the active tool."""

        tool = QueryModelTool._tool_from_retry_state(retry_state)
        if tool is None:
            return None
        return tool._progress_callback

    @staticmethod
    def _log_attempt(retry_state: RetryCallState):
        tool = QueryModelTool._tool_from_retry_state(retry_state)
        n = retry_state.attempt_number
        max_attempts = tool._remote_attempts if tool is not None else QueryModelTool.REMOTE_ATTEMPTS
        if tool is not None:
            tool._current_attempt = n
            tool._current_source = "remote"
            tool._emit_progress_event(
                {
                    "state": "queued remote request",
                    "attempt": n,
                    "max_attempts": max_attempts,
                    "source": "remote",
                }
            )
        if QueryModelTool._progress_style_from_retry_state(retry_state) != "verbose":
            return
        plural = "" if n == 1 else " again"
        print(f"{GREEN}Attempting to connect to a remote LLM{plural}...{RESET}\n")

    @staticmethod
    def _log_retry_state(retry_state: RetryCallState):
        callback = QueryModelTool._progress_callback_from_retry_state(retry_state)
        tool = QueryModelTool._tool_from_retry_state(retry_state)
        max_attempts = tool._remote_attempts if tool is not None else QueryModelTool.REMOTE_ATTEMPTS
        if callback is not None:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            next_action = getattr(retry_state, "next_action", None)
            callback(
                {
                    "state": "retrying after remote failure",
                    "attempt": retry_state.attempt_number,
                    "max_attempts": max_attempts,
                    "source": "remote",
                    "warning": (
                        f"{type(exc).__name__}: {exc}" if exc is not None else "remote retry"
                    ),
                    "retry_delay_seconds": (
                        float(next_action.sleep)
                        if next_action is not None and next_action.sleep is not None
                        else None
                    ),
                }
            )
        if QueryModelTool._progress_style_from_retry_state(retry_state) == "silent":
            return
        n = retry_state.attempt_number
        print(f"{PINK}Connection attempt failed! ({n + 1}/{max_attempts} attempts made){RESET}\n")

    def _emit_progress_event(self, payload: dict[str, object]) -> None:
        """Publish structured LLM status updates to higher-level CLI renderers.

        Phrase translation now needs heartbeat-style reassurance during long
        remote calls. Keeping those updates as structured payloads here lets the
        translation layer render richer status lines without coupling this tool
        to a specific CLI presentation.
        """

        if self._progress_callback is None:
            return
        self._progress_callback(dict(payload))

    def _start_heartbeat(
        self,
        *,
        state_provider: Callable[[], dict[str, object]],
    ) -> None:
        """Emit periodic liveness updates while a blocking model call runs.

        Remote streaming can block for long stretches before the first token
        arrives. A lightweight heartbeat thread gives the CLI a steady elapsed
        timer and current sub-state so users can distinguish slow progress from
        a dead process.
        """

        if self._progress_callback is None:
            return
        self._stop_heartbeat()
        stop_event = threading.Event()
        self._heartbeat_stop = stop_event

        def _heartbeat_loop() -> None:
            while not stop_event.wait(self.HEARTBEAT_INTERVAL_SECONDS):
                snapshot = state_provider()
                snapshot["heartbeat"] = True
                self._emit_progress_event(snapshot)

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="query-model-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread = heartbeat_thread
        heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop any active heartbeat thread for the current model call."""

        stop_event = self._heartbeat_stop
        heartbeat_thread = self._heartbeat_thread
        self._heartbeat_stop = None
        self._heartbeat_thread = None
        if stop_event is not None:
            stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.1)

    @staticmethod
    def _get_random_spinner():
        SHARK_TEXT = [
            "Let us admire the pretty shark while we wait for LLM's response... ",
            "Such a happy shark! (we're waiting... 🙃) ",
            "Marooned in shark-waters. 🏝️ But it's okay, the LLM-ship will come save us! ",
            "Waiting on the LLM. In the meantime, enjoy this shark. ",
            "Sharks as a species are older than Saturn's rings. Not exactly relevant, but cool. ",
            "If I were a shark, I, too, would swim laps while waiting for LLM API calls. ",
            "CLI aquariums are too small for LLM-sharks. Waiting to set the critter free... ",
            "Look! A shark! A distraction and indication that the stream didn't crash! ",
            "What is a shark doing swimming on a stream?? Let's wait and find out... ",
            "Note: the shark is here for decoration, not because it speaks Enochian and can help us. ",
            "Our shark friend promises: no chomp until the LLM answers. ",
            "Shark in the console! Not an error—just LLM still thinking... ",
            "A shark sighting means the process is still afloat. Stay tuned! ",
        ]
        BALL_TEXT = [
            "Ping... Pong... the LLM is about to return your serve! 🏓 ",
            "Paddle at the ready—waiting for the LLM's smash shot! ",
            "Volley in progress... LLM's response coming any moment! ",
            "Keep your eye on the ball... the LLM's return is next! ",
            "Fast-paced Pong at CPU speed—still waiting... 🤖 ",
            "Don't let the ball drop—LLM response inbound! 🎾 ",
            "Serve, return, repeat... LLM's turn to play! ",
        ]
        SHY_TEXT = [
            "I'm blushing... but the LLM isn't ready yet 😳 ",
            "So shy... please say something, LLM? 😢 ",
            "I'm hiding behind my code... waiting on you, LLM! ",
            "Shy mode activated—LLM, you first! 🤫 ",
            "Please, LLM—where are you?? 👀 ",
            f"Um, um um um... {GRAY}Psst, LLM... are you there...?{RESET} 🤐 ",
            "Shyness level: maximum. Oh no, please save me LLM! ",
            "Heart racing... LLM, won't you say hello? ❤️ ",
        ]
        EARTH_TEXT = [
            "The world is a distraction. Sort of. This one anyway. ",
            "Earth's heartbeat is steady—LLM pulse arriving soon! ",
            "We've planted the prompt, let's see what we grow. 🌱 ",
            "... One eternity later... ",
            "The globe represents how at home we feel with AI's progress. ",
            "(Please be aware, this Earth is not to scale)",
        ]
        MOON_TEXT = [
            "The number of moon emojis really eclipses everything but faces. ",
            "If the moon were made of cheese, would it be too much cheese? 🧀 ",
            "Waiting is just phase; it'll pass soon enough. ",
            "Even the moon takes 27 days to orbit; good things come with time. ",
            "The AI will happily wax eloquent about Enochian for us in a moment. ",
            "You wane some, you lose some—and it's how we respond that matters.",
        ]
        chosen_shark_text = random.choice(SHARK_TEXT)
        chosen_ball_text = random.choice(BALL_TEXT)
        chosen_shy_text = random.choice(SHY_TEXT)
        chosen_earth_text = random.choice(EARTH_TEXT)
        chosen_moon_text = random.choice(MOON_TEXT)
        spinner_names = ["SHARK", "BALL", "SHY", "EARTH", "MOON"]
        spinners = {
            "SHARK": yaspin(
                ellipsis="...", text=chosen_shark_text
            ).white.bold.shark.on_blue,
            "BALL": yaspin(
                ellipsis="...", text=chosen_ball_text
            ).bold.blink.magenta.bouncingBall.on_cyan,
            "SHY": yaspin(
                Spinner(
                    [
                        "👉    👈🥺",
                        "👉    👈🥺",
                        "👉    👈🥺",
                        "👉    👈🥺",
                        "👉    👈🥺",
                        "👉    👈🥺",
                        " 👉  👈 🥺",
                        " 👉  👈 🥺",
                        "  👉👈  🥺",
                        " 👉  👈 🥺",
                        "  👉👈  🥺",
                        "  👉👈  🥺",
                        "  👉👈  🥺",
                        " 👉  👈 🥺",
                    ],
                    175,
                ),
                text=chosen_shy_text,
                ellipsis="...",
            ),
            "EARTH": yaspin(Spinners.earth, text=chosen_earth_text, ellipsis="..."),
            "MOON": yaspin(Spinners.moon, text=chosen_moon_text, ellipsis="..."),
        }

        return spinners[random.choice(spinner_names)]

    @retry(
        reraise=True,
        stop=stop_after_attempt(REMOTE_ATTEMPTS),  # max attempts
        wait=wait_exponential(multiplier=1, min=2, max=62),
        before=_log_attempt,
        before_sleep=_log_retry_state,
    )
    def _try_remote(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        return self._try_remote_once(
            prompt=prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
            progress_message=progress_message,
            progress_callback=progress_callback,
        )
    
    def attach_logging(
        self,
        db: sqlite3.Connection | None,
        run_id: str | None,
    ) -> None:
        """Attach optional logging metadata for downstream persistence.

        CrewAI tools derive from :class:`pydantic.BaseModel`, which performs
        type validation on attribute assignment.  We therefore accept
        ``None`` values here so callers can skip logging without tripping
        validation errors when running in contexts where a database handle or
        run id is unavailable.
        """

        self._db = db
        self._run_id = run_id

    def _emit(self, print_chunks, stream_callback, role, content):
        if print_chunks:
            print(content, end="", flush=True)
        elif stream_callback:
            stream_callback(role, content)

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("ROOT_LLM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    def _debug(self, msg: str) -> None:
        if self._debug_enabled():
            print(f"{YELLOW}[LLM DEBUG]{RESET} {msg}")

    def _print_progress(self, message: str, *, style: str | None = None) -> None:
        mode = style or self.progress_style
        if mode == "silent" or not message:
            return
        print(message)

    def _print_connection_retry(self, exc: Exception) -> None:
        if self.progress_style == "silent":
            return
        print(f"⚠️ {YELLOW}Remote call failure: {type(exc).__name__}: {exc}{RESET}")

    def _print_fallback_notice(self) -> None:
        if self.progress_style == "silent":
            return
        print(f"⚠️ {YELLOW}Falling back to utilizing a local LLM instead...\n{RESET}")

    @staticmethod
    def _extract_chunk_text(chunk) -> str:
        """Extract visible text from streaming chunks across API variants."""
        try:
            delta = chunk.choices[0].delta
        except Exception:
            return ""

        content = getattr(delta, "content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or ""
                else:
                    text = getattr(item, "text", "") or ""
                if text:
                    parts.append(text)
            return "".join(parts)
        return ""

    def _llm_call(
        self,
        api_base_env: str,
        api_key_env: str,
        model_env: str,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        base_url = os.getenv(api_base_env, "[ERROR] could not get base URL!")
        api_key  = os.getenv(api_key_env, "[ERROR] could not get API key!")
        model    = os.getenv(model_env, "[ERROR] could not identify model!")
        role     = role_name or self.name
        temperature = 0.2
        self._progress_callback = progress_callback or self._progress_callback
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        source = "remote" if api_base_env == "REMOTE_OPENAI_API_BASE" else "local"
        if source != "remote" and self._current_attempt <= 0:
            self._current_attempt = 1
        self._current_source = source
        call_started = time.monotonic()
        progress_state: dict[str, object] = {
            "state": "connecting",
            "attempt": self._current_attempt or 1,
            "max_attempts": self._remote_attempts if source == "remote" else 1,
            "source": source,
            "elapsed_seconds": 0.0,
            "chunk_count": 0,
            "char_count": 0,
        }

        def _snapshot() -> dict[str, object]:
            return {
                **progress_state,
                "elapsed_seconds": max(0.0, time.monotonic() - call_started),
            }

        self._emit_progress_event(_snapshot())
        self._start_heartbeat(state_provider=_snapshot)

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(
                self._read_timeout_seconds,
                read=self._read_timeout_seconds,
                write=10.0,
                connect=5.0,
            ),
        )
        self._debug(f"role={role!r} model={model!r} base_url={base_url!r}")

        # --- LLM JOB logging (optional) ---
        job_id = None
        if self._db and self._run_id:
            phash = make_prompt_hash(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                role=role,
                model=model,
                temperature=temperature,
                base_url=base_url,
            )
            # 0) try cache first
            use_cache = os.getenv("ROOT_LLM_DISABLE_CACHE", "").strip().lower() not in {"1", "true", "yes", "on"}
            cached = llm_job_try_cache(self._db, phash, run_id=self._run_id) if use_cache else None
            if cached:
                # mark as cached (idempotent)
                try:
                    job_id = llm_job_start(
                        self._db, run_id=self._run_id, prompt_hash=phash, role=role,
                        model=model, base_url=base_url, temperature=temperature,
                        system_prompt=self.system_prompt, user_prompt=prompt,
                        request_json={"model": model, "messages": [{"role":"system","content": self.system_prompt},{"role":"user","content": prompt}], "temperature": temperature, "stream": self._stream_response}
                    )
                    llm_job_finish(self._db, job_id, response_text=cached["response_text"], status="cached")
                except Exception:
                    pass  # don’t let logging failures break the call
                if self.progress_style == "verbose":
                    print(f"{YELLOW}↺ Using cached LLM response for {role}.{RESET}")
                elif self.progress_style == "compact" and progress_message:
                    self._print_progress(f"{progress_message} (cached)", style="compact")
                self._debug(f"cache hit for role={role!r}; chars={len(cached.get('response_text',''))}")
                cached_state = _snapshot()
                cached_state["state"] = "using cached response"
                cached_state["cached"] = True
                self._emit_progress_event(cached_state)
                self._stop_heartbeat()
                return cached

            # 1) log queued
            try:
                job_id = llm_job_start(
                    self._db, run_id=self._run_id, prompt_hash=phash, role=role,
                    model=model, base_url=base_url, temperature=temperature,
                    system_prompt=self.system_prompt, user_prompt=prompt,
                    request_json={"model": model, "messages": [{"role":"system","content": self.system_prompt},{"role":"user","content": prompt}], "temperature": temperature, "stream": self._stream_response}
                )
            except Exception:
                job_id = None  # proceed without logging

        if self.progress_style == "compact" and progress_message:
            self._print_progress(progress_message, style="compact")

        response_text = ""
        role = role_name or self.name
        if "Glossator" in role:
            self.gloss_model = os.getenv(
                model_env, "([Error] Not able to retrieve the model!)"
            )

        if self._stream_response:
            try:
                progress_state["state"] = "queued request with provider"
                self._emit_progress_event(_snapshot())
                completion = client.chat.completions.create(
                    model=os.getenv(model_env, ""),
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    stream=True,
                    seed=93,
                )
                progress_state["state"] = "waiting for first token"
                self._emit_progress_event(_snapshot())
            except Exception as exc:
                self._debug(f"stream create failed for role={role!r}: {type(exc).__name__}: {exc}")
                failure_state = _snapshot()
                failure_state["state"] = "provider connection failed"
                failure_state["warning"] = f"{type(exc).__name__}: {exc}"
                self._emit_progress_event(failure_state)
                self._stop_heartbeat()
                raise

            if self.progress_style == "verbose":
                print(
                    f"{GREEN}🤝 Connection successful! 🥰{RESET}\n\nWhat next, you might ask? We wait...\n"
                )

            response_text = ""

            # Streaming is best when callers surface incremental text. For
            # short JSON-only jobs, callers can disable it and use the
            # non-stream branch below to avoid long first-token stalls.
            if self.progress_style == "verbose":
                with self._get_random_spinner() as sp:
                    while True:
                        try:
                            chunk = next(completion)
                        except StopIteration:
                            break

                        content = self._extract_chunk_text(chunk)
                        if not content:
                            continue

                        sp.hide()
                        sys.stdout.write("\r\033[2K")
                        sys.stdout.write(RESET)
                        sys.stdout.flush()
                        print(
                            f"{GREEN}Waiting complete! 😊 Let's see what they have to say!{RESET}\n"
                        )
                        if role_name:
                            role_label = f">>>{role_name}"
                        if role_name != "TLDR":
                            role_label += " speaking"
                            print(f"{WHITE}{role_label}:{RESET}")

                        response_text += content
                        progress_state["state"] = "streaming response"
                        progress_state["chunk_count"] = int(progress_state.get("chunk_count") or 0) + 1
                        progress_state["char_count"] = int(progress_state.get("char_count") or 0) + len(content)
                        self._emit_progress_event(_snapshot())
                        self._emit(
                            print_chunks,
                            stream_callback,
                            role_name or self.name,
                            f"{GRAY}{content}{RESET}",
                        )
                        break
            else:
                while True:
                    try:
                        chunk = next(completion)
                    except StopIteration:
                        break

                    content = self._extract_chunk_text(chunk)
                    if not content:
                        continue

                    response_text += content
                    progress_state["state"] = "streaming response"
                    progress_state["chunk_count"] = int(progress_state.get("chunk_count") or 0) + 1
                    progress_state["char_count"] = int(progress_state.get("char_count") or 0) + len(content)
                    self._emit_progress_event(_snapshot())
                    self._emit(
                        print_chunks,
                        stream_callback,
                        role_name or self.name,
                        f"{GRAY}{content}{RESET}",
                    )
                    break

            for chunk in completion:
                content = self._extract_chunk_text(chunk)
                if not content:
                    continue
                response_text += content
                progress_state["state"] = "streaming response"
                progress_state["chunk_count"] = int(progress_state.get("chunk_count") or 0) + 1
                progress_state["char_count"] = int(progress_state.get("char_count") or 0) + len(content)
                self._emit(
                    print_chunks,
                    stream_callback,
                    role_name or self.name,
                    f"{GRAY}{content}{RESET}",
                )
                if chunk.choices[0].finish_reason is not None:
                    break
        else:
            try:
                progress_state["state"] = "waiting for non-stream response"
                self._emit_progress_event(_snapshot())
                non_stream = client.chat.completions.create(
                    model=os.getenv(model_env, ""),
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    stream=False,
                    seed=93,
                )
            except Exception as exc:
                self._debug(
                    f"non-stream create failed for role={role!r}: {type(exc).__name__}: {exc}"
                )
                failure_state = _snapshot()
                failure_state["state"] = "provider connection failed"
                failure_state["warning"] = f"{type(exc).__name__}: {exc}"
                self._emit_progress_event(failure_state)
                self._stop_heartbeat()
                raise

            response_text = (non_stream.choices[0].message.content or "").strip()
            if response_text:
                progress_state["state"] = "response complete"
                progress_state["chunk_count"] = 1
                progress_state["char_count"] = len(response_text)
                self._emit(
                    print_chunks,
                    stream_callback,
                    role_name or self.name,
                    f"{GRAY}{response_text}{RESET}",
                )

        # 6) Final fallback if nothing arrived
        if self._stream_response and not response_text:
            # Some providers may stream only reasoning metadata, while final text
            # remains available in a non-stream completion response.
            try:
                progress_state["state"] = "waiting for non-stream fallback"
                self._emit_progress_event(_snapshot())
                non_stream = client.chat.completions.create(
                    model=os.getenv(model_env, ""),
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    stream=False,
                    seed=93,
                )
                response_text = (non_stream.choices[0].message.content or "").strip()
            except Exception as exc:
                logger.warning("Non-stream fallback failed: %s", exc)
                failure_state = _snapshot()
                failure_state["state"] = "non-stream fallback failed"
                failure_state["warning"] = f"{type(exc).__name__}: {exc}"
                self._emit_progress_event(failure_state)

        if not response_text:
            response_text = "[ERROR] No content returned from remote/local model."

        if self._db and job_id:
            try:
                llm_job_finish(self._db, job_id, response_text=response_text, status="ok")
            except Exception:
                pass

        complete_state = _snapshot()
        complete_state["state"] = "response complete"
        complete_state["char_count"] = len(response_text)
        self._emit_progress_event(complete_state)
        self._stop_heartbeat()

        return {
            "response_text": response_text,
            "gloss_model": getattr(self, "gloss_model", "<unset>"),
        }

    def _try_remote_once(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        """Run one remote attempt with the shared call machinery.

        Phrase rendering now uses a shorter retry budget than extraction jobs.
        Keeping the single-attempt body separate lets `_run` choose between the
        legacy Tenacity policy and a per-instance fast-fail loop.
        """

        return self._llm_call(
            api_base_env="REMOTE_OPENAI_API_BASE",
            api_key_env="REMOTE_OPENAI_API_KEY",
            model_env="REMOTE_MODEL_NAME",
            prompt=prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
            progress_message=progress_message,
            progress_callback=progress_callback,
        )

    def _remote_retry_delay_seconds(self, attempt_number: int) -> float:
        """Mirror the default exponential retry cadence for custom budgets."""

        return float(min(62, max(2, 2 ** max(1, attempt_number))))

    def _run_remote_with_policy(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        """Run remote calls with either the legacy or instance-specific retry budget.

        Extraction jobs still want the long-standing Tenacity behavior, while
        phrase rendering now needs a shorter, per-call budget to avoid
        multi-minute stalls. This method chooses the cheaper path without
        changing defaults for existing callers.
        """

        if self._remote_attempts == self.REMOTE_ATTEMPTS:
            return self._try_remote(
                prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
                progress_message=progress_message,
                progress_callback=progress_callback,
            )

        last_exc: Exception | None = None
        for attempt in range(1, self._remote_attempts + 1):
            self._current_attempt = attempt
            self._current_source = "remote"
            self._emit_progress_event(
                {
                    "state": "queued remote request",
                    "attempt": attempt,
                    "max_attempts": self._remote_attempts,
                    "source": "remote",
                }
            )
            if self.progress_style == "verbose":
                plural = "" if attempt == 1 else " again"
                print(f"{GREEN}Attempting to connect to a remote LLM{plural}...{RESET}\n")
            try:
                return self._try_remote_once(
                    prompt,
                    stream_callback=stream_callback,
                    print_chunks=print_chunks,
                    role_name=role_name,
                    progress_message=progress_message,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self._remote_attempts:
                    break
                retry_delay_seconds = self._remote_retry_delay_seconds(attempt)
                self._emit_progress_event(
                    {
                        "state": "retrying after remote failure",
                        "attempt": attempt,
                        "max_attempts": self._remote_attempts,
                        "source": "remote",
                        "warning": f"{type(exc).__name__}: {exc}",
                        "retry_delay_seconds": retry_delay_seconds,
                    }
                )
                self._print_connection_retry(exc)
                if self.progress_style != "silent":
                    print(
                        f"{PINK}Connection attempt failed! ({attempt + 1}/{self._remote_attempts} attempts made){RESET}\n"
                    )
                time.sleep(retry_delay_seconds)

        assert last_exc is not None
        raise last_exc

    def _run(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        self._progress_callback = progress_callback
        try:
            if self._use_remote:
                try:
                    return self._run_remote_with_policy(
                        prompt,
                        stream_callback=stream_callback,
                        print_chunks=print_chunks,
                        role_name=role_name,
                        progress_message=progress_message,
                        progress_callback=progress_callback,
                    )
                except Exception as exc:
                    if not self._local_fallback_enabled:
                        raise
                    logger.exception("Remote LLM call failed; attempting local fallback.")
                    self._emit_progress_event(
                        {
                            "state": "falling back to local model",
                            "attempt": self._current_attempt or 1,
                            "max_attempts": self._remote_attempts,
                            "source": "remote",
                            "warning": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self._print_connection_retry(exc)
                    self._print_fallback_notice()
                    self._current_attempt = 1
                    self._current_source = "local"
                    return self._llm_call(
                        api_base_env="LOCAL_OPENAI_API_BASE",
                        api_key_env="LOCAL_OPENAI_API_KEY",
                        model_env="LOCAL_MODEL_NAME",
                        prompt=prompt,
                        stream_callback=stream_callback,
                        print_chunks=print_chunks,
                        role_name=role_name,
                        progress_message=progress_message,
                        progress_callback=progress_callback,
                    )
            return self._llm_call(
                api_base_env="LOCAL_OPENAI_API_BASE",
                api_key_env="LOCAL_OPENAI_API_KEY",
                model_env="LOCAL_MODEL_NAME",
                prompt=prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
                progress_message=progress_message,
                progress_callback=progress_callback,
            )
        finally:
            self._stop_heartbeat()
            self._progress_callback = None

    async def _arun(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
        progress_message: str | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, str]:
        """
        Async entrypoint. Delegate straight to the sync _run.
        """
        return self._run(
            prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
            progress_message=progress_message,
            progress_callback=progress_callback,
        )
