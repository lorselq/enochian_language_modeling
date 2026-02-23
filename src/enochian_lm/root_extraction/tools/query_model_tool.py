from __future__ import annotations

from enochian_lm.common.sqlite_bootstrap import sqlite3
import os
import sys
import logging
import httpx
import random
import time
from yaspin import yaspin, Spinner
from yaspin.spinners import Spinners
from tenacity import retry, stop_after_attempt, wait_exponential, RetryCallState
from collections.abc import Callable
from typing import ClassVar
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
    default_name: ClassVar[str] = "Query LLM"
    default_description: ClassVar[str] = (
        "Sends a prompt to the LLM for linguistic analysis."
    )
    system_prompt: str = (
        "You are a research linguist specializing in rare, obscure, dead constructed-languages."
    )
    gloss_model: str = ""
    # private attribute (not a field)
    _use_remote: bool = PrivateAttr(default=True)
    _db: sqlite3.Connection | None = PrivateAttr(default=None)
    _run_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        system_prompt: str,
        name: str | None = None,
        description: str | None = None,
        use_remote: bool = True,
    ):
        super().__init__(
            name=name or self.default_name,
            description=description or self.default_description,
        )
        # now Pydantic knows system_prompt exists and is a string
        self.system_prompt = system_prompt
        self._use_remote = use_remote

    @staticmethod
    def _log_attempt(retry_state: RetryCallState):
        n = retry_state.attempt_number
        plural = "" if n == 1 else " again"
        print(f"{GREEN}Attempting to connect to a remote LLM{plural}...{RESET}\n")

    @staticmethod
    def _log_retry_state(retry_state: RetryCallState):
        n = retry_state.attempt_number
        print(f"{PINK}Connection attempt failed! ({n + 1}/5 attempts made){RESET}\n")

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
        stop=stop_after_attempt(5),  # max attempts
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
    ) -> dict[str, str]:
        return self._llm_call(
            api_base_env="REMOTE_OPENAI_API_BASE",
            api_key_env="REMOTE_OPENAI_API_KEY",
            model_env="REMOTE_MODEL_NAME",
            prompt=prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
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
    ) -> dict[str, str]:
        base_url = os.getenv(api_base_env, "[ERROR] could not get base URL!")
        api_key  = os.getenv(api_key_env, "[ERROR] could not get API key!")
        model    = os.getenv(model_env, "[ERROR] could not identify model!")
        role     = role_name or self.name
        temperature = 0.2
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(120.0, read=120.0, write=10.0, connect=5.0),
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
            cached = llm_job_try_cache(self._db, phash)
            if cached:
                # mark as cached (idempotent)
                try:
                    job_id = llm_job_start(
                        self._db, run_id=self._run_id, prompt_hash=phash, role=role,
                        model=model, base_url=base_url, temperature=temperature,
                        system_prompt=self.system_prompt, user_prompt=prompt,
                        request_json={"model": model, "messages": [{"role":"system","content": self.system_prompt},{"role":"user","content": prompt}], "temperature": temperature, "stream": True}
                    )
                    llm_job_finish(self._db, job_id, response_text=cached["response_text"], status="cached")
                except Exception:
                    pass  # don’t let logging failures break the call
                self._debug(f"cache hit for role={role!r}; chars={len(cached.get('response_text',''))}")
                return cached

            # 1) log queued
            try:
                job_id = llm_job_start(
                    self._db, run_id=self._run_id, prompt_hash=phash, role=role,
                    model=model, base_url=base_url, temperature=temperature,
                    system_prompt=self.system_prompt, user_prompt=prompt,
                    request_json={"model": model, "messages": [{"role":"system","content": self.system_prompt},{"role":"user","content": prompt}], "temperature": temperature, "stream": True}
                )
            except Exception:
                job_id = None  # proceed without logging

        response_text = ""
        role = role_name or self.name
        if "Glossator" in role:
            self.gloss_model = os.getenv(
                model_env, "([Error] Not able to retrieve the model!)"
            )

        try:
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
        except Exception as exc:
            self._debug(f"stream create failed for role={role!r}: {type(exc).__name__}: {exc}")
            raise

        print(
            f"{GREEN}🤝 Connection successful! 🥰{RESET}\n\nWhat next, you might ask? We wait...\n"
        )

        response_text = ""

        # 4) Spinner until the first real chunk arrives
        with self._get_random_spinner() as sp:
            while True:
                try:
                    chunk = next(completion)
                except StopIteration:
                    # no data at all
                    break

                content = self._extract_chunk_text(chunk)
                if not content:
                    continue

                # first real token → clear spinner and print header
                sp.hide()
                sys.stdout.write("\r\033[2K")  # Erase line
                sys.stdout.write(RESET)  # Reset styling
                sys.stdout.flush()
                print(
                    f"{GREEN}Waiting complete! 😊 Let's see what they have to say!{RESET}\n"
                )
                if role_name:
                    role_label = f">>>{role_name}"
                    if role_name != "TLDR":
                        role_label += " speaking"
                    print(f"{WHITE}{role_label}:{RESET}")

                # emit that first bit
                response_text += content
                self._emit(
                    print_chunks,
                    stream_callback,
                    role_name or self.name,
                    f"{GRAY}{content}{RESET}",
                )
                break

        # 5) Consume the rest of the stream
        for chunk in completion:
            content = self._extract_chunk_text(chunk)
            if not content:
                continue
            response_text += content
            self._emit(
                print_chunks,
                stream_callback,
                role_name or self.name,
                f"{GRAY}{content}{RESET}",
            )
            if chunk.choices[0].finish_reason is not None:
                break

        # 6) Final fallback if nothing arrived
        if not response_text:
            # Some providers may stream only reasoning metadata, while final text
            # remains available in a non-stream completion response.
            try:
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

        if not response_text:
            response_text = "[ERROR] No content returned from remote/local model."

        if self._db and job_id:
            try:
                llm_job_finish(self._db, job_id, response_text=response_text, status="ok")
            except Exception:
                pass

        return {
            "response_text": response_text,
            "gloss_model": getattr(self, "gloss_model", "<unset>"),
        }

    def _run(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
    ) -> dict[str, str]:
        if self._use_remote:
            try:
                return self._try_remote(
                    prompt,
                    stream_callback=stream_callback,
                    print_chunks=print_chunks,
                    role_name=role_name,
                )
            except Exception as exc:
                logger.exception("Remote LLM call failed; attempting local fallback.")
                print(f"⚠️ {YELLOW}Remote call failure: {type(exc).__name__}: {exc}{RESET}")
                # exit if out of OpenRouter calls
                # print(
                #     f"⚠️ [{time.ctime()}] Clearly we're out of LLM calls for the day. Stopping for now. Goodbye for now. 🫡"
                # )
                # sys.exit()
                # fallback to local
                print(
                    f"⚠️ {YELLOW}Falling back to utilizing a local LLM instead...\n{RESET}"
                )
                return self._llm_call(
                    api_base_env="LOCAL_OPENAI_API_BASE",
                    api_key_env="LOCAL_OPENAI_API_KEY",
                    model_env="LOCAL_MODEL_NAME",
                    prompt=prompt,
                    stream_callback=stream_callback,
                    print_chunks=print_chunks,
                    role_name=role_name,
                )
        else:
            return self._llm_call(
                api_base_env="LOCAL_OPENAI_API_BASE",
                api_key_env="LOCAL_OPENAI_API_KEY",
                model_env="LOCAL_MODEL_NAME",
                prompt=prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )

    async def _arun(
        self,
        prompt: str,
        stream_callback: Callable[[str, str], None] | None = None,
        print_chunks: bool = False,
        role_name: str | None = None,
    ) -> dict[str, str]:
        """
        Async entrypoint. Delegate straight to the sync _run.
        """
        return self._run(
            prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
        )
