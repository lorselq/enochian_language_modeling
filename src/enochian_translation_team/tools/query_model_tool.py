import os
import sys
import logging
import httpx
import random
from yaspin import yaspin
from tenacity import retry, stop_after_attempt, wait_exponential, RetryCallState
from typing import Optional, Callable, ClassVar
from openai import OpenAI
from crewai.tools import BaseTool

# Silence those INFO logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.api_requestor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

GRAY = "\033[90m"
WHITE = "\033[97m"
GREEN = "\033[32m"
PINK = "\033[38;5;213m"
YELLOW = "\033[38;5;190m"
RESET = "\033[0m"
SHARK_TEXT = [
    "Let us admire the pretty shark while we wait for LLM's response...",
    "Such a happy shark! (we're waiting... 🙃)",
    "Marooned in shark-waters. 🏝️ But it's okay, the LLM-ship will come save us!",
    "Waiting on the LLM. In the meantime, enjoy this shark.",
    "Sharks have been on earth for longer than Jupiter's rings have existed. Fortunately, that's more time than this API call should take.",
    "If I were a shark, I, too, would swim laps while waiting for LLM API calls.",
    "CLI aquariums are too small for LLM-sharks. Waiting to set the critter free...",
    "Look! A shark! A distraction and indication that the stream didn't crash!",
    "What is a shark doing swimming on a stream?? Let's wait and find out...",
    "Note: the shark is here for decoration, not because it speaks Enochian and can help us. We need the LLM for that.",
    "Our shark mascot promises: no chomp until the LLM answers.",
    "The shark waves hello while we wait for the LLM's reply. 🌊🦈",
    "Shark in the console! Not an error—just LLM still thinking...",
    "A shark sighting means the process is still afloat. Stay tuned!",
    "Shark on watch! 🦈 Awaiting the LLM's next move.",
    "Shark patrol on duty. ⚓️ Holding the line for the LLM!",
]


class QueryModelTool(BaseTool):
    MAX_ATTEMPTS: int = 10
    default_name: ClassVar[str] = "Query LLM"
    default_description: ClassVar[str] = (
        "Sends a prompt to the LLM for linguistic analysis."
    )
    system_prompt: str = (
        "You are a research linguist specializing in rare, obscure, dead constructed-languages."
    )

    def __init__(
        self,
        *,
        system_prompt: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            name=name or self.default_name,
            description=description or self.default_description,
        )
        # now Pydantic knows system_prompt exists and is a string
        self.system_prompt = system_prompt

    @staticmethod
    def _log_attempt(retry_state: RetryCallState):
        n = retry_state.attempt_number
        plural = "" if n == 1 else " again"
        print(f"{GREEN}Attempting to connect to a remote LLM{plural}...{RESET}\n")

    @staticmethod
    def _log_retry_state(retry_state: RetryCallState):
        n = retry_state.attempt_number
        print(
            f"{PINK}Connection attempt failed! ({n}/{QueryModelTool.MAX_ATTEMPTS} attempts made){RESET}\n"
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before=_log_attempt,
        before_sleep=_log_retry_state,
    )
    def _try_remote(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:
        return self._llm_call(
            api_base_env="REMOTE_OPENAI_API_BASE",
            api_key_env="REMOTE_OPENAI_API_KEY",
            model_env="REMOTE_MODEL_NAME",
            prompt=prompt,
            stream_callback=stream_callback,
            print_chunks=print_chunks,
            role_name=role_name,
        )

    def _emit(self, print_chunks, stream_callback, role, content):
        if print_chunks:
            print(content, end="", flush=True)
        elif stream_callback:
            stream_callback(role, content)

    def _llm_call(
        self,
        api_base_env: str,
        api_key_env: str,
        model_env: str,
        prompt: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:
        client = OpenAI(
            base_url=os.getenv(api_base_env, ""),
            api_key=os.getenv(api_key_env, ""),
            timeout=httpx.Timeout(120.0, read=120.0, write=10.0, connect=3.0),
        )

        chosen_shark_text = random.choice(SHARK_TEXT)
        role_not_spoken_yet = True
        response_text = ""
        buffer: list[str] = []
        role = role_name or self.name

        completion = client.chat.completions.create(
            model=os.getenv(model_env, ""),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=True,
        )

        print(f"{GREEN}🤝 Connection successful! 🥰{RESET}\nNow, we wait...\n")

        with yaspin(ellipsis="...", text=chosen_shark_text).white.bold.shark.on_blue as sp:
            for chunk in completion:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", "")
                if not content:
                    continue

                if role_name and role_not_spoken_yet:
                    sp.hide()
                    sys.stdout.write("\r\033[2K")
                    sys.stdout.write(RESET)
                    sys.stdout.flush()
                    print(f"{GREEN}Waiting complete! 😊 Let's see what they have to say!{RESET}\n")
                    role_not_spoken_yet = False
                    print(f"{WHITE}>>>{role_name} speaking:{RESET}")

                buffer.append(content)
                response_text += content

                self._emit(
                    print_chunks, stream_callback, role, f"{GRAY}{content}{RESET}"
                )
                if chunk.choices[0].finish_reason is not None:
                    break

            return response_text or "[ERROR] No content returned."

    def _run(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:
        try:
            return self._try_remote(
                prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )
        except Exception:
            # fallback to local
            print(f"⚠️ {YELLOW}Falling back to utilizing a local LLM instead...{RESET}")
            return self._llm_call(
                api_base_env="LOCAL_OPENAI_API_BASE",
                api_key_env="LOCAL_OPENAI_API_KEY",
                model_env="LOCAL_MODEL_NAME",
                prompt=prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )
