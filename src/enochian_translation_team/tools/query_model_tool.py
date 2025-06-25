import os
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
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
RESET = "\033[0m"


class QueryModelTool(BaseTool):
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

    def _llm_call(
        self,
        api_base_env: str,
        api_key_env: str,
        model_env: str,
        prompt: str,
        stream_callback: Optional[Callable[[str, str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:

        client = OpenAI(
            base_url=os.getenv(api_base_env, ""),
            api_key=os.getenv(api_key_env, ""),
            timeout=90,
        )

        response_text = ""
        buffer: list[str] = []
        role = role_name or self.name
        idle_deadline = time.time() + 45

        if role_name:
            print(f"{WHITE}>>>{role} speaking...{RESET}")

        completion = client.chat.completions.create(
            model=os.getenv(model_env, ""),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=True,
        )

        for chunk in completion:
            idle_deadline = time.time() + 45
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "")
            if not content:
                if time.time() > idle_deadline:
                    raise TimeoutError("[Error] Timed out because remote LLM stream idle > 30 seconds")
                continue

            buffer.append(content)
            response_text += content

            if print_chunks:
                print(f"{GRAY}{content}{RESET}", end="", flush=True)
            elif stream_callback:
                # thinking tokens
                stream_callback(role, f"{GRAY}{content}{RESET}", "thinking")

            if chunk.choices[0].finish_reason is not None:
                break

            if time.time() > idle_deadline:
                raise TimeoutError("[Error] Timed out because remote LLM stream idle > 30 seconds")

        # emit final answer tokens
        if print_chunks:
            print()  # newline after thinking
            for token in buffer:
                print(f"{WHITE}{token}{RESET}", end="", flush=True)
            print()
        elif stream_callback:
            for token in buffer:
                stream_callback(role, f"{WHITE}{token}{RESET}", "answer")

        return response_text or "[ERROR] No content returned."

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    def _try_remote(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
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

    def _run(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str, Optional[str]], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:
        try:
            print(f"{GREEN}Attempting to connect to a remote LLM...{RESET}")
            return self._try_remote(
                prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )
        except Exception:
            # fallback to local
            print(f"{GREEN}Attempting to utilize a local LLM...{RESET}")
            return self._llm_call(
                api_base_env="LOCAL_OPENAI_API_BASE",
                api_key_env="LOCAL_OPENAI_API_KEY",
                model_env="LOCAL_MODEL_NAME",
                prompt=prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )
