import os
import sys
import logging
import httpx
import random
import time
from yaspin import yaspin, Spinner
from yaspin.spinners import Spinners
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
        print(f"{PINK}Connection attempt failed! ({n + 1}/5 attempts made){RESET}\n")

    @staticmethod
    def _get_random_spinner():
        SHARK_TEXT = [
            "Let us admire the pretty shark while we wait for LLM's response... ",
            "Such a happy shark! (we're waiting... 🙃) ",
            "Marooned in shark-waters. 🏝️ But it's okay, the LLM-ship will come save us! ",
            "Waiting on the LLM. In the meantime, enjoy this shark. ",
            "Sharks as a species are older than Jupiter's rings. Not exactly relevant, but cool. ",
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
                    ], # type: ignore
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
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
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
    ) -> dict[str, str]:
        client = OpenAI(
            base_url=os.getenv(api_base_env, ""),
            api_key=os.getenv(api_key_env, ""),
            timeout=httpx.Timeout(120.0, read=120.0, write=10.0, connect=5.0),
        )
        response_text = ""
        role = role_name or self.name
        if "Glossator" in role:
            self.gloss_model = os.getenv(
                model_env, "([Error] Not able to retrieve the model!)"
            )

        completion = client.chat.completions.create(
            model=os.getenv(model_env, ""),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            stream=True,
        )

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

                content = getattr(chunk.choices[0].delta, "content", "") or ""
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
            content = getattr(chunk.choices[0].delta, "content", "") or ""
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
            response_text = "[ERROR] No content returned."

        return {
            "response_text": response_text,
            "gloss_model": getattr(self, "gloss_model", "<unset>"),
        }

    def _run(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> dict[str, str]:
        try:
            return self._try_remote(
                prompt,
                stream_callback=stream_callback,
                print_chunks=print_chunks,
                role_name=role_name,
            )
        except Exception:
            # exit if out of OpenRouter calls
            print(f"⚠️ [{time.ctime()}] Clearly we're out of LLM calls for the day. Stopping for now. Goodbye for now. 🫡")
            sys.exit()
            # fallback to local
            # print(
            #     f"⚠️ {YELLOW}Falling back to utilizing a local LLM instead...\n{RESET}"
            # )
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
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
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