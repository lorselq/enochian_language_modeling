import os
from typing import Optional, Callable
from openai import OpenAI
from crewai.tools import BaseTool


class QueryModelTool(BaseTool):
    name: str = "Query LM Studio"
    description: str = "Sends a prompt to the local LLM for linguistic analysis."
    system_prompt: str = (
        "You are a computational linguist specializing in dead and obscure languages."
    )

    def _run(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        print_chunks: bool = False,
        role_name: Optional[str] = None,
    ) -> str:
        GRAY = "\033[90m"
        RESET = "\033[0m"
        try:
            if role_name and "Junior" in role_name:
                print(f"{GRAY}")
            elif stream_callback:
                print(f"{RESET}>>>{role_name} speaking...")
            client = OpenAI(
                base_url=os.getenv(
                    "OPENAI_API_BASE", "this-will-fail-if-doesn't-work-lol"
                ),
                api_key=os.getenv("OPENAI_API_KEY", "sk-local-testing"),
            )

            response_text = ""
            role = role_name or self.name  # Fallback to tool name if no role given

            completion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "deepseek/deepseek-r1-0528:free"),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=1.15,
                stream=True,
            )

            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    if content:
                        response_text += content
                        if stream_callback:
                            stream_callback(role, content)
                        elif print_chunks:
                            print(f"{content}", end="", flush=True)
                except Exception as inner:
                    print(f"[!] Inner stream failure: {inner}")

            if not response_text:
                print(
                    f"[!] No content returned for role {role_name}. Stream probably failed."
                )

            return response_text or "[ERROR] No content returned from model."

        except Exception as e:
            return f"[ERROR] Query failed: {e}"
