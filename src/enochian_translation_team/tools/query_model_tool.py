import os
import time
from openai import OpenAI
from typing import Optional, Callable
from crewai.tools import BaseTool


class QueryModelTool(BaseTool):
    name: str = "Query LM Studio"
    description: str = "Sends a prompt to the local LLM for linguistic analysis."
    system_prompt: str = (
        "You are a computational linguist specializing in dead and obscure languages."
    )

    def _run(
        self, prompt: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        try:
            client = OpenAI(
                base_url=os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "sk-local-testing"),
            )

            response_text = ""

            completion = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "deepseek-r1-distill-qwen-7b"),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                stream=True,
            )

            for chunk in completion:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "")
                    if content:
                        response_text += content
                        if stream_callback:
                            stream_callback(content)
                            time.sleep(0.01)
                except Exception as inner:
                    print(f"[!] Inner stream failure: {inner}")

            return response_text or "[ERROR] No content returned from model."

        except Exception as e:
            return f"[ERROR] Query failed: {e}"
