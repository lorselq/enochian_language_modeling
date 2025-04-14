import os
from openai import OpenAI
from crewai.tools import BaseTool

class QueryModelTool(BaseTool):
    name: str = "Query LM Studio"
    description: str = "Sends a prompt to the local LLM for linguistic analysis."
    system_prompt: str = "You are a computational linguist specializing in dead and obscure languages."

    def _run(self, prompt: str) -> str:
        try:
            client = OpenAI(
                base_url=os.getenv("OPENAI_API_BASE", "failure-if-not-in-env-lol"),
                api_key=os.getenv("OPENAI_API_KEY", "sk-local-testing")
            )
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "openai/local-model"),
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
            )
            content = response.choices[0].message.content
            return content if content else "[ERROR] Model returned no content."
        except Exception as e:
            return f"[ERROR] Query failed: {e}"
