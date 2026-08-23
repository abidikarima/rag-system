import yaml
from groq import Groq
from typing import List, Optional


class GroqLLM:
    """
    Robust Groq LLM wrapper with fallback models
    """

    def __init__(
        self,
        config_path: str = "./secrets/config.yaml",
        primary_model: str = "openai/gpt-oss-120b",
        fallback_models: Optional[List[str]] = None,
    ):

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.api_key = config["GROQ_API_KEY"]

        self.client = Groq(api_key=self.api_key)

        self.primary_model = primary_model

        # fallback chain
        self.fallback_models = fallback_models or [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:

        models_to_try = [self.primary_model] + self.fallback_models

        last_error = None

        for model in models_to_try:

            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                print(f"[WARN] Model failed: {model} | Error: {e}")

                # if token limit or rate limit → try next model
                continue

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        )
