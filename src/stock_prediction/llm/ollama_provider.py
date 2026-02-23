"""Ollama LLM provider implementation."""

import ollama as ollama_client

from stock_prediction.config import get_setting
from stock_prediction.llm.base import LLMProvider
from stock_prediction.utils.logging import get_logger

logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """LLM provider using local Ollama instance."""

    def __init__(self):
        self.model = get_setting("llm", "ollama", "model", default="llama3.1:8b")
        self.base_url = get_setting("llm", "ollama", "base_url", default="http://localhost:11434")
        self.timeout = get_setting("llm", "ollama", "timeout", default=120)
        self._client = ollama_client.Client(host=self.base_url, timeout=self.timeout)

    def analyze(self, prompt: str) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return ""

    def analyze_batch(self, prompts: list[str]) -> list[str]:
        results = []
        for prompt in prompts:
            results.append(self.analyze(prompt))
        return results

    def is_available(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False
