"""LLM provider factory."""

from stock_prediction.llm.base import LLMProvider


def get_llm_provider(name: str = "ollama") -> LLMProvider:
    """Factory to get an LLM provider by name."""
    if name == "ollama":
        from stock_prediction.llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    raise ValueError(f"Unknown LLM provider: {name}")


__all__ = ["get_llm_provider", "LLMProvider"]
