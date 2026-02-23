"""Abstract LLM provider interface."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def analyze(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response."""
        ...

    @abstractmethod
    def analyze_batch(self, prompts: list[str]) -> list[str]:
        """Send multiple prompts and return responses."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        ...
