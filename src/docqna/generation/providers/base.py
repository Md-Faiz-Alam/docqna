from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLMProvider(ABC):
    """Base interface for all LLM providers."""

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Yield the response incrementally.

        Providers without native streaming fall back to one chunk.
        """
        yield self.generate(prompt, system_prompt=system_prompt)

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a text response.

        Args:
            prompt:
                User prompt or final RAG prompt.

            system_prompt:
                Optional system-level instructions.

        Returns:
            Generated text response.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check whether the LLM provider is reachable.

        Returns:
            True when the provider is available.
        """
        raise NotImplementedError
