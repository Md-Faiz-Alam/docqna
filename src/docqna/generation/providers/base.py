from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Tokens one LLM call used, as reported by the provider."""

    input_tokens: int
    output_tokens: int
    # Hidden reasoning tokens (Gemini "thinking"); billed like output tokens.
    thinking_tokens: int = 0


class BaseLLMProvider(ABC):
    """Base interface for all LLM providers."""

    # Usage of the most recent completed call, when the provider reports it.
    # Used by experiments/measure_speed_and_cost.py.
    last_usage: TokenUsage | None = None

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
