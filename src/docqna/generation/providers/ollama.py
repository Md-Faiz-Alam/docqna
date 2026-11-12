from collections.abc import Iterator
from typing import Any

import ollama

from docqna.exceptions import GenerationError, LLMConnectionError, LLMModelError
from docqna.generation.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """LLM provider backed by a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 320,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = ollama.Client(host=self.base_url, timeout=timeout_seconds)

    def _messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _chat(self, messages: list[dict[str, str]], *, stream: bool) -> Any:
        options = {"temperature": self.temperature, "num_predict": self.max_tokens}
        try:
            if stream:
                return self.client.chat(
                    model=self.model, messages=messages, stream=True, options=options
                )
            return self.client.chat(model=self.model, messages=messages, options=options)
        except Exception as exc:
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> Exception:
        if isinstance(exc, ConnectionError):
            return LLMConnectionError(f"Could not connect to Ollama at {self.base_url}.")
        if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
            return LLMModelError(f"Ollama model '{self.model}' is not available.")
        return GenerationError(f"Ollama generation failed using model '{self.model}'.")

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a complete response."""

        response = self._chat(self._messages(prompt, system_prompt), stream=False)
        content = response["message"]["content"].strip()
        if not content:
            raise GenerationError("Ollama returned an empty response.")
        return content

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Yield response text as Ollama produces it."""

        stream = self._chat(self._messages(prompt, system_prompt), stream=True)
        produced = False
        try:
            for part in stream:
                text = part["message"]["content"]
                if text:
                    produced = True
                    yield text
        except Exception as exc:
            raise self._translate(exc) from exc

        if not produced:
            raise GenerationError("Ollama returned an empty response.")

    def health_check(self) -> bool:
        """Return True when the Ollama server is reachable."""

        try:
            self.client.list()
            return True
        except Exception:
            return False
