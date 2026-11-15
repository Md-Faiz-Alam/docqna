"""LLM provider backed by Google's hosted Gemini API (official `google-genai` SDK)."""

from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import errors, types

from docqna.exceptions import (
    ConfigurationError,
    GenerationError,
    LLMConnectionError,
    LLMModelError,
    LLMRateLimitError,
)
from docqna.generation.providers.base import BaseLLMProvider

THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class GeminiProvider(BaseLLMProvider):
    """Generate grounded answers with a hosted Gemini model.

    The API key is only passed to the SDK client; it is never logged or
    included in error messages.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 320,
        timeout_seconds: float = 60.0,
        thinking_level: str | None = "minimal",
        client: Any | None = None,
    ) -> None:
        if not api_key.strip() and client is None:
            raise ConfigurationError("GEMINI_API_KEY must be set when LLM_PROVIDER=gemini.")

        level = (thinking_level or "").strip().lower() or None
        if level is not None and level not in THINKING_LEVELS:
            raise ConfigurationError(
                f"GEMINI_THINKING_LEVEL must be one of {sorted(THINKING_LEVELS)} or empty."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_level = level
        # The SDK takes its timeout in milliseconds.
        self.client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000))
        )

    def _config(self, system_prompt: str | None) -> types.GenerateContentConfig:
        thinking = (
            types.ThinkingConfig(thinking_level=types.ThinkingLevel(self.thinking_level.upper()))
            if self.thinking_level
            else None
        )
        return types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            temperature=self.temperature,
            # Gemini counts thinking tokens against this limit too.
            max_output_tokens=self.max_tokens,
            thinking_config=thinking,
        )

    @staticmethod
    def _check_prompt(prompt: str) -> None:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

    def _translate(self, exc: Exception) -> Exception:
        if isinstance(exc, errors.APIError):
            code = exc.code
            message = (exc.message or "").lower()
            if code == 429:
                return LLMRateLimitError(
                    "The answer service is busy (Gemini quota reached). Please try again shortly."
                )
            if code in (401, 403) or "api key" in message:
                return ConfigurationError("Gemini rejected the configured API key.")
            if code == 404:
                return LLMModelError(f"Gemini model '{self.model}' is not available.")
            if isinstance(exc, errors.ServerError):
                return LLMConnectionError("Gemini is temporarily unavailable.")
            return GenerationError(f"Gemini generation failed using model '{self.model}'.")
        if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or any(
            word in type(exc).__name__ for word in ("Connect", "Timeout", "Network")
        ):
            return LLMConnectionError("Could not connect to Gemini.")
        return GenerationError(f"Gemini generation failed using model '{self.model}'.")

    @staticmethod
    def _blocked(response: Any) -> bool:
        feedback = getattr(response, "prompt_feedback", None)
        return bool(feedback is not None and getattr(feedback, "block_reason", None))

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a complete response."""

        self._check_prompt(prompt)
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config(system_prompt)
            )
        except Exception as exc:
            raise self._translate(exc) from exc

        if self._blocked(response):
            raise GenerationError("Gemini declined to answer this request.")
        content = (response.text or "").strip()
        if not content:
            raise GenerationError("Gemini returned an empty response.")
        return content

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Yield response text as Gemini produces it."""

        self._check_prompt(prompt)
        produced = False
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config(system_prompt)
            )
            for chunk in stream:
                if self._blocked(chunk):
                    raise GenerationError("Gemini declined to answer this request.")
                text = chunk.text
                if text:
                    produced = True
                    yield text
        except GenerationError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

        if not produced:
            raise GenerationError("Gemini returned an empty response.")

    def health_check(self) -> bool:
        """Return True when the model is reachable with this key.

        Reads model metadata only, so it uses no generation quota.
        """

        try:
            self.client.models.get(model=self.model)
            return True
        except Exception:
            return False
