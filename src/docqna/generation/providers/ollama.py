import ollama

from docqna.exceptions import GenerationError, LLMConnectionError
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

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response using Ollama."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
            )

        except ConnectionError as exc:
            raise LLMConnectionError(f"Could not connect to Ollama at {self.base_url}.") from exc

        except Exception as exc:
            raise GenerationError(f"Ollama generation failed using model '{self.model}'.") from exc

        content = response["message"]["content"].strip()

        if not content:
            raise GenerationError("Ollama returned an empty response.")

        return content

    def health_check(self) -> bool:
        """Return True when the Ollama server is reachable."""

        try:
            self.client.list()
            return True

        except Exception:
            return False
