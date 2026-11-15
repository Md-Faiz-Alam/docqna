"""GeminiProvider and provider selection, using a fake SDK client (no network, no key)."""

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors, types

from docqna.api.dependencies import build_llm
from docqna.api.errors import resolve_error_spec
from docqna.exceptions import (
    ConfigurationError,
    GenerationError,
    LLMConnectionError,
    LLMModelError,
    LLMRateLimitError,
)
from docqna.generation.providers import GeminiProvider, OllamaProvider
from docqna.generation.providers.base import TokenUsage
from docqna.settings import Settings


def _response(text: str | None, block_reason: str | None = None) -> Any:
    return SimpleNamespace(text=text, prompt_feedback=SimpleNamespace(block_reason=block_reason))


class FakeModels:
    def __init__(self, *, reply: Any = None, chunks: list[Any] | None = None, error: Any = None):
        self.reply = reply
        self.chunks = chunks or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.reply

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        yield from self.chunks

    def get(self, *, model: str) -> Any:
        if self.error:
            raise self.error
        return SimpleNamespace(name=f"models/{model}")


def make_provider(models: FakeModels, **kwargs: Any) -> GeminiProvider:
    return GeminiProvider(
        api_key="", model="gemini-test", client=SimpleNamespace(models=models), **kwargs
    )


def test_generate_sends_system_prompt_and_generation_config() -> None:
    models = FakeModels(reply=_response("  Meals are capped at 60 [1].  "))
    provider = make_provider(models, temperature=0.0, max_tokens=320)

    assert provider.generate("question", system_prompt="be grounded") == (
        "Meals are capped at 60 [1]."
    )

    call = models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "question"
    config = call["config"]
    assert config.system_instruction == "be grounded"
    assert config.temperature == 0.0
    assert config.max_output_tokens == 320
    assert config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_empty_thinking_level_uses_model_default() -> None:
    models = FakeModels(reply=_response("ok"))
    make_provider(models, thinking_level="").generate("q")

    assert models.calls[0]["config"].thinking_config is None


def test_invalid_thinking_level_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        make_provider(FakeModels(), thinking_level="extreme")


def test_missing_api_key_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="  ", model="gemini-test")


@pytest.mark.parametrize("reply", [_response(""), _response(None), _response("x", "SAFETY")])
def test_empty_or_blocked_response_raises(reply: Any) -> None:
    with pytest.raises(GenerationError):
        make_provider(FakeModels(reply=reply)).generate("q")


def test_empty_prompt_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_provider(FakeModels()).generate("   ")


def test_stream_yields_text_chunks_and_skips_empty_ones() -> None:
    chunks = [_response("Meals "), _response(None), _response("are capped [1].")]
    provider = make_provider(FakeModels(chunks=chunks))

    assert list(provider.generate_stream("q")) == ["Meals ", "are capped [1]."]


def test_empty_stream_raises() -> None:
    with pytest.raises(GenerationError):
        list(make_provider(FakeModels(chunks=[_response(None)])).generate_stream("q"))


@pytest.mark.parametrize(
    ("error", "expected", "http_status"),
    [
        (errors.ClientError(429, {"error": {"message": "quota"}}), LLMRateLimitError, 503),
        (errors.ClientError(404, {"error": {"message": "not found"}}), LLMModelError, 503),
        (errors.ClientError(403, {"error": {"message": "denied"}}), ConfigurationError, 500),
        (
            errors.ClientError(400, {"error": {"message": "API key not valid."}}),
            ConfigurationError,
            500,
        ),
        (errors.ServerError(503, {"error": {"message": "overloaded"}}), LLMConnectionError, 503),
        (ConnectionError("down"), LLMConnectionError, 503),
        (RuntimeError("boom"), GenerationError, 500),
    ],
)
def test_sdk_errors_map_to_domain_errors(
    error: Exception, expected: type[Exception], http_status: int
) -> None:
    for call in (lambda p: p.generate("q"), lambda p: list(p.generate_stream("q"))):
        with pytest.raises(expected) as caught:
            call(make_provider(FakeModels(error=error)))
        assert resolve_error_spec(caught.value).status_code == http_status  # type: ignore[arg-type]


def test_error_messages_never_contain_the_api_key() -> None:
    secret = "AIza-secret-key"
    error = errors.ClientError(400, {"error": {"message": f"API key not valid: {secret}"}})
    provider = GeminiProvider(
        api_key=secret, model="gemini-test", client=SimpleNamespace(models=FakeModels(error=error))
    )

    with pytest.raises(ConfigurationError) as caught:
        provider.generate("q")
    assert secret not in str(caught.value)


def test_health_check() -> None:
    assert make_provider(FakeModels()).health_check() is True
    assert make_provider(FakeModels(error=ConnectionError())).health_check() is False


def test_build_llm_defaults_to_ollama() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_provider == "ollama"
    assert isinstance(build_llm(settings), OllamaProvider)


def test_build_llm_selects_gemini() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="gemini",
        gemini_api_key="test-key",
        gemini_model="gemini-x",
    )

    llm = build_llm(settings)

    assert isinstance(llm, GeminiProvider)
    assert llm.model == "gemini-x"
    assert settings.llm_model == "gemini-x"
    assert "test-key" not in repr(settings)


def test_build_llm_gemini_without_key_fails_clearly() -> None:
    settings = Settings(_env_file=None, llm_provider="gemini")  # type: ignore[call-arg]

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        build_llm(settings)


def test_each_provider_uses_its_own_measured_grounding_threshold() -> None:
    ollama = Settings(_env_file=None)  # type: ignore[call-arg]
    gemini = Settings(_env_file=None, llm_provider="gemini")  # type: ignore[call-arg]

    assert ollama.active_grounding_min_score == 3.3
    assert gemini.active_grounding_min_score == 1.5


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, llm_provider="openai")  # type: ignore[call-arg]


def _usage(prompt: int, output: int, thoughts: int | None = None) -> Any:
    return SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts
    )


def test_token_usage_is_recorded_for_cost_reporting() -> None:
    reply = _response("Answer [1].")
    reply.usage_metadata = _usage(900, 25, 4)
    provider = make_provider(FakeModels(reply=reply))

    provider.generate("prompt")

    assert provider.last_usage == TokenUsage(input_tokens=900, output_tokens=25, thinking_tokens=4)


def test_streaming_keeps_the_final_token_usage() -> None:
    first, last = _response("Ans"), _response("wer [1].")
    first.usage_metadata = _usage(900, 2)
    last.usage_metadata = _usage(900, 25)
    provider = make_provider(FakeModels(chunks=[first, last]))

    assert "".join(provider.generate_stream("prompt")) == "Answer [1]."
    assert provider.last_usage == TokenUsage(input_tokens=900, output_tokens=25)


def test_missing_usage_metadata_leaves_usage_empty() -> None:
    provider = make_provider(FakeModels(reply=_response("Answer [1].")))

    provider.generate("prompt")

    assert provider.last_usage is None


def test_ollama_token_counts_are_recorded() -> None:
    from ollama import ChatResponse

    provider = OllamaProvider(base_url="http://127.0.0.1:1", model="phi3:mini")
    reply = ChatResponse(
        model="phi3:mini",
        message={"role": "assistant", "content": "Answer [1]."},
        done=True,
        prompt_eval_count=850,
        eval_count=40,
    )
    provider.client = SimpleNamespace(chat=lambda **kwargs: reply)  # type: ignore[assignment]

    assert provider.generate("prompt") == "Answer [1]."
    assert provider.last_usage == TokenUsage(input_tokens=850, output_tokens=40)
