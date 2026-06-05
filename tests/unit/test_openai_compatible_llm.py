from __future__ import annotations

from types import SimpleNamespace

from dagqa.config import LLMConfig
from dagqa.llm import openai_compatible
from dagqa.schemas import LLMRequest


async def test_openai_compatible_llm_uses_selected_model_and_messages(monkeypatch) -> None:
    captured: dict = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=1),
            )

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("CLUSTER_API_KEY", "secret")
    monkeypatch.setattr(openai_compatible, "AsyncOpenAI", Client)
    llm = openai_compatible.OpenAICompatibleLanguageModel(
        LLMConfig(
            provider="cluster",
            model="google/gemma-4-31B-it",
            api_key_env="CLUSTER_API_KEY",
            api_base="http://cluster/inference",
        )
    )

    response = await llm.complete(
        LLMRequest(system="System instruction", prompt="Question", temperature=0.2)
    )

    assert captured["client"] == {
        "api_key": "secret",
        "base_url": "http://cluster/inference",
        "timeout": 120.0,
        "max_retries": 0,
    }
    assert captured["model"] == "google/gemma-4-31B-it"
    assert captured["messages"] == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Question"},
    ]
    assert response.text == "ready"
    assert response.model == "google/gemma-4-31B-it"
    assert response.metadata["retry_count"] == 0


async def test_openai_compatible_llm_retries_retryable_errors(monkeypatch) -> None:
    expected_attempts = 2
    attempts = 0

    class RetryableError(Exception):
        status_code = 429

    class Completions:
        async def create(self, **kwargs):  # noqa: ANN001, ARG002
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RetryableError("rate limited")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
                usage=None,
            )

    class Client:
        def __init__(self, **kwargs):  # noqa: ANN001, ARG002
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setenv("CLUSTER_API_KEY", "secret")
    monkeypatch.setattr(openai_compatible, "AsyncOpenAI", Client)
    llm = openai_compatible.OpenAICompatibleLanguageModel(
        LLMConfig(
            provider="cluster",
            model="model",
            api_key_env="CLUSTER_API_KEY",
            api_base="http://cluster/inference",
            max_retries=1,
            retry_initial_delay_seconds=0.001,
        )
    )

    response = await llm.complete(LLMRequest(system="System", prompt="Question"))

    assert response.text == "ready"
    assert response.metadata["retry_count"] == 1
    assert attempts == expected_attempts
