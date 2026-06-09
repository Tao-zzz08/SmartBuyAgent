from __future__ import annotations

from typing import Any

import pytest

from app.services import llm as llm_module
from app.services.llm import (
    BaseLLMService,
    LLMMessage,
    MockLLMService,
    OpenAICompatibleLLMService,
    get_llm_service,
)


class FakeLLMResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeLLMClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeLLMResponse:
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return FakeLLMResponse(self.payload)


def test_mock_llm_service_returns_stable_response() -> None:
    service = MockLLMService(model="mock-chat")
    messages = [
        LLMMessage(role="system", content="You are a shopping assistant."),
        LLMMessage(role="user", content="推荐一款拍照好的手机"),
    ]

    first = service.chat(messages)
    second = service.chat(messages)

    assert first.content
    assert first.content == second.content
    assert "推荐一款拍照好的手机" in first.content
    assert first.provider == "mock"
    assert first.model == "mock-chat"


def test_openai_compatible_llm_service_calls_chat_completions() -> None:
    fake_client = FakeLLMClient(
        {
            "choices": [
                {
                    "message": {
                        "content": "推荐回答",
                    }
                }
            ],
            "model": "test-model",
        }
    )
    service = OpenAICompatibleLLMService(
        api_base="https://llm.example.com/v1/",
        api_key="test-key",
        model="test-model",
        timeout_seconds=12.5,
        max_tokens=512,
        temperature=0.1,
        http_client=fake_client,
    )
    messages = [
        LLMMessage(role="system", content="system prompt"),
        LLMMessage(role="user", content="user prompt"),
    ]

    response = service.chat(messages)

    assert response.content == "推荐回答"
    assert response.model == "test-model"
    assert response.provider == "openai_compatible"
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["url"] == "https://llm.example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in response.content
    assert call["json"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }
    assert call["timeout"] == 12.5


def test_openai_compatible_chat_allows_runtime_generation_options() -> None:
    fake_client = FakeLLMClient(
        {"choices": [{"message": {"content": "runtime options"}}]}
    )
    service = OpenAICompatibleLLMService(
        api_base="https://llm.example.com/v1",
        api_key="test-key",
        model="test-model",
        http_client=fake_client,
    )

    response = service.chat(
        [LLMMessage(role="user", content="hello")],
        max_tokens=128,
        temperature=0.7,
    )

    assert response.content == "runtime options"
    assert fake_client.calls[0]["json"]["max_tokens"] == 128
    assert fake_client.calls[0]["json"]["temperature"] == 0.7


def test_openai_compatible_requires_api_base_and_key() -> None:
    with pytest.raises(ValueError, match="LLM_API_BASE"):
        OpenAICompatibleLLMService(
            api_base=None,
            api_key="test-key",
            model="test-model",
        )

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatibleLLMService(
            api_base="https://llm.example.com/v1",
            api_key=None,
            model="test-model",
        )


def test_openai_compatible_raises_for_invalid_response() -> None:
    service = OpenAICompatibleLLMService(
        api_base="https://llm.example.com/v1",
        api_key="test-key",
        model="test-model",
        http_client=FakeLLMClient({"choices": [{"message": {}}]}),
    )

    with pytest.raises(ValueError, match="content"):
        service.chat([LLMMessage(role="user", content="hello")])


def test_get_llm_service_returns_mock_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(llm_module.settings, "LLM_MODEL", "mock-chat")

    service = get_llm_service()

    assert isinstance(service, BaseLLMService)
    assert isinstance(service, MockLLMService)
    assert service.chat([LLMMessage(role="user", content="hi")]).provider == "mock"


def test_get_llm_service_returns_openai_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "openai_compatible")
    monkeypatch.setattr(llm_module.settings, "LLM_API_BASE", "https://llm.example.com/v1")
    monkeypatch.setattr(llm_module.settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.settings, "LLM_MODEL", "test-model")

    service = get_llm_service()

    assert isinstance(service, OpenAICompatibleLLMService)


def test_get_llm_service_treats_real_as_openai_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "real")
    monkeypatch.setattr(llm_module.settings, "LLM_API_BASE", "https://llm.example.com/v1")
    monkeypatch.setattr(llm_module.settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm_module.settings, "LLM_MODEL", "test-model")

    service = get_llm_service()

    assert isinstance(service, OpenAICompatibleLLMService)


def test_get_llm_service_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_module.settings, "LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_service()
