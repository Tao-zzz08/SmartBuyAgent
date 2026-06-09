from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    raw: dict[str, Any] | None = None


class BaseLLMService(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Return one chat response for the given messages."""


class MockLLMService(BaseLLMService):
    provider = "mock"

    def __init__(self, model: str = "mock-chat") -> None:
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        user_message = _last_user_message(messages)
        content = (
            "这是一个模拟 LLM 回答。请基于候选商品和引用信息进行最终推荐。"
            f" 用户问题摘要：{user_message or '无'}"
        )
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            raw=None,
        )


class OpenAICompatibleLLMService(BaseLLMService):
    provider = "openai_compatible"

    def __init__(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        timeout_seconds: float = 30.0,
        max_tokens: int = 800,
        temperature: float = 0.2,
        http_client: Any | None = None,
    ) -> None:
        if not api_base:
            raise ValueError("LLM_API_BASE is required for openai_compatible provider")
        if not api_key:
            raise ValueError("LLM_API_KEY is required for openai_compatible provider")
        if not model:
            raise ValueError("LLM_MODEL is required for openai_compatible provider")

        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._http_client = http_client

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._post_chat(payload=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc

        content = _extract_chat_content(body)
        response_model = body.get("model") if isinstance(body, dict) else None
        return LLMResponse(
            content=content,
            model=response_model or self.model,
            provider=self.provider,
            raw=body,
        )

    def _post_chat(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        endpoint = f"{self.api_base}/chat/completions"
        if self._http_client is not None:
            return self._http_client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )

        with httpx.Client(timeout=self.timeout_seconds) as client:
            return client.post(endpoint, json=payload, headers=headers)


def get_llm_service() -> BaseLLMService:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "mock":
        return MockLLMService(model=settings.LLM_MODEL)

    if provider in {"openai_compatible", "real"}:
        return OpenAICompatibleLLMService(
            api_base=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _last_user_message(messages: list[LLMMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _extract_chat_content(body: Any) -> str:
    if not isinstance(body, dict):
        raise ValueError("LLM response must be a JSON object")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("LLM response choice must be an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response choice missing message")

    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("LLM response message missing content")

    return content
