from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from typing import Any, Iterator

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

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield text deltas for the given messages."""
        response = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield response.content


class MockLLMService(BaseLLMService):
    provider = "mock"
    QUERY_UNDERSTANDING_MARKER = "SMARTBUY_QUERY_UNDERSTANDING_JSON"

    def __init__(self, model: str = "mock-chat") -> None:
        self.model = model

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if _is_query_understanding_request(messages):
            return LLMResponse(
                content=_mock_query_understanding_json(messages),
                model=self.model,
                provider=self.provider,
                raw=None,
            )

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

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        content = self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ).content
        for index in range(0, len(content), 12):
            yield content[index : index + 12]


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

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
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
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        endpoint = f"{self.api_base}/chat/completions"
        if self._http_client is not None:
            response = self._http_client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            yield from _iter_openai_compatible_deltas(response.iter_lines())
            return

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    yield from _iter_openai_compatible_deltas(response.iter_lines())
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM streaming request failed: {exc}") from exc

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


def get_llm_service(*, timeout_seconds: float | None = None) -> BaseLLMService:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "mock":
        return MockLLMService(model=settings.LLM_MODEL)

    if provider in {"openai_compatible", "real"}:
        return OpenAICompatibleLLMService(
            api_base=settings.LLM_API_BASE,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else settings.LLM_TIMEOUT_SECONDS,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _last_user_message(messages: list[LLMMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _is_query_understanding_request(messages: list[LLMMessage]) -> bool:
    return any(
        MockLLMService.QUERY_UNDERSTANDING_MARKER in message.content
        for message in messages
    )


def _mock_query_understanding_json(messages: list[LLMMessage]) -> str:
    try:
        payload = json.loads(_last_user_message(messages))
    except json.JSONDecodeError:
        payload = {}
    query = str(payload.get("current_query") or "")
    memory = payload.get("shopping_memory") if isinstance(payload, dict) else {}
    if not isinstance(memory, dict):
        memory = {}
    budget = memory.get("budget") if isinstance(memory.get("budget"), dict) else {}
    memory_budget_max = _int_or_none(budget.get("max")) if isinstance(budget, dict) else None
    category = memory.get("category") if memory.get("category") in {"phone", "shoes", "skincare"} else None

    output: dict[str, Any] = {
        "is_follow_up": bool(memory),
        "intent": "shopping_guide" if memory else "clarification",
        "category": category,
        "budget": {
            "min": None,
            "max": memory_budget_max,
            "currency": "CNY",
        },
        "preferences": [],
        "negative_preferences": [],
        "compare_product_ids": [],
        "referenced_product_indices": [],
        "confidence": 0.72,
        "reason": "ambiguous_query",
    }

    budget_match = re.search(r"(?P<value>\d{3,6})", query)
    if budget_match:
        output["budget"]["max"] = int(budget_match.group("value"))
        output["confidence"] = 0.84
        output["reason"] = "budget_update_follow_up"
    elif ("贵" in query or "放宽" in query) and memory_budget_max:
        output["budget"]["max"] = int(memory_budget_max * 1.5)
        output["confidence"] = 0.84
        output["reason"] = "budget_and_preference_update_follow_up"

    if any(token in query for token in ["轻", "重"]):
        output["preferences"].append("轻便")
    if any(token in query for token in ["通勤", "上班"]):
        output["preferences"].append("通勤")
    if "拍" in query or "vlog" in query.lower():
        output["preferences"].append("拍照")
    if "苹果" in query and any(token in query for token in ["不要", "不考虑", "别"]):
        output["negative_preferences"].append("苹果")
    if any(token in query for token in ["对比", "比较", "哪个", "比"]):
        output["intent"] = "compare"
        output["reason"] = "compare_follow_up"
    if "第一个" in query:
        output["referenced_product_indices"].append(1)
    if "第二个" in query:
        output["referenced_product_indices"].append(2)

    return json.dumps(output, ensure_ascii=False)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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


def _iter_openai_compatible_deltas(lines: Any) -> Iterator[str]:
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="ignore")
        else:
            line = str(raw_line)
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield content
            continue
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                yield content
