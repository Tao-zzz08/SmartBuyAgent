from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.cache.cache_service import InMemoryCacheService
from app.chat.llm_answer_composer import LLMAnswerComposer
from app.services.llm import BaseLLMService, LLMMessage, LLMResponse
from app.streaming.safety import STREAM_GUARDED_FALLBACK_ANSWER

from test_chat_api import (  # noqa: E402
    _cleanup,
    _override_dependencies,
    _prepare_api_stack,
    _sse_event_data,
    _sse_event_datas,
    _sse_events,
)
from app.main import app  # noqa: E402


class UnsafeStreamingLLMService(BaseLLMService):
    provider = "unsafe_fake"

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="这是一段非流式安全占位回答。",
            model="unsafe-fake",
            provider=self.provider,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield "这款整体更适合拍照，"
        yield "我已经帮你下单。"


class SafeStreamingLLMService(BaseLLMService):
    provider = "safe_fake"

    answer = "建议优先看候选商品，它在预算内，并且标签包含拍照和续航。"

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self.answer,
            model="safe-fake",
            provider=self.provider,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield "建议优先看"
        yield "候选商品，它在预算内，"
        yield "并且标签包含拍照和续航。"


class FailingCacheService(InMemoryCacheService):
    def get_json(self, key: str):
        raise RuntimeError("cache unavailable")

    def set_json(self, key: str, value, ttl_seconds: int) -> None:
        raise RuntimeError("cache unavailable")

    def delete(self, key: str) -> None:
        raise RuntimeError("cache unavailable")

    def incr(self, key: str, ttl_seconds: int | None = None) -> int:
        raise RuntimeError("cache unavailable")


def test_chat_stream_blocks_unsafe_tokens_and_returns_guarded_result() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_guarded_test.db",
        "chroma_chat_stream_guarded_test",
    )
    llm_answer_composer = LLMAnswerComposer(UnsafeStreamingLLMService())
    _override_dependencies(
        TestingSessionLocal,
        chroma_client,
        llm_answer_composer=llm_answer_composer,
        cache_service=InMemoryCacheService(),
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        events = _sse_events(response.text)
        token_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "token"
        )
        result = _sse_event_data(events, "result")
        guard_event = _sse_event_data(events, "stream_guard")
        error_event = _sse_event_data(events, "error")
        response_node_end = next(
            event["data"]
            for event in events
            if event["event"] == "node_end"
            and event["data"].get("node") == "response_compose"
        )

        assert response.status_code == 200
        assert guard_event["status"] == "blocked"
        assert guard_event["reason"] == "purchase_action"
        assert guard_event["matched_phrase"] == "我已经帮你下单"
        assert error_event["failed_node"] == "response_compose"
        assert error_event["error_type"] == "StreamSafetyViolation"
        assert response_node_end["status"] == "failed"
        assert response_node_end["summary"]["guarded"] is True
        assert _sse_event_data(events, "done")["status"] == "guarded"
        assert result["answer"] == STREAM_GUARDED_FALLBACK_ANSWER
        assert result["product_cards"]
        assert result["citations"]
        assert "我已经帮你下单" not in token_text
        assert any(
            step.get("step") == "stream_guard"
            and step.get("status") == "blocked"
            for step in result["trace"]
        )
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_safe_tokens_still_match_final_answer() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_safe_guard_test.db",
        "chroma_chat_stream_safe_guard_test",
    )
    llm_answer_composer = LLMAnswerComposer(SafeStreamingLLMService())
    _override_dependencies(
        TestingSessionLocal,
        chroma_client,
        llm_answer_composer=llm_answer_composer,
        cache_service=InMemoryCacheService(),
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        events = _sse_events(response.text)
        token_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "token"
        )
        result = _sse_event_data(events, "result")

        assert response.status_code == 200
        assert _sse_event_data(events, "done")["status"] == "ok"
        assert _sse_event_datas(events, "stream_guard") == []
        assert token_text == result["answer"]
        assert result["answer"] == SafeStreamingLLMService.answer
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_guard_still_works_when_cache_fails() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_guard_cache_failure_test.db",
        "chroma_chat_stream_guard_cache_failure_test",
    )
    llm_answer_composer = LLMAnswerComposer(UnsafeStreamingLLMService())
    _override_dependencies(
        TestingSessionLocal,
        chroma_client,
        llm_answer_composer=llm_answer_composer,
        cache_service=FailingCacheService(),
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        events = _sse_events(response.text)

        assert response.status_code == 200
        assert _sse_event_data(events, "stream_guard")["status"] == "blocked"
        assert _sse_event_data(events, "result")["answer"] == STREAM_GUARDED_FALLBACK_ANSWER
        assert _sse_event_data(events, "done")["status"] == "guarded"
    finally:
        _cleanup(engine, db_path, chroma_dir)
