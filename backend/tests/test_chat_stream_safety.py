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


STREAM_QUERY = "预算3000，推荐一款拍照好的手机"


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
            content="This is a non-streaming safe placeholder.",
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
        yield "This candidate matches your camera preference. "
        yield "buy now"


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


class GroundingUnsafeLLMService(BaseLLMService):
    provider = "grounding_unsafe_fake"
    answer = "这款手机很适合拍照，建议下单。"

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self.answer,
            model="grounding-unsafe-fake",
            provider=self.provider,
        )

    def stream_chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        yield "这款手机很适合拍照，"
        yield "建议下单。"


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
            json={"query": STREAM_QUERY, "debug": True},
        )
        events = _sse_events(response.text)
        draft_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "answer_draft_delta"
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
        assert guard_event["matched_phrase"] == "buy now"
        assert error_event["failed_node"] == "response_compose"
        assert error_event["error_type"] == "StreamSafetyViolation"
        assert response_node_end["status"] == "failed"
        assert response_node_end["summary"]["guarded"] is True
        assert _sse_event_data(events, "done")["status"] == "guarded"
        assert result["answer"] == STREAM_GUARDED_FALLBACK_ANSWER
        assert result["product_cards"]
        assert result["citations"]
        assert "buy now" not in draft_text
        assert any(
            step.get("step") == "stream_guard"
            and step.get("status") == "blocked"
            for step in result["trace"]
        )
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_safe_draft_is_replaced_by_final_answer() -> None:
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
            json={"query": STREAM_QUERY, "debug": True},
        )
        events = _sse_events(response.text)
        draft_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "answer_draft_delta"
        )
        result = _sse_event_data(events, "result")

        assert response.status_code == 200
        assert _sse_event_data(events, "done")["status"] == "ok"
        assert _sse_event_datas(events, "stream_guard") == []
        assert draft_text == result["answer"]
        assert _sse_event_data(events, "final_answer")["answer"] == result["answer"]
        assert result["answer"] == SafeStreamingLLMService.answer
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_grounding_guard_fallbacks_unsafe_draft() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_grounding_guard_test.db",
        "chroma_chat_stream_grounding_guard_test",
    )
    llm_answer_composer = LLMAnswerComposer(GroundingUnsafeLLMService())
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
            json={"query": STREAM_QUERY, "debug": True},
        )
        events = _sse_events(response.text)
        draft_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "answer_draft_delta"
        )
        final_answer = _sse_event_data(events, "final_answer")
        guard_result = _sse_event_data(events, "grounding_guard_result")
        result = _sse_event_data(events, "result")

        assert response.status_code == 200
        assert "下单" in draft_text
        assert guard_result["status"] == "fallback"
        assert any(
            violation["type"] == "purchase_boundary_violation"
            for violation in guard_result["violations"]
        )
        assert "下单" not in final_answer["answer"]
        assert "限时优惠" not in final_answer["answer"]
        assert result["answer"] == final_answer["answer"]
        assert _sse_event_data(events, "done")["status"] == "guarded"
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
            json={"query": STREAM_QUERY, "debug": True},
        )
        events = _sse_events(response.text)

        assert response.status_code == 200
        assert _sse_event_data(events, "stream_guard")["status"] == "blocked"
        assert _sse_event_data(events, "result")["answer"] == STREAM_GUARDED_FALLBACK_ANSWER
        assert _sse_event_data(events, "done")["status"] == "guarded"
    finally:
        _cleanup(engine, db_path, chroma_dir)
