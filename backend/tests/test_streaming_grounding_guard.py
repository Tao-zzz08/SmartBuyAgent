from __future__ import annotations

from collections.abc import Iterator

from fastapi.testclient import TestClient

from app.cache.cache_service import InMemoryCacheService
from app.chat.llm_answer_composer import LLMAnswerComposer
from app.main import app
from app.services.llm import BaseLLMService, LLMMessage, LLMResponse

from test_chat_api import (  # noqa: E402
    _cleanup,
    _override_dependencies,
    _prepare_api_stack,
    _sse_event_data,
    _sse_events,
)


STREAM_QUERY = "预算3000，推荐一款拍照好的手机"
UNSAFE_DRAFT_TERMS = ["下单"]
FORBIDDEN_FINAL_TERMS = ["限时优惠", "立即下单", "下单", "优惠券", "购买链接"]


class PurchaseOverreachStreamingLLMService(BaseLLMService):
    provider = "purchase_overreach_fake"
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
            model="purchase-overreach-fake",
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


def test_streaming_grounding_guard_uses_fallback_for_official_answer() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_grounding_regression_test.db",
        "chroma_chat_stream_grounding_regression_test",
    )
    llm_answer_composer = LLMAnswerComposer(PurchaseOverreachStreamingLLMService())
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
        guard_result = _sse_event_data(events, "grounding_guard_result")
        final_answer = _sse_event_data(events, "final_answer")
        result = _sse_event_data(events, "result")

        event_names = [event["event"] for event in events]
        assert response.status_code == 200
        assert "answer_draft_delta" in event_names
        assert "grounding_guard_result" in event_names
        assert "final_answer" in event_names
        assert event_names.index("grounding_guard_result") < event_names.index("final_answer")

        assert any(term in draft_text for term in UNSAFE_DRAFT_TERMS)
        assert guard_result["status"] == "fallback"
        assert guard_result["action"] == "fallback"
        assert any(
            violation["type"] == "purchase_boundary_violation"
            for violation in guard_result["violations"]
        )

        assert final_answer["answer"] == result["answer"]
        for term in FORBIDDEN_FINAL_TERMS:
            assert term not in final_answer["answer"]
            assert term not in result["answer"]

        assert result["product_cards"]
        assert result["citations"]
        assert _sse_event_data(events, "done")["status"] == "guarded"
    finally:
        _cleanup(engine, db_path, chroma_dir)
