from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.agent.context import AgentRuntimeContext
from app.agent.stream_runner import AgentStreamRunner
from app.chat.query_understanding import QueryUnderstandingService
from app.chat.shopping_memory import Budget, ShoppingMemory
from app.services.llm import LLMMessage, LLMResponse


class FakeLLMService:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            model="fake",
            provider="fake",
            raw=None,
        )


class FakeConversationMemoryService:
    last_cache_status = "disabled"

    def __init__(self, turns: list[Any]) -> None:
        self.turns = turns

    def get_recent_turns(self, session_id: str, limit: int = 5) -> list[Any]:
        return self.turns[-limit:]


def test_stream_trace_contains_llm_fallback_status() -> None:
    previous = ShoppingMemory(
        category="shoes",
        budget=Budget(max=800),
        preferences=["通勤"],
        negative_preferences=[],
        last_intent="shopping_guide",
    )
    turn = SimpleNamespace(
        user_query="预算800，推荐通勤鞋",
        assistant_answer="已推荐几款通勤鞋。",
        intent="shopping_guide",
        category_id="cat_shoes",
        category_path="服装/鞋靴",
        budget_min=None,
        budget_max=800,
        preferences_json=json.dumps(
            {"shopping_memory": previous.to_dict()},
            ensure_ascii=False,
        ),
        product_ids_json=json.dumps(["shoe_001", "shoe_002"]),
    )
    llm = FakeLLMService(
        json.dumps(
            {
                "is_follow_up": True,
                "intent": "shopping_guide",
                "category": "shoes",
                "budget": {"max": 1200, "currency": "CNY"},
                "preferences": ["轻便"],
                "confidence": 0.84,
                "reason": "budget_and_preference_update_follow_up",
            },
            ensure_ascii=False,
        )
    )
    context = AgentRuntimeContext(
        query_understanding_service=QueryUnderstandingService(
            llm_service=llm,
            llm_enabled=True,
        ),
        conversation_memory_service=FakeConversationMemoryService([turn]),
    )
    runner = AgentStreamRunner(context)

    events = list(
        runner.stream(
            "那贵一点但别太重的呢",
            request_id="req_test",
            session_id="session_test",
            event_session_id="session_test",
        )
    )
    trace_events = [
        event.data
        for event in events
        if event.event == "trace" and event.data.get("step") == "query_understanding"
    ]

    assert trace_events
    trace = trace_events[-1]
    assert trace["llm_fallback_attempted"] is True
    assert trace["llm_fallback_status"] == "success"
    assert trace["source"] == "mixed"
    assert trace["category"] == "shoes"
    assert trace["budget"]["max"] == 1200
    assert "轻便" in trace["preferences"]
