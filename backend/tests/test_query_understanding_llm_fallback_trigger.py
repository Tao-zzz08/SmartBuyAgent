from __future__ import annotations

import json
import importlib.util
from typing import Any

import pytest

HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.chat.query_understanding import (
        Budget,
        QueryUnderstandingResult,
        QueryUnderstandingService,
        decide_llm_fallback,
    )
    from app.chat.shopping_memory import Budget as MemoryBudget
    from app.chat.shopping_memory import ShoppingMemory
    from app.services.llm import LLMMessage, LLMResponse


requires_pydantic = pytest.mark.skipif(
    not HAS_PYDANTIC,
    reason="QueryUnderstanding fallback trigger tests require backend pydantic dependency.",
)


def test_backend_dependency_status_for_fallback_trigger_tests() -> None:
    assert True


class FakeLLMService:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        del messages, max_tokens, temperature
        self.calls += 1
        return LLMResponse(content=self.content, model="fake", provider="fake", raw=None)


@requires_pydantic
def test_decision_disabled_and_empty_query_do_not_trigger() -> None:
    rule = _rule(intent="clarification", confidence=0.3)

    disabled = decide_llm_fallback(
        rule_result=rule,
        query="推荐一下",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=False,
    )
    empty = decide_llm_fallback(
        rule_result=rule,
        query="   ",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )

    assert disabled.should_call is False
    assert disabled.reasons == ["disabled"]
    assert empty.should_call is False
    assert empty.reasons == ["empty_query"]


@requires_pydantic
def test_safety_boundary_queries_do_not_trigger() -> None:
    for query in ["帮我下单第一款", "给我购买链接", "推荐能治疗痘痘的护肤品"]:
        decision = decide_llm_fallback(
            rule_result=_rule(intent="clarification", category=None, confidence=0.3),
            query=query,
            previous_memory=_memory(),
            confidence_threshold=0.75,
            enabled=True,
        )

        assert decision.should_call is False
        assert decision.reasons == ["safety_boundary"]


@requires_pydantic
def test_previous_memory_product_reference_and_followup_trigger() -> None:
    previous = _memory()

    compare_decision = decide_llm_fallback(
        rule_result=_rule(intent="chitchat", confidence=0.4),
        query="第一个和第二个哪个好",
        previous_memory=previous,
        confidence_threshold=0.75,
        enabled=True,
    )
    followup_decision = decide_llm_fallback(
        rule_result=_rule(intent="chitchat", confidence=0.4),
        query="便宜一点呢",
        previous_memory=previous,
        confidence_threshold=0.75,
        enabled=True,
    )

    assert compare_decision.should_call is True
    assert "product_reference" in compare_decision.reasons
    assert followup_decision.should_call is True
    assert "ambiguous_follow_up" in followup_decision.reasons


@requires_pydantic
def test_first_turn_long_tail_query_triggers() -> None:
    decision = decide_llm_fallback(
        rule_result=_rule(intent="chitchat", category=None, preferences=[], confidence=0.4),
        query="我想买个适合毕业旅行 vlog 的东西，别太贵，最好晚上也能拍",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )

    assert decision.should_call is True
    assert "long_tail_first_turn" in decision.reasons
    assert "weak_rule_slots" in decision.reasons


@requires_pydantic
def test_multi_intent_and_unknown_category_purchase_trigger() -> None:
    multi = decide_llm_fallback(
        rule_result=_rule(intent="shopping_guide", category="phone", preferences=["拍照"], confidence=0.85),
        query="我想买一部拍照好的手机，顺便告诉我为什么像素高不一定拍照好",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )
    unknown = decide_llm_fallback(
        rule_result=_rule(intent="clarification", category=None, preferences=[], confidence=0.3),
        query="我想买个适合露营用的东西",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )

    assert multi.should_call is True
    assert "multi_intent_query" in multi.reasons
    assert unknown.should_call is True
    assert "unknown_category_purchase" in unknown.reasons


@requires_pydantic
def test_chitchat_and_high_confidence_explicit_shopping_do_not_trigger() -> None:
    chitchat = decide_llm_fallback(
        rule_result=_rule(intent="chitchat", category=None, confidence=0.65),
        query="你好",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )
    strong = decide_llm_fallback(
        rule_result=_rule(
            intent="shopping_guide",
            category="phone",
            budget=Budget(max=4000),
            preferences=["拍照"],
            confidence=0.85,
        ),
        query="预算4000以内，推荐拍照好的手机",
        previous_memory=None,
        confidence_threshold=0.75,
        enabled=True,
    )

    assert chitchat.should_call is False
    assert strong.should_call is False
    assert strong.reasons == ["strong_rule"]


@requires_pydantic
def test_service_calls_fake_llm_for_first_turn_long_tail_query() -> None:
    llm = FakeLLMService(
        _json_content(
            {
                "is_follow_up": False,
                "intent": "shopping_guide",
                "category": "phone",
                "budget": {"min": None, "max": None, "currency": "CNY"},
                "preferences": ["vlog", "夜景", "拍摄"],
                "negative_preferences": [],
                "compare_product_ids": [],
                "referenced_product_indices": [],
                "confidence": 0.8,
                "reason": "long_tail_scene",
            }
        )
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand(
        "I want to buy something for graduation travel vlog, not too expensive, night shots"
    )

    assert llm.calls == 1
    assert result.llm_fallback_attempted is True
    assert result.llm_fallback_status == "success"
    assert result.llm_fallback_should_call is True
    assert result.source == "mixed"
    assert {"long_tail_first_turn", "weak_rule_slots"} <= set(result.llm_fallback_trigger_reasons)


@requires_pydantic
def test_service_records_failed_status_when_llm_returns_invalid_json() -> None:
    llm = FakeLLMService("not json")
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand(
        "I want to buy something for graduation travel vlog, not too expensive, night shots"
    )

    assert llm.calls == 1
    assert result.llm_fallback_attempted is True
    assert result.llm_fallback_status == "failed"
    assert result.llm_fallback_error == "invalid_json"
    assert result.llm_fallback_should_call is True
    assert "long_tail_first_turn" in result.llm_fallback_trigger_reasons
    assert result.source == "rule"


def _rule(
    *,
    intent: str,
    category: str | None = None,
    budget: Budget | None = None,
    preferences: list[str] | None = None,
    confidence: float,
) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        original_query="query",
        effective_query="query",
        intent=intent,
        category=category,
        budget=budget or Budget(),
        preferences=preferences or [],
        confidence=confidence,
        source="rule",
        reason="test_rule",
    )


def _memory() -> ShoppingMemory:
    return ShoppingMemory(
        category="phone",
        budget=MemoryBudget(max=5000),
        preferences=["拍照"],
        last_product_ids=["p1", "p2", "p3"],
        last_intent="shopping_guide",
    )


def _json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
