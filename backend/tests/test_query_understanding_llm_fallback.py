from __future__ import annotations

import json
from typing import Any

from app.chat.query_understanding import (
    LLMQueryUnderstandingOutput,
    QueryUnderstandingService,
    sanitize_llm_understanding,
)
from app.chat.shopping_memory import Budget, ShoppingMemory
from app.services.llm import LLMMessage, LLMResponse


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
        self.calls += 1
        return LLMResponse(
            content=self.content,
            model="fake",
            provider="fake",
            raw=None,
        )


def _json_content(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_high_confidence_rule_does_not_call_llm() -> None:
    llm = FakeLLMService(
        _json_content(
            {
                "intent": "shopping_guide",
                "category": "phone",
                "confidence": 0.9,
            }
        )
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand("预算5000，推荐拍照好的手机")

    assert llm.calls == 0
    assert result.source == "rule"
    assert result.llm_fallback_attempted is False
    assert result.category == "phone"
    assert result.budget_max == 5000
    assert "拍照" in result.preferences


def test_low_confidence_follow_up_uses_llm_slots() -> None:
    previous = ShoppingMemory(
        category="shoes",
        budget=Budget(max=800),
        preferences=["通勤"],
        negative_preferences=[],
        last_intent="shopping_guide",
    )
    llm = FakeLLMService(
        _json_content(
            {
                "is_follow_up": True,
                "intent": "shopping_guide",
                "category": "shoes",
                "budget": {"min": None, "max": 1200, "currency": "CNY"},
                "preferences": ["轻便"],
                "negative_preferences": [],
                "compare_product_ids": [],
                "referenced_product_indices": [],
                "confidence": 0.84,
                "reason": "budget_and_preference_update_follow_up",
            }
        )
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand(
        "那贵一点但别太重的呢",
        previous_memory=previous,
    )

    assert llm.calls == 1
    assert result.source == "mixed"
    assert result.llm_fallback_attempted is True
    assert result.llm_fallback_status == "success"
    assert result.category == "shoes"
    assert result.budget_max == 1200
    assert result.preferences == ["通勤", "轻便"]
    assert result.is_follow_up is True


def test_invalid_llm_json_falls_back_to_rule_result() -> None:
    previous = ShoppingMemory(
        category="shoes",
        budget=Budget(max=800),
        preferences=["通勤"],
        last_intent="shopping_guide",
    )
    service = QueryUnderstandingService(
        llm_service=FakeLLMService("我觉得你可以考虑以下商品..."),
        llm_enabled=True,
    )

    result = service.understand("那更适合妈妈的呢", previous_memory=previous)

    assert result.llm_fallback_attempted is True
    assert result.llm_fallback_status == "failed"
    assert result.llm_fallback_error == "invalid_json"
    assert result.source == "rule"
    assert result.intent in {"chitchat", "clarification", "shopping_guide"}


def test_invalid_category_is_filtered() -> None:
    output = LLMQueryUnderstandingOutput.model_validate(
        {
            "intent": "shopping_guide",
            "category": "laptop",
            "budget": {"max": 5000, "currency": "CNY"},
            "preferences": ["轻薄"],
            "confidence": 0.8,
            "reason": "invalid_category_test",
        }
    )

    sanitized = sanitize_llm_understanding(
        output,
        allowed_product_ids=set(),
        max_reference_index=0,
    )

    assert sanitized.category is None
    assert sanitized.budget.max == 5000


def test_invalid_product_references_are_filtered() -> None:
    output = LLMQueryUnderstandingOutput.model_validate(
        {
            "intent": "compare",
            "compare_product_ids": ["p1", "fake_id"],
            "referenced_product_indices": [1, 3],
            "confidence": 0.83,
            "reason": "compare_follow_up",
        }
    )

    sanitized = sanitize_llm_understanding(
        output,
        allowed_product_ids={"p1", "p2"},
        max_reference_index=2,
    )

    assert sanitized.compare_product_ids == ["p1"]
    assert sanitized.referenced_product_indices == [1]


def test_referenced_indices_are_removed_without_last_products() -> None:
    output = LLMQueryUnderstandingOutput.model_validate(
        {
            "intent": "compare",
            "referenced_product_indices": [1, 2],
            "confidence": 0.8,
            "reason": "compare_follow_up",
        }
    )

    sanitized = sanitize_llm_understanding(
        output,
        allowed_product_ids=set(),
        max_reference_index=0,
    )

    assert sanitized.referenced_product_indices == []


def test_referenced_indices_keep_only_existing_product_positions() -> None:
    output = LLMQueryUnderstandingOutput.model_validate(
        {
            "intent": "compare",
            "referenced_product_indices": [0, 1, 2, 3, -1],
            "confidence": 0.8,
            "reason": "compare_follow_up",
        }
    )

    sanitized = sanitize_llm_understanding(
        output,
        allowed_product_ids=set(),
        max_reference_index=2,
    )

    assert sanitized.referenced_product_indices == [1, 2]


def test_skincare_medical_claims_are_sanitized_from_llm_slots() -> None:
    previous = ShoppingMemory(
        category="skincare",
        budget=Budget(max=300),
        preferences=[],
        negative_preferences=[],
        last_intent="shopping_guide",
    )
    llm = FakeLLMService(
        _json_content(
            {
                "is_follow_up": True,
                "intent": "shopping_guide",
                "category": "skincare",
                "budget": {"max": 300, "currency": "CNY"},
                "preferences": ["治疗痘痘", "药效强"],
                "confidence": 0.86,
                "reason": "skincare_medical_claim",
            }
        )
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand("那更适合她的呢", previous_memory=previous)

    assert result.category == "skincare"
    assert result.budget_max == 300
    assert {"清爽", "控油", "温和"} <= set(result.preferences)
    assert "治疗" not in result.effective_query
    assert "治愈" not in result.effective_query
    assert "药效" not in result.effective_query
    assert "处方" not in result.effective_query
    assert "医学修复" not in result.effective_query


def test_three_turn_budget_follow_up_regression_stays_rule_based() -> None:
    llm = FakeLLMService(
        _json_content(
            {
                "intent": "shopping_guide",
                "budget": {"max": 9999, "currency": "CNY"},
                "confidence": 0.9,
            }
        )
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)
    first = service.understand("预算3000，推荐一款拍照好的手机")
    second = service.understand(
        "我的预算增加到4000呢",
        previous_memory=first.to_shopping_memory(),
    )
    third = service.understand(
        "增加到5000呢",
        previous_memory=second.to_shopping_memory(),
    )

    assert llm.calls == 0
    assert third.source == "rule"
    assert third.category == "phone"
    assert third.budget_max == 5000
    assert "拍照" in third.preferences
    assert third.reason == "budget_update_follow_up"
