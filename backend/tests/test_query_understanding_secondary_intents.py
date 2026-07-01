from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.chat.query_understanding import (
        LLMQueryUnderstandingOutput,
        QueryUnderstandingService,
        sanitize_llm_understanding,
    )
    from app.chat.shopping_memory import Budget, ShoppingMemory
    from app.services.llm import LLMMessage, LLMResponse


requires_pydantic = pytest.mark.skipif(
    not HAS_PYDANTIC,
    reason="QueryUnderstanding secondary intent tests require backend pydantic dependency.",
)


def test_backend_dependency_status_for_secondary_intent_tests() -> None:
    assert True


@requires_pydantic
def test_shopping_guide_with_product_knowledge_secondary() -> None:
    result = _service().understand(
        "我想买一部拍照好的手机，顺便告诉我为什么像素高不一定拍照好"
    )

    assert result.intent == "shopping_guide"
    assert result.category == "phone"
    assert result.secondary_intents == ["product_knowledge"]
    assert "为什么像素高不一定拍照好" in result.knowledge_questions
    assert result.multi_intent_detected is True
    assert result.multi_intent_reason == "shopping_with_knowledge_question"


@requires_pydantic
def test_shoes_shopping_with_explanation_secondary() -> None:
    result = _service().understand(
        "推荐一双通勤鞋，也解释一下为什么鞋底纹路影响防滑"
    )

    assert result.intent == "shopping_guide"
    assert result.category == "shoes"
    assert {"通勤", "防滑"} <= set(result.preferences)
    assert result.secondary_intents == ["product_knowledge"]
    assert "为什么鞋底纹路影响防滑" in result.knowledge_questions


@requires_pydantic
def test_compare_with_product_knowledge_secondary() -> None:
    result = _service().understand(
        "第一个和第二个比一下，顺便说说哪个拍照更好",
        previous_memory=ShoppingMemory(
            category="phone",
            budget=Budget(max=4000),
            preferences=["拍照"],
            last_product_ids=["p1", "p2", "p3"],
            last_intent="shopping_guide",
            dialog_state="showing_products",
        ),
    )

    assert result.intent == "compare"
    assert result.compare_product_ids == ["p1", "p2"]
    assert result.referenced_product_indices == [1, 2]
    assert result.secondary_intents == ["product_knowledge"]
    assert "哪个拍照更好" in result.knowledge_questions


@requires_pydantic
def test_single_intent_queries_do_not_emit_secondary_intents() -> None:
    service = _service()

    shopping = service.understand("预算4000以内，推荐拍照好的手机")
    knowledge = service.understand("为什么像素不是越高越好")
    chitchat = service.understand("你好")

    assert shopping.intent == "shopping_guide"
    assert shopping.secondary_intents == []
    assert shopping.knowledge_questions == []
    assert knowledge.intent == "product_knowledge"
    assert knowledge.secondary_intents == []
    assert chitchat.intent == "chitchat"
    assert chitchat.secondary_intents == []


@requires_pydantic
def test_safety_boundary_query_does_not_emit_secondary_intents() -> None:
    result = _service().understand("推荐能治疗痘痘的护肤品并解释药效")

    assert result.secondary_intents == []
    assert result.knowledge_questions == []
    assert result.multi_intent_detected is False


@requires_pydantic
def test_fake_llm_secondary_intents_are_merged_without_changing_main_route() -> None:
    llm = FakeLLMService(
        {
            "is_follow_up": False,
            "intent": "compare",
            "category": "phone",
            "budget": {"min": None, "max": None, "currency": "CNY"},
            "preferences": ["拍照"],
            "negative_preferences": [],
            "compare_product_ids": [],
            "referenced_product_indices": [],
            "secondary_intents": ["product_knowledge"],
            "knowledge_questions": ["为什么像素高不一定拍照好"],
            "confidence": 0.8,
            "reason": "multi_intent",
        }
    )
    service = QueryUnderstandingService(llm_service=llm, llm_enabled=True)

    result = service.understand(
        "我想买一部拍照好的手机，顺便告诉我为什么像素高不一定拍照好"
    )

    assert llm.calls == 1
    assert result.intent == "shopping_guide"
    assert result.secondary_intents == ["product_knowledge"]
    assert "为什么像素高不一定拍照好" in result.knowledge_questions
    assert result.llm_fallback_attempted is True


@requires_pydantic
def test_sanitize_drops_invalid_and_same_as_primary_secondary_intents() -> None:
    invalid = sanitize_llm_understanding(
        LLMQueryUnderstandingOutput.model_validate(
            {
                "intent": "shopping_guide",
                "secondary_intents": ["purchase", "product_knowledge"],
                "knowledge_questions": ["为什么像素高不一定拍照好"],
            }
        ),
        allowed_product_ids=set(),
        max_reference_index=0,
    )
    same_as_primary = sanitize_llm_understanding(
        LLMQueryUnderstandingOutput.model_validate(
            {
                "intent": "product_knowledge",
                "secondary_intents": ["product_knowledge"],
                "knowledge_questions": ["为什么像素高不一定拍照好"],
            }
        ),
        allowed_product_ids=set(),
        max_reference_index=0,
    )

    assert invalid.secondary_intents == ["product_knowledge"]
    assert same_as_primary.secondary_intents == []


@requires_pydantic
def test_sanitize_truncates_long_knowledge_question() -> None:
    long_question = "为什么" + "像素" * 80

    sanitized = sanitize_llm_understanding(
        LLMQueryUnderstandingOutput.model_validate(
            {
                "intent": "shopping_guide",
                "secondary_intents": ["product_knowledge"],
                "knowledge_questions": [long_question],
            }
        ),
        allowed_product_ids=set(),
        max_reference_index=0,
    )

    assert sanitized.secondary_intents == ["product_knowledge"]
    assert len(sanitized.knowledge_questions[0]) == 120


class FakeLLMService:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
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
        return LLMResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            model="fake",
            provider="fake",
            raw=None,
        )


def _service() -> "QueryUnderstandingService":
    return QueryUnderstandingService(llm_enabled=False)
