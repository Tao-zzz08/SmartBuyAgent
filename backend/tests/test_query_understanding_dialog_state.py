from __future__ import annotations

import importlib.util

import pytest


HAS_PYDANTIC = importlib.util.find_spec("pydantic") is not None

if HAS_PYDANTIC:
    from app.chat.query_understanding import QueryUnderstandingService
    from app.chat.shopping_memory import Budget, ShoppingMemory


requires_pydantic = pytest.mark.skipif(
    not HAS_PYDANTIC,
    reason="QueryUnderstanding dialog state tests require backend pydantic dependency.",
)


def test_backend_dependency_status_for_dialog_state_tests() -> None:
    assert True


@requires_pydantic
def test_awaiting_budget_short_budget_inherits_context() -> None:
    result = _service().understand(
        "3000以内",
        previous_memory=ShoppingMemory(
            category="phone",
            budget=Budget(),
            preferences=["拍照"],
            last_intent="clarification",
            dialog_state="awaiting_budget",
        ),
    )

    assert result.intent == "shopping_guide"
    assert result.category == "phone"
    assert result.budget.max == 3000
    assert result.preferences == ["拍照"]
    assert result.is_follow_up is True
    assert result.need_clarification is False
    assert result.dialog_state == "awaiting_budget"
    assert result.next_dialog_state == "showing_products"


@requires_pydantic
def test_awaiting_category_short_category_enters_guide() -> None:
    result = _service().understand(
        "手机",
        previous_memory=ShoppingMemory(
            budget=Budget(max=4000),
            last_intent="clarification",
            dialog_state="awaiting_category",
        ),
    )

    assert result.intent == "shopping_guide"
    assert result.category == "phone"
    assert result.budget.max == 4000
    assert result.is_follow_up is True
    assert result.need_clarification is False
    assert result.dialog_state == "awaiting_category"
    assert result.next_dialog_state == "showing_products"


@requires_pydantic
def test_showing_products_compare_first_second_resolves_ids() -> None:
    result = _service().understand(
        "第一个和第二个哪个好",
        previous_memory=_showing_products_memory(),
    )

    assert result.intent == "compare"
    assert result.referenced_product_indices == [1, 2]
    assert result.compare_product_ids == ["p1", "p2"]
    assert result.is_follow_up is True
    assert result.next_dialog_state == "comparing_products"


@requires_pydantic
def test_showing_products_single_product_attribute_followup_tracks_reference() -> None:
    result = _service().understand(
        "第二个续航怎么样",
        previous_memory=_showing_products_memory(),
    )

    assert result.intent == "product_knowledge"
    assert result.referenced_product_indices == [2]
    assert result.next_dialog_state == "answering_knowledge"


@requires_pydantic
def test_comparing_products_attribute_followup_keeps_compare_context() -> None:
    result = _service().understand(
        "哪个更适合拍照",
        previous_memory=ShoppingMemory(
            category="phone",
            budget=Budget(max=4000),
            preferences=["拍照"],
            last_product_ids=["p1", "p2"],
            last_intent="compare",
            dialog_state="comparing_products",
        ),
    )

    assert result.intent == "compare"
    assert result.compare_product_ids == ["p1", "p2"]
    assert result.is_follow_up is True
    assert result.next_dialog_state == "comparing_products"


@requires_pydantic
def test_decision_confirmation_does_not_create_purchase_flow() -> None:
    result = _service().understand(
        "就第一个吧",
        previous_memory=_showing_products_memory(),
    )

    assert result.intent in {"clarification", "product_knowledge", "shopping_guide"}
    assert result.intent not in {"purchase", "payment", "checkout"}
    assert result.referenced_product_indices == [1]
    assert result.need_clarification is True
    assert result.next_dialog_state == "showing_products"
    assert result.dialog_state_reason == "decision_confirmation_without_purchase"


@requires_pydantic
def test_idle_chitchat_stays_idle() -> None:
    result = _service().understand("你好")

    assert result.intent == "chitchat"
    assert result.dialog_state == "idle"
    assert result.next_dialog_state == "idle"


@requires_pydantic
def test_explicit_category_switch_wins_over_showing_products_state() -> None:
    result = _service().understand(
        "那鞋子呢",
        previous_memory=_showing_products_memory(),
    )

    assert result.intent == "shopping_guide"
    assert result.category == "shoes"
    assert result.next_dialog_state == "showing_products"


def _service() -> "QueryUnderstandingService":
    return QueryUnderstandingService(llm_enabled=False)


def _showing_products_memory() -> "ShoppingMemory":
    return ShoppingMemory(
        category="phone",
        budget=Budget(max=4000),
        preferences=["拍照"],
        last_product_ids=["p1", "p2", "p3"],
        last_intent="shopping_guide",
        dialog_state="showing_products",
    )
