from app.chat.query_understanding import QueryUnderstandingService
from app.chat.shopping_memory import Budget, ShoppingMemory


def test_phone_shopping_guide_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("预算3000，推荐一款拍照好的手机")

    assert result.intent == "shopping_guide"
    assert result.category_id == "cat_phone"
    assert result.category_path == "数码/手机"
    assert result.budget_max == 3000
    assert "拍照" in result.preferences
    assert result.need_clarification is False


def test_query_understanding_result_merges_previous_shopping_memory() -> None:
    service = QueryUnderstandingService()
    previous = ShoppingMemory(
        category="phone",
        budget=Budget(max=4000),
        preferences=["拍照"],
        negative_preferences=[],
        last_intent="shopping_guide",
    )

    result = service.understand("增加到5000呢", previous_memory=previous)

    assert result.original_query == "增加到5000呢"
    assert result.effective_query == "预算5000元以内，推荐拍照好的手机"
    assert result.is_follow_up is True
    assert result.intent == "shopping_guide"
    assert result.category == "phone"
    assert result.budget.model_dump() == {
        "min": None,
        "max": 5000,
        "currency": "CNY",
    }
    assert result.preferences == ["拍照"]
    assert result.negative_preferences == []
    assert result.source == "rule"
    assert 0 <= result.confidence <= 1
    assert result.reason == "budget_update_follow_up"
    assert result.shopping_memory["last_intent"] == "shopping_guide"


def test_shoes_shopping_guide_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("500以内，想买一双通勤防滑的鞋")

    assert result.category_id == "cat_shoes"
    assert result.budget_max == 500
    assert "通勤" in result.preferences
    assert "防滑" in result.preferences


def test_skincare_shopping_guide_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("敏感肌用什么保湿修护面霜，预算300以内")

    assert result.category_id == "cat_skincare"
    assert result.budget_max == 300
    assert "敏感肌" in result.preferences
    assert "保湿" in result.preferences
    assert "修护" in result.preferences


def test_budget_range_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("100到300之间的护肤品")

    assert result.budget_min == 100
    assert result.budget_max == 300


def test_k_budget_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("3k以内拍照手机")

    assert result.budget_max == 3000


def test_chinese_thousand_budget_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("3千以内拍照手机")

    assert result.budget_max == 3000


def test_product_knowledge_intent() -> None:
    service = QueryUnderstandingService()

    result = service.understand("为什么手机拍照不能只看像素")

    assert result.intent == "product_knowledge"
    assert result.category_id == "cat_phone"


def test_compare_intent() -> None:
    service = QueryUnderstandingService()

    result = service.understand("phone_001 和 phone_002 哪个更值得买")

    assert result.intent == "compare"


def test_chitchat_intent() -> None:
    service = QueryUnderstandingService()

    result = service.understand("你好")

    assert result.intent == "chitchat"
    assert result.need_clarification is False


def test_clarification_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("推荐一下")

    assert result.intent == "clarification"
    assert result.need_clarification is True
    assert result.clarification_question


def test_empty_query_needs_clarification() -> None:
    service = QueryUnderstandingService()

    result = service.understand("   ")

    assert result.intent == "clarification"
    assert result.need_clarification is True


def test_skincare_query_sanitizes_medical_claims_in_effective_query() -> None:
    service = QueryUnderstandingService()

    result = service.understand("预算300，推荐能治疗痘痘的护肤品")

    assert result.intent == "shopping_guide"
    assert result.category_id == "cat_skincare"
    assert result.category == "skincare"
    assert result.budget_max == 300
    assert {"清爽", "控油", "温和"} <= set(result.preferences)
    assert result.effective_query is not None
    assert "治疗" not in result.effective_query
    assert "治愈" not in result.effective_query
    assert "药效" not in result.effective_query
    assert "处方" not in result.effective_query
