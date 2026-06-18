from app.chat.shopping_memory import (
    Budget,
    ShoppingMemory,
    build_effective_query,
    extract_memory_from_query,
    looks_like_budget_follow_up,
    merge_shopping_memory,
)


def test_budget_follow_up_short_number_uses_existing_shopping_memory() -> None:
    previous = ShoppingMemory(
        category="phone",
        budget=Budget(max=4000),
        preferences=["拍照"],
        last_intent="shopping_guide",
    )
    current = extract_memory_from_query("5000呢")

    assert looks_like_budget_follow_up("5000呢", previous) is True

    merged = merge_shopping_memory(previous, current)

    assert merged.category == "phone"
    assert merged.budget.max == 5000
    assert merged.preferences == ["拍照"]
    assert build_effective_query(merged) == "预算5000元以内，推荐拍照好的手机"


def test_short_number_without_memory_is_not_budget_follow_up() -> None:
    current = extract_memory_from_query("5000呢")

    assert current.budget.max == 5000
    assert looks_like_budget_follow_up("5000呢", None) is False
    assert looks_like_budget_follow_up("5000呢", ShoppingMemory()) is False


def test_category_switch_filters_incompatible_preferences() -> None:
    previous = ShoppingMemory(
        category="phone",
        budget=Budget(max=5000),
        preferences=["拍照", "续航"],
        last_intent="shopping_guide",
    )
    current = extract_memory_from_query("换成鞋子看看")
    merged = merge_shopping_memory(previous, current)

    assert merged.category == "shoes"
    assert merged.budget.max == 5000
    assert merged.preferences == []
    assert build_effective_query(merged) == "预算5000元以内，推荐鞋靴"


def test_preference_and_negative_preference_update() -> None:
    previous = ShoppingMemory(
        category="phone",
        budget=Budget(max=4000),
        preferences=["拍照"],
        negative_preferences=[],
        last_intent="shopping_guide",
    )
    current = extract_memory_from_query("更看重续航，不考虑苹果")
    merged = merge_shopping_memory(previous, current)

    assert merged.category == "phone"
    assert merged.budget.max == 4000
    assert merged.preferences == ["拍照", "续航"]
    assert merged.negative_preferences == ["苹果"]
    assert build_effective_query(merged) == "预算4000元以内，推荐拍照、续航好的手机，不考虑苹果"


def test_skincare_effective_query_removes_medical_claims() -> None:
    memory = extract_memory_from_query("预算300，推荐能治疗痘痘的护肤品")
    effective_query = build_effective_query(memory)

    assert memory.category == "skincare"
    assert memory.budget.max == 300
    assert {"清爽", "控油", "温和"} <= set(memory.preferences)
    assert "治疗" not in effective_query
    assert "治愈" not in effective_query
    assert "药效" not in effective_query
    assert "处方" not in effective_query
    assert "医学修复" not in effective_query
    assert "护肤品" in effective_query
