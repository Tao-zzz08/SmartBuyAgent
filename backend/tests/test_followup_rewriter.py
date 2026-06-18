import json

from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.models import ChatTurn


def _turn(
    turn_index: int = 1,
    product_ids: list[str] | None = None,
    preferences: list[str] | None = None,
    budget_max: int = 3000,
    category_id: str = "cat_phone",
    category_path: str = "数码/手机",
) -> ChatTurn:
    return ChatTurn(
        session_id="session_test",
        turn_index=turn_index,
        user_query="预算3000，推荐一款拍照好的手机",
        assistant_answer="answer",
        intent="shopping_guide",
        category_id=category_id,
        category_path=category_path,
        budget_min=None,
        budget_max=budget_max,
        preferences_json=json.dumps(preferences or ["拍照", "续航"], ensure_ascii=False),
        product_ids_json=json.dumps(
            product_ids or ["phone_001", "phone_002", "phone_003"],
            ensure_ascii=False,
        ),
        citation_chunk_ids_json="[]",
    )


def test_budget_follow_up_rewrite_inherits_context() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="预算提高到4000呢",
        recent_turns=[_turn()],
    )

    assert result.is_follow_up is True
    assert result.reason == "budget_update_follow_up"
    assert "4000" in result.rewritten_query
    assert "手机" in result.rewritten_query
    assert "拍照" in result.rewritten_query
    assert result.source_turn_index == 1


def test_abbreviated_budget_follow_up_rewrite_inherits_context() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="增加到5000呢",
        recent_turns=[_turn(budget_max=4000, preferences=["拍照"])],
    )

    assert result.is_follow_up is True
    assert result.reason == "budget_update_follow_up"
    assert result.rewritten_query == "预算5000元以内，推荐拍照好的手机"
    assert result.context_used["category"] == "phone"
    assert result.context_used["budget"]["max"] == 5000
    assert result.context_used["preferences"] == ["拍照"]


def test_category_switch_rewrite_filters_phone_preferences() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="换成鞋子看看",
        recent_turns=[_turn(budget_max=5000, preferences=["拍照", "续航"])],
    )

    assert result.is_follow_up is True
    assert result.reason == "category_switch_follow_up"
    assert result.rewritten_query == "预算5000元以内，推荐鞋靴"
    assert result.context_used["shopping_memory"]["category"] == "shoes"
    assert result.context_used["shopping_memory"]["preferences"] == []


def test_preference_update_rewrite_merges_positive_and_negative_preferences() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="更看重续航，不考虑苹果",
        recent_turns=[_turn(budget_max=4000, preferences=["拍照"])],
    )

    assert result.is_follow_up is True
    assert result.reason == "preference_update_follow_up"
    assert result.context_used["shopping_memory"]["preferences"] == ["拍照", "续航"]
    assert result.context_used["shopping_memory"]["negative_preferences"] == ["苹果"]
    assert result.rewritten_query == "预算4000元以内，推荐拍照、续航好的手机，不考虑苹果"


def test_vague_reference_records_referenced_product_ids() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="这几款哪个更适合拍照",
        recent_turns=[_turn(product_ids=["phone_001", "phone_002"])],
    )

    assert result.is_follow_up is True
    assert result.reason == "vague_product_reference"
    assert result.context_used["referenced_product_ids"] == ["phone_001", "phone_002"]
    assert result.context_used["resolved_product_ids"] == []
    assert "上一轮推荐" in result.rewritten_query


def test_ordinal_reference_resolves_product_ids() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="第一个和第二个有什么区别",
        recent_turns=[_turn(product_ids=["phone_001", "phone_002", "phone_003"])],
    )

    assert result.is_follow_up is True
    assert result.reason == "ordinal_reference"
    assert result.context_used["resolved_product_ids"] == ["phone_001", "phone_002"]
    assert "phone_001" in result.rewritten_query
    assert "phone_002" in result.rewritten_query


def test_no_recent_turns_does_not_rewrite() -> None:
    result = FollowUpQueryRewriter().rewrite(
        query="预算提高到4000呢",
        recent_turns=[],
    )

    assert result.is_follow_up is False
    assert result.rewritten_query == "预算提高到4000呢"
    assert result.reason == "no_recent_turns"
