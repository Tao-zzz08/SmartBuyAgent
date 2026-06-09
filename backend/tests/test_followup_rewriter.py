import json

from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.models import ChatTurn


def _turn(
    turn_index: int = 1,
    product_ids: list[str] | None = None,
    preferences: list[str] | None = None,
) -> ChatTurn:
    return ChatTurn(
        session_id="session_test",
        turn_index=turn_index,
        user_query="预算3000，推荐一款拍照好的手机",
        assistant_answer="answer",
        intent="shopping_guide",
        category_id="cat_phone",
        category_path="数码/手机",
        budget_min=None,
        budget_max=3000,
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
    assert result.reason == "budget_update"
    assert "4000" in result.rewritten_query
    assert "手机" in result.rewritten_query
    assert "拍照" in result.rewritten_query
    assert result.source_turn_index == 1


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
