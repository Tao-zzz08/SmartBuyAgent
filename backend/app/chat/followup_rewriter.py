from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from app.chat.query_understanding import QueryUnderstandingService
from app.chat.shopping_memory import (
    ShoppingMemory,
    build_effective_query,
    empty_shopping_memory,
    extract_memory_from_query,
    looks_like_budget_follow_up,
    memory_from_turn,
    merge_shopping_memory,
)
from app.models import ChatTurn


ORDINAL_INDEXES = {
    "第一个": 0,
    "第一款": 0,
    "第1个": 0,
    "第1款": 0,
    "第二个": 1,
    "第二款": 1,
    "第2个": 1,
    "第2款": 1,
    "第三个": 2,
    "第三款": 2,
    "第3个": 2,
    "第3款": 2,
}
VAGUE_REFERENCE_KEYWORDS = [
    "这几款",
    "这几个",
    "刚才推荐",
    "刚才这几款",
    "上一轮推荐",
    "哪个更好",
    "哪个好",
    "有什么区别",
]
@dataclass(frozen=True)
class FollowUpRewriteResult:
    is_follow_up: bool
    rewritten_query: str
    reason: str | None = None
    source_turn_index: int | None = None
    context_used: dict[str, Any] = field(default_factory=dict)
    shopping_memory: dict[str, Any] | None = None


class FollowUpQueryRewriter:
    def __init__(
        self,
        query_understanding_service: QueryUnderstandingService | None = None,
    ) -> None:
        self.query_understanding_service = (
            query_understanding_service or QueryUnderstandingService()
        )

    def rewrite(
        self,
        query: str,
        recent_turns: list[ChatTurn] | None,
    ) -> FollowUpRewriteResult:
        turns = recent_turns or []
        if not turns:
            return FollowUpRewriteResult(
                is_follow_up=False,
                rewritten_query=query,
                reason="no_recent_turns",
                context_used={},
            )

        source_turn = _latest_context_turn(turns)
        if source_turn is None:
            return FollowUpRewriteResult(
                is_follow_up=False,
                rewritten_query=query,
                reason="no_usable_context",
                context_used={},
            )

        stripped_query = query.strip()
        parsed_query = self.query_understanding_service.understand(stripped_query)
        previous_memory = _memory_from_turns(turns)
        current_memory = extract_memory_from_query(
            stripped_query,
            intent=parsed_query.intent,
        )
        merged_memory = merge_shopping_memory(previous_memory, current_memory)
        product_ids = _json_list(source_turn.product_ids_json)
        preferences = _merge_preferences(
            previous_memory.preferences,
            parsed_query.preferences,
        )
        resolved_product_ids = _resolve_ordinal_product_ids(stripped_query, product_ids)
        budget_max = merged_memory.budget.max

        if current_memory.category and previous_memory.has_shopping_context():
            return _shopping_memory_result(
                query=stripped_query,
                reason="category_switch_follow_up"
                if previous_memory.category != current_memory.category
                else "category_follow_up",
                source_turn=source_turn,
                product_ids=product_ids,
                memory=merged_memory,
            )

        if looks_like_budget_follow_up(stripped_query, previous_memory):
            rewritten_query = build_effective_query(merged_memory)
            return FollowUpRewriteResult(
                is_follow_up=True,
                rewritten_query=rewritten_query,
                reason="budget_update_follow_up",
                source_turn_index=source_turn.turn_index,
                context_used={
                    "referenced_product_ids": product_ids,
                    "resolved_product_ids": [],
                    "category": merged_memory.category,
                    "category_id": source_turn.category_id,
                    "category_path": source_turn.category_path,
                    "budget_max": budget_max,
                    "budget": merged_memory.budget.__dict__,
                    "preferences": merged_memory.preferences,
                    "negative_preferences": merged_memory.negative_preferences,
                    "shopping_memory": merged_memory.to_dict(),
                },
                shopping_memory=merged_memory.to_dict(),
            )

        if resolved_product_ids:
            rewritten_query = _build_ordinal_rewrite(
                query=stripped_query,
                resolved_product_ids=resolved_product_ids,
                preferences=preferences,
            )
            return FollowUpRewriteResult(
                is_follow_up=True,
                rewritten_query=rewritten_query,
                reason="ordinal_reference",
                source_turn_index=source_turn.turn_index,
                context_used={
                    "referenced_product_ids": product_ids,
                    "resolved_product_ids": resolved_product_ids,
                    "preferences": preferences,
                    "shopping_memory": previous_memory.to_dict(),
                },
                shopping_memory=previous_memory.to_dict(),
            )

        if product_ids and _looks_like_vague_product_follow_up(stripped_query):
            rewritten_query = _build_vague_reference_rewrite(
                query=stripped_query,
                preferences=preferences,
            )
            return FollowUpRewriteResult(
                is_follow_up=True,
                rewritten_query=rewritten_query,
                reason="vague_product_reference",
                source_turn_index=source_turn.turn_index,
                context_used={
                    "referenced_product_ids": product_ids,
                    "resolved_product_ids": [],
                    "preferences": preferences,
                    "shopping_memory": previous_memory.to_dict(),
                },
                shopping_memory=previous_memory.to_dict(),
            )

        if (
            current_memory.preferences or current_memory.negative_preferences
        ) and previous_memory.has_shopping_context():
            return _shopping_memory_result(
                query=stripped_query,
                reason="preference_update_follow_up",
                source_turn=source_turn,
                product_ids=product_ids,
                memory=merged_memory,
            )

        return FollowUpRewriteResult(
            is_follow_up=False,
            rewritten_query=query,
            reason="not_follow_up",
            source_turn_index=source_turn.turn_index,
            context_used={},
        )


def _latest_context_turn(turns: list[ChatTurn]) -> ChatTurn | None:
    for turn in reversed(turns):
        if turn.category_id or _json_list(turn.product_ids_json):
            return turn
    return turns[-1] if turns else None


def _memory_from_turns(turns: list[ChatTurn]) -> ShoppingMemory:
    memory = empty_shopping_memory()
    for turn in turns:
        turn_memory = memory_from_turn(turn)
        if turn_memory.has_shopping_context():
            memory = merge_shopping_memory(memory, turn_memory)
    return memory


def _shopping_memory_result(
    *,
    query: str,
    reason: str,
    source_turn: ChatTurn,
    product_ids: list[str],
    memory: ShoppingMemory,
) -> FollowUpRewriteResult:
    rewritten_query = build_effective_query(memory)
    return FollowUpRewriteResult(
        is_follow_up=True,
        rewritten_query=rewritten_query,
        reason=reason,
        source_turn_index=source_turn.turn_index,
        context_used={
            "referenced_product_ids": product_ids,
            "resolved_product_ids": [],
            "category": memory.category,
            "category_id": source_turn.category_id,
            "category_path": source_turn.category_path,
            "budget": memory.budget.__dict__,
            "budget_max": memory.budget.max,
            "preferences": memory.preferences,
            "negative_preferences": memory.negative_preferences,
            "shopping_memory": memory.to_dict(),
            "original_query": query,
        },
        shopping_memory=memory.to_dict(),
    )


def _json_list(text: str | None) -> list[str]:
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _merge_preferences(previous: list[str], current: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*previous, *current]:
        if item and item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _looks_like_vague_product_follow_up(query: str) -> bool:
    return any(keyword in query for keyword in VAGUE_REFERENCE_KEYWORDS)


def _resolve_ordinal_product_ids(query: str, product_ids: list[str]) -> list[str]:
    resolved: list[str] = []
    for keyword, index in ORDINAL_INDEXES.items():
        if keyword in query and index < len(product_ids):
            product_id = product_ids[index]
            if product_id not in resolved:
                resolved.append(product_id)
    return resolved


def _build_ordinal_rewrite(
    query: str,
    resolved_product_ids: list[str],
    preferences: list[str],
) -> str:
    focus = _focus_text(query=query, preferences=preferences)
    product_text = "、".join(resolved_product_ids)
    return f"比较上一轮推荐的 {product_text}，重点关注{focus}"


def _build_vague_reference_rewrite(query: str, preferences: list[str]) -> str:
    focus = _focus_text(query=query, preferences=preferences)
    return f"对上一轮推荐的商品进行比较，重点关注{focus}"


def _focus_text(query: str, preferences: list[str]) -> str:
    if preferences:
        return "、".join(preferences)
    if re.search(r"区别|不同", query):
        return "主要区别"
    return "综合表现"
