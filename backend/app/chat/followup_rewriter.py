from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from app.chat.query_understanding import QueryUnderstandingService
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
BUDGET_FOLLOW_UP_KEYWORDS = ["预算", "以内", "以下", "提高到", "降到", "如果"]
CATEGORY_FALLBACK_NAMES = {
    "cat_phone": "手机",
    "cat_shoes": "鞋靴",
    "cat_skincare": "护肤品",
}


@dataclass(frozen=True)
class FollowUpRewriteResult:
    is_follow_up: bool
    rewritten_query: str
    reason: str | None = None
    source_turn_index: int | None = None
    context_used: dict[str, Any] = field(default_factory=dict)


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
        product_ids = _json_list(source_turn.product_ids_json)
        preferences = _merge_preferences(
            _json_list(source_turn.preferences_json),
            parsed_query.preferences,
        )
        category_name = _category_name(source_turn.category_id, source_turn.category_path)
        resolved_product_ids = _resolve_ordinal_product_ids(stripped_query, product_ids)
        budget_max = _parse_budget_max(stripped_query, parsed_query.budget_max)

        if budget_max is not None and _looks_like_budget_follow_up(stripped_query):
            rewritten_query = _build_budget_rewrite(
                budget_max=budget_max,
                preferences=preferences,
                category_name=category_name,
            )
            return FollowUpRewriteResult(
                is_follow_up=True,
                rewritten_query=rewritten_query,
                reason="budget_update",
                source_turn_index=source_turn.turn_index,
                context_used={
                    "referenced_product_ids": product_ids,
                    "resolved_product_ids": [],
                    "category_id": source_turn.category_id,
                    "category_path": source_turn.category_path,
                    "budget_max": budget_max,
                    "preferences": preferences,
                },
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
                },
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
                },
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


def _category_name(category_id: str | None, category_path: str | None) -> str:
    if category_path:
        return category_path.split("/")[-1]
    if category_id:
        return CATEGORY_FALLBACK_NAMES.get(category_id, "商品")
    return "商品"


def _looks_like_budget_follow_up(query: str) -> bool:
    return any(keyword in query for keyword in BUDGET_FOLLOW_UP_KEYWORDS)


def _parse_budget_max(query: str, parsed_budget_max: int | None) -> int | None:
    if parsed_budget_max is not None:
        return parsed_budget_max
    match = re.search(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)", query)
    if not match:
        return None

    multiplier = 1
    unit = match.group("unit")
    if unit in {"k", "K", "千"}:
        multiplier = 1000
    elif unit == "万":
        multiplier = 10000
    return int(float(match.group("value")) * multiplier)


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


def _build_budget_rewrite(
    budget_max: int,
    preferences: list[str],
    category_name: str,
) -> str:
    preference_text = "、".join(preferences) if preferences else "当前偏好"
    return f"预算{budget_max}以内，推荐{preference_text}相关的{category_name}"


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
