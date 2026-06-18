from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.chat.shopping_memory import (
    Budget as ShoppingBudget,
    ShoppingMemory,
    build_effective_query,
    category_from_id_or_path,
    category_to_id,
    category_to_path,
    extract_category,
    extract_memory_from_query,
    looks_like_budget_follow_up,
    merge_shopping_memory,
    merge_turns_to_memory,
    parse_budget_max,
    shopping_memory_from_dict,
)


VALID_INTENTS = {
    "shopping_guide",
    "product_knowledge",
    "compare",
    "clarification",
    "chitchat",
}
VALID_SOURCES = {"rule", "llm", "mixed"}


class Budget(BaseModel):
    min: int | None = None
    max: int | None = None
    currency: str = "CNY"

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="after")
    def _normalize_budget(self) -> "Budget":
        if self.min is not None and not 0 < self.min <= 100000:
            self.min = None
        if self.max is not None and not 0 < self.max <= 100000:
            self.max = None
        if not self.currency:
            self.currency = "CNY"
        return self


class QueryUnderstandingResult(BaseModel):
    original_query: str
    effective_query: str
    is_follow_up: bool = False
    intent: Literal[
        "shopping_guide",
        "product_knowledge",
        "compare",
        "clarification",
        "chitchat",
    ] = "clarification"
    category: Literal["phone", "shoes", "skincare"] | None = None
    budget: Budget = Field(default_factory=Budget)
    preferences: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)
    compare_product_ids: list[str] = Field(default_factory=list)
    referenced_product_indices: list[int] = Field(default_factory=list)
    confidence: float = 0.0
    source: Literal["rule", "llm", "mixed"] = "rule"
    reason: str = ""
    memory_updated: bool = False

    category_id: str | None = None
    category_path: str | None = None
    need_clarification: bool = False
    clarification_question: str | None = None
    shopping_memory: dict[str, Any] | None = None

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if not data.get("original_query"):
            data["original_query"] = str(data.get("raw_query") or "")
        if not data.get("effective_query"):
            data["effective_query"] = str(data.get("rewritten_query") or data["original_query"])

        intent = str(data.get("intent") or "clarification")
        data["intent"] = intent if intent in VALID_INTENTS else "clarification"

        source = str(data.get("source") or "rule")
        data["source"] = source if source in VALID_SOURCES else "rule"

        category = data.get("category")
        if category not in {"phone", "shoes", "skincare", None}:
            category = None
        if category is None:
            category = category_from_id_or_path(
                data.get("category_id"),
                data.get("category_path"),
            )
        data["category"] = category

        budget_value = data.get("budget")
        if isinstance(budget_value, BaseModel):
            budget_value = budget_value.model_dump()
        elif not isinstance(budget_value, dict):
            budget_value = {
                "min": data.get("budget_min"),
                "max": data.get("budget_max"),
                "currency": "CNY",
            }
        data["budget"] = budget_value

        data["preferences"] = _dedupe_short_terms(data.get("preferences"))
        data["negative_preferences"] = _dedupe_short_terms(
            data.get("negative_preferences")
        )
        data["compare_product_ids"] = _list_of_str(data.get("compare_product_ids"))
        data["referenced_product_indices"] = _list_of_int(
            data.get("referenced_product_indices")
        )
        data["reason"] = str(data.get("reason") or "")
        if "need_clarification" not in data:
            data["need_clarification"] = data["intent"] == "clarification"
        return data

    @model_validator(mode="after")
    def _fill_derived_fields(self) -> "QueryUnderstandingResult":
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        if self.category:
            self.category_id = self.category_id or category_to_id(self.category)
            self.category_path = self.category_path or category_to_path(self.category)

        if not self.effective_query:
            self.effective_query = self.original_query

        self.preferences = _dedupe_short_terms(self.preferences)
        self.negative_preferences = _dedupe_short_terms(self.negative_preferences)
        self.preferences = [
            item for item in self.preferences if item not in self.negative_preferences
        ]

        if self.shopping_memory is None:
            self.shopping_memory = self.to_shopping_memory().to_dict()
        return self

    @property
    def raw_query(self) -> str:
        return self.original_query

    @property
    def budget_min(self) -> int | None:
        return self.budget.min

    @property
    def budget_max(self) -> int | None:
        return self.budget.max

    def to_shopping_memory(self) -> ShoppingMemory:
        return ShoppingMemory(
            category=self.category,
            budget=ShoppingBudget(
                min=self.budget.min,
                max=self.budget.max,
                currency=self.budget.currency,
            ),
            preferences=list(self.preferences),
            negative_preferences=list(self.negative_preferences),
            last_product_ids=list(self.compare_product_ids),
            last_intent=self.intent,
        )

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "effective_query": self.effective_query,
            "is_follow_up": self.is_follow_up,
            "intent": self.intent,
            "category": self.category,
            "category_id": self.category_id,
            "category_path": self.category_path,
            "budget": self.budget.model_dump(),
            "budget_min": self.budget_min,
            "budget_max": self.budget_max,
            "preferences": list(self.preferences),
            "negative_preferences": list(self.negative_preferences),
            "compare_product_ids": list(self.compare_product_ids),
            "referenced_product_indices": list(self.referenced_product_indices),
            "source": self.source,
            "confidence": self.confidence,
            "reason": self.reason,
            "memory_updated": self.memory_updated,
            "shopping_memory": self.shopping_memory,
            "need_clarification": self.need_clarification,
        }


@dataclass(frozen=True)
class CategoryRule:
    category_id: str
    category_path: str
    keywords: list[str]


class QueryUnderstandingService:
    CLARIFICATION_QUESTION = "你想看哪个品类的商品？目前我可以帮你选手机、鞋靴和护肤品。"

    CATEGORY_RULES = [
        CategoryRule(
            category_id="cat_phone",
            category_path="数码/手机",
            keywords=[
                "手机",
                "拍照",
                "续航",
                "性能",
                "影像",
                "像素",
                "安卓",
                "iphone",
                "5g",
                "存储",
                "处理器",
            ],
        ),
        CategoryRule(
            category_id="cat_shoes",
            category_path="服装/鞋靴",
            keywords=[
                "鞋",
                "鞋靴",
                "通勤",
                "防滑",
                "尺码",
                "跑步",
                "运动鞋",
                "透气",
                "材质",
                "脚感",
            ],
        ),
        CategoryRule(
            category_id="cat_skincare",
            category_path="美妆/护肤",
            keywords=[
                "护肤",
                "敏感肌",
                "保湿",
                "成分",
                "面霜",
                "精华",
                "洁面",
                "防晒",
                "修护",
                "爽肤水",
                "护肤品",
            ],
        ),
    ]

    PREFERENCE_KEYWORDS = [
        "敏感肌",
        "性价比",
        "拍照",
        "续航",
        "性能",
        "游戏",
        "轻薄",
        "学生",
        "影像",
        "存储",
        "通勤",
        "防滑",
        "透气",
        "舒适",
        "跑步",
        "运动",
        "尺码",
        "材质",
        "耐磨",
        "保湿",
        "修护",
        "成分",
        "清爽",
        "防晒",
        "控油",
        "温和",
    ]

    COMPARE_KEYWORDS = ["对比", "比较", "哪个更好", "哪个更值得"]
    KNOWLEDGE_KEYWORDS = [
        "为什么",
        "怎么选",
        "怎么看",
        "区别",
        "原理",
        "售后",
        "七天无理由",
        "保养",
    ]
    SHOPPING_KEYWORDS = ["推荐", "买", "想要", "适合", "预算", "以内", "以下"]
    CLARIFICATION_KEYWORDS = ["推荐一下", "帮我选一个", "买哪个比较好"]
    CHITCHAT_QUERIES = {"你好", "hi", "hello", "在吗"}

    def understand(
        self,
        query: str,
        session_id: str | None = None,
        history: list[Any] | None = None,
        previous_memory: ShoppingMemory | None = None,
        rewrite_result: Any | None = None,
    ) -> QueryUnderstandingResult:
        del session_id
        raw_query = query
        normalized_query = query.strip()
        lower_query = normalized_query.lower()

        if not normalized_query:
            return self._clarification_result(raw_query)

        memory = extract_memory_from_query(normalized_query)
        category_value = memory.category
        category = (
            CategoryRule(
                category_id=category_to_id(category_value) or "",
                category_path=category_to_path(category_value) or "",
                keywords=[],
            )
            if category_value
            else self._detect_category(lower_query)
        )
        if category is not None and category_value is None:
            category_value = category_from_id_or_path(
                category.category_id,
                category.category_path,
            )
        budget_min, budget_max = self._parse_budget(lower_query)
        if budget_max is None:
            budget_max = memory.budget.max
        preferences = self._extract_preferences(lower_query)
        if memory.preferences:
            preferences = memory.preferences
        intent = self._detect_intent(
            normalized_query=normalized_query,
            lower_query=lower_query,
            has_category=category is not None,
        )
        need_clarification = intent == "clarification"
        current_memory = ShoppingMemory(
            category=category_value,
            budget=ShoppingBudget(min=budget_min, max=budget_max),
            preferences=preferences,
            negative_preferences=memory.negative_preferences,
            last_intent=intent,
        )
        previous = previous_memory or (
            merge_turns_to_memory(list(history or [])) if history else None
        )
        resolved_memory = current_memory
        is_follow_up = False
        reason = "direct_rule"
        memory_updated = current_memory.has_shopping_context()
        effective_query = (
            build_effective_query(current_memory)
            if intent == "shopping_guide" and current_memory.category
            else raw_query
        )
        compare_product_ids: list[str] = []
        referenced_product_indices: list[int] = []

        if rewrite_result is not None and getattr(rewrite_result, "is_follow_up", False):
            is_follow_up = True
            reason = str(getattr(rewrite_result, "reason", None) or "follow_up")
            rewrite_memory = getattr(rewrite_result, "shopping_memory", None)
            if isinstance(rewrite_memory, dict):
                resolved_memory = shopping_memory_from_dict(rewrite_memory)
            elif previous is not None:
                resolved_memory = merge_shopping_memory(previous, current_memory)
            context_used = getattr(rewrite_result, "context_used", {}) or {}
            resolved_ids = _list_of_str(context_used.get("resolved_product_ids"))
            referenced_ids = _list_of_str(context_used.get("referenced_product_ids"))
            compare_product_ids = resolved_ids or referenced_ids
            referenced_product_indices = _indices_for_product_ids(
                referenced_ids,
                compare_product_ids,
            )
            if resolved_memory.category:
                effective_query = build_effective_query(resolved_memory)
                if intent in {"chitchat", "clarification"}:
                    intent = "shopping_guide"
                    need_clarification = False
            else:
                effective_query = str(
                    getattr(rewrite_result, "rewritten_query", None) or effective_query
                )
            memory_updated = resolved_memory.has_shopping_context()
        elif previous is not None and previous.has_shopping_context():
            maybe_merged = merge_shopping_memory(previous, current_memory)
            if (
                looks_like_budget_follow_up(normalized_query, previous)
                or current_memory.has_shopping_context()
            ) and (
                maybe_merged != previous
                and maybe_merged.category
            ):
                resolved_memory = maybe_merged
                is_follow_up = True
                reason = (
                    "budget_update_follow_up"
                    if looks_like_budget_follow_up(normalized_query, previous)
                    else "memory_merge_follow_up"
                )
                effective_query = build_effective_query(resolved_memory)
                intent = "shopping_guide" if intent in {"chitchat", "clarification"} else intent
                need_clarification = False
                memory_updated = True

        confidence = _confidence_for_result(
            intent=intent,
            is_follow_up=is_follow_up,
            category=resolved_memory.category or category_value,
            budget_max=resolved_memory.budget.max,
            preferences=resolved_memory.preferences,
        )

        return QueryUnderstandingResult(
            original_query=raw_query,
            effective_query=effective_query,
            is_follow_up=is_follow_up,
            intent=intent,
            category=resolved_memory.category or category_value,
            category_id=category_to_id(resolved_memory.category or category_value)
            or (category.category_id if category else None),
            category_path=category_to_path(resolved_memory.category or category_value)
            or (category.category_path if category else None),
            budget=Budget(
                min=resolved_memory.budget.min,
                max=resolved_memory.budget.max,
                currency=resolved_memory.budget.currency,
            ),
            preferences=resolved_memory.preferences,
            negative_preferences=resolved_memory.negative_preferences,
            compare_product_ids=compare_product_ids,
            referenced_product_indices=referenced_product_indices,
            confidence=confidence,
            source="rule",
            reason=reason,
            memory_updated=memory_updated,
            need_clarification=need_clarification,
            clarification_question=self.CLARIFICATION_QUESTION
            if need_clarification
            else None,
            shopping_memory=resolved_memory.to_dict(),
        )

    def _clarification_result(self, raw_query: str) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            original_query=raw_query,
            effective_query=raw_query,
            intent="clarification",
            category_id=None,
            category_path=None,
            budget=Budget(),
            preferences=[],
            need_clarification=True,
            clarification_question=self.CLARIFICATION_QUESTION,
            confidence=0.3,
            reason="empty_query",
        )

    def _detect_category(self, lower_query: str) -> CategoryRule | None:
        best_rule: CategoryRule | None = None
        best_count = 0
        best_first_index: int | None = None

        for rule in self.CATEGORY_RULES:
            positions = [
                lower_query.find(keyword)
                for keyword in rule.keywords
                if lower_query.find(keyword) != -1
            ]
            if not positions:
                continue

            count = len(positions)
            first_index = min(positions)
            if best_rule is None:
                best_rule = rule
                best_count = count
                best_first_index = first_index
                continue

            if count > best_count:
                best_rule = rule
                best_count = count
                best_first_index = first_index
            elif count == best_count and best_first_index is not None:
                if first_index < best_first_index:
                    best_rule = rule
                    best_first_index = first_index

        return best_rule

    def _parse_budget(self, lower_query: str) -> tuple[int | None, int | None]:
        range_match = re.search(
            r"(?P<min_value>\d+(?:\.\d+)?)\s*(?P<min_unit>[kK千万]?)\s*(?:元)?"
            r"\s*(?:到|至|-|~|～)\s*"
            r"(?P<max_value>\d+(?:\.\d+)?)\s*(?P<max_unit>[kK千万]?)\s*(?:元)?"
            r"(?:之间)?",
            lower_query,
        )
        if range_match:
            min_value = self._parse_amount(
                range_match.group("min_value"),
                range_match.group("min_unit"),
            )
            max_value = self._parse_amount(
                range_match.group("max_value"),
                range_match.group("max_unit"),
            )
            return min_value, max_value

        max_patterns = [
            r"(?:预算|价格|价位)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元)?\s*(?:以内|以下|内)",
            r"(?:预算|价格|价位)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kK千万]?)\s*(?:元)?",
        ]
        for pattern in max_patterns:
            match = re.search(pattern, lower_query)
            if match:
                return None, self._parse_amount(match.group("value"), match.group("unit"))

        return None, parse_budget_max(lower_query)

    @staticmethod
    def _parse_amount(value: str, unit: str | None) -> int:
        multiplier = 1
        if unit in {"k", "K", "千"}:
            multiplier = 1000
        elif unit == "万":
            multiplier = 10000
        return int(float(value) * multiplier)

    def _extract_preferences(self, lower_query: str) -> list[str]:
        matches: list[tuple[int, str]] = []
        for keyword in self.PREFERENCE_KEYWORDS:
            index = lower_query.find(keyword.lower())
            if index != -1:
                matches.append((index, keyword))

        preferences: list[str] = []
        seen: set[str] = set()
        for _, keyword in sorted(matches, key=lambda item: item[0]):
            if keyword not in seen:
                preferences.append(keyword)
                seen.add(keyword)
        return preferences

    def _detect_intent(
        self,
        normalized_query: str,
        lower_query: str,
        has_category: bool,
    ) -> str:
        if lower_query in self.CHITCHAT_QUERIES:
            return "chitchat"

        if self._contains_any(lower_query, self.COMPARE_KEYWORDS) or re.search(
            r"\b[a-z]+_\d+\s*(?:和|跟|与|vs|VS)\s*[a-z]+_\d+\b",
            normalized_query,
        ):
            return "compare"

        if self._contains_any(lower_query, self.KNOWLEDGE_KEYWORDS):
            return "product_knowledge"

        if has_category or self._contains_any(lower_query, self.SHOPPING_KEYWORDS):
            if not has_category and self._looks_like_clarification(lower_query):
                return "clarification"
            return "shopping_guide"

        if self._looks_like_clarification(lower_query):
            return "clarification"

        return "chitchat"

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword.lower() in text for keyword in keywords)

    def _looks_like_clarification(self, lower_query: str) -> bool:
        return self._contains_any(lower_query, self.CLARIFICATION_KEYWORDS)


def _dedupe_short_terms(value: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in _list_of_str(value):
        term = item.strip()
        if not term or len(term) > 16:
            continue
        if term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _list_of_int(value: Any) -> list[int]:
    if value is None or not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _indices_for_product_ids(
    referenced_product_ids: list[str],
    compare_product_ids: list[str],
) -> list[int]:
    if not referenced_product_ids or not compare_product_ids:
        return []
    indices: list[int] = []
    for product_id in compare_product_ids:
        try:
            indices.append(referenced_product_ids.index(product_id) + 1)
        except ValueError:
            continue
    return indices


def _confidence_for_result(
    *,
    intent: str,
    is_follow_up: bool,
    category: str | None,
    budget_max: int | None,
    preferences: list[str],
) -> float:
    if is_follow_up:
        return 0.9
    if intent == "clarification":
        return 0.3
    if category and (budget_max is not None or preferences):
        return 0.85
    if category or budget_max is not None or preferences:
        return 0.75
    if intent in {"product_knowledge", "compare", "chitchat"}:
        return 0.65
    return 0.55
