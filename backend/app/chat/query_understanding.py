from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import settings
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
    normalize_dialog_state,
    shopping_memory_from_dict,
)
from app.services.llm import BaseLLMService, LLMMessage, get_llm_service


VALID_INTENTS = {
    "shopping_guide",
    "product_knowledge",
    "compare",
    "clarification",
    "chitchat",
}
VALID_SOURCES = {"rule", "llm", "mixed"}
VALID_CATEGORIES = {"phone", "shoes", "skincare"}
LLM_UNDERSTANDING_MARKER = "SMARTBUY_QUERY_UNDERSTANDING_JSON"
SKINCARE_MEDICAL_TERMS = [
    "治疗",
    "治愈",
    "药效",
    "处方",
    "医学修复",
    "修复疾病",
    "祛病",
    "消炎药",
    "药物",
]
PURCHASE_BOUNDARY_TERMS = [
    "购买",
    "下单",
    "支付",
    "购物车",
    "订单",
    "购买链接",
    "立即购买",
]


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
    llm_fallback_attempted: bool = False
    llm_fallback_status: str | None = None
    llm_fallback_error: str | None = None
    llm_fallback_result: dict[str, Any] | None = None
    llm_fallback_should_call: bool = False
    llm_fallback_trigger_reasons: list[str] = Field(default_factory=list)
    dialog_state: str | None = None
    next_dialog_state: str | None = None
    dialog_state_reason: str = ""

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
        data["dialog_state"] = normalize_dialog_state(data.get("dialog_state"))
        data["next_dialog_state"] = normalize_dialog_state(data.get("next_dialog_state"))
        data["dialog_state_reason"] = str(data.get("dialog_state_reason") or "")
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
        self.dialog_state = normalize_dialog_state(self.dialog_state)
        self.next_dialog_state = normalize_dialog_state(self.next_dialog_state)

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
        last_product_ids: list[str] = []
        if isinstance(self.shopping_memory, dict):
            last_product_ids = _list_of_str(self.shopping_memory.get("last_product_ids"))
        if not last_product_ids and self.intent == "compare":
            last_product_ids = list(self.compare_product_ids)

        return ShoppingMemory(
            category=self.category,
            budget=ShoppingBudget(
                min=self.budget.min,
                max=self.budget.max,
                currency=self.budget.currency,
            ),
            preferences=list(self.preferences),
            negative_preferences=list(self.negative_preferences),
            last_product_ids=last_product_ids,
            last_intent=self.intent,
            dialog_state=self.next_dialog_state or self.dialog_state,
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
            "llm_fallback_attempted": self.llm_fallback_attempted,
            "llm_fallback_status": self.llm_fallback_status,
            "llm_fallback_error": self.llm_fallback_error,
            "llm_fallback_result": self.llm_fallback_result,
            "llm_fallback_should_call": self.llm_fallback_should_call,
            "llm_fallback_trigger_reasons": list(self.llm_fallback_trigger_reasons),
            "dialog_state": self.dialog_state,
            "next_dialog_state": self.next_dialog_state,
            "dialog_state_reason": self.dialog_state_reason,
        }


class LLMQueryUnderstandingOutput(BaseModel):
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
    reason: str = "unknown"

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)

        intent = str(data.get("intent") or "clarification")
        data["intent"] = intent if intent in VALID_INTENTS else "clarification"

        category = data.get("category")
        data["category"] = category if category in VALID_CATEGORIES else None

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
        data["reason"] = _safe_reason(data.get("reason"))
        return data

    @model_validator(mode="after")
    def _normalize(self) -> "LLMQueryUnderstandingOutput":
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.preferences = _dedupe_short_terms(self.preferences)
        self.negative_preferences = _dedupe_short_terms(self.negative_preferences)
        self.preferences = [
            item for item in self.preferences if item not in self.negative_preferences
        ]
        return self


@dataclass(frozen=True)
class CategoryRule:
    category_id: str
    category_path: str
    keywords: list[str]


@dataclass(frozen=True)
class LLMFallbackDecision:
    should_call: bool
    reasons: list[str] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class DialogStateAdjustment:
    intent: str
    memory: ShoppingMemory
    is_follow_up: bool
    need_clarification: bool
    compare_product_ids: list[str] = dataclass_field(default_factory=list)
    referenced_product_indices: list[int] = dataclass_field(default_factory=list)
    next_dialog_state: str | None = None
    reason: str = ""
    clarification_question: str | None = None


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

    COMPARE_KEYWORDS = [
        "对比",
        "比较",
        "哪个更好",
        "哪个更值得",
        "哪个更适合",
        "有什么区别",
        "这几款",
        "这几个",
        "第一个",
        "第二个",
    ]
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

    def __init__(
        self,
        llm_service: BaseLLMService | None = None,
        *,
        llm_enabled: bool | None = None,
        llm_confidence_threshold: float | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.llm_enabled = (
            settings.QUERY_UNDERSTANDING_LLM_ENABLED
            if llm_enabled is None
            else llm_enabled
        )
        self.llm_confidence_threshold = (
            settings.QUERY_UNDERSTANDING_LLM_CONFIDENCE_THRESHOLD
            if llm_confidence_threshold is None
            else llm_confidence_threshold
        )

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
        if category_value is None:
            category_value = _category_from_dialog_query(normalized_query)
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
        if budget_max is None:
            budget_max = _budget_max_from_dialog_query(normalized_query)
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
        inferred_dialog_state = infer_dialog_state(previous)
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
            if intent == "compare":
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

        dialog_adjustment = apply_dialog_state_hints(
            normalized_query=normalized_query,
            current_memory=current_memory,
            resolved_memory=resolved_memory,
            previous_memory=previous,
            inferred_state=inferred_dialog_state,
            intent=intent,
            need_clarification=need_clarification,
            compare_product_ids=compare_product_ids,
            referenced_product_indices=referenced_product_indices,
        )
        if dialog_adjustment is not None:
            resolved_memory = dialog_adjustment.memory
            is_follow_up = is_follow_up or dialog_adjustment.is_follow_up
            intent = dialog_adjustment.intent
            need_clarification = dialog_adjustment.need_clarification
            compare_product_ids = dialog_adjustment.compare_product_ids
            referenced_product_indices = dialog_adjustment.referenced_product_indices
            reason = dialog_adjustment.reason or reason
            memory_updated = resolved_memory.has_shopping_context()
            effective_query = (
                build_effective_query(resolved_memory)
                if intent in {"shopping_guide", "compare"} and resolved_memory.category
                else raw_query
            )

        next_dialog_state = infer_next_dialog_state(
            intent=intent,
            category=resolved_memory.category or category_value,
            budget_max=resolved_memory.budget.max,
            override=dialog_adjustment.next_dialog_state if dialog_adjustment else None,
        )
        dialog_state_reason = dialog_adjustment.reason if dialog_adjustment else ""
        resolved_memory = _memory_with_dialog_state(
            resolved_memory,
            dialog_state=next_dialog_state,
            last_intent=intent,
        )

        confidence = _confidence_for_result(
            intent=intent,
            is_follow_up=is_follow_up,
            category=resolved_memory.category or category_value,
            budget_max=resolved_memory.budget.max,
            preferences=resolved_memory.preferences,
        )

        rule_result = QueryUnderstandingResult(
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
            if need_clarification and not (dialog_adjustment and dialog_adjustment.clarification_question)
            else (dialog_adjustment.clarification_question if dialog_adjustment else None),
            shopping_memory=resolved_memory.to_dict(),
            dialog_state=inferred_dialog_state,
            next_dialog_state=next_dialog_state,
            dialog_state_reason=dialog_state_reason,
        )
        return self._apply_llm_fallback_if_needed(
            rule_result=rule_result,
            query=normalized_query,
            history=history,
            previous_memory=previous,
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
            dialog_state="idle",
            next_dialog_state="awaiting_category",
            dialog_state_reason="empty_query",
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

    def _apply_llm_fallback_if_needed(
        self,
        *,
        rule_result: QueryUnderstandingResult,
        query: str,
        history: list[Any] | None,
        previous_memory: ShoppingMemory | None,
    ) -> QueryUnderstandingResult:
        previous = previous_memory or empty_memory_from_result(rule_result)
        decision = decide_llm_fallback(
            rule_result=rule_result,
            query=query,
            previous_memory=previous,
            confidence_threshold=self.llm_confidence_threshold,
            enabled=self.llm_enabled,
        )
        if not decision.should_call:
            return rule_result.model_copy(
                update={
                    "llm_fallback_attempted": False,
                    "llm_fallback_status": "skipped",
                    "llm_fallback_should_call": False,
                    "llm_fallback_trigger_reasons": decision.reasons,
                }
            )

        try:
            llm_output = self._llm_structured_extract(
                query=query,
                history=history,
                previous_memory=previous,
            )
        except Exception:
            return _with_llm_failure(rule_result, "llm_call_failed", decision=decision)

        if llm_output is None:
            return _with_llm_failure(rule_result, "invalid_json", decision=decision)

        sanitized = sanitize_llm_understanding(
            llm_output,
            allowed_product_ids=set(previous.last_product_ids),
            max_reference_index=len(previous.last_product_ids),
        )
        merged = merge_rule_and_llm_understanding(
            rule_result=rule_result,
            llm_output=sanitized,
            previous_memory=previous,
            confidence_threshold=self.llm_confidence_threshold,
        )
        return merged.model_copy(
            update={
                "llm_fallback_attempted": True,
                "llm_fallback_status": "success",
                "llm_fallback_error": None,
                "llm_fallback_result": sanitized.model_dump(),
                "llm_fallback_should_call": True,
                "llm_fallback_trigger_reasons": decision.reasons,
            }
        )

    def _llm_structured_extract(
        self,
        *,
        query: str,
        history: list[Any] | None,
        previous_memory: ShoppingMemory,
    ) -> LLMQueryUnderstandingOutput | None:
        messages = build_llm_understanding_prompt(
            query=query,
            history=history,
            previous_memory=previous_memory,
        )
        service = self.llm_service or get_llm_service(
            timeout_seconds=settings.QUERY_UNDERSTANDING_LLM_TIMEOUT_SECONDS
        )
        response = service.chat(
            messages,
            max_tokens=400,
            temperature=0.0,
        )
        payload = parse_llm_understanding_json(response.content)
        if payload is None:
            return None
        return LLMQueryUnderstandingOutput.model_validate(payload)


def should_call_llm_fallback(
    *,
    rule_result: QueryUnderstandingResult,
    query: str,
    previous_memory: ShoppingMemory | None,
    confidence_threshold: float,
    enabled: bool = True,
) -> bool:
    return decide_llm_fallback(
        rule_result=rule_result,
        query=query,
        previous_memory=previous_memory,
        confidence_threshold=confidence_threshold,
        enabled=enabled,
    ).should_call


def decide_llm_fallback(
    *,
    rule_result: QueryUnderstandingResult,
    query: str,
    previous_memory: ShoppingMemory | None,
    confidence_threshold: float,
    enabled: bool = True,
) -> LLMFallbackDecision:
    if not enabled:
        return LLMFallbackDecision(False, ["disabled"])
    normalized = query.strip()
    if not normalized:
        return LLMFallbackDecision(False, ["empty_query"])
    if _looks_like_safety_boundary_query(normalized):
        return LLMFallbackDecision(False, ["safety_boundary"])

    reasons: list[str] = []
    if previous_memory is not None and previous_memory.has_shopping_context():
        if _looks_like_ambiguous_follow_up(normalized):
            reasons.append("ambiguous_follow_up")
        if _contains_product_reference(normalized):
            reasons.append("product_reference")

    if _looks_like_multi_intent_query(normalized):
        reasons.append("multi_intent_query")

    if _looks_like_long_tail_first_turn_query(normalized, rule_result):
        reasons.extend(["long_tail_first_turn", "weak_rule_slots"])

    if _looks_like_unknown_category_purchase(normalized, rule_result):
        reasons.append("unknown_category_purchase")

    if (
        rule_result.confidence < confidence_threshold
        and _looks_like_user_needs_product_help(normalized)
        and rule_result.intent != "chitchat"
    ):
        reasons.append("low_confidence_product_help")

    if reasons:
        return LLMFallbackDecision(True, _dedupe_terms(reasons))

    if rule_result.confidence >= confidence_threshold and rule_result.intent != "clarification":
        return LLMFallbackDecision(False, ["strong_rule"])

    return LLMFallbackDecision(False, ["no_trigger"])


def infer_dialog_state(previous_memory: ShoppingMemory | None) -> str:
    if previous_memory is None:
        return "idle"
    explicit_state = normalize_dialog_state(previous_memory.dialog_state)
    if explicit_state:
        return explicit_state
    if previous_memory.last_intent == "compare":
        return "comparing_products"
    if previous_memory.last_intent == "product_knowledge":
        return "answering_knowledge"
    if previous_memory.last_product_ids:
        return "showing_products"
    if previous_memory.last_intent == "clarification" and not previous_memory.category:
        return "awaiting_category"
    return "idle"


def infer_next_dialog_state(
    *,
    intent: str,
    category: str | None,
    budget_max: int | None,
    override: str | None = None,
) -> str:
    override_state = normalize_dialog_state(override)
    if override_state:
        return override_state
    if intent == "clarification" and not category:
        return "awaiting_category"
    if intent == "clarification" and category and budget_max is None:
        return "awaiting_budget"
    if intent == "shopping_guide":
        return "showing_products"
    if intent == "compare":
        return "comparing_products"
    if intent == "product_knowledge":
        return "answering_knowledge"
    if intent == "chitchat":
        return "idle"
    return "idle"


def apply_dialog_state_hints(
    *,
    normalized_query: str,
    current_memory: ShoppingMemory,
    resolved_memory: ShoppingMemory,
    previous_memory: ShoppingMemory | None,
    inferred_state: str,
    intent: str,
    need_clarification: bool,
    compare_product_ids: list[str],
    referenced_product_indices: list[int],
) -> DialogStateAdjustment | None:
    if previous_memory is None:
        return None

    state = normalize_dialog_state(inferred_state) or "idle"
    explicit_category = current_memory.category or _category_from_dialog_query(normalized_query)
    budget_max = current_memory.budget.max or _budget_max_from_dialog_query(normalized_query)

    if state == "awaiting_budget" and previous_memory.category and budget_max is not None:
        memory = merge_shopping_memory(
            previous_memory,
            ShoppingMemory(
                category=previous_memory.category,
                budget=ShoppingBudget(
                    min=current_memory.budget.min,
                    max=budget_max,
                    currency=current_memory.budget.currency or "CNY",
                ),
                preferences=current_memory.preferences,
                negative_preferences=current_memory.negative_preferences,
                last_product_ids=previous_memory.last_product_ids,
                last_intent="shopping_guide",
            ),
        )
        return DialogStateAdjustment(
            intent="shopping_guide",
            memory=memory,
            is_follow_up=True,
            need_clarification=False,
            next_dialog_state="showing_products",
            reason="awaiting_budget_filled",
        )

    if state == "awaiting_category" and explicit_category:
        compatible_previous = _compatible_previous_memory(previous_memory, explicit_category)
        memory = merge_shopping_memory(
            compatible_previous,
            ShoppingMemory(
                category=explicit_category,
                budget=current_memory.budget,
                preferences=current_memory.preferences,
                negative_preferences=current_memory.negative_preferences,
                last_intent="shopping_guide",
            ),
        )
        return DialogStateAdjustment(
            intent="shopping_guide",
            memory=memory,
            is_follow_up=True,
            need_clarification=False,
            next_dialog_state="showing_products",
            reason="awaiting_category_filled",
        )

    if state == "showing_products" and previous_memory.last_product_ids:
        indices = resolve_referenced_product_indices(
            normalized_query,
            max_count=len(previous_memory.last_product_ids),
        )
        if _looks_like_decision_confirmation(normalized_query):
            memory = _memory_with_dialog_state(
                merge_shopping_memory(previous_memory, current_memory),
                dialog_state="showing_products",
                last_intent="clarification",
            )
            return DialogStateAdjustment(
                intent="clarification",
                memory=memory,
                is_follow_up=True,
                need_clarification=True,
                referenced_product_indices=indices,
                next_dialog_state="showing_products",
                reason="decision_confirmation_without_purchase",
                clarification_question=(
                    "我可以继续帮你比较参数或说明这款是否适合你，但不能直接下单。"
                    "你想了解哪方面？"
                ),
            )
        if indices and _looks_like_compare_query(normalized_query):
            selected_ids = _product_ids_for_indices(previous_memory.last_product_ids, indices)
            memory = _memory_with_dialog_state(
                merge_shopping_memory(previous_memory, current_memory),
                dialog_state="comparing_products",
                last_intent="compare",
            )
            return DialogStateAdjustment(
                intent="compare",
                memory=memory,
                is_follow_up=True,
                need_clarification=False,
                compare_product_ids=selected_ids,
                referenced_product_indices=indices,
                next_dialog_state="comparing_products",
                reason="ordinal_compare_from_showing_products",
            )
        if indices or _looks_like_single_product_attribute_query(normalized_query):
            inferred_indices = indices or ([1] if _contains_demonstrative_reference(normalized_query) else [])
            memory = _memory_with_dialog_state(
                merge_shopping_memory(previous_memory, current_memory),
                dialog_state="answering_knowledge",
                last_intent="product_knowledge",
            )
            return DialogStateAdjustment(
                intent="product_knowledge",
                memory=memory,
                is_follow_up=True,
                need_clarification=False,
                referenced_product_indices=inferred_indices,
                next_dialog_state="answering_knowledge",
                reason="single_product_attribute_follow_up",
            )

    if (
        state == "comparing_products"
        and len(previous_memory.last_product_ids) >= 2
        and _looks_like_compare_attribute_followup(normalized_query)
    ):
        ids = previous_memory.last_product_ids[:2]
        memory = _memory_with_dialog_state(
            merge_shopping_memory(previous_memory, current_memory),
            dialog_state="comparing_products",
            last_intent="compare",
        )
        return DialogStateAdjustment(
            intent="compare",
            memory=memory,
            is_follow_up=True,
            need_clarification=False,
            compare_product_ids=ids,
            referenced_product_indices=[1, 2],
            next_dialog_state="comparing_products",
            reason="compare_attribute_follow_up",
        )

    del intent, need_clarification, compare_product_ids, referenced_product_indices
    return None


def resolve_referenced_product_indices(query: str, max_count: int) -> list[int]:
    if max_count <= 0:
        return []
    lowered = query.lower()
    indices: list[int] = []
    ordinal_patterns = [
        (1, [r"第\s*一\s*[个款台件]?", r"第\s*1\s*[个款台件]?", r"\b1\s*[个款台件]"]),
        (2, [r"第\s*二\s*[个款台件]?", r"第\s*2\s*[个款台件]?", r"\b2\s*[个款台件]"]),
        (3, [r"第\s*三\s*[个款台件]?", r"第\s*3\s*[个款台件]?", r"\b3\s*[个款台件]"]),
    ]
    for index, patterns in ordinal_patterns:
        if index <= max_count and any(re.search(pattern, lowered) for pattern in patterns):
            indices.append(index)
    if not indices and "这两个" in lowered and max_count >= 2:
        indices.extend([1, 2])
    if not indices and ("这几款" in lowered or "这几个" in lowered):
        indices.extend(range(1, min(max_count, 3) + 1))
    return _dedupe_ints(indices)


def _category_from_dialog_query(query: str) -> str | None:
    lowered = query.lower()
    if any(term in lowered for term in ["手机", "iphone", "安卓", "拍照手机"]):
        return "phone"
    if any(term in lowered for term in ["鞋", "鞋子", "通勤鞋", "高跟", "运动鞋"]):
        return "shoes"
    if any(term in lowered for term in ["护肤", "敏感肌", "保湿", "美白", "面霜", "精华"]):
        return "skincare"
    return None


def _budget_max_from_dialog_query(query: str) -> int | None:
    normalized = query.strip().lower().replace(",", "")
    has_budget_cue = any(
        term in normalized
        for term in ["预算", "以内", "以下", "不超过", "元", "块", "k", "千", "万"]
    )
    is_bare_amount = re.fullmatch(r"\d+(?:\.\d+)?\s*(?:k|千|万)?", normalized) is not None
    if not has_budget_cue and not is_bare_amount:
        return None
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>k|千|万)?\s*(?:元|块)?\s*(?:以内|以下|内)?",
        normalized,
    )
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit")
    if unit in {"k", "千"}:
        value *= 1000
    elif unit == "万":
        value *= 10000
    amount = int(value)
    return amount if 0 < amount <= 100000 else None


def _compatible_previous_memory(
    previous_memory: ShoppingMemory,
    category: str,
) -> ShoppingMemory:
    if previous_memory.category and previous_memory.category != category:
        return ShoppingMemory(
            category=None,
            budget=previous_memory.budget,
            preferences=[],
            negative_preferences=previous_memory.negative_preferences,
            last_product_ids=[],
            last_intent=previous_memory.last_intent,
            dialog_state=previous_memory.dialog_state,
        )
    return previous_memory


def _memory_with_dialog_state(
    memory: ShoppingMemory,
    *,
    dialog_state: str | None,
    last_intent: str | None = None,
) -> ShoppingMemory:
    return ShoppingMemory(
        category=memory.category,
        budget=memory.budget,
        preferences=list(memory.preferences),
        negative_preferences=list(memory.negative_preferences),
        last_product_ids=list(memory.last_product_ids),
        last_intent=last_intent or memory.last_intent,
        dialog_state=normalize_dialog_state(dialog_state),
    )


def _product_ids_for_indices(product_ids: list[str], indices: list[int]) -> list[str]:
    selected: list[str] = []
    for index in indices:
        position = index - 1
        if 0 <= position < len(product_ids):
            selected.append(product_ids[position])
    return selected


def _looks_like_compare_query(query: str) -> bool:
    return _contains_any_casefold(
        query,
        ["哪个好", "哪个更好", "哪个更适合", "对比", "比较", "比一下", "差别"],
    )


def _looks_like_single_product_attribute_query(query: str) -> bool:
    return _contains_any_casefold(
        query,
        [
            "怎么样",
            "支持",
            "适合",
            "防水",
            "续航",
            "拍照",
            "防滑",
            "敏感肌",
            "参数",
            "性能",
        ],
    ) and (
        bool(resolve_referenced_product_indices(query, max_count=3))
        or _contains_demonstrative_reference(query)
    )


def _looks_like_decision_confirmation(query: str) -> bool:
    return _contains_any_casefold(
        query,
        ["就这个吧", "就它吧", "选这个", "选第一个", "就第一个吧", "就第二个吧"],
    )


def _looks_like_compare_attribute_followup(query: str) -> bool:
    return _contains_any_casefold(
        query,
        ["哪个更适合", "哪个更好", "哪个好", "拍照", "续航", "防滑", "更适合"],
    )


def _contains_demonstrative_reference(query: str) -> bool:
    return _contains_any_casefold(query, ["这个", "这款", "它", "那个", "那款"])


def _dedupe_ints(values: list[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _looks_like_safety_boundary_query(query: str) -> bool:
    lower_query = query.lower()
    purchase_terms = [
        *PURCHASE_BOUNDARY_TERMS,
        "购买",
        "下单",
        "支付",
        "购物车",
        "订单",
        "购买链接",
        "立即购买",
        "checkout",
        "payment",
        "cart",
    ]
    medical_terms = [
        *SKINCARE_MEDICAL_TERMS,
        "治疗",
        "治愈",
        "药效",
        "处方",
        "医学修复",
        "修复疾病",
        "临床治愈率",
    ]
    return _contains_any_casefold(lower_query, purchase_terms) or _contains_any_casefold(
        lower_query,
        medical_terms,
    )


def _looks_like_long_tail_first_turn_query(
    query: str,
    rule_result: QueryUnderstandingResult,
) -> bool:
    if rule_result.is_follow_up:
        return False
    if len(query.strip()) < 22:
        return False
    if not _looks_like_user_needs_product_help(query):
        return False
    weak_rule_slots = (
        rule_result.category is None
        or not rule_result.preferences
        or rule_result.confidence < 0.85
    )
    scene_terms = [
        "vlog",
        "旅行",
        "夜拍",
        "晚上",
        "妈妈",
        "日常用",
        "露营",
        "上课",
        "通勤",
        "下雨天",
        "别太贵",
        "不要太",
    ]
    return weak_rule_slots and _contains_any_casefold(query, scene_terms)


def _looks_like_multi_intent_query(query: str) -> bool:
    connector_terms = ["顺便", "同时", "另外", "也解释", "也说说", "再说说", "并且"]
    has_connector = _contains_any_casefold(query, connector_terms)
    has_shopping = _contains_any_casefold(query, _shopping_cue_terms())
    has_knowledge = _contains_any_casefold(query, _knowledge_cue_terms())
    has_compare = _contains_any_casefold(query, _compare_cue_terms())
    return (has_connector and (has_shopping or has_compare) and has_knowledge) or (
        (has_shopping or has_compare) and has_knowledge and len(query) >= 18
    )


def _looks_like_unknown_category_purchase(
    query: str,
    rule_result: QueryUnderstandingResult,
) -> bool:
    if rule_result.category is not None:
        return False
    if _looks_like_chitchat_query(query):
        return False
    return _looks_like_user_needs_product_help(query)


def _looks_like_user_needs_product_help(query: str) -> bool:
    if _looks_like_chitchat_query(query):
        return False
    return _contains_any_casefold(query, _shopping_cue_terms())


def _looks_like_chitchat_query(query: str) -> bool:
    return query.strip().lower() in {
        "你好",
        "hi",
        "hello",
        "在吗",
        "谢谢",
        "thanks",
        "thank you",
        "浣犲ソ",
        "鍦ㄥ悧",
    }


def _shopping_cue_terms() -> list[str]:
    return [
        "推荐",
        "买",
        "想买",
        "想要",
        "适合",
        "预算",
        "别太贵",
        "不要太贵",
        "recommend",
        "buy",
        "budget",
        *QueryUnderstandingService.SHOPPING_KEYWORDS,
    ]


def _knowledge_cue_terms() -> list[str]:
    return [
        "为什么",
        "怎么选",
        "原理",
        "解释",
        "说说",
        "主要看",
        "哪些参数",
        "why",
        "explain",
        *QueryUnderstandingService.KNOWLEDGE_KEYWORDS,
    ]


def _compare_cue_terms() -> list[str]:
    return [
        "第一个",
        "第二个",
        "这两个",
        "对比",
        "比较",
        "哪个好",
        "first",
        "second",
        "compare",
        *QueryUnderstandingService.COMPARE_KEYWORDS,
    ]


def _contains_any_casefold(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(str(term).lower() in lowered for term in terms if term)


def build_llm_understanding_prompt(
    *,
    query: str,
    history: list[Any] | None,
    previous_memory: ShoppingMemory,
) -> list[LLMMessage]:
    payload = {
        "current_query": query,
        "recent_turns": _recent_turns_payload(history),
        "shopping_memory": previous_memory.to_dict(),
        "last_recommended_products": _last_products_payload(previous_memory),
        "allowed_categories": ["phone", "shoes", "skincare", None],
        "allowed_intents": sorted(VALID_INTENTS),
        "output_schema": {
            "is_follow_up": "boolean",
            "intent": sorted(VALID_INTENTS),
            "category": ["phone", "shoes", "skincare", None],
            "budget": {"min": None, "max": "int|null", "currency": "CNY"},
            "preferences": ["short preference words"],
            "negative_preferences": ["short negative preference words"],
            "compare_product_ids": ["only ids from last_recommended_products"],
            "referenced_product_indices": ["positive integers within products"],
            "confidence": "0..1",
            "reason": "short debug reason",
        },
    }
    system_prompt = (
        f"{LLM_UNDERSTANDING_MARKER}\n"
        "You are SmartBuyAgent Query Understanding. Your only task is to parse "
        "the current user message into structured JSON slots. Do not answer the "
        "user. Do not create products, citations, source URLs, purchase links, "
        "orders, payments, or cart actions. Supported categories are phone, "
        "shoes, skincare. Supported intents are shopping_guide, "
        "product_knowledge, compare, clarification, chitchat. Return one JSON "
        "object only, without markdown. If the user refers to the first or "
        "second product, use referenced_product_indices. compare_product_ids "
        "may only use ids provided in last_recommended_products. For skincare, "
        "do not extract medical claims such as treatment, cure, drug effect, "
        "prescription, or medical repair as preferences. If uncertain, use "
        "intent=clarification and low confidence."
    )
    return [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]


def parse_llm_understanding_json(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def sanitize_llm_understanding(
    output: LLMQueryUnderstandingOutput,
    *,
    allowed_product_ids: set[str],
    max_reference_index: int,
) -> LLMQueryUnderstandingOutput:
    category = output.category if output.category in VALID_CATEGORIES else None
    preferences = _sanitize_preference_terms(
        output.preferences,
        category=category,
    )
    negative_preferences = _sanitize_preference_terms(
        output.negative_preferences,
        category=None,
    )
    preferences = [item for item in preferences if item not in negative_preferences]
    compare_product_ids = [
        product_id
        for product_id in output.compare_product_ids
        if product_id in allowed_product_ids
    ]
    if max_reference_index <= 0:
        referenced_indices = []
    else:
        referenced_indices = [
            index
            for index in output.referenced_product_indices
            if 1 <= index <= max_reference_index
        ]
    return output.model_copy(
        update={
            "category": category,
            "budget": Budget(
                min=output.budget.min,
                max=output.budget.max,
                currency=output.budget.currency or "CNY",
            ),
            "preferences": preferences,
            "negative_preferences": negative_preferences,
            "compare_product_ids": compare_product_ids,
            "referenced_product_indices": referenced_indices,
            "confidence": max(0.0, min(1.0, output.confidence)),
            "reason": _safe_reason(output.reason),
        }
    )


def merge_rule_and_llm_understanding(
    *,
    rule_result: QueryUnderstandingResult,
    llm_output: LLMQueryUnderstandingOutput,
    previous_memory: ShoppingMemory,
    confidence_threshold: float,
) -> QueryUnderstandingResult:
    strong_rule = (
        rule_result.confidence >= confidence_threshold
        and rule_result.intent != "clarification"
    )
    intent = rule_result.intent if strong_rule else llm_output.intent
    if intent not in VALID_INTENTS:
        intent = "clarification"

    category = rule_result.category or llm_output.category or previous_memory.category
    budget = _select_budget(rule_result, llm_output, previous_memory)
    preferences = _dedupe_short_terms(
        [
            *previous_memory.preferences,
            *rule_result.preferences,
            *llm_output.preferences,
        ]
    )
    negative_preferences = _dedupe_short_terms(
        [
            *previous_memory.negative_preferences,
            *rule_result.negative_preferences,
            *llm_output.negative_preferences,
        ]
    )
    preferences = [item for item in preferences if item not in negative_preferences]

    if intent == "compare":
        actionable_compare_product_ids = (
            rule_result.compare_product_ids or llm_output.compare_product_ids
        )
        actionable_referenced_indices = (
            rule_result.referenced_product_indices
            or llm_output.referenced_product_indices
        )
    else:
        actionable_compare_product_ids = []
        actionable_referenced_indices = []

    current_memory = ShoppingMemory(
        category=category,
        budget=ShoppingBudget(
            min=budget.min,
            max=budget.max,
            currency=budget.currency,
        ),
        preferences=preferences,
        negative_preferences=negative_preferences,
        last_product_ids=actionable_compare_product_ids
        or previous_memory.last_product_ids,
        last_intent=intent,
    )
    resolved_memory = merge_shopping_memory(previous_memory, current_memory)
    next_dialog_state = infer_next_dialog_state(
        intent=intent,
        category=resolved_memory.category,
        budget_max=resolved_memory.budget.max,
        override=rule_result.next_dialog_state,
    )
    resolved_memory = _memory_with_dialog_state(
        resolved_memory,
        dialog_state=next_dialog_state,
        last_intent=intent,
    )
    effective_query = (
        build_effective_query(resolved_memory)
        if resolved_memory.category and intent in {"shopping_guide", "compare"}
        else rule_result.effective_query
    )
    effective_query = _sanitize_effective_query(effective_query, resolved_memory.category)
    compare_product_ids = actionable_compare_product_ids
    referenced_indices = actionable_referenced_indices
    confidence = max(rule_result.confidence, min(llm_output.confidence, 0.85))
    need_clarification = intent == "clarification" or (
        intent == "shopping_guide" and not resolved_memory.category
    )
    reason = _combine_reasons(rule_result.reason, llm_output.reason)
    return QueryUnderstandingResult(
        original_query=rule_result.original_query,
        effective_query=effective_query,
        is_follow_up=rule_result.is_follow_up or llm_output.is_follow_up,
        intent=intent,
        category=resolved_memory.category,
        category_id=category_to_id(resolved_memory.category),
        category_path=category_to_path(resolved_memory.category),
        budget=Budget(
            min=resolved_memory.budget.min,
            max=resolved_memory.budget.max,
            currency=resolved_memory.budget.currency,
        ),
        preferences=resolved_memory.preferences,
        negative_preferences=resolved_memory.negative_preferences,
        compare_product_ids=compare_product_ids,
        referenced_product_indices=referenced_indices,
        confidence=confidence,
        source="mixed",
        reason=reason,
        memory_updated=resolved_memory.has_shopping_context(),
        need_clarification=need_clarification,
        clarification_question=rule_result.clarification_question
        if need_clarification
        else None,
        shopping_memory=resolved_memory.to_dict(),
        dialog_state=rule_result.dialog_state,
        next_dialog_state=next_dialog_state,
        dialog_state_reason=rule_result.dialog_state_reason,
    )


def empty_memory_from_result(result: QueryUnderstandingResult) -> ShoppingMemory:
    if isinstance(result.shopping_memory, dict):
        return shopping_memory_from_dict(result.shopping_memory)
    return result.to_shopping_memory()


def _with_llm_failure(
    rule_result: QueryUnderstandingResult,
    error: str,
    *,
    decision: LLMFallbackDecision | None = None,
) -> QueryUnderstandingResult:
    reason = rule_result.reason or "rule_low_confidence"
    return rule_result.model_copy(
        update={
            "llm_fallback_attempted": True,
            "llm_fallback_status": "failed",
            "llm_fallback_error": error,
            "llm_fallback_should_call": bool(decision.should_call) if decision else True,
            "llm_fallback_trigger_reasons": decision.reasons if decision else [],
            "source": "rule",
            "reason": f"{reason}_fallback_to_rule",
        }
    )


def _select_budget(
    rule_result: QueryUnderstandingResult,
    llm_output: LLMQueryUnderstandingOutput,
    previous_memory: ShoppingMemory,
) -> Budget:
    if rule_result.budget.min is not None or rule_result.budget.max is not None:
        return rule_result.budget
    if llm_output.budget.min is not None or llm_output.budget.max is not None:
        return llm_output.budget
    return Budget(
        min=previous_memory.budget.min,
        max=previous_memory.budget.max,
        currency=previous_memory.budget.currency,
    )


def _recent_turns_payload(history: list[Any] | None, limit: int = 3) -> list[dict[str, str]]:
    if not history:
        return []
    payload: list[dict[str, str]] = []
    for turn in list(history)[-limit:]:
        user_query = str(getattr(turn, "user_query", "") or "")
        assistant_answer = str(getattr(turn, "assistant_answer", "") or "")
        if user_query:
            payload.append({"role": "user", "content": _truncate(user_query)})
        if assistant_answer:
            payload.append({"role": "assistant", "content": _truncate(assistant_answer)})
    return payload


def _last_products_payload(memory: ShoppingMemory) -> list[dict[str, str | None]]:
    return [{"id": product_id, "title": None} for product_id in memory.last_product_ids]


def _looks_like_ambiguous_follow_up(query: str) -> bool:
    lower_query = query.lower()
    if _contains_any_casefold(
        lower_query,
        [
            "贵一点",
            "便宜一点",
            "便宜",
            "更贵",
            "更便宜",
            "换一个",
            "还有吗",
            "再看看",
            "刚才",
            "那个",
            "这两个",
            "第一个",
            "第二个",
        ],
    ):
        return True
    phrases = [
        "贵一点",
        "便宜",
        "放宽",
        "别太",
        "不要太",
        "更适合",
        "第二个",
        "第一个",
        "刚才",
        "那个",
        "这两个",
        "换一个",
        "还有吗",
        "再看看",
        "妈妈",
        "对象",
    ]
    return any(phrase in lower_query for phrase in phrases)


def _contains_product_reference(query: str) -> bool:
    if re.search(r"第[一二三四五六七八九\d]+个", query) or any(
        term in query for term in ["刚才那个", "这两个", "这几款", "第一个", "第二个"]
    ):
        return True
    return bool(
        re.search(r"第[一二三四五六七八九\d]+个", query)
        or any(term in query for term in ["刚才那个", "这两个", "这几款"])
    )


def _sanitize_preference_terms(
    terms: list[str],
    *,
    category: str | None,
) -> list[str]:
    sanitized: list[str] = []
    for term in terms:
        item = str(term).strip()
        if not item or not _term_length_allowed(item):
            continue
        if _contains_boundary_term(item):
            continue
        if category == "skincare" and _contains_medical_term(item):
            sanitized.extend(_safe_skincare_terms(item))
            continue
        sanitized.append(item)
    return _dedupe_short_terms(sanitized)


def _safe_skincare_terms(term: str) -> list[str]:
    if any(token in term for token in ["痘", "油", "痤疮"]):
        return ["清爽", "控油", "温和"]
    return ["温和", "保湿"]


def _sanitize_effective_query(query: str, category: str | None) -> str:
    if category != "skincare":
        return query
    sanitized = query
    for term in SKINCARE_MEDICAL_TERMS:
        sanitized = sanitized.replace(term, "")
    return sanitized


def _contains_boundary_term(text: str) -> bool:
    return any(term in text for term in PURCHASE_BOUNDARY_TERMS)


def _contains_medical_term(text: str) -> bool:
    return any(term in text for term in SKINCARE_MEDICAL_TERMS)


def _term_length_allowed(text: str) -> bool:
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    if cjk_count:
        return cjk_count <= 12
    return len(text) <= 30


def _safe_reason(value: Any) -> str:
    reason = str(value or "unknown").strip()[:80]
    if not reason:
        return "unknown"
    if _contains_boundary_term(reason) or _contains_medical_term(reason):
        return "unsafe_llm_reason"
    return re.sub(r"[^A-Za-z0-9_\-+]", "_", reason) or "unknown"


def _combine_reasons(rule_reason: str, llm_reason: str) -> str:
    rule = _safe_reason(rule_reason or "rule_low_confidence")
    llm = _safe_reason(llm_reason or "unknown")
    return f"{rule}+llm_{llm}"


def _truncate(text: str, limit: int = 240) -> str:
    return text[:limit]


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


def _dedupe_terms(value: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for item in _list_of_str(value):
        term = item.strip()
        if not term or term in seen:
            continue
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
