from __future__ import annotations

from dataclasses import dataclass, field
import re

from app.chat.shopping_memory import (
    Budget,
    ShoppingMemory,
    build_effective_query,
    category_from_id_or_path,
    category_to_id,
    category_to_path,
    extract_category,
    extract_memory_from_query,
    parse_budget_max,
)


@dataclass(frozen=True)
class QueryUnderstandingResult:
    raw_query: str
    intent: str
    category_id: str | None
    category_path: str | None
    budget_min: int | None
    budget_max: int | None
    preferences: list[str]
    need_clarification: bool
    clarification_question: str | None
    category: str | None = None
    negative_preferences: list[str] = field(default_factory=list)
    shopping_memory: dict | None = None
    effective_query: str | None = None
    is_follow_up: bool = False
    source: str = "rule"
    confidence: float = 0.8
    reason: str | None = None


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

    def understand(self, query: str) -> QueryUnderstandingResult:
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

        return QueryUnderstandingResult(
            raw_query=raw_query,
            intent=intent,
            category_id=category.category_id if category else None,
            category_path=category.category_path if category else None,
            budget_min=budget_min,
            budget_max=budget_max,
            preferences=preferences,
            need_clarification=need_clarification,
            clarification_question=self.CLARIFICATION_QUESTION
            if need_clarification
            else None,
            category=category_value,
            negative_preferences=memory.negative_preferences,
            shopping_memory=ShoppingMemory(
                category=category_value,
                budget=Budget(min=budget_min, max=budget_max),
                preferences=preferences,
                negative_preferences=memory.negative_preferences,
                last_intent=intent,
            ).to_dict(),
            effective_query=build_effective_query(
                ShoppingMemory(
                    category=category_value,
                    budget=Budget(min=budget_min, max=budget_max),
                    preferences=preferences,
                    negative_preferences=memory.negative_preferences,
                    last_intent=intent,
                )
            )
            if intent == "shopping_guide" and category_value
            else raw_query,
            reason="direct_rule",
        )

    def _clarification_result(self, raw_query: str) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            raw_query=raw_query,
            intent="clarification",
            category_id=None,
            category_path=None,
            budget_min=None,
            budget_max=None,
            preferences=[],
            need_clarification=True,
            clarification_question=self.CLARIFICATION_QUESTION,
            effective_query=raw_query,
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
