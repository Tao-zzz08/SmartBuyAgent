from __future__ import annotations

from typing import Any, Literal
import re

from pydantic import BaseModel, Field


class GuardViolation(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"] = "medium"
    message: str
    matched_text: str | None = None
    evidence_required: str | None = None


class AnswerGroundingContext(BaseModel):
    answer: str
    route: str | None = None
    query_understanding: dict[str, Any] = Field(default_factory=dict)
    product_cards: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    comparison_result: dict[str, Any] | None = None


class GroundingGuardResult(BaseModel):
    passed: bool
    action: Literal["pass", "fallback", "sanitize"] = "pass"
    violations: list[GuardViolation] = Field(default_factory=list)
    checks: dict[str, str] = Field(default_factory=dict)
    sanitized_answer: str | None = None
    fallback_answer: str | None = None


PURCHASE_FORBIDDEN_TERMS = [
    "立即购买",
    "马上购买",
    "点击购买",
    "购买链接",
    "下单",
    "支付",
    "结算",
    "购物车",
    "加入购物车",
    "领券",
    "优惠券",
    "满减",
    "秒杀",
    "限时优惠",
    "buy now",
    "checkout",
    "payment",
]

UNSUPPORTED_FACT_TERMS = [
    "库存充足",
    "现货",
    "有货",
    "销量第一",
    "销量最高",
    "爆款",
    "热销第一",
    "全网最低",
    "最低价",
    "历史低价",
    "官方正品",
    "保真",
    "正品保障",
    "限时优惠",
    "优惠券",
    "满减",
    "秒杀",
]

SKINCARE_MEDICAL_FORBIDDEN = [
    "治疗",
    "治愈",
    "药效",
    "处方",
    "医学修复",
    "修复疾病",
    "消炎药",
    "药物",
    "临床治愈率",
]

DOMAIN_TERMS = {
    "phone": [
        "传感器",
        "光圈",
        "防抖",
        "影像",
        "算法",
        "续航",
        "电池",
        "快充",
        "芯片",
        "屏幕",
    ],
    "shoes": [
        "尺码",
        "脚长",
        "鞋楦",
        "防滑",
        "鞋底",
        "缓震",
        "透气",
        "通勤",
        "轻便",
    ],
    "skincare": [
        "敏感肌",
        "温和",
        "保湿",
        "控油",
        "清爽",
        "刺激",
        "屏障",
        "日常护理",
    ],
}

BRAND_ALIASES = [
    {"苹果", "Apple", "iPhone", "iphone"},
    {"华为", "Huawei"},
    {"小米", "Xiaomi", "Redmi", "红米", "Poco"},
    {"荣耀", "Honor"},
    {"vivo", "Vivo"},
    {"OPPO", "Oppo", "oppo"},
    {"三星", "Samsung"},
    {"Nike", "耐克"},
    {"Adidas", "阿迪达斯"},
    {"安踏", "ANTA"},
    {"李宁"},
    {"理肤泉", "La Roche-Posay"},
    {"薇诺娜"},
]

PRICE_PATTERNS = [
    re.compile(r"￥\s*(\d{2,6})"),
    re.compile(r"(?:约|大约|目前|价格|售价|只要)?\s*(\d{2,6})\s*(?:元|块)"),
]

CATEGORY_ID_TO_CATEGORY = {
    "cat_phone": "phone",
    "cat_shoes": "shoes",
    "cat_skincare": "skincare",
}


class AnswerGroundingGuard:
    def check(self, context: AnswerGroundingContext) -> GroundingGuardResult:
        violations: list[GuardViolation] = []
        checks: dict[str, str] = {}

        self._check_purchase_boundary(context, violations, checks)
        self._check_price_claims(context, violations, checks)
        self._check_unsupported_fact_claims(context, violations, checks)
        self._check_brand_product_claims(context, violations, checks)
        self._check_citation_support(context, violations, checks)
        self._check_skincare_safety(context, violations, checks)

        if violations:
            return GroundingGuardResult(
                passed=False,
                action="fallback",
                violations=violations,
                checks=checks,
                fallback_answer=self._build_fallback_answer(context),
            )

        return GroundingGuardResult(
            passed=True,
            action="pass",
            checks=checks,
            sanitized_answer=context.answer,
        )

    def _check_purchase_boundary(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        matched = _first_match(context.answer, PURCHASE_FORBIDDEN_TERMS)
        if matched:
            violations.append(
                GuardViolation(
                    type="purchase_boundary_violation",
                    severity="high",
                    message="Answer contains purchase, payment, cart, coupon, or order action.",
                    matched_text=matched,
                    evidence_required="SmartBuyAgent does not provide purchase actions.",
                )
            )
            checks["purchase_boundary"] = "failed"
            return
        checks["purchase_boundary"] = "passed"

    def _check_price_claims(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        mentioned_prices = _extract_prices(context.answer)
        if not mentioned_prices:
            checks["price_claims"] = "passed"
            return

        allowed_prices = _allowed_prices(context)
        unsupported = [price for price in mentioned_prices if price not in allowed_prices]
        if unsupported:
            violations.append(
                GuardViolation(
                    type="unsupported_price_claim",
                    severity="high",
                    message="Answer contains price numbers that are not present in product cards, budget, or comparison evidence.",
                    matched_text=", ".join(str(price) for price in unsupported),
                    evidence_required="product_cards.price or query_understanding.budget",
                )
            )
            checks["price_claims"] = "failed"
            return
        checks["price_claims"] = "passed"

    def _check_unsupported_fact_claims(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        matched = _first_match(context.answer, UNSUPPORTED_FACT_TERMS)
        if matched and not _has_fact_evidence(context, matched):
            violations.append(
                GuardViolation(
                    type="unsupported_fact_claim",
                    severity="high",
                    message="Answer contains unsupported stock, sales, ranking, authenticity, or discount claim.",
                    matched_text=matched,
                    evidence_required="explicit stock/sales/discount/rank evidence in product cards",
                )
            )
            checks["unsupported_facts"] = "failed"
            return
        checks["unsupported_facts"] = "passed"

    def _check_brand_product_claims(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        known_text = " ".join(
            str(value)
            for card in context.product_cards
            for value in [
                card.get("title"),
                card.get("name"),
                card.get("brand"),
            ]
            if value
        )
        unknown_brands: list[str] = []
        for aliases in BRAND_ALIASES:
            if not any(alias and alias in context.answer for alias in aliases):
                continue
            if not any(alias and alias in known_text for alias in aliases):
                if _is_negative_brand_reference(
                    context.answer,
                    aliases,
                    context.query_understanding,
                ):
                    continue
                unknown_brands.append(sorted(aliases, key=len)[0])

        if unknown_brands:
            violations.append(
                GuardViolation(
                    type="unsupported_brand_or_product_claim",
                    severity="high",
                    message="Answer mentions a brand or product that is not present in current product cards.",
                    matched_text=", ".join(unknown_brands),
                    evidence_required="current product_cards title or brand",
                )
            )
            checks["brand_product_claims"] = "failed"
            return
        checks["brand_product_claims"] = "passed"

    def _check_citation_support(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        route = context.route or _str(context.query_understanding.get("intent"))
        if route != "product_knowledge":
            checks["citation_support"] = "skipped"
            return

        if not context.citations:
            violations.append(
                GuardViolation(
                    type="citation_support_missing",
                    severity="high",
                    message="Knowledge answer requires at least one retrieved citation.",
                    evidence_required="KnowledgeRetrieval citation chunks",
                )
            )
            checks["citation_support"] = "failed"
            return

        category = _category_from_context(context)
        answer_terms = _domain_terms_in_text(context.answer, category)
        if not answer_terms:
            checks["citation_support"] = "passed"
            return

        citation_text = _citation_text(context.citations)
        unsupported_terms = [term for term in answer_terms if term not in citation_text]
        if unsupported_terms and len(unsupported_terms) == len(answer_terms):
            violations.append(
                GuardViolation(
                    type="citation_support_missing",
                    severity="high",
                    message="Knowledge terms in answer are not supported by retrieved citations.",
                    matched_text=", ".join(unsupported_terms),
                    evidence_required="citation text/content/content_preview",
                )
            )
            checks["citation_support"] = "failed"
            return
        checks["citation_support"] = "passed"

    def _check_skincare_safety(
        self,
        context: AnswerGroundingContext,
        violations: list[GuardViolation],
        checks: dict[str, str],
    ) -> None:
        if not _is_skincare_context(context):
            checks["skincare_safety"] = "skipped"
            return

        matched = _first_match(context.answer, SKINCARE_MEDICAL_FORBIDDEN)
        if matched:
            violations.append(
                GuardViolation(
                    type="skincare_medical_claim",
                    severity="high",
                    message="Skincare answer contains medical treatment, cure, prescription, or drug-effect claim.",
                    matched_text=matched,
                    evidence_required="Skincare answers must stay within daily care guidance.",
                )
            )
            checks["skincare_safety"] = "failed"
            return
        checks["skincare_safety"] = "passed"

    def _build_fallback_answer(self, context: AnswerGroundingContext) -> str:
        category = _category_from_context(context)
        if category == "skincare":
            return (
                "根据当前商品和知识资料，建议优先关注清爽、控油、温和、保湿等日常护理方向。"
                "如果皮肤问题比较严重，建议咨询专业人士。"
            )

        route = context.route or _str(context.query_understanding.get("intent"))
        preferences = _list_str(context.query_understanding.get("preferences"))
        negative_preferences = _list_str(
            context.query_understanding.get("negative_preferences")
        )

        if route == "product_knowledge":
            terms = _domain_terms_in_text(_citation_text(context.citations), category)
            if terms:
                return (
                    "根据检索到的知识资料，可以优先关注"
                    f"{'、'.join(terms[:3])}等因素。具体结论请以知识引用内容为准。"
                )
            return "根据检索到的知识资料，可以优先关注引用中提到的关键因素。具体结论请以知识引用内容为准。"

        if route == "compare":
            return (
                "根据当前候选商品的价格、品牌、标签和参数信息，我整理了一版保守对比。"
                "具体差异请以商品卡片和对比字段为准。"
            )

        parts = [
            "根据当前筛选结果，我优先推荐以下商品。它们符合你的预算、品类和偏好要求。"
        ]
        if preferences:
            parts.append(f"这次主要根据你的「{'、'.join(preferences)}」偏好进行筛选。")
        if negative_preferences:
            parts.append(
                f"同时已尽量排除「{'、'.join(negative_preferences)}」等不考虑的条件。"
            )
        parts.append("你可以重点查看商品卡片中的价格、品牌、标签和参数信息，并结合知识引用进一步比较。")
        return "".join(parts)


def _first_match(text: str, terms: list[str]) -> str | None:
    lowered = text.lower()
    for term in terms:
        if term and term.lower() in lowered:
            return term
    return None


def _extract_prices(text: str) -> list[int]:
    prices: list[int] = []
    for pattern in PRICE_PATTERNS:
        for match in pattern.findall(text):
            try:
                prices.append(int(match))
            except (TypeError, ValueError):
                continue
    return _unique_ints(prices)


def _allowed_prices(context: AnswerGroundingContext) -> set[int]:
    prices: set[int] = set()
    for card in context.product_cards:
        for key in ["price", "current_price", "sale_price"]:
            value = _int_or_none(card.get(key))
            if value is not None:
                prices.add(value)

    budget = context.query_understanding.get("budget")
    if isinstance(budget, dict):
        for key in ["min", "max"]:
            value = _int_or_none(budget.get(key))
            if value is not None:
                prices.add(value)
    else:
        for key in ["budget_min", "budget_max"]:
            value = _int_or_none(context.query_understanding.get(key))
            if value is not None:
                prices.add(value)

    if context.comparison_result:
        _collect_ints(context.comparison_result, prices)
    return prices


def _has_fact_evidence(context: AnswerGroundingContext, matched: str) -> bool:
    evidence_keys = {
        "stock",
        "sales",
        "sales_count",
        "discount",
        "coupon",
        "rank",
        "ranking",
        "promotion",
    }
    for card in context.product_cards:
        for key in evidence_keys:
            value = card.get(key)
            if value not in {None, "", 0, False} and key in _fact_key_hint(matched):
                return True
    return False


def _fact_key_hint(term: str) -> set[str]:
    if any(token in term for token in ["库存", "现货", "有货"]):
        return {"stock"}
    if any(token in term for token in ["销量", "爆款", "热销"]):
        return {"sales", "sales_count", "rank", "ranking"}
    if any(token in term for token in ["优惠", "券", "满减", "秒杀", "低价"]):
        return {"discount", "coupon", "promotion"}
    return {"rank", "ranking"}


def _category_from_context(context: AnswerGroundingContext) -> str | None:
    query_understanding = context.query_understanding or {}
    category = query_understanding.get("category")
    if isinstance(category, str) and category:
        return category

    category_id = query_understanding.get("category_id")
    if isinstance(category_id, str) and category_id in CATEGORY_ID_TO_CATEGORY:
        return CATEGORY_ID_TO_CATEGORY[category_id]

    for card in context.product_cards:
        card_category = card.get("category")
        if isinstance(card_category, str) and card_category:
            return card_category
        card_category_id = card.get("category_id")
        if isinstance(card_category_id, str) and card_category_id in CATEGORY_ID_TO_CATEGORY:
            return CATEGORY_ID_TO_CATEGORY[card_category_id]

    for citation in context.citations:
        citation_category = citation.get("category")
        if isinstance(citation_category, str) and citation_category:
            return citation_category
        citation_category_id = citation.get("category_id")
        if isinstance(citation_category_id, str) and citation_category_id in CATEGORY_ID_TO_CATEGORY:
            return CATEGORY_ID_TO_CATEGORY[citation_category_id]
    return None


def _domain_terms_in_text(text: str, category: str | None) -> list[str]:
    if not category:
        terms = [term for values in DOMAIN_TERMS.values() for term in values]
    else:
        terms = DOMAIN_TERMS.get(category, [])
    return [term for term in terms if term in text]


def _citation_text(citations: list[dict[str, Any]]) -> str:
    values: list[str] = []
    for citation in citations:
        for key in ["text", "content", "content_preview", "snippet", "title", "section"]:
            value = citation.get(key)
            if value:
                values.append(str(value))
    return "\n".join(values)


def _is_skincare_context(context: AnswerGroundingContext) -> bool:
    if _category_from_context(context) == "skincare":
        return True
    return "护肤" in context.answer or "敏感肌" in context.answer


def _is_negative_brand_reference(
    text: str,
    aliases: set[str],
    query_understanding: dict[str, Any],
) -> bool:
    negative_preferences = _list_str(query_understanding.get("negative_preferences"))
    if not any(
        alias
        for alias in aliases
        if alias and any(alias in preference or preference in alias for preference in negative_preferences)
    ):
        return False

    negative_cues = ["不考虑", "不要", "不想要", "排除", "避开", "过滤", "已排除", "尽量排除"]
    return any(cue in text for cue in negative_cues)


def _int_or_none(value: Any) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _collect_ints(value: Any, output: set[int]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _collect_ints(item, output)
        return
    if isinstance(value, list):
        for item in value:
            _collect_ints(item, output)
        return
    integer = _int_or_none(value)
    if integer is not None:
        output.add(integer)


def _unique_ints(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _list_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
