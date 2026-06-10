from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.orm import Session

from app.retrieval.retrieval_service import ProductCandidate, load_product_detail


FOCUS_KEYWORDS: dict[str, list[str]] = {
    "拍照": ["拍照", "影像", "像素", "相机", "主摄", "防抖", "夜景", "人像"],
    "影像": ["拍照", "影像", "像素", "相机", "主摄", "防抖", "夜景", "人像"],
    "续航": ["续航", "电池", "电池容量", "快充", "充电", "mAh"],
    "性能": ["性能", "处理器", "运行内存", "游戏", "高刷", "刷新率"],
    "通勤": ["通勤", "舒适", "防滑", "透气", "耐磨", "鞋底"],
    "防滑": ["防滑", "鞋底", "橡胶", "雨天", "耐磨"],
    "油皮": ["油皮", "控油", "清爽", "成分", "肤质"],
    "控油": ["油皮", "控油", "清爽", "成分", "肤质"],
}


@dataclass(frozen=True)
class CompareContext:
    product_ids: list[str]
    source: str
    focus_preferences: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProductComparisonResult:
    product_candidates: list[ProductCandidate]
    answer: str
    trace: dict[str, Any]
    focus_preferences: list[str]


class ProductComparisonService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def compare(
        self,
        query: str,
        product_ids: list[str],
        focus_preferences: list[str] | None = None,
        source: str = "unknown",
    ) -> ProductComparisonResult:
        requested_product_ids = _unique_ordered(product_ids)
        focus = _infer_focus_preferences(query, focus_preferences or [])
        product_candidates: list[ProductCandidate] = []
        missing_product_ids: list[str] = []

        for product_id in requested_product_ids:
            candidate = load_product_detail(self.db, product_id)
            if candidate is None:
                missing_product_ids.append(product_id)
                continue
            product_candidates.append(_with_comparison_score(candidate, focus))

        returned_product_ids = [
            candidate.product_id for candidate in product_candidates
        ]
        status = "compared" if len(product_candidates) >= 2 else "insufficient_products"
        trace = {
            "step": "product_comparison",
            "status": status,
            "source": source,
            "requested_product_ids": requested_product_ids,
            "returned_product_ids": returned_product_ids,
            "missing_product_ids": missing_product_ids,
            "focus_preferences": focus,
        }

        return ProductComparisonResult(
            product_candidates=product_candidates,
            answer=_build_comparison_answer(
                query=query,
                product_candidates=product_candidates,
                focus_preferences=focus,
                status=status,
            ),
            trace=trace,
            focus_preferences=focus,
        )


def _infer_focus_preferences(query: str, preferences: list[str]) -> list[str]:
    focus = _unique_ordered(preferences)
    for preference, keywords in FOCUS_KEYWORDS.items():
        if preference in focus:
            continue
        if any(keyword.lower() in query.lower() for keyword in keywords):
            focus.append(preference)
    return focus


def _with_comparison_score(
    candidate: ProductCandidate,
    focus_preferences: list[str],
) -> ProductCandidate:
    bonus = _focus_match_score(candidate, focus_preferences)
    budget_bonus = 0.2 if candidate.price > 0 else 0.0
    return replace(candidate, score=round(candidate.score + bonus + budget_bonus, 4))


def _focus_match_score(
    candidate: ProductCandidate,
    focus_preferences: list[str],
) -> float:
    if not focus_preferences:
        return 0.0

    text = " ".join(
        [
            candidate.title,
            candidate.description or "",
            *candidate.tags,
            *candidate.attributes.keys(),
            *candidate.attributes.values(),
        ]
    ).lower()

    score = 0.0
    for preference in focus_preferences:
        keywords = FOCUS_KEYWORDS.get(preference, [preference])
        for keyword in keywords:
            if keyword.lower() in text:
                score += 1.0
    return score


def _build_comparison_answer(
    query: str,
    product_candidates: list[ProductCandidate],
    focus_preferences: list[str],
    status: str,
) -> str:
    prefix = "下面只基于上一轮推荐的商品做比较。"
    if status == "insufficient_products":
        return f"{prefix}目前可比较商品不足，建议先补充更多候选商品后再比较。"

    focus_text = "、".join(focus_preferences) if focus_preferences else "综合表现"
    ranked_candidates = sorted(
        product_candidates,
        key=lambda candidate: (-candidate.score, candidate.price, candidate.product_id),
    )
    best = ranked_candidates[0]
    parts = [
        prefix,
        f"本轮重点关注：{focus_text}。",
        f"综合当前商品信息，{best.title} 更匹配这个关注点；如果你更看重价格，可以优先看价格更低的候选。",
    ]

    if "区别" in query or "不同" in query:
        parts.append("主要区别如下：")
    else:
        parts.append("各商品可参考这些差异点：")

    for candidate in product_candidates:
        evidence = _candidate_evidence(candidate, focus_preferences)
        parts.append(f"{candidate.title}：价格 {candidate.price} 元，{evidence}。")

    return "".join(parts)


def _candidate_evidence(
    candidate: ProductCandidate,
    focus_preferences: list[str],
) -> str:
    matched_tags = [
        tag
        for tag in candidate.tags
        if not focus_preferences
        or any(
            keyword.lower() in tag.lower()
            for preference in focus_preferences
            for keyword in FOCUS_KEYWORDS.get(preference, [preference])
        )
    ]
    if matched_tags:
        return f"标签包含 {', '.join(matched_tags[:3])}"

    if candidate.attributes:
        pairs = list(candidate.attributes.items())[:3]
        return "；".join(f"{key}={value}" for key, value in pairs)

    return "当前商品信息较少，建议结合商品详情进一步判断"


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
