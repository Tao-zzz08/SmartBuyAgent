from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate
from app.services.llm import BaseLLMService, LLMMessage
from app.streaming.safety import (
    STREAM_SAFE_RELEASE_HOLD_CHARS,
    StreamSafetyGuard,
    StreamSafetyViolation,
)


MAX_PRODUCTS = 3
MAX_CITATIONS = 5
MAX_ATTRIBUTES = 8
MAX_EVIDENCE_LENGTH = 260
MAX_CITATION_PREVIEW_LENGTH = 280
SAFE_LLM_FALLBACK_ANSWER = "当前没有找到足够匹配的商品或知识依据，建议你调整预算、品类或偏好后再试。"
PRODUCT_ID_PATTERN = re.compile(r"\b[a-zA-Z]+_\d+\b")
URL_PATTERN = re.compile(r"https?://[^\s)）\]】>\"']+")

PURCHASE_ACTION_KEYWORDS = [
    "已经下单",
    "已经帮你下单",
    "已帮你下单",
    "已加入购物车",
    "加入购物车",
    "立即下单",
    "现在下单",
    "已完成购买",
    "已支付",
    "支付成功",
    "已锁价",
    "锁价成功",
]
DISCOUNT_HALLUCINATION_KEYWORDS = [
    "全网最低价",
    "保证最低价",
    "一定最低",
    "官方补贴",
    "限时优惠",
    "独家优惠",
    "优惠券已领取",
    "我帮你领取优惠券",
]
SKINCARE_MEDICAL_CLAIM_KEYWORDS = [
    "治疗",
    "治愈",
    "根治",
    "药效",
    "医用疗效",
    "治好",
    "修复皮肤病",
    "治疗痘痘",
    "治疗湿疹",
]


@dataclass(frozen=True)
class LLMAnswerValidationResult:
    is_valid: bool
    reason: str | None = None


class LLMAnswerComposer:
    def __init__(self, llm_service: BaseLLMService) -> None:
        self.llm_service = llm_service

    def compose(
        self,
        query: str,
        query_result: QueryUnderstandingResult,
        product_candidates: list[ProductCandidate] | None = None,
        citations: list[Citation] | None = None,
    ) -> str:
        products = product_candidates or []
        citation_list = citations or []
        if not products and not citation_list:
            return _safe_fallback_answer()

        messages = _build_messages(
            query=query,
            query_result=query_result,
            product_candidates=products,
            citations=citation_list,
        )

        try:
            response = self.llm_service.chat(messages)
        except Exception:
            return _safe_fallback_answer()

        content = response.content.strip()
        if not content:
            return _safe_fallback_answer()

        validation = validate_llm_answer(
            content,
            product_candidates=products,
            citations=citation_list,
        )
        if not validation.is_valid:
            return _safe_fallback_answer()

        return content

    def stream_compose(
        self,
        query: str,
        query_result: QueryUnderstandingResult,
        product_candidates: list[ProductCandidate] | None = None,
        citations: list[Citation] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        products = product_candidates or []
        citation_list = citations or []
        if not products and not citation_list:
            return _safe_fallback_answer()

        messages = _build_messages(
            query=query,
            query_result=query_result,
            product_candidates=products,
            citations=citation_list,
        )

        parts: list[str] = []
        pending_safe_text = ""
        guard = StreamSafetyGuard()
        category_context = query_result.category_id or query_result.category_path
        try:
            for delta in self.llm_service.stream_chat(messages):
                if not delta:
                    continue
                decision = guard.check_delta(
                    delta,
                    category=category_context,
                    intent=query_result.intent,
                )
                if not decision.safe:
                    raise StreamSafetyViolation(
                        reason=decision.reason or "unsafe_stream_delta",
                        matched_phrase=decision.matched_phrase,
                        severity=decision.severity,
                    )
                parts.append(delta)
                pending_safe_text += delta
                if len(pending_safe_text) > STREAM_SAFE_RELEASE_HOLD_CHARS:
                    releasable = pending_safe_text[:-STREAM_SAFE_RELEASE_HOLD_CHARS]
                    pending_safe_text = pending_safe_text[-STREAM_SAFE_RELEASE_HOLD_CHARS:]
                    if releasable and on_token is not None:
                        on_token(releasable)
        except StreamSafetyViolation:
            raise
        except Exception:
            fallback_answer = self.compose(
                query=query,
                query_result=query_result,
                product_candidates=products,
                citations=citation_list,
            )
            for delta in _chunk_text(fallback_answer):
                if on_token is not None:
                    on_token(delta)
            return fallback_answer

        content = "".join(parts).strip()
        if not content:
            return _safe_fallback_answer()

        stream_validation = guard.check_buffer(
            content,
            category=category_context,
            intent=query_result.intent,
        )
        if not stream_validation.safe:
            raise StreamSafetyViolation(
                reason=stream_validation.reason or "unsafe_stream_answer",
                matched_phrase=stream_validation.matched_phrase,
                severity=stream_validation.severity,
            )

        validation = validate_llm_answer(
            content,
            product_candidates=products,
            citations=citation_list,
        )
        if not validation.is_valid:
            return _safe_fallback_answer()

        if pending_safe_text and on_token is not None:
            on_token(pending_safe_text)

        return content


def validate_llm_answer(
    answer: str,
    product_candidates: list[ProductCandidate] | None = None,
    citations: list[Citation] | None = None,
) -> LLMAnswerValidationResult:
    normalized = answer.strip()
    if not normalized:
        return LLMAnswerValidationResult(False, "empty")

    if normalized == SAFE_LLM_FALLBACK_ANSWER:
        return LLMAnswerValidationResult(False, "safe_fallback")

    if _contains_any(normalized, PURCHASE_ACTION_KEYWORDS):
        return LLMAnswerValidationResult(False, "purchase_action")

    if _contains_any(normalized, DISCOUNT_HALLUCINATION_KEYWORDS):
        return LLMAnswerValidationResult(False, "discount_hallucination")

    if _contains_any(normalized, SKINCARE_MEDICAL_CLAIM_KEYWORDS):
        return LLMAnswerValidationResult(False, "skincare_medical_claim")

    compact = normalized.strip()
    if (compact.startswith("{") and compact.endswith("}")) or (
        compact.startswith("[") and compact.endswith("]")
    ):
        return LLMAnswerValidationResult(False, "json_output")

    if "|" in normalized and "---" in normalized and len(normalized.splitlines()) > 1:
        return LLMAnswerValidationResult(False, "markdown_table")

    products = product_candidates or []
    if products:
        allowed_product_ids = {product.product_id for product in products}
        mentioned_ids = set(PRODUCT_ID_PATTERN.findall(normalized))
        unknown_ids = mentioned_ids - allowed_product_ids
        if unknown_ids:
            return LLMAnswerValidationResult(False, "unknown_product_id")

    unknown_urls = _unknown_urls(normalized, product_candidates=products)
    if unknown_urls:
        return LLMAnswerValidationResult(False, "unknown_url")

    return LLMAnswerValidationResult(True)


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _unknown_urls(
    answer: str,
    product_candidates: list[ProductCandidate],
) -> list[str]:
    allowed_urls = {
        url
        for product in product_candidates
        for url in (product.source_url, product.compare_url)
        if url
    }
    urls = [
        url.rstrip(".,;:!?。，；：！？")
        for url in URL_PATTERN.findall(answer)
    ]
    return [url for url in urls if url not in allowed_urls]


def _build_system_prompt() -> str:
    return "\n".join(
        [
            "你是 SmartBuyAgent 的电商导购回答生成器。",
            "你只能基于输入中的候选商品和引用信息回答。",
            "不能编造商品、价格、库存、链接、优惠、平台政策。",
            "不能推荐候选列表之外的商品。",
            "不能声称已经下单、加入购物车、锁价、支付或完成购买。",
            "不能承诺护肤品具有治疗、治愈、药效。",
            "回答要简洁、自然、中文。",
            "如果证据不足，要明确说明不确定，并建议用户补充预算、品类或偏好。",
            "需要优先解释推荐理由，而不是只堆商品名。",
        ]
    )


def _build_user_prompt(
    query: str,
    query_result: QueryUnderstandingResult,
    product_candidates: list[ProductCandidate],
    citations: list[Citation],
) -> str:
    sections = [
        "用户原始问题：",
        query,
        "",
        "QueryUnderstandingResult：",
        f"- intent: {query_result.intent}",
        f"- category_id: {query_result.category_id or '-'}",
        f"- category_path: {query_result.category_path or '-'}",
        f"- budget_min: {query_result.budget_min if query_result.budget_min is not None else '-'}",
        f"- budget_max: {query_result.budget_max if query_result.budget_max is not None else '-'}",
        f"- preferences: {', '.join(query_result.preferences) if query_result.preferences else '-'}",
        f"- negative_preferences: {', '.join(query_result.negative_preferences) if query_result.negative_preferences else '-'}",
        "",
        "候选商品：",
        _format_products(product_candidates),
        "",
        "引用信息：",
        _format_citations(citations),
        "",
        "输出要求：",
        "- 先给 1 段总体建议。",
        "- 再给 2-3 个推荐理由。",
        "- 如果有商品，必须只围绕候选商品回答。",
        "- 如果有 citation，可以使用“依据引用信息”表达，但不要生成不存在的 citation 编号。",
        "- 不要输出 JSON。",
        "- 不要输出 Markdown 表格。",
        "- 不要输出购买、下单、支付、加入购物车等动作指令。",
    ]
    return "\n".join(sections)


def _build_messages(
    query: str,
    query_result: QueryUnderstandingResult,
    product_candidates: list[ProductCandidate],
    citations: list[Citation],
) -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content=_build_system_prompt()),
        LLMMessage(
            role="user",
            content=_build_user_prompt(
                query=query,
                query_result=query_result,
                product_candidates=product_candidates,
                citations=citations,
            ),
        ),
    ]


def _format_products(product_candidates: list[ProductCandidate]) -> str:
    if not product_candidates:
        return "- 无候选商品"

    lines: list[str] = []
    for index, product in enumerate(product_candidates[:MAX_PRODUCTS], start=1):
        lines.extend(
            [
                f"{index}. product_id: {product.product_id}",
                f"   title: {product.title}",
                f"   brand: {product.brand or '-'}",
                f"   price: {product.price}",
                f"   stock: {product.stock}",
                f"   tags: {', '.join(product.tags) if product.tags else '-'}",
                f"   attributes: {_format_attributes(product.attributes)}",
                "   recommend_evidence: "
                f"{_truncate_text(product.description or product.product_text, MAX_EVIDENCE_LENGTH)}",
            ]
        )
    return "\n".join(lines)


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "- 无引用信息"

    lines: list[str] = []
    for index, citation in enumerate(citations[:MAX_CITATIONS], start=1):
        lines.extend(
            [
                f"{index}. chunk_id: {citation.chunk_id}",
                f"   title: {citation.title or '-'}",
                f"   section: {citation.section or '-'}",
                f"   source_file: {citation.source_file or '-'}",
                "   content_preview: "
                f"{_truncate_text(citation.content_preview, MAX_CITATION_PREVIEW_LENGTH)}",
            ]
        )
    return "\n".join(lines)


def _format_attributes(attributes: dict[str, str]) -> str:
    if not attributes:
        return "-"

    pairs = list(attributes.items())[:MAX_ATTRIBUTES]
    return "; ".join(f"{key}={value}" for key, value in pairs)


def _safe_fallback_answer() -> str:
    return SAFE_LLM_FALLBACK_ANSWER


def _truncate_text(text: str | None, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _chunk_text(text: str, chunk_size: int = 18) -> list[str]:
    return [
        text[index : index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ]
