from __future__ import annotations

from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate
from app.services.llm import BaseLLMService, LLMMessage


MAX_PRODUCTS = 3
MAX_CITATIONS = 5
MAX_ATTRIBUTES = 8
MAX_EVIDENCE_LENGTH = 260
MAX_CITATION_PREVIEW_LENGTH = 320


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

        messages = [
            LLMMessage(role="system", content=_build_system_prompt()),
            LLMMessage(
                role="user",
                content=_build_user_prompt(
                    query=query,
                    query_result=query_result,
                    product_candidates=products,
                    citations=citation_list,
                ),
            ),
        ]

        try:
            response = self.llm_service.chat(messages)
        except Exception:
            return _safe_fallback_answer()

        content = response.content.strip()
        if not content:
            return _safe_fallback_answer()
        return content


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
    return "当前没有找到足够匹配的商品或知识依据，建议你调整预算、品类或偏好后再试。"


def _truncate_text(text: str | None, limit: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."
