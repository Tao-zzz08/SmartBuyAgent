from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.chat.query_understanding import QueryUnderstandingResult
from app.retrieval.retrieval_service import Citation, ProductCandidate


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str | None
    price: int
    image_url: str | None
    tags: list[str]
    attributes: dict[str, str]
    source_url: str | None
    compare_url: str | None
    recommend_reason: str


@dataclass(frozen=True)
class CitationView:
    chunk_id: str
    title: str | None
    section: str | None
    section_path: str | None
    source_file: str | None
    content_preview: str
    score: float


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    product_cards: list[ProductCard]
    citations: list[CitationView]
    trace: list[dict[str, Any]]


class ResponseComposer:
    def compose(
        self,
        query_result: QueryUnderstandingResult,
        product_candidates: list[ProductCandidate] | None = None,
        citations: list[Citation] | None = None,
    ) -> ChatResponse:
        product_candidates = product_candidates or []
        citations = citations or []

        if query_result.need_clarification:
            product_cards: list[ProductCard] = []
            citation_views: list[CitationView] = []
            return ChatResponse(
                answer=query_result.clarification_question
                or "我需要再确认一下你的需求。",
                product_cards=product_cards,
                citations=citation_views,
                trace=self._build_trace(query_result, product_cards, citation_views),
            )

        if query_result.intent == "chitchat":
            product_cards = []
            citation_views = []
            return ChatResponse(
                answer="你好，我可以帮你挑选手机、鞋靴和护肤品。你可以告诉我预算、用途和偏好。",
                product_cards=product_cards,
                citations=citation_views,
                trace=self._build_trace(query_result, product_cards, citation_views),
            )

        if query_result.intent == "compare":
            product_cards = []
            citation_views = []
            return ChatResponse(
                answer="我已经识别到你想做商品对比，但当前阶段还没有实现对比服务。后续会基于商品详情和知识文档生成对比表。",
                product_cards=product_cards,
                citations=citation_views,
                trace=self._build_trace(query_result, product_cards, citation_views),
            )

        if query_result.intent == "product_knowledge":
            product_cards = []
            citation_views = self._build_citation_views(citations)
            if citation_views:
                answer = "我先根据知识文档整理了几个参考点，下面是可引用的资料片段。"
            else:
                answer = "当前知识库里还没有找到足够依据。"
            return ChatResponse(
                answer=answer,
                product_cards=product_cards,
                citations=citation_views,
                trace=self._build_trace(query_result, product_cards, citation_views),
            )

        if query_result.intent == "shopping_guide":
            product_cards = [
                self._build_product_card(query_result, candidate)
                for candidate in product_candidates
            ]
            citation_views = self._build_citation_views(citations)
            answer = self._build_shopping_answer(query_result, product_cards)
            return ChatResponse(
                answer=answer,
                product_cards=product_cards,
                citations=citation_views,
                trace=self._build_trace(query_result, product_cards, citation_views),
            )

        product_cards = []
        citation_views = self._build_citation_views(citations)
        return ChatResponse(
            answer="我已经收到你的问题，但当前阶段只支持基础导购、知识问答参考和澄清回复模板。",
            product_cards=product_cards,
            citations=citation_views,
            trace=self._build_trace(query_result, product_cards, citation_views),
        )

    def _build_shopping_answer(
        self,
        query_result: QueryUnderstandingResult,
        product_cards: list[ProductCard],
    ) -> str:
        if not product_cards:
            return "暂时没有找到满足条件的商品。你可以尝试放宽预算、品类或偏好后再试。"

        parts = ["我根据你的需求整理了下面几款候选商品。"]
        if query_result.category_path:
            parts.append(f"品类：{query_result.category_path}。")
        if query_result.budget_max is not None:
            parts.append(f"预算：{query_result.budget_max} 元以内。")
        if query_result.preferences:
            parts.append(f"主要偏好：{', '.join(query_result.preferences)}。")
        parts.append("下面几款是基于当前商品数据和知识文档整理的候选。")
        return "".join(parts)

    def _build_product_card(
        self,
        query_result: QueryUnderstandingResult,
        candidate: ProductCandidate,
    ) -> ProductCard:
        return ProductCard(
            product_id=candidate.product_id,
            title=candidate.title,
            brand=candidate.brand,
            price=candidate.price,
            image_url=candidate.image_url,
            tags=list(candidate.tags),
            attributes=dict(candidate.attributes),
            source_url=candidate.source_url,
            compare_url=candidate.compare_url,
            recommend_reason=self._build_recommend_reason(query_result, candidate),
        )

    def _build_recommend_reason(
        self,
        query_result: QueryUnderstandingResult,
        candidate: ProductCandidate,
    ) -> str:
        reasons: list[str] = []
        searchable_parts = [
            *candidate.tags,
            *candidate.attributes.keys(),
            *candidate.attributes.values(),
        ]
        searchable_text = " ".join(searchable_parts)

        for preference in query_result.preferences:
            if preference in searchable_text:
                reasons.append(f"匹配你的{preference}偏好")

        if (
            query_result.budget_max is not None
            and candidate.price <= query_result.budget_max
        ):
            reasons.append("价格在你的预算范围内")

        if candidate.tags:
            reasons.append(f"商品标签包含：{', '.join(candidate.tags[:3])}")

        if not reasons:
            return "基于当前商品信息进入候选列表"

        deduped: list[str] = []
        seen: set[str] = set()
        for reason in reasons:
            if reason not in seen:
                deduped.append(reason)
                seen.add(reason)
        return "；".join(deduped)

    def _build_citation_views(self, citations: list[Citation]) -> list[CitationView]:
        return [
            CitationView(
                chunk_id=citation.chunk_id,
                title=citation.title,
                section=citation.section,
                section_path=citation.section_path,
                source_file=citation.source_file,
                content_preview=citation.content_preview,
                score=citation.score,
            )
            for citation in citations
        ]

    def _build_trace(
        self,
        query_result: QueryUnderstandingResult,
        product_cards: list[ProductCard],
        citations: list[CitationView],
    ) -> list[dict[str, Any]]:
        step = {
            "step": "response_composer",
            "intent": query_result.intent,
            "product_count": len(product_cards),
            "citation_count": len(citations),
        }
        if query_result.need_clarification:
            step["need_clarification"] = True
        return [step]
