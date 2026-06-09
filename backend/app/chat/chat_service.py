from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.chat.query_understanding import (
    QueryUnderstandingResult,
    QueryUnderstandingService,
)
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
    ProductSearchFilters,
)
from app.services.embedding import BaseEmbeddingService


class ChatService:
    def __init__(
        self,
        db: Session,
        embedding_service: BaseEmbeddingService,
        chroma_client=None,
        query_understanding_service: QueryUnderstandingService | None = None,
        response_composer: ResponseComposer | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        self.query_understanding_service = (
            query_understanding_service or QueryUnderstandingService()
        )
        self.response_composer = response_composer or ResponseComposer()
        self.product_retrieval_service = ProductRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )
        self.knowledge_retrieval_service = KnowledgeRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

    def handle_message(self, query: str) -> ChatResponse:
        query_result = self.query_understanding_service.understand(query)
        trace: list[dict[str, Any]] = [self._query_understanding_trace(query_result)]

        if query_result.need_clarification:
            return self._with_trace(self.response_composer.compose(query_result), trace)

        if query_result.intent in {"chitchat", "compare"}:
            return self._with_trace(self.response_composer.compose(query_result), trace)

        if query_result.intent == "shopping_guide":
            product_filters = ProductSearchFilters(
                category_id=query_result.category_id,
                budget_min=query_result.budget_min,
                budget_max=query_result.budget_max,
                stock_only=True,
            )
            product_candidates = self.product_retrieval_service.search_products(
                query=query,
                filters=product_filters,
                top_k=3,
            )
            trace.append(
                {
                    "step": "product_retrieval",
                    "category_id": query_result.category_id,
                    "budget_min": query_result.budget_min,
                    "budget_max": query_result.budget_max,
                    "candidate_count": len(product_candidates),
                }
            )

            citations = self.knowledge_retrieval_service.search_knowledge(
                query=query,
                category_id=query_result.category_id,
                top_k=3,
            )
            trace.append(
                {
                    "step": "knowledge_retrieval",
                    "category_id": query_result.category_id,
                    "citation_count": len(citations),
                }
            )

            response = self.response_composer.compose(
                query_result,
                product_candidates=product_candidates,
                citations=citations,
            )
            return self._with_trace(response, trace)

        if query_result.intent == "product_knowledge":
            citations = self.knowledge_retrieval_service.search_knowledge(
                query=query,
                category_id=query_result.category_id,
                top_k=5,
            )
            trace.append(
                {
                    "step": "knowledge_retrieval",
                    "category_id": query_result.category_id,
                    "citation_count": len(citations),
                }
            )

            response = self.response_composer.compose(
                query_result,
                citations=citations,
            )
            return self._with_trace(response, trace)

        return self._with_trace(self.response_composer.compose(query_result), trace)

    @staticmethod
    def _query_understanding_trace(
        query_result: QueryUnderstandingResult,
    ) -> dict[str, Any]:
        return {
            "step": "query_understanding",
            "intent": query_result.intent,
            "category_id": query_result.category_id,
            "category_path": query_result.category_path,
            "budget_min": query_result.budget_min,
            "budget_max": query_result.budget_max,
            "preferences": query_result.preferences,
            "need_clarification": query_result.need_clarification,
        }

    @staticmethod
    def _with_trace(response: ChatResponse, trace: list[dict[str, Any]]) -> ChatResponse:
        return ChatResponse(
            answer=response.answer,
            product_cards=response.product_cards,
            citations=response.citations,
            trace=[*trace, *response.trace],
        )
