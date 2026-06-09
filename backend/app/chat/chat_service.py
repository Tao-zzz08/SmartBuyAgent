from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.chat.llm_answer_composer import LLMAnswerComposer, SAFE_LLM_FALLBACK_ANSWER
from app.chat.query_understanding import (
    QueryUnderstandingResult,
    QueryUnderstandingService,
)
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.retrieval.retrieval_service import (
    Citation,
    KnowledgeRetrievalService,
    ProductCandidate,
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
        llm_answer_composer: LLMAnswerComposer | None = None,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        self.query_understanding_service = (
            query_understanding_service or QueryUnderstandingService()
        )
        self.response_composer = response_composer or ResponseComposer()
        self.llm_answer_composer = llm_answer_composer
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
                preferences=query_result.preferences,
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
            response = self._compose_with_optional_llm(
                query=query,
                query_result=query_result,
                base_response=response,
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
            response = self._compose_with_optional_llm(
                query=query,
                query_result=query_result,
                base_response=response,
                product_candidates=[],
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

    def _compose_with_optional_llm(
        self,
        query: str,
        query_result: QueryUnderstandingResult,
        base_response: ChatResponse,
        product_candidates: list[ProductCandidate] | None = None,
        citations: list[Citation] | None = None,
    ) -> ChatResponse:
        if self.llm_answer_composer is None:
            return self._append_response_trace(
                base_response,
                {
                    "step": "llm_answer",
                    "enabled": False,
                    "status": "disabled",
                },
            )

        provider = self._llm_provider()
        try:
            llm_answer = self.llm_answer_composer.compose(
                query=query,
                query_result=query_result,
                product_candidates=product_candidates or [],
                citations=citations or [],
            )
        except Exception:
            return self._append_response_trace(
                base_response,
                {
                    "step": "llm_answer",
                    "enabled": True,
                    "status": "fallback",
                    "provider": provider,
                },
            )

        normalized_answer = llm_answer.strip()
        if not normalized_answer or normalized_answer == SAFE_LLM_FALLBACK_ANSWER:
            return self._append_response_trace(
                base_response,
                {
                    "step": "llm_answer",
                    "enabled": True,
                    "status": "fallback",
                    "provider": provider,
                },
            )

        return ChatResponse(
            answer=normalized_answer,
            product_cards=base_response.product_cards,
            citations=base_response.citations,
            trace=[
                *base_response.trace,
                {
                    "step": "llm_answer",
                    "enabled": True,
                    "status": "success",
                    "provider": provider,
                },
            ],
        )

    def _llm_provider(self) -> str:
        llm_service = getattr(self.llm_answer_composer, "llm_service", None)
        provider = getattr(llm_service, "provider", None)
        if isinstance(provider, str) and provider:
            return provider
        provider = getattr(self.llm_answer_composer, "provider", None)
        if isinstance(provider, str) and provider:
            return provider
        return "unknown"

    @staticmethod
    def _append_response_trace(
        response: ChatResponse,
        trace_step: dict[str, Any],
    ) -> ChatResponse:
        return ChatResponse(
            answer=response.answer,
            product_cards=response.product_cards,
            citations=response.citations,
            trace=[*response.trace, trace_step],
        )
