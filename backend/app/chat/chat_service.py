from __future__ import annotations

from sqlalchemy.orm import Session

from app.agent.context import AgentRuntimeContext
from app.agent.workflow import AgentWorkflow
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.llm_answer_composer import LLMAnswerComposer
from app.chat.product_comparison import CompareContext, ProductComparisonService
from app.chat.query_understanding import QueryUnderstandingService
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
)
from app.services.embedding import BaseEmbeddingService


class ChatService:
    def __init__(
        self,
        db: Session | None,
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
        self.product_retrieval_service = (
            ProductRetrievalService(
                db=db,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
            )
            if db is not None
            else None
        )
        self.knowledge_retrieval_service = (
            KnowledgeRetrievalService(
                db=db,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
            )
            if db is not None
            else None
        )
        self.conversation_memory_service = (
            ConversationMemoryService(db) if db is not None else None
        )
        self.product_comparison_service = (
            ProductComparisonService(db=db) if db is not None else None
        )

    def handle_message(
        self,
        query: str,
        session_id: str | None = None,
        compare_context: CompareContext | None = None,
    ) -> ChatResponse:
        try:
            workflow = AgentWorkflow(self._runtime_context())
            state = workflow.run(
                query=query,
                session_id=session_id,
                compare_context=compare_context,
            )
            return ChatResponse(
                answer=state.answer
                or "当前导购服务暂时没有生成回答，请稍后再试。",
                product_cards=list(state.product_cards),
                citations=list(state.citations),
                trace=list(state.trace),
            )
        except Exception as exc:
            return ChatResponse(
                answer="当前导购服务暂时不可用，请稍后再试。",
                product_cards=[],
                citations=[],
                trace=[
                    {
                        "step": "agent_workflow",
                        "status": "failed",
                        "error": str(exc),
                    }
                ],
            )

    def _runtime_context(self) -> AgentRuntimeContext:
        return AgentRuntimeContext(
            db=self.db,
            embedding_service=self.embedding_service,
            chroma_client=self.chroma_client,
            query_understanding_service=self.query_understanding_service,
            product_retrieval_service=self.product_retrieval_service,
            knowledge_retrieval_service=self.knowledge_retrieval_service,
            response_composer=self.response_composer,
            llm_answer_composer=self.llm_answer_composer,
            conversation_memory_service=self.conversation_memory_service,
            product_comparison_service=self.product_comparison_service,
        )
