from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.cache.cache_service import CacheService
    from app.chat.conversation_memory import ConversationMemoryService
    from app.chat.followup_rewriter import FollowUpQueryRewriter
    from app.chat.llm_answer_composer import LLMAnswerComposer
    from app.chat.product_comparison import ProductComparisonService
    from app.chat.query_understanding import QueryUnderstandingService
    from app.chat.response_composer import ResponseComposer
    from app.retrieval.retrieval_service import (
        KnowledgeRetrievalService,
        ProductRetrievalService,
    )
    from app.services.embedding import BaseEmbeddingService
    from app.services.answer_grounding_guard import AnswerGroundingGuard


@dataclass
class AgentRuntimeContext:
    db: Session | None = None
    embedding_service: BaseEmbeddingService | None = None
    chroma_client: Any | None = None
    query_understanding_service: QueryUnderstandingService | None = None
    product_retrieval_service: ProductRetrievalService | None = None
    knowledge_retrieval_service: KnowledgeRetrievalService | None = None
    response_composer: ResponseComposer | None = None
    llm_answer_composer: LLMAnswerComposer | None = None
    conversation_memory_service: ConversationMemoryService | None = None
    followup_rewriter: FollowUpQueryRewriter | None = None
    product_comparison_service: ProductComparisonService | None = None
    answer_grounding_guard: AnswerGroundingGuard | None = None
    cache_service: CacheService | None = None
