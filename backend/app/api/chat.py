from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.chat.chat_service import ChatService
from app.chat.response_composer import ChatResponse
from app.core.db import get_db
from app.retrieval.chroma_indexer import get_chroma_client
from app.schemas.chat import (
    ChatRequest,
    ChatResponseSchema,
    CitationResponse,
    ProductCardResponse,
)
from app.services.embedding import BaseEmbeddingService, get_embedding_service


router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_chroma_client():
    return get_chroma_client()


def get_chat_embedding_service() -> BaseEmbeddingService:
    return get_embedding_service()


@router.post("/chat", response_model=ChatResponseSchema)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chroma_client=Depends(get_chat_chroma_client),
    embedding_service: BaseEmbeddingService = Depends(get_chat_embedding_service),
) -> ChatResponseSchema:
    chat_service = ChatService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )
    chat_response = chat_service.handle_message(request.query)
    return _to_response_schema(
        chat_response,
        session_id=request.session_id,
        include_trace=request.debug,
    )


def _to_response_schema(
    response: ChatResponse,
    session_id: str | None,
    include_trace: bool,
) -> ChatResponseSchema:
    return ChatResponseSchema(
        answer=response.answer,
        product_cards=[
            ProductCardResponse(
                product_id=card.product_id,
                title=card.title,
                brand=card.brand,
                price=card.price,
                image_url=card.image_url,
                tags=card.tags,
                attributes=card.attributes,
                source_url=card.source_url,
                compare_url=card.compare_url,
                recommend_reason=card.recommend_reason,
            )
            for card in response.product_cards
        ],
        citations=[
            CitationResponse(
                chunk_id=citation.chunk_id,
                title=citation.title,
                section=citation.section,
                section_path=citation.section_path,
                source_file=citation.source_file,
                content_preview=citation.content_preview,
                score=citation.score,
            )
            for citation in response.citations
        ],
        trace=response.trace if include_trace else [],
        session_id=session_id,
    )
