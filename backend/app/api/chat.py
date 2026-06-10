import json
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.chat.chat_service import ChatService
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.llm_answer_composer import LLMAnswerComposer
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
from app.services.llm import BaseLLMService, get_llm_service


router = APIRouter(prefix="/api", tags=["chat"])


def get_chat_chroma_client():
    return get_chroma_client()


def get_chat_embedding_service() -> BaseEmbeddingService:
    return get_embedding_service()


def get_chat_llm_service() -> BaseLLMService:
    return get_llm_service()


def get_chat_llm_answer_composer(
    llm_service: BaseLLMService = Depends(get_chat_llm_service),
) -> LLMAnswerComposer:
    return LLMAnswerComposer(llm_service)


@router.post("/chat", response_model=ChatResponseSchema)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chroma_client=Depends(get_chat_chroma_client),
    embedding_service: BaseEmbeddingService = Depends(get_chat_embedding_service),
    llm_answer_composer: LLMAnswerComposer = Depends(get_chat_llm_answer_composer),
) -> ChatResponseSchema:
    chat_service = ChatService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        llm_answer_composer=llm_answer_composer,
    )
    session_id = request.session_id or uuid4().hex
    chat_response = chat_service.handle_message(
        query=request.query,
        session_id=request.session_id,
    )
    chat_response = _save_conversation_turn(
        db=db,
        session_id=session_id,
        user_query=request.query,
        chat_response=chat_response,
    )
    return _to_response_schema(
        chat_response,
        session_id=session_id,
        include_trace=request.debug,
    )


<<<<<<< Updated upstream
=======
@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chroma_client=Depends(get_chat_chroma_client),
    embedding_service: BaseEmbeddingService = Depends(get_chat_embedding_service),
    llm_answer_composer: LLMAnswerComposer = Depends(get_chat_llm_answer_composer),
) -> StreamingResponse:
    def event_generator():
        session_id = request.session_id or uuid4().hex
        yield _sse_event("session", {"session_id": session_id})

        try:
            chat_service = ChatService(
                db=db,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
                llm_answer_composer=llm_answer_composer,
            )
            chat_response = chat_service.handle_message(
                query=request.query,
                session_id=request.session_id,
            )
            chat_response = _save_conversation_turn(
                db=db,
                session_id=session_id,
                user_query=request.query,
                chat_response=chat_response,
            )

            if request.debug:
                for trace_step in chat_response.trace:
                    yield _sse_event("trace", trace_step)

            response_schema = _to_response_schema(
                chat_response,
                session_id=session_id,
                include_trace=request.debug,
            )
            yield _sse_event("result", response_schema.model_dump())
            yield _sse_event("done", {"status": "ok"})
        except Exception:
            yield _sse_event("error", {"message": "chat stream failed"})
            yield _sse_event("done", {"status": "error"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


>>>>>>> Stashed changes
def _save_conversation_turn(
    db: Session,
    session_id: str,
    user_query: str,
    chat_response: ChatResponse,
) -> ChatResponse:
    try:
        turn = ConversationMemoryService(db).save_turn(
            session_id=session_id,
            user_query=user_query,
            chat_response=chat_response,
        )
        trace_step = {
            "step": "conversation_memory",
            "status": "saved",
            "session_id": session_id,
            "turn_index": turn.turn_index,
        }
    except Exception:
        trace_step = {
            "step": "conversation_memory",
            "status": "failed",
            "session_id": session_id,
        }

    return ChatResponse(
        answer=chat_response.answer,
        product_cards=chat_response.product_cards,
        citations=chat_response.citations,
        trace=[*chat_response.trace, trace_step],
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


def _sse_event(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )
