import json
import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.context import AgentRuntimeContext
from app.agent.stream_runner import AgentStreamRunner
from app.cache.cache_service import CacheService, get_cache_service
from app.cache.rate_limit import RateLimitExceeded, check_rate_limit
from app.chat.chat_service import ChatService
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.chat.llm_answer_composer import LLMAnswerComposer
from app.chat.product_comparison import ProductComparisonService
from app.chat.query_understanding import QueryUnderstandingService
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.core.config import settings
from app.core.db import get_db
from app.retrieval.chroma_indexer import get_chroma_client
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponseSchema,
    CitationResponse,
    ProductCardResponse,
)
from app.services.embedding import BaseEmbeddingService, get_embedding_service
from app.services.llm import BaseLLMService, get_llm_service
from app.services.answer_grounding_guard import AnswerGroundingGuard
from app.streaming.events import StreamEvent, sse_event


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


def get_chat_cache_service() -> CacheService:
    return get_cache_service()


@router.post("/chat", response_model=ChatResponseSchema)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chroma_client=Depends(get_chat_chroma_client),
    embedding_service: BaseEmbeddingService = Depends(get_chat_embedding_service),
    llm_answer_composer: LLMAnswerComposer = Depends(get_chat_llm_answer_composer),
    cache_service: CacheService = Depends(get_chat_cache_service),
) -> ChatResponseSchema:
    session_id = request.session_id or uuid4().hex
    _enforce_rate_limit(cache_service, session_id)
    chat_service = ChatService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        llm_answer_composer=llm_answer_composer,
        cache_service=cache_service,
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
        cache_service=cache_service,
    )
    return _to_response_schema(
        chat_response,
        session_id=session_id,
        include_trace=request.debug,
    )


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    chroma_client=Depends(get_chat_chroma_client),
    embedding_service: BaseEmbeddingService = Depends(get_chat_embedding_service),
    llm_answer_composer: LLMAnswerComposer = Depends(get_chat_llm_answer_composer),
    cache_service: CacheService = Depends(get_chat_cache_service),
) -> StreamingResponse:
    session_id = request.session_id or uuid4().hex
    request_id = uuid4().hex
    _enforce_rate_limit(cache_service, session_id)

    def event_generator():
        def emit(stream_event: StreamEvent):
            _cache_sse_event(
                cache_service,
                session_id,
                request_id,
                stream_event.event,
                stream_event.data,
            )
            return sse_event(stream_event.event, stream_event.data)

        session_event = StreamEvent(
            event="session",
            data={"session_id": session_id, "request_id": request_id},
        )
        yield emit(session_event)

        try:
            context = _stream_runtime_context(
                db=db,
                embedding_service=embedding_service,
                chroma_client=chroma_client,
                llm_answer_composer=llm_answer_composer,
                cache_service=cache_service,
            )
            runner = AgentStreamRunner(context)
            stream = runner.stream(
                request.query,
                request_id=request_id,
                session_id=request.session_id,
                event_session_id=session_id,
            )

            try:
                while True:
                    stream_event = next(stream)
                    yield emit(stream_event)
            except StopIteration as stop:
                state = stop.value

            chat_response = _chat_response_from_state(state)
            for stream_event in _save_conversation_turn_stream_events(
                db=db,
                session_id=session_id,
                user_query=request.query,
                chat_response=chat_response,
                cache_service=cache_service,
                request_id=request_id,
            ):
                yield emit(stream_event)
                if stream_event.event == "trace":
                    chat_response = ChatResponse(
                        answer=chat_response.answer,
                        product_cards=chat_response.product_cards,
                        citations=chat_response.citations,
                        trace=[*chat_response.trace, stream_event.data],
                    )

            if request.debug:
                # Trace steps are already emitted in realtime; this keeps the
                # final result schema compatible without replaying duplicates.
                pass

            response_schema = _to_response_schema(
                chat_response,
                session_id=session_id,
                include_trace=request.debug,
            )
            result_event = {
                **response_schema.model_dump(),
                "request_id": request_id,
            }
            done_event = {
                "request_id": request_id,
                "session_id": session_id,
                "status": getattr(state, "_stream_done_status", "ok"),
            }
            yield emit(StreamEvent(event="result", data=result_event))
            yield emit(StreamEvent(event="done", data=done_event))
        except Exception as exc:
            error_event = {
                "request_id": request_id,
                "session_id": session_id,
                "failed_node": "agent_stream",
                "error_type": type(exc).__name__,
                "message": "chat stream failed",
            }
            done_event = {
                "request_id": request_id,
                "session_id": session_id,
                "status": "error",
            }
            yield emit(StreamEvent(event="error", data=error_event))
            yield emit(StreamEvent(event="done", data=done_event))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _save_conversation_turn(
    db: Session,
    session_id: str,
    user_query: str,
    chat_response: ChatResponse,
    cache_service: CacheService | None = None,
) -> ChatResponse:
    try:
        turn = ConversationMemoryService(db, cache_service=cache_service).save_turn(
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


def _stream_runtime_context(
    db: Session,
    embedding_service: BaseEmbeddingService,
    chroma_client,
    llm_answer_composer: LLMAnswerComposer,
    cache_service: CacheService | None,
) -> AgentRuntimeContext:
    return AgentRuntimeContext(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
        query_understanding_service=QueryUnderstandingService(),
        product_retrieval_service=ProductRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
            cache_service=cache_service,
        ),
        knowledge_retrieval_service=KnowledgeRetrievalService(
            db=db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
            cache_service=cache_service,
        ),
        response_composer=ResponseComposer(),
        llm_answer_composer=llm_answer_composer,
        conversation_memory_service=ConversationMemoryService(
            db,
            cache_service=cache_service,
        ),
        followup_rewriter=FollowUpQueryRewriter(),
        product_comparison_service=ProductComparisonService(db),
        answer_grounding_guard=AnswerGroundingGuard(),
        cache_service=cache_service,
    )


def _chat_response_from_state(state) -> ChatResponse:
    return ChatResponse(
        answer=state.answer
        or "当前导购服务暂时没有生成回答，请稍后再试。",
        product_cards=list(state.product_cards),
        citations=list(state.citations),
        trace=list(state.trace),
    )


def _save_conversation_turn_stream_events(
    db: Session,
    session_id: str,
    user_query: str,
    chat_response: ChatResponse,
    cache_service: CacheService | None,
    request_id: str,
):
    started = time.perf_counter()
    yield StreamEvent(
        event="node_start",
        data={
            "request_id": request_id,
            "session_id": session_id,
            "node": "conversation_memory",
            "label": "会话保存",
            "status": "running",
        },
    )

    saved_response = _save_conversation_turn(
        db=db,
        session_id=session_id,
        user_query=user_query,
        chat_response=chat_response,
        cache_service=cache_service,
    )
    trace_step = saved_response.trace[-1]
    duration_ms = max(0, int((time.perf_counter() - started) * 1000))
    yield StreamEvent(
        event="trace",
        data={
            "request_id": request_id,
            "session_id": session_id,
            **trace_step,
        },
    )
    if trace_step.get("status") == "failed":
        yield StreamEvent(
            event="error",
            data={
                "request_id": request_id,
                "session_id": session_id,
                "failed_node": "conversation_memory",
                "error_type": "MemorySaveFailed",
                "message": "conversation memory save failed",
                "duration_ms": duration_ms,
            },
        )
    yield StreamEvent(
        event="node_end",
        data={
            "request_id": request_id,
            "session_id": session_id,
            "node": "conversation_memory",
            "label": "会话保存",
            "status": trace_step.get("status", "success"),
            "duration_ms": duration_ms,
            "summary": {
                "session_id": session_id,
                "turn_index": trace_step.get("turn_index"),
            },
        },
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


def _enforce_rate_limit(cache_service: CacheService, session_id: str) -> None:
    try:
        check_rate_limit(
            cache_service,
            key=f"smartbuy:rate:{session_id}",
            limit=settings.RATE_LIMIT_MAX_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests for this session. Please try again later.",
        )


def _cache_sse_event(
    cache_service: CacheService,
    session_id: str,
    request_id: str,
    event: str,
    data: dict,
) -> None:
    key = f"smartbuy:sse:{session_id}:{request_id}:trace"
    try:
        cached = cache_service.get_json(key)
        events = cached if isinstance(cached, list) else []
        events.append({"event": event, "data": data})
        cache_service.set_json(
            key,
            events,
            ttl_seconds=settings.SSE_TRACE_TTL_SECONDS,
        )
    except Exception:
        return
