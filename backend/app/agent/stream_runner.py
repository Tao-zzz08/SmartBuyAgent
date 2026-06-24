from __future__ import annotations

from collections.abc import Generator
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import queue
import threading
import time
from typing import Any, Callable

from app.agent.context import AgentRuntimeContext
from app.agent.nodes import (
    append_trace,
    chitchat_node,
    clarification_node,
    compare_node,
    follow_up_rewrite_node,
    intent_router_node,
    load_context_node,
    save_trace_node,
    _search_knowledge_structured,
    _structured_filters_from_state,
)
from app.agent.state import AgentState, create_initial_agent_state
from app.agent.workflow import _route_name_for_state
from app.chat.llm_answer_composer import SAFE_LLM_FALLBACK_ANSWER
from app.chat.product_comparison import CompareContext
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.services.answer_grounding_guard import (
    AnswerGroundingContext,
    AnswerGroundingGuard,
    GroundingGuardResult,
)
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
    ProductSearchFilters,
)
from app.streaming.events import StreamEvent
from app.streaming.event_emitter import StreamEventEmitter
from app.streaming.safety import (
    STREAM_GUARDED_FALLBACK_ANSWER,
    StreamSafetyViolation,
)


NodeFunc = Callable[[AgentState, AgentRuntimeContext], AgentState]


NODE_LABELS = {
    "load_context": "Load context",
    "follow_up_rewrite": "Follow-up rewrite",
    "intent_router": "Intent routing",
    "route_by_intent": "Route by intent",
    "product_retrieval": "Product retrieval",
    "knowledge_retrieval": "Knowledge retrieval",
    "product_comparison": "Product comparison",
    "clarification": "Clarification",
    "chitchat": "Chitchat",
    "response_compose": "Response compose",
    "save_trace": "Save trace",
}


class AgentStreamRunner:
    """Run agent nodes synchronously while yielding realtime stream events."""

    def __init__(self, context: AgentRuntimeContext) -> None:
        self.context = context

    def stream(
        self,
        query: str,
        *,
        request_id: str,
        session_id: str | None = None,
        event_session_id: str | None = None,
        compare_context: CompareContext | None = None,
    ) -> Generator[StreamEvent, None, AgentState]:
        state = create_initial_agent_state(query=query, session_id=session_id)
        state.compare_context = compare_context
        emitter = StreamEventEmitter(
            request_id=request_id,
            session_id=event_session_id or session_id,
        )
        token_answer_emitted = False

        try:
            yield from self._run_node(state, emitter, "load_context", load_context_node)
            yield from self._run_node(
                state,
                emitter,
                "follow_up_rewrite",
                follow_up_rewrite_node,
            )
            yield from self._run_node(state, emitter, "intent_router", intent_router_node)
            yield from self._run_route_node(state, emitter)

            route = _route_name_for_state(state)
            if route == "compare":
                yield from self._run_node(
                    state,
                    emitter,
                    "product_comparison",
                    compare_node,
                )
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True
            elif route == "clarification":
                yield from self._run_node(
                    state,
                    emitter,
                    "clarification",
                    clarification_node,
                )
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True
            elif route == "shopping_guide":
                yield from self._run_product_retrieval_node(state, emitter)
                yield from self._run_knowledge_retrieval_node(state, emitter, top_k=3)
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True
            elif route == "product_knowledge":
                yield from self._run_knowledge_retrieval_node(state, emitter, top_k=5)
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True
            elif route == "chitchat":
                yield from self._run_node(
                    state,
                    emitter,
                    "chitchat",
                    chitchat_node,
                )
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True
            else:
                yield from self._run_response_node(state, emitter)
                token_answer_emitted = True

            if not token_answer_emitted and state.answer:
                yield from self._emit_answer_tokens(emitter, state.answer)

            yield from self._run_node(state, emitter, "save_trace", save_trace_node)
            return state
        except Exception as exc:
            state.errors.append(f"agent_stream: {exc}")
            state.trace.append(
                {
                    "step": "agent_workflow",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise

    def _run_route_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, None]:
        def route_node(inner_state: AgentState, _context: AgentRuntimeContext) -> AgentState:
            append_trace(
                inner_state,
                "route_by_intent",
                status="routed",
                route=_route_name_for_state(inner_state),
            )
            return inner_state

        yield from self._run_node(state, emitter, "route_by_intent", route_node)

    def _run_product_retrieval_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, None]:
        def product_retrieval_node(
            inner_state: AgentState,
            context: AgentRuntimeContext,
        ) -> AgentState:
            product_service = _product_retrieval_service(context)
            if product_service is None:
                inner_state.errors.append("product_retrieval: service unavailable")
                return append_trace(inner_state, "product_retrieval", status="skipped")

            product_filters = ProductSearchFilters(
                category_id=inner_state.category_id,
                budget_min=inner_state.budget_min,
                budget_max=inner_state.budget_max,
                stock_only=True,
                brand_exclude=inner_state.negative_preferences,
                preferences=inner_state.preferences,
            )
            inner_state.product_candidates = product_service.search_products(
                query=inner_state.effective_query,
                filters=product_filters,
                top_k=3,
            )
            inner_state.trace.append(
                {
                    "step": "product_retrieval",
                    "query": inner_state.effective_query,
                    "category_id": inner_state.category_id,
                    "category": inner_state.category,
                    "budget_min": inner_state.budget_min,
                    "budget_max": inner_state.budget_max,
                    "preferences": inner_state.preferences,
                    "negative_preferences": inner_state.negative_preferences,
                    "structured_filters": _structured_filters_from_state(inner_state),
                    "candidate_count": len(inner_state.product_candidates),
                    "filtered_count": getattr(
                        product_service,
                        "last_filtered_count",
                        len(inner_state.product_candidates),
                    ),
                    "negative_filtered_count": getattr(
                        product_service,
                        "last_negative_filtered_count",
                        0,
                    ),
                    "negative_filter_fallback": getattr(
                        product_service,
                        "last_negative_filter_fallback",
                        False,
                    ),
                    "product_ids": [
                        candidate.product_id
                        for candidate in inner_state.product_candidates
                    ],
                    "cache_status": getattr(product_service, "last_cache_status", None),
                }
            )
            return inner_state

        yield from self._run_node(
            state,
            emitter,
            "product_retrieval",
            product_retrieval_node,
        )

    def _run_knowledge_retrieval_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
        *,
        top_k: int,
    ) -> Generator[StreamEvent, None, None]:
        def knowledge_retrieval_node(
            inner_state: AgentState,
            context: AgentRuntimeContext,
        ) -> AgentState:
            knowledge_service = _knowledge_retrieval_service(context)
            if knowledge_service is None:
                inner_state.errors.append("knowledge_retrieval: service unavailable")
                return append_trace(inner_state, "knowledge_retrieval", status="skipped")

            inner_state.citations = _search_knowledge_structured(
                knowledge_service,
                query=inner_state.effective_query,
                category_id=inner_state.category_id,
                top_k=top_k,
                preferences=inner_state.preferences,
                negative_preferences=inner_state.negative_preferences,
            )
            inner_state.trace.append(
                {
                    "step": "knowledge_retrieval",
                    "query": getattr(
                        knowledge_service,
                        "last_query",
                        inner_state.effective_query,
                    ),
                    "category_id": inner_state.category_id,
                    "category": inner_state.category,
                    "preferences": inner_state.preferences,
                    "negative_preferences": inner_state.negative_preferences,
                    "citation_count": len(inner_state.citations),
                    "chunk_ids": [
                        citation.chunk_id for citation in inner_state.citations
                    ],
                    "cache_status": getattr(knowledge_service, "last_cache_status", None),
                }
            )
            return inner_state

        yield from self._run_node(
            state,
            emitter,
            "knowledge_retrieval",
            knowledge_retrieval_node,
        )

    def _run_response_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, None]:
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        trace_start = len(state.trace)
        token_answer_emitted = False

        emitter.emit(
            "node_start",
            {
                "node": "response_compose",
                "label": NODE_LABELS["response_compose"],
                "started_at": started_at,
                "status": "running",
            },
        )
        yield from emitter.drain()

        try:
            response, token_answer_emitted = yield from self._compose_response(
                state,
                emitter,
            )
            _apply_response(state, response)
            append_trace(state, "response_compose", status="composed")
        except StreamSafetyViolation as exc:
            duration_ms = _duration_ms(start_time)
            setattr(state, "_stream_done_status", "guarded")
            guarded_response = _guarded_response(state, self.context, exc)
            _apply_response(state, guarded_response)
            guard_trace = {
                "step": "stream_guard",
                "node": "response_compose",
                "status": "blocked",
                "reason": exc.reason,
                "matched_phrase": exc.matched_phrase,
                "severity": exc.severity,
            }
            state.trace.append(guard_trace)
            append_trace(state, "response_compose", status="failed")
            emitter.emit(
                "stream_guard",
                {
                    "node": "response_compose",
                    "status": "blocked",
                    "reason": exc.reason,
                    "matched_phrase": exc.matched_phrase,
                    "severity": exc.severity,
                },
            )
            emitter.emit("trace", guard_trace)
            emitter.emit(
                "error",
                {
                    "failed_node": "response_compose",
                    "error_type": "StreamSafetyViolation",
                    "message": "Streaming output blocked by safety guard",
                    "duration_ms": duration_ms,
                },
            )
            emitter.emit(
                "node_end",
                {
                    "node": "response_compose",
                    "label": NODE_LABELS["response_compose"],
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "summary": {
                        "guarded": True,
                        "reason": exc.reason,
                        "matched_phrase": exc.matched_phrase,
                    },
                },
            )
            yield from emitter.drain()
            return
        except Exception as exc:
            duration_ms = _duration_ms(start_time)
            emitter.emit(
                "error",
                {
                    "failed_node": "response_compose",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "duration_ms": duration_ms,
                },
            )
            emitter.emit(
                "node_end",
                {
                    "node": "response_compose",
                    "label": NODE_LABELS["response_compose"],
                    "status": "failed",
                    "duration_ms": duration_ms,
                },
            )
            yield from emitter.drain()
            raise

        new_trace_steps = state.trace[trace_start:]
        for trace_step in new_trace_steps:
            emitter.emit("trace", trace_step)
            yield from emitter.drain()

        if state.answer and not token_answer_emitted:
            yield from self._emit_answer_tokens(emitter, state.answer)

        duration_ms = _duration_ms(start_time)
        status = _node_status("response_compose", new_trace_steps)
        emitter.emit(
            "node_end",
            {
                "node": "response_compose",
                "label": NODE_LABELS["response_compose"],
                "status": status,
                "duration_ms": duration_ms,
                "summary": _node_summary("response_compose", state, new_trace_steps),
            },
        )
        yield from emitter.drain()

    def _compose_response(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, tuple[ChatResponse, bool]]:
        if state.query_result is None:
            state.errors.append("response_compose: query_result missing")
            return (
                ChatResponse(
                    answer="",
                    product_cards=[],
                    citations=[],
                    trace=[
                        {
                            "step": "agent_node",
                            "node": "response_compose",
                            "status": "failed",
                        }
                    ],
                ),
                False,
            )

        response_composer = self.context.response_composer or ResponseComposer()
        base_response = response_composer.compose(
            state.query_result,
            product_candidates=state.product_candidates,
            citations=state.citations,
        )
        _emit_response_payload_events(emitter, base_response)

        if state.answer:
            guarded_response, emitted = _guard_stream_response(
                state,
                self.context,
                emitter,
                ChatResponse(
                    answer=state.answer,
                    product_cards=base_response.product_cards,
                    citations=base_response.citations,
                    trace=base_response.trace,
                ),
            )
            yield from emitter.drain()
            return guarded_response, emitted

        if state.intent not in {"shopping_guide", "product_knowledge"}:
            guarded_response, emitted = _guard_stream_response(
                state,
                self.context,
                emitter,
                base_response,
            )
            yield from emitter.drain()
            return guarded_response, emitted

        if self.context.llm_answer_composer is None:
            guarded_response, emitted = _guard_stream_response(
                state,
                self.context,
                emitter,
                _append_response_trace(
                    base_response,
                    {"step": "llm_answer", "enabled": False, "status": "disabled"},
                ),
            )
            yield from emitter.drain()
            return guarded_response, emitted

        llm_answer, emitted_draft = yield from self._stream_llm_answer(
            state,
            emitter,
        )
        normalized_answer = llm_answer.strip()
        if not normalized_answer or normalized_answer == SAFE_LLM_FALLBACK_ANSWER:
            guarded_response, emitted = _guard_stream_response(
                state,
                self.context,
                emitter,
                _append_response_trace(
                    base_response,
                    {
                        "step": "llm_answer",
                        "enabled": True,
                        "status": "fallback",
                        "draft_streamed": emitted_draft,
                    },
                ),
            )
            yield from emitter.drain()
            return guarded_response, emitted

        guarded_response, emitted = _guard_stream_response(
            state,
            self.context,
            emitter,
            ChatResponse(
                answer=normalized_answer,
                product_cards=base_response.product_cards,
                citations=base_response.citations,
                trace=[
                    *base_response.trace,
                    {
                        "step": "llm_answer",
                        "enabled": True,
                        "status": "success",
                        "draft_streamed": emitted_draft,
                    },
                ],
            ),
        )
        yield from emitter.drain()
        return guarded_response, emitted

    def _stream_llm_answer(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, tuple[str, bool]]:
        if self.context.llm_answer_composer is None or state.query_result is None:
            return "", False

        token_queue: queue.Queue[str | None] = queue.Queue()
        result: dict[str, Any] = {}

        def on_token(delta: str) -> None:
            token_queue.put(delta)

        def run_compose() -> None:
            try:
                result["answer"] = self.context.llm_answer_composer.stream_compose(
                    query=state.effective_query,
                    query_result=state.query_result,
                    product_candidates=state.product_candidates,
                    citations=state.citations,
                    on_token=on_token,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                result["error"] = exc
            finally:
                token_queue.put(None)

        thread = threading.Thread(target=run_compose, daemon=True)
        thread.start()

        emitted_tokens = False
        while True:
            delta = token_queue.get()
            if delta is None:
                break
            emitted_tokens = True
            emitter.emit(
                "answer_draft_delta",
                {
                    "node": "response_compose",
                    "delta": delta,
                },
            )
            yield from emitter.drain()

        thread.join()
        error = result.get("error")
        if isinstance(error, StreamSafetyViolation):
            raise error
        if isinstance(error, Exception):
            return SAFE_LLM_FALLBACK_ANSWER, emitted_tokens
        answer = result.get("answer")
        return str(answer or ""), emitted_tokens

    def _run_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
        node_name: str,
        node_func: NodeFunc,
        *,
        emit_tokens: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        trace_start = len(state.trace)
        answer_before = state.answer or ""

        emitter.emit(
            "node_start",
            {
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
                "started_at": started_at,
                "status": "running",
            },
        )
        yield from emitter.drain()

        try:
            node_func(state, self.context)
        except Exception as exc:
            duration_ms = _duration_ms(start_time)
            emitter.emit(
                "error",
                {
                    "failed_node": node_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "duration_ms": duration_ms,
                },
            )
            emitter.emit(
                "node_end",
                {
                    "node": node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                    "status": "failed",
                    "duration_ms": duration_ms,
                },
            )
            yield from emitter.drain()
            raise

        new_trace_steps = state.trace[trace_start:]
        for trace_step in new_trace_steps:
            for retrieval_event in _retrieval_events_from_trace(
                trace_step,
                query=state.effective_query,
                category_id=state.category_id,
                budget_min=state.budget_min,
                budget_max=state.budget_max,
            ):
                emitter.emit("retrieval", retrieval_event)
            emitter.emit("trace", trace_step)
            yield from emitter.drain()

        if emit_tokens and state.answer:
            answer_after = state.answer
            token_source = answer_after if answer_after != answer_before else answer_after
            yield from self._emit_answer_tokens(emitter, token_source)

        duration_ms = _duration_ms(start_time)
        status = _node_status(node_name, new_trace_steps)
        if status == "failed":
            emitter.emit(
                "error",
                {
                    "failed_node": node_name,
                    "error_type": "NodeFailed",
                    "message": _failed_node_message(state, node_name),
                    "duration_ms": duration_ms,
                },
            )
            yield from emitter.drain()
        emitter.emit(
            "node_end",
            {
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
                "status": status,
                "duration_ms": duration_ms,
                "summary": _node_summary(node_name, state, new_trace_steps),
            },
        )
        yield from emitter.drain()

    def _emit_answer_tokens(
        self,
        emitter: StreamEventEmitter,
        answer: str,
    ) -> Generator[StreamEvent, None, None]:
        for delta in _chunk_text(answer):
            emitter.emit(
                "token",
                {
                    "node": "response_compose",
                    "delta": delta,
                },
            )
            yield from emitter.drain()


def _duration_ms(start_time: float) -> int:
    return max(0, int((time.perf_counter() - start_time) * 1000))


def _node_status(node_name: str, trace_steps: list[dict[str, Any]]) -> str:
    for step in reversed(trace_steps):
        if step.get("status") in {"failed", "error"}:
            return "failed"
        if step.get("step") == "agent_node" and step.get("node") == node_name:
            status = step.get("status")
            return str(status or "success")
        if step.get("step") == _primary_step_for_node(node_name):
            status = step.get("status")
            return str(status or "success")
    return "success"


def _failed_node_message(state: AgentState, node_name: str) -> str:
    for error in reversed(state.errors):
        if error.startswith(f"{node_name}:") or node_name in error:
            return error
    return f"{node_name} failed"


def _primary_step_for_node(node_name: str) -> str:
    return {
        "intent_router": "query_understanding",
        "product_retrieval": "product_retrieval",
        "knowledge_retrieval": "knowledge_retrieval",
        "product_comparison": "product_comparison",
        "response_compose": "response_composer",
    }.get(node_name, node_name)


def _node_summary(
    node_name: str,
    state: AgentState,
    trace_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if node_name == "product_retrieval":
        return {
            "returned_products": len(state.product_candidates),
            "candidate_product_ids": [
                candidate.product_id for candidate in state.product_candidates
            ],
            "cache_status": _last_trace_value(trace_steps, "product_retrieval", "cache_status"),
        }
    if node_name == "knowledge_retrieval":
        return {
            "returned_chunks": len(state.citations),
            "chunk_ids": [citation.chunk_id for citation in state.citations],
            "cache_status": _last_trace_value(trace_steps, "knowledge_retrieval", "cache_status"),
        }
    if node_name == "product_comparison":
        comparison = next(
            (
                step
                for step in trace_steps
                if step.get("step") == "product_comparison"
            ),
            {},
        )
        return {
            "returned_products": len(state.product_candidates),
            "requested_product_ids": comparison.get("requested_product_ids", []),
            "returned_product_ids": comparison.get("returned_product_ids", []),
            "missing_product_ids": comparison.get("missing_product_ids", []),
        }
    if node_name == "response_compose":
        return {
            "answer_length": len(state.answer or ""),
            "product_cards": len(state.product_cards),
            "citations": len(state.citations),
        }
    if node_name == "intent_router":
        return {
            "intent": state.intent,
            "category": state.category,
            "category_id": state.category_id,
            "budget_max": state.budget_max,
            "preferences": state.preferences,
            "negative_preferences": state.negative_preferences,
        }
    return {}


def _last_trace_value(
    trace_steps: list[dict[str, Any]],
    step_name: str,
    key: str,
) -> Any:
    for step in reversed(trace_steps):
        if step.get("step") == step_name:
            return step.get(key)
    return None


def _retrieval_events_from_trace(
    trace_step: dict[str, Any],
    *,
    query: str,
    category_id: str | None,
    budget_min: int | None,
    budget_max: int | None,
) -> list[dict[str, Any]]:
    step = trace_step.get("step")
    if step == "product_retrieval":
        return [
            {
                "type": "product",
                "query": trace_step.get("query", query),
                "category": trace_step.get("category"),
                "category_id": trace_step.get("category_id") or category_id,
                "budget_min": trace_step.get("budget_min", budget_min),
                "budget_max": trace_step.get("budget_max", budget_max),
                "preferences": trace_step.get("preferences", []),
                "negative_preferences": trace_step.get("negative_preferences", []),
                "structured_filters": trace_step.get("structured_filters", {}),
                "returned_products": trace_step.get("candidate_count", 0),
                "filtered_count": trace_step.get("filtered_count"),
                "negative_filtered_count": trace_step.get("negative_filtered_count", 0),
                "negative_filter_fallback": trace_step.get(
                    "negative_filter_fallback",
                    False,
                ),
                "candidate_product_ids": trace_step.get("product_ids", []),
                "cache_status": trace_step.get("cache_status"),
                "status": trace_step.get("status", "success"),
            }
        ]
    if step == "knowledge_retrieval":
        return [
            {
                "type": "knowledge",
                "query": trace_step.get("query", query),
                "category": trace_step.get("category"),
                "category_id": trace_step.get("category_id") or category_id,
                "preferences": trace_step.get("preferences", []),
                "negative_preferences": trace_step.get("negative_preferences", []),
                "returned_chunks": trace_step.get("citation_count", 0),
                "chunk_ids": trace_step.get("chunk_ids", []),
                "cache_status": trace_step.get("cache_status"),
                "status": trace_step.get("status", "success"),
            }
        ]
    if step == "product_comparison":
        return [
            {
                "type": "comparison",
                "query": query,
                "requested_product_ids": trace_step.get("requested_product_ids", []),
                "returned_product_ids": trace_step.get("returned_product_ids", []),
                "missing_product_ids": trace_step.get("missing_product_ids", []),
                "focus_preferences": trace_step.get("focus_preferences", []),
                "status": trace_step.get("status", "success"),
            }
        ]
    return []


def _product_retrieval_service(
    context: AgentRuntimeContext,
) -> ProductRetrievalService | None:
    if context.product_retrieval_service is not None:
        return context.product_retrieval_service
    if context.db is None or context.embedding_service is None:
        return None
    context.product_retrieval_service = ProductRetrievalService(
        db=context.db,
        embedding_service=context.embedding_service,
        chroma_client=context.chroma_client,
        cache_service=context.cache_service,
    )
    return context.product_retrieval_service


def _knowledge_retrieval_service(
    context: AgentRuntimeContext,
) -> KnowledgeRetrievalService | None:
    if context.knowledge_retrieval_service is not None:
        return context.knowledge_retrieval_service
    if context.db is None or context.embedding_service is None:
        return None
    context.knowledge_retrieval_service = KnowledgeRetrievalService(
        db=context.db,
        embedding_service=context.embedding_service,
        chroma_client=context.chroma_client,
        cache_service=context.cache_service,
    )
    return context.knowledge_retrieval_service


def _apply_response(state: AgentState, response: ChatResponse) -> None:
    state.answer = response.answer
    state.product_cards = response.product_cards
    state.citations = response.citations
    state.trace.extend(response.trace)


def _emit_response_payload_events(
    emitter: StreamEventEmitter,
    response: ChatResponse,
) -> None:
    emitter.emit(
        "product_cards",
        {"product_cards": [_object_to_dict(card) for card in response.product_cards]},
    )
    emitter.emit(
        "citations",
        {"citations": [_object_to_dict(citation) for citation in response.citations]},
    )


def _guard_stream_response(
    state: AgentState,
    context: AgentRuntimeContext,
    emitter: StreamEventEmitter,
    response: ChatResponse,
) -> tuple[ChatResponse, bool]:
    guard = context.answer_grounding_guard or AnswerGroundingGuard()
    result = guard.check(_grounding_context_from_state(state, response))
    trace_step = _grounding_trace_step(result)
    emitter.emit("grounding_guard_result", _grounding_event_payload(result))

    if result.passed:
        guarded_response = _append_response_trace(response, trace_step)
    else:
        setattr(state, "_stream_done_status", "guarded")
        guarded_response = ChatResponse(
            answer=result.fallback_answer or response.answer,
            product_cards=response.product_cards,
            citations=response.citations,
            trace=[*response.trace, trace_step],
        )

    emitter.emit(
        "final_answer",
        {
            "answer": guarded_response.answer,
            "status": "passed" if result.passed else result.action,
        },
    )
    return guarded_response, True


def _grounding_context_from_state(
    state: AgentState,
    response: ChatResponse,
) -> AnswerGroundingContext:
    query_understanding = (
        state.query_result.to_trace_dict()
        if state.query_result is not None and hasattr(state.query_result, "to_trace_dict")
        else dict(state.query_understanding or {})
    )
    return AnswerGroundingContext(
        answer=response.answer,
        route=state.intent,
        query_understanding=query_understanding,
        product_cards=[_object_to_dict(card) for card in response.product_cards],
        citations=[_object_to_dict(citation) for citation in response.citations],
        comparison_result=_object_to_dict(state.compare_context)
        if state.compare_context is not None
        else None,
    )


def _grounding_trace_step(result: GroundingGuardResult) -> dict[str, Any]:
    return {
        "step": "answer_grounding_guard",
        "status": "passed" if result.passed else result.action,
        "action": result.action,
        "checks": dict(result.checks),
        "violations": [
            violation.model_dump(exclude_none=True)
            for violation in result.violations
        ],
    }


def _grounding_event_payload(result: GroundingGuardResult) -> dict[str, Any]:
    return {
        "status": "passed" if result.passed else result.action,
        "action": result.action,
        "passed": result.passed,
        "checks": dict(result.checks),
        "violations": [
            violation.model_dump(exclude_none=True)
            for violation in result.violations
        ],
    }


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


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


def _guarded_response(
    state: AgentState,
    context: AgentRuntimeContext,
    exc: StreamSafetyViolation,
) -> ChatResponse:
    if state.query_result is None:
        return ChatResponse(
            answer=STREAM_GUARDED_FALLBACK_ANSWER,
            product_cards=[],
            citations=[],
            trace=[
                {
                    "step": "llm_answer",
                    "enabled": True,
                    "status": "guarded",
                    "reason": exc.reason,
                }
            ],
        )

    response_composer = context.response_composer or ResponseComposer()
    base_response = response_composer.compose(
        state.query_result,
        product_candidates=state.product_candidates,
        citations=state.citations,
    )
    return ChatResponse(
        answer=STREAM_GUARDED_FALLBACK_ANSWER,
        product_cards=base_response.product_cards,
        citations=base_response.citations,
        trace=[
            *base_response.trace,
            {
                "step": "llm_answer",
                "enabled": True,
                "status": "guarded",
                "reason": exc.reason,
                "matched_phrase": exc.matched_phrase,
                "severity": exc.severity,
            },
        ],
    )


def _chunk_text(text: str, chunk_size: int = 18) -> list[str]:
    normalized = text or ""
    if not normalized:
        return []
    return [
        normalized[index : index + chunk_size]
        for index in range(0, len(normalized), chunk_size)
    ]
