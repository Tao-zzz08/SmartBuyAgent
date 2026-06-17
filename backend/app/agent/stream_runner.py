from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
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
    product_knowledge_node,
    response_compose_node,
    save_trace_node,
    shopping_guide_node,
)
from app.agent.state import AgentState, create_initial_agent_state
from app.agent.workflow import _route_name_for_state
from app.chat.product_comparison import CompareContext
from app.streaming.events import StreamEvent
from app.streaming.event_emitter import StreamEventEmitter


NodeFunc = Callable[[AgentState, AgentRuntimeContext], AgentState]


NODE_LABELS = {
    "load_context": "上下文读取",
    "follow_up_rewrite": "追问改写",
    "intent_router": "意图识别",
    "route_by_intent": "意图路由",
    "shopping_guide": "导购召回",
    "product_knowledge": "知识问答",
    "compare": "候选商品比较",
    "clarification": "澄清回复",
    "chitchat": "闲聊回复",
    "response_compose": "回答生成",
    "save_trace": "Trace 记录",
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
            for event in self._run_node(
                state,
                emitter,
                "load_context",
                load_context_node,
            ):
                yield event
            for event in self._run_node(
                state,
                emitter,
                "follow_up_rewrite",
                follow_up_rewrite_node,
            ):
                yield event
            for event in self._run_node(
                state,
                emitter,
                "intent_router",
                intent_router_node,
            ):
                yield event
            for event in self._run_route_node(state, emitter):
                yield event

            route = _route_name_for_state(state)
            if route == "compare":
                for event in self._run_node(state, emitter, "compare", compare_node):
                    yield event
                for event in self._run_response_node(state, emitter):
                    token_answer_emitted = True
                    yield event
            elif route == "clarification":
                for event in self._run_node(
                    state,
                    emitter,
                    "clarification",
                    clarification_node,
                    emit_tokens=True,
                ):
                    token_answer_emitted = True
                    yield event
            elif route == "shopping_guide":
                for event in self._run_node(
                    state,
                    emitter,
                    "shopping_guide",
                    shopping_guide_node,
                ):
                    yield event
                for event in self._run_response_node(state, emitter):
                    token_answer_emitted = True
                    yield event
            elif route == "product_knowledge":
                for event in self._run_node(
                    state,
                    emitter,
                    "product_knowledge",
                    product_knowledge_node,
                ):
                    yield event
                for event in self._run_response_node(state, emitter):
                    token_answer_emitted = True
                    yield event
            elif route == "chitchat":
                for event in self._run_node(
                    state,
                    emitter,
                    "chitchat",
                    chitchat_node,
                    emit_tokens=True,
                ):
                    token_answer_emitted = True
                    yield event
            else:
                for event in self._run_response_node(state, emitter):
                    token_answer_emitted = True
                    yield event

            if not token_answer_emitted and state.answer:
                for event in self._emit_answer_tokens(emitter, state.answer):
                    yield event

            for event in self._run_node(
                state,
                emitter,
                "save_trace",
                save_trace_node,
            ):
                yield event
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

    def _run_response_node(
        self,
        state: AgentState,
        emitter: StreamEventEmitter,
    ) -> Generator[StreamEvent, None, None]:
        yield from self._run_node(
            state,
            emitter,
            "response_compose",
            response_compose_node,
            emit_tokens=True,
        )

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

        yield from self._run_node(
            state,
            emitter,
            "route_by_intent",
            route_node,
        )

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
            for event in self._emit_answer_tokens(emitter, token_source):
                yield event

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
        "shopping_guide": "product_retrieval",
        "product_knowledge": "knowledge_retrieval",
        "compare": "product_comparison",
        "response_compose": "response_composer",
    }.get(node_name, node_name)


def _node_summary(
    node_name: str,
    state: AgentState,
    trace_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    if node_name == "shopping_guide":
        return {
            "returned_products": len(state.product_candidates),
            "returned_citations": len(state.citations),
            "product_ids": [
                candidate.product_id for candidate in state.product_candidates
            ],
        }
    if node_name == "product_knowledge":
        return {
            "returned_chunks": len(state.citations),
            "chunk_ids": [citation.chunk_id for citation in state.citations],
        }
    if node_name == "compare":
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
            "category_id": state.category_id,
            "budget_max": state.budget_max,
            "preferences": state.preferences,
        }
    return {}


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
                "query": query,
                "category_id": trace_step.get("category_id") or category_id,
                "budget_min": trace_step.get("budget_min", budget_min),
                "budget_max": trace_step.get("budget_max", budget_max),
                "returned_products": trace_step.get("candidate_count", 0),
                "candidate_product_ids": trace_step.get("product_ids", []),
                "cache_status": trace_step.get("cache_status"),
                "status": trace_step.get("status", "success"),
            }
        ]
    if step == "knowledge_retrieval":
        return [
            {
                "type": "knowledge",
                "query": query,
                "category_id": trace_step.get("category_id") or category_id,
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


def _chunk_text(text: str, chunk_size: int = 18) -> list[str]:
    normalized = text or ""
    if not normalized:
        return []
    return [
        normalized[index : index + chunk_size]
        for index in range(0, len(normalized), chunk_size)
    ]
