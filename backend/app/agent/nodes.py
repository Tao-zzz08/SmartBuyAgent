from __future__ import annotations

from typing import Any

from app.agent.context import AgentRuntimeContext
from app.agent.state import AgentState
from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.chat.llm_answer_composer import SAFE_LLM_FALLBACK_ANSWER
from app.chat.product_comparison import CompareContext, ProductComparisonService
from app.chat.query_understanding import QueryUnderstandingService
from app.chat.response_composer import ChatResponse, ResponseComposer
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
    ProductSearchFilters,
)


def append_trace(
    state: AgentState,
    node_name: str,
    status: str = "stub",
    **extra: Any,
) -> AgentState:
    state.trace.append(
        {
            "step": "agent_node",
            "node": node_name,
            "status": status,
            **extra,
        }
    )
    return state


def load_context_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "load_context")

    if not state.session_id or context.conversation_memory_service is None:
        return append_trace(state, "load_context", status="skipped", turn_count=0)

    try:
        state.recent_turns = context.conversation_memory_service.get_recent_turns(
            session_id=state.session_id,
            limit=3,
        )
        return append_trace(
            state,
            "load_context",
            status="loaded",
            turn_count=len(state.recent_turns),
        )
    except Exception as exc:
        state.errors.append(f"load_context: {exc}")
        return append_trace(state, "load_context", status="failed", turn_count=0)


def follow_up_rewrite_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "follow_up_rewrite")

    if not state.session_id or not state.recent_turns:
        _append_step(
            state,
            "follow_up_rewrite",
            status="skipped",
            original_query=state.original_query,
            rewritten_query=state.effective_query,
        )
        return state

    try:
        rewriter = context.followup_rewriter or FollowUpQueryRewriter()
        result = rewriter.rewrite(
            query=state.original_query,
            recent_turns=state.recent_turns,
        )
        state.rewrite_result = result
        if result.is_follow_up:
            state.effective_query = result.rewritten_query
            state.compare_context = _build_compare_context(result)

        _append_follow_up_trace(
            state,
            status="rewritten" if result.is_follow_up else "not_follow_up",
            result=result,
        )
        return state
    except Exception as exc:
        state.errors.append(f"follow_up_rewrite: {exc}")
        _append_step(
            state,
            "follow_up_rewrite",
            status="failed",
            original_query=state.original_query,
            rewritten_query=state.effective_query,
        )
        return state


def intent_router_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "intent_router")

    try:
        service = context.query_understanding_service or QueryUnderstandingService()
        result = service.understand(state.effective_query)
        state.query_result = result
        state.intent = result.intent
        state.category_id = result.category_id
        state.category_path = result.category_path
        state.budget_min = result.budget_min
        state.budget_max = result.budget_max
        state.preferences = list(result.preferences)
        state.need_clarification = result.need_clarification
        state.clarification_question = result.clarification_question
        _append_step(
            state,
            "query_understanding",
            intent=result.intent,
            category_id=result.category_id,
            category_path=result.category_path,
            budget_min=result.budget_min,
            budget_max=result.budget_max,
            preferences=result.preferences,
            need_clarification=result.need_clarification,
        )
        return state
    except Exception as exc:
        state.errors.append(f"intent_router: {exc}")
        _append_step(state, "query_understanding", status="failed")
        return state


def shopping_guide_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "shopping_guide")

    product_service = _product_retrieval_service(context)
    knowledge_service = _knowledge_retrieval_service(context)
    if product_service is None or knowledge_service is None:
        state.errors.append("shopping_guide: retrieval services unavailable")
        return append_trace(state, "shopping_guide", status="skipped")

    try:
        product_filters = ProductSearchFilters(
            category_id=state.category_id,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            stock_only=True,
            preferences=state.preferences,
        )
        state.product_candidates = product_service.search_products(
            query=state.effective_query,
            filters=product_filters,
            top_k=3,
        )
        _append_step(
            state,
            "product_retrieval",
            category_id=state.category_id,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            candidate_count=len(state.product_candidates),
        )

        state.citations = knowledge_service.search_knowledge(
            query=state.effective_query,
            category_id=state.category_id,
            top_k=3,
        )
        _append_step(
            state,
            "knowledge_retrieval",
            category_id=state.category_id,
            citation_count=len(state.citations),
        )
        return state
    except Exception as exc:
        state.errors.append(f"shopping_guide: {exc}")
        return append_trace(state, "shopping_guide", status="failed")


def product_knowledge_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "product_knowledge")

    knowledge_service = _knowledge_retrieval_service(context)
    if knowledge_service is None:
        state.errors.append("product_knowledge: knowledge service unavailable")
        return append_trace(state, "product_knowledge", status="skipped")

    try:
        state.citations = knowledge_service.search_knowledge(
            query=state.effective_query,
            category_id=state.category_id,
            top_k=5,
        )
        _append_step(
            state,
            "knowledge_retrieval",
            category_id=state.category_id,
            citation_count=len(state.citations),
        )
        return state
    except Exception as exc:
        state.errors.append(f"product_knowledge: {exc}")
        return append_trace(state, "product_knowledge", status="failed")


def compare_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "compare")

    product_ids, source, focus_preferences = _compare_context_parts(
        state.compare_context
    )
    if not product_ids:
        _append_step(
            state,
            "product_comparison",
            status="skipped",
            requested_product_ids=[],
            returned_product_ids=[],
            missing_product_ids=[],
        )
        return state

    service = _product_comparison_service(context)
    if service is None:
        state.errors.append("compare: product comparison service unavailable")
        _append_step(
            state,
            "product_comparison",
            status="failed",
            requested_product_ids=product_ids,
            returned_product_ids=[],
            missing_product_ids=[],
        )
        return state

    try:
        result = service.compare(
            query=state.effective_query,
            product_ids=product_ids,
            focus_preferences=focus_preferences,
            source=source,
        )
        state.product_candidates = result.product_candidates
        state.answer = result.answer
        state.preferences = result.focus_preferences
        state.trace.append(result.trace)
        return state
    except Exception as exc:
        state.errors.append(f"compare: {exc}")
        _append_step(
            state,
            "product_comparison",
            status="failed",
            requested_product_ids=product_ids,
            returned_product_ids=[],
            missing_product_ids=[],
        )
        return state


def clarification_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "clarification")
    return _compose_simple_response(state, context, "clarification")


def chitchat_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "chitchat")
    return _compose_simple_response(state, context, "chitchat")


def response_compose_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "response_compose")

    if state.query_result is None:
        state.errors.append("response_compose: query_result missing")
        return append_trace(state, "response_compose", status="failed")

    response_composer = context.response_composer or ResponseComposer()
    response = response_composer.compose(
        state.query_result,
        product_candidates=state.product_candidates,
        citations=state.citations,
    )

    preserved_answer = state.answer
    if preserved_answer:
        response = ChatResponse(
            answer=preserved_answer,
            product_cards=response.product_cards,
            citations=response.citations,
            trace=response.trace,
        )
    else:
        response = _compose_with_optional_llm(state, context, response)

    _apply_response(state, response)
    return state


def save_trace_node(
    state: AgentState,
    context: AgentRuntimeContext | None = None,
) -> AgentState:
    if context is None:
        return append_trace(state, "save_trace")
    return append_trace(
        state,
        "save_trace",
        status="recorded",
        trace_count=len(state.trace),
    )


def _append_step(state: AgentState, step: str, **extra: Any) -> None:
    state.trace.append({"step": step, **extra})


def _append_follow_up_trace(state: AgentState, status: str, result: Any) -> None:
    context_used = getattr(result, "context_used", {}) or {}
    _append_step(
        state,
        "follow_up_rewrite",
        status=status,
        original_query=state.original_query,
        rewritten_query=getattr(result, "rewritten_query", state.effective_query),
        reason=getattr(result, "reason", None),
        source_turn_index=getattr(result, "source_turn_index", None),
        referenced_product_ids=context_used.get("referenced_product_ids", []),
        resolved_product_ids=context_used.get("resolved_product_ids", []),
        context_used=context_used,
    )


def _build_compare_context(result: Any) -> CompareContext | None:
    context_used = getattr(result, "context_used", {}) or {}
    resolved_product_ids = context_used.get("resolved_product_ids") or []
    referenced_product_ids = context_used.get("referenced_product_ids") or []
    focus_preferences = context_used.get("preferences") or []

    if resolved_product_ids:
        return CompareContext(
            product_ids=list(resolved_product_ids),
            source="resolved_product_ids",
            focus_preferences=list(focus_preferences),
        )

    if getattr(result, "reason", None) == "vague_product_reference" and referenced_product_ids:
        return CompareContext(
            product_ids=list(referenced_product_ids),
            source="referenced_product_ids",
            focus_preferences=list(focus_preferences),
        )

    return None


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
    )
    return context.knowledge_retrieval_service


def _product_comparison_service(
    context: AgentRuntimeContext,
) -> ProductComparisonService | None:
    if context.product_comparison_service is not None:
        return context.product_comparison_service
    if context.db is None:
        return None
    context.product_comparison_service = ProductComparisonService(db=context.db)
    return context.product_comparison_service


def _compare_context_parts(compare_context: Any) -> tuple[list[str], str, list[str]]:
    if isinstance(compare_context, CompareContext):
        return (
            list(compare_context.product_ids),
            compare_context.source,
            list(compare_context.focus_preferences),
        )
    if isinstance(compare_context, dict):
        return (
            list(compare_context.get("product_ids") or []),
            str(compare_context.get("source") or "unknown"),
            list(compare_context.get("focus_preferences") or []),
        )
    return [], "unknown", []


def _compose_simple_response(
    state: AgentState,
    context: AgentRuntimeContext,
    node_name: str,
) -> AgentState:
    if state.query_result is None:
        state.errors.append(f"{node_name}: query_result missing")
        return append_trace(state, node_name, status="failed")

    response_composer = context.response_composer or ResponseComposer()
    response = response_composer.compose(state.query_result)
    _apply_response(state, response)
    return state


def _apply_response(state: AgentState, response: ChatResponse) -> None:
    state.answer = response.answer
    state.product_cards = response.product_cards
    state.citations = response.citations
    state.trace.extend(response.trace)


def _compose_with_optional_llm(
    state: AgentState,
    context: AgentRuntimeContext,
    base_response: ChatResponse,
) -> ChatResponse:
    if (
        context.llm_answer_composer is None
        or state.intent not in {"shopping_guide", "product_knowledge"}
    ):
        return base_response

    try:
        llm_answer = context.llm_answer_composer.compose(
            query=state.effective_query,
            query_result=state.query_result,
            product_candidates=state.product_candidates,
            citations=state.citations,
        )
    except Exception:
        return _append_response_trace(
            base_response,
            {"step": "llm_answer", "enabled": True, "status": "fallback"},
        )

    normalized_answer = llm_answer.strip()
    if not normalized_answer or normalized_answer == SAFE_LLM_FALLBACK_ANSWER:
        return _append_response_trace(
            base_response,
            {"step": "llm_answer", "enabled": True, "status": "fallback"},
        )

    return ChatResponse(
        answer=normalized_answer,
        product_cards=base_response.product_cards,
        citations=base_response.citations,
        trace=[
            *base_response.trace,
            {"step": "llm_answer", "enabled": True, "status": "success"},
        ],
    )


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
