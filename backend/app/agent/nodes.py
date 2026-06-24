from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any

from app.agent.context import AgentRuntimeContext
from app.agent.state import AgentState
from app.chat.followup_rewriter import FollowUpQueryRewriter
from app.chat.llm_answer_composer import SAFE_LLM_FALLBACK_ANSWER
from app.chat.product_comparison import CompareContext, ProductComparisonService
from app.chat.query_understanding import (
    QueryUnderstandingResult,
    QueryUnderstandingService,
)
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
            cache_status=getattr(
                context.conversation_memory_service,
                "last_cache_status",
                None,
            ),
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

    if not state.session_id:
        _append_step(
            state,
            "follow_up_rewrite",
            status="skipped",
            original_query=state.original_query,
            rewritten_query=state.effective_query,
        )
        return state

    if not state.recent_turns:
        load_context_failed = any(
            error.startswith("load_context:") for error in state.errors
        )
        _append_step(
            state,
            "follow_up_rewrite",
            status="failed" if load_context_failed else "not_follow_up",
            original_query=state.original_query,
            rewritten_query=state.effective_query,
            reason="load_context_failed" if load_context_failed else "no_recent_turns",
            referenced_product_ids=[],
            resolved_product_ids=[],
            context_used={},
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
        try:
            result = service.understand(
                state.original_query,
                session_id=state.session_id,
                history=state.recent_turns,
                rewrite_result=state.rewrite_result,
            )
        except TypeError:
            result = service.understand(state.effective_query)
        state.effective_query = result.effective_query

        state.query_result = result
        state.query_understanding = result.to_trace_dict()
        state.intent = result.intent
        state.category = result.category
        state.category_id = result.category_id
        state.category_path = result.category_path
        state.budget_min = result.budget_min
        state.budget_max = result.budget_max
        state.preferences = list(result.preferences)
        state.negative_preferences = list(getattr(result, "negative_preferences", []))
        state.compare_product_ids = list(result.compare_product_ids)
        state.referenced_product_indices = list(result.referenced_product_indices)
        state.shopping_memory = getattr(result, "shopping_memory", None)
        state.need_clarification = result.need_clarification
        state.clarification_question = result.clarification_question
        _apply_structured_compare_context(state)
        _append_step(
            state,
            "query_understanding",
            **state.query_understanding,
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
            brand_exclude=state.negative_preferences,
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
            query=state.effective_query,
            category_id=state.category_id,
            category=state.category,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            preferences=state.preferences,
            negative_preferences=state.negative_preferences,
            structured_filters=_structured_filters_from_state(state),
            candidate_count=len(state.product_candidates),
            filtered_count=getattr(
                product_service,
                "last_filtered_count",
                len(state.product_candidates),
            ),
            negative_filtered_count=getattr(
                product_service,
                "last_negative_filtered_count",
                0,
            ),
            negative_filter_fallback=getattr(
                product_service,
                "last_negative_filter_fallback",
                False,
            ),
            product_ids=[
                candidate.product_id for candidate in state.product_candidates
            ],
            cache_status=getattr(product_service, "last_cache_status", None),
        )

        state.citations = _search_knowledge_structured(
            knowledge_service,
            query=state.effective_query,
            category_id=state.category_id,
            top_k=3,
            preferences=state.preferences,
            negative_preferences=state.negative_preferences,
        )
        _append_step(
            state,
            "knowledge_retrieval",
            query=getattr(knowledge_service, "last_query", state.effective_query),
            category_id=state.category_id,
            category=state.category,
            preferences=state.preferences,
            negative_preferences=state.negative_preferences,
            citation_count=len(state.citations),
            cache_status=getattr(knowledge_service, "last_cache_status", None),
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
        state.citations = _search_knowledge_structured(
            knowledge_service,
            query=state.effective_query,
            category_id=state.category_id,
            top_k=5,
            preferences=state.preferences,
            negative_preferences=state.negative_preferences,
        )
        _append_step(
            state,
            "knowledge_retrieval",
            query=getattr(knowledge_service, "last_query", state.effective_query),
            category_id=state.category_id,
            category=state.category,
            preferences=state.preferences,
            negative_preferences=state.negative_preferences,
            citation_count=len(state.citations),
            cache_status=getattr(knowledge_service, "last_cache_status", None),
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

    (
        product_ids,
        source,
        focus_preferences,
        referenced_product_indices,
        resolved_from_last_products,
    ) = _compare_context_parts(
        state.compare_context
    )
    if not product_ids:
        _append_step(
            state,
            "product_comparison",
            status="skipped",
            reason="missing_compare_product_ids",
            compare_product_ids=[],
            referenced_product_indices=state.referenced_product_indices,
            resolved_from_last_products=False,
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
            referenced_product_indices=referenced_product_indices,
            resolved_from_last_products=resolved_from_last_products,
        )
        state.product_candidates = result.product_candidates
        state.answer = result.answer
        state.preferences = result.focus_preferences
        if result.product_candidates:
            state.category_id = result.product_candidates[0].category_id
        state.query_result = QueryUnderstandingResult(
            raw_query=state.effective_query,
            intent="shopping_guide",
            category_id=state.category_id,
            category_path=state.category_path,
            budget_min=state.budget_min,
            budget_max=state.budget_max,
            preferences=result.focus_preferences,
            need_clarification=False,
            clarification_question=None,
        )
        state.trace.append(result.trace)
        knowledge_service = _knowledge_retrieval_service(context)
        if knowledge_service is not None and state.category_id:
            state.citations = _search_knowledge_structured(
                knowledge_service,
                query=state.effective_query,
                category_id=state.category_id,
                top_k=3,
                preferences=state.preferences,
                negative_preferences=state.negative_preferences,
            )
            _append_step(
                state,
                "knowledge_retrieval",
                query=getattr(knowledge_service, "last_query", state.effective_query),
                category_id=state.category_id,
                category=state.category,
                preferences=state.preferences,
                negative_preferences=state.negative_preferences,
                citation_count=len(state.citations),
                cache_status=getattr(knowledge_service, "last_cache_status", None),
            )
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

    response = _apply_answer_grounding_guard(state, context, response)
    _apply_response(state, response)
    append_trace(state, "response_compose", status="composed")
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
    shopping_memory = context_used.get("shopping_memory") or getattr(
        result,
        "shopping_memory",
        None,
    )
    budget = context_used.get("budget") or (
        shopping_memory.get("budget") if isinstance(shopping_memory, dict) else None
    )
    _append_step(
        state,
        "follow_up_rewrite",
        status=status,
        original_query=state.original_query,
        rewritten_query=getattr(result, "rewritten_query", state.effective_query),
        reason=getattr(result, "reason", None),
        source_turn_index=getattr(result, "source_turn_index", None),
        category=context_used.get("category"),
        budget=budget,
        preferences=context_used.get("preferences", []),
        negative_preferences=context_used.get("negative_preferences", []),
        referenced_product_ids=context_used.get("referenced_product_ids", []),
        resolved_product_ids=context_used.get("resolved_product_ids", []),
        shopping_memory=shopping_memory,
        context_used=context_used,
    )


def _apply_structured_compare_context(state: AgentState) -> None:
    if state.intent != "compare":
        state.compare_product_ids = []
        state.referenced_product_indices = []
        state.compare_context = None
        if state.query_result is not None and (
            getattr(state.query_result, "compare_product_ids", None)
            or getattr(state.query_result, "referenced_product_indices", None)
        ):
            state.query_result = state.query_result.model_copy(
                update={
                    "compare_product_ids": [],
                    "referenced_product_indices": [],
                }
            )
            state.query_understanding = state.query_result.to_trace_dict()
        return

    existing_product_ids, _, _, _, _ = _compare_context_parts(state.compare_context)
    if existing_product_ids:
        return

    if state.compare_product_ids:
        state.compare_context = CompareContext(
            product_ids=list(state.compare_product_ids),
            source="compare_product_ids",
            focus_preferences=list(state.preferences),
            referenced_product_indices=list(state.referenced_product_indices),
            resolved_from_last_products=False,
        )
        return

    last_product_ids = _last_product_ids_from_state(state)
    if state.referenced_product_indices:
        resolved_ids = [
            last_product_ids[index - 1]
            for index in state.referenced_product_indices
            if 1 <= index <= len(last_product_ids)
        ]
        if resolved_ids:
            state.compare_product_ids = resolved_ids
            state.compare_context = CompareContext(
                product_ids=resolved_ids,
                source="referenced_product_indices",
                focus_preferences=list(state.preferences),
                referenced_product_indices=list(state.referenced_product_indices),
                resolved_from_last_products=True,
            )
            return

        state.need_clarification = True
        _set_compare_clarification(state)
        _append_step(
            state,
            "product_comparison",
            status="skipped",
            reason="missing_last_products",
            compare_product_ids=[],
            referenced_product_indices=list(state.referenced_product_indices),
            resolved_from_last_products=False,
            requested_product_ids=[],
            returned_product_ids=[],
            missing_product_ids=[],
            comparison_product_count=0,
        )
        return

    if state.intent == "compare" and not last_product_ids:
        state.need_clarification = True
        _set_compare_clarification(state)
        _append_step(
            state,
            "product_comparison",
            status="skipped",
            reason="missing_last_products",
            compare_product_ids=[],
            referenced_product_indices=[],
            resolved_from_last_products=False,
            requested_product_ids=[],
            returned_product_ids=[],
            missing_product_ids=[],
            comparison_product_count=0,
        )


def _set_compare_clarification(state: AgentState) -> None:
    question = "我还没有上一轮可引用的商品，请先让我推荐几款商品后再比较。"
    state.need_clarification = True
    state.clarification_question = state.clarification_question or question
    if state.query_result is not None:
        state.query_result = state.query_result.model_copy(
            update={
                "intent": "clarification",
                "need_clarification": True,
                "clarification_question": state.clarification_question,
            }
        )
        state.query_understanding = state.query_result.to_trace_dict()


def _last_product_ids_from_state(state: AgentState) -> list[str]:
    for payload in [state.shopping_memory, state.query_understanding.get("shopping_memory")]:
        if isinstance(payload, dict):
            ids = _list_of_str(payload.get("last_product_ids"))
            if ids:
                return ids

    for turn in reversed(state.recent_turns):
        raw_ids = getattr(turn, "product_ids_json", None)
        if not raw_ids:
            continue
        try:
            value = json.loads(raw_ids)
        except (TypeError, json.JSONDecodeError):
            continue
        ids = _list_of_str(value)
        if ids:
            return ids
    return []


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
            resolved_from_last_products=True,
        )

    if getattr(result, "reason", None) == "vague_product_reference" and referenced_product_ids:
        return CompareContext(
            product_ids=list(referenced_product_ids),
            source="referenced_product_ids",
            focus_preferences=list(focus_preferences),
            resolved_from_last_products=True,
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


def _product_comparison_service(
    context: AgentRuntimeContext,
) -> ProductComparisonService | None:
    if context.product_comparison_service is not None:
        return context.product_comparison_service
    if context.db is None:
        return None
    context.product_comparison_service = ProductComparisonService(db=context.db)
    return context.product_comparison_service


def _compare_context_parts(
    compare_context: Any,
) -> tuple[list[str], str, list[str], list[int], bool]:
    if isinstance(compare_context, CompareContext):
        return (
            list(compare_context.product_ids),
            compare_context.source,
            list(compare_context.focus_preferences),
            list(compare_context.referenced_product_indices),
            compare_context.resolved_from_last_products,
        )
    if isinstance(compare_context, dict):
        return (
            list(compare_context.get("product_ids") or []),
            str(compare_context.get("source") or "unknown"),
            list(compare_context.get("focus_preferences") or []),
            list(compare_context.get("referenced_product_indices") or []),
            bool(compare_context.get("resolved_from_last_products")),
        )
    return [], "unknown", [], [], False


def _structured_filters_from_state(state: AgentState) -> dict[str, Any]:
    return {
        "category": state.category,
        "category_id": state.category_id,
        "budget_min": state.budget_min,
        "budget_max": state.budget_max,
        "preferences": list(state.preferences),
        "negative_preferences": list(state.negative_preferences),
    }


def _search_knowledge_structured(
    knowledge_service: Any,
    *,
    query: str,
    category_id: str | None,
    top_k: int,
    preferences: list[str],
    negative_preferences: list[str],
):
    try:
        return knowledge_service.search_knowledge(
            query=query,
            category_id=category_id,
            top_k=top_k,
            preferences=preferences,
            negative_preferences=negative_preferences,
        )
    except TypeError:
        return knowledge_service.search_knowledge(
            query=query,
            category_id=category_id,
            top_k=top_k,
        )


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None and str(item)]


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
    append_trace(state, node_name, status="composed")
    return state


def _apply_response(state: AgentState, response: ChatResponse) -> None:
    state.answer = response.answer
    state.product_cards = response.product_cards
    state.citations = response.citations
    state.trace.extend(response.trace)


def _apply_answer_grounding_guard(
    state: AgentState,
    context: AgentRuntimeContext,
    response: ChatResponse,
) -> ChatResponse:
    guard = context.answer_grounding_guard or AnswerGroundingGuard()
    result = guard.check(_grounding_context_from_state(state, response))
    trace_step = _grounding_trace_step(result)
    if result.passed:
        return _append_response_trace(response, trace_step)

    fallback_answer = result.fallback_answer or response.answer
    return ChatResponse(
        answer=fallback_answer,
        product_cards=response.product_cards,
        citations=response.citations,
        trace=[*response.trace, trace_step],
    )


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


def _compose_with_optional_llm(
    state: AgentState,
    context: AgentRuntimeContext,
    base_response: ChatResponse,
) -> ChatResponse:
    if state.intent not in {"shopping_guide", "product_knowledge"}:
        return base_response

    if context.llm_answer_composer is None:
        return _append_response_trace(
            base_response,
            {"step": "llm_answer", "enabled": False, "status": "disabled"},
        )

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
