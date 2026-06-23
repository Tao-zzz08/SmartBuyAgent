from __future__ import annotations

from dataclasses import replace
import sys
import types
from types import SimpleNamespace

try:
    import langgraph.graph  # noqa: F401
except ModuleNotFoundError:
    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")

    class _FakeStateGraph:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_node(self, *args, **kwargs) -> None:
            pass

        def add_edge(self, *args, **kwargs) -> None:
            pass

        def add_conditional_edges(self, *args, **kwargs) -> None:
            pass

        def compile(self):
            return self

    graph_module.START = "__start__"
    graph_module.END = "__end__"
    graph_module.StateGraph = _FakeStateGraph
    langgraph_module.graph = graph_module
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module

from app.agent.context import AgentRuntimeContext
from app.agent.nodes import intent_router_node
from app.agent.state import create_initial_agent_state
from app.agent.workflow import _route_name_for_state
from app.chat.product_comparison import CompareContext
from app.chat.query_understanding import QueryUnderstandingResult, QueryUnderstandingService


PHONE = "\u624b\u673a"
PHOTO = "\u62cd\u7167"


class StaticQueryUnderstandingService:
    def __init__(self, result: QueryUnderstandingResult) -> None:
        self.result = result

    def understand(self, *args, **kwargs) -> QueryUnderstandingResult:
        return self.result


def test_budget_follow_up_route_ignores_polluted_compare_context() -> None:
    state = create_initial_agent_state(
        "\u9884\u7b97\u589e\u52a0\u52304000\u5462",
        session_id="session_test",
    )
    state.intent = "shopping_guide"
    state.category = "phone"
    state.category_id = "cat_phone"
    state.budget_max = 4000
    state.preferences = [PHOTO]
    state.compare_product_ids = ["p1", "p2", "p3"]
    state.referenced_product_indices = [1, 2, 3]
    state.compare_context = CompareContext(
        product_ids=["p1", "p2", "p3"],
        source="compare_product_ids",
        focus_preferences=[PHOTO],
    )

    assert _route_name_for_state(state) == "shopping_guide"


def test_non_compare_intent_clears_compare_fields_before_routing() -> None:
    state = create_initial_agent_state(
        "\u9884\u7b97\u589e\u52a0\u52304000\u5462",
        session_id="session_test",
    )
    result = QueryUnderstandingResult(
        original_query=state.original_query,
        effective_query=f"\u9884\u7b974000\u5143\u4ee5\u5185\uff0c\u63a8\u8350{PHOTO}\u597d\u7684{PHONE}",
        is_follow_up=True,
        intent="shopping_guide",
        category="phone",
        budget={"max": 4000, "currency": "CNY"},
        preferences=[PHOTO],
        compare_product_ids=["p1", "p2", "p3"],
        referenced_product_indices=[1, 2, 3],
        confidence=0.9,
        source="rule",
        reason="budget_update_follow_up",
        shopping_memory={
            "category": "phone",
            "budget": {"min": None, "max": 4000, "currency": "CNY"},
            "preferences": [PHOTO],
            "negative_preferences": [],
            "last_product_ids": ["p1", "p2", "p3"],
            "last_intent": "shopping_guide",
        },
    )

    intent_router_node(
        state,
        AgentRuntimeContext(
            query_understanding_service=StaticQueryUnderstandingService(result)
        ),
    )

    assert state.intent == "shopping_guide"
    assert state.compare_product_ids == []
    assert state.referenced_product_indices == []
    assert state.compare_context is None
    assert _route_name_for_state(state) == "shopping_guide"
    query_trace = next(step for step in state.trace if step.get("step") == "query_understanding")
    assert query_trace["compare_product_ids"] == []
    assert query_trace["referenced_product_indices"] == []


def test_budget_rewrite_referenced_products_do_not_become_compare_ids() -> None:
    service = QueryUnderstandingService(llm_enabled=False)
    first = service.understand(f"\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e{PHOTO}\u597d\u7684{PHONE}")
    rewrite_result = SimpleNamespace(
        is_follow_up=True,
        reason="budget_update_follow_up",
        rewritten_query=f"\u9884\u7b974000\u5143\u4ee5\u5185\uff0c\u63a8\u8350{PHOTO}\u597d\u7684{PHONE}",
        shopping_memory={
            "category": "phone",
            "budget": {"min": None, "max": 4000, "currency": "CNY"},
            "preferences": [PHOTO],
            "negative_preferences": [],
            "last_product_ids": ["p1", "p2", "p3"],
            "last_intent": "shopping_guide",
        },
        context_used={
            "category": "phone",
            "budget": {"min": None, "max": 4000, "currency": "CNY"},
            "preferences": [PHOTO],
            "referenced_product_ids": ["p1", "p2", "p3"],
            "resolved_product_ids": [],
        },
    )

    result = service.understand(
        "\u9884\u7b97\u589e\u52a0\u52304000\u5462",
        previous_memory=first.to_shopping_memory(),
        rewrite_result=rewrite_result,
    )

    assert result.intent == "shopping_guide"
    assert result.category == "phone"
    assert result.budget_max == 4000
    assert PHOTO in result.preferences
    assert result.compare_product_ids == []
    assert result.referenced_product_indices == []
    assert result.source == "rule"


def test_real_compare_follow_up_still_routes_to_compare() -> None:
    service = QueryUnderstandingService(llm_enabled=False)
    previous = service.understand(
        f"\u9884\u7b975000\uff0c\u63a8\u8350{PHOTO}\u597d\u7684{PHONE}"
    ).to_shopping_memory()
    previous = replace(previous, last_product_ids=["p1", "p2", "p3"])
    rewrite_result = SimpleNamespace(
        is_follow_up=True,
        reason="ordinal_product_reference",
        rewritten_query="\u5bf9\u4e0a\u4e00\u8f6e\u63a8\u8350\u7684\u7b2c\u4e00\u4e2a\u548c\u7b2c\u4e8c\u4e2a\u5546\u54c1\u8fdb\u884c\u6bd4\u8f83",
        shopping_memory=previous.to_dict(),
        context_used={
            "preferences": [PHOTO],
            "referenced_product_ids": ["p1", "p2", "p3"],
            "resolved_product_ids": ["p1", "p2"],
        },
    )

    result = service.understand(
        "\u7b2c\u4e00\u4e2a\u548c\u7b2c\u4e8c\u4e2a\u54ea\u4e2a\u597d",
        previous_memory=previous,
        rewrite_result=rewrite_result,
    )
    state = create_initial_agent_state(result.original_query, session_id="session_test")

    intent_router_node(
        state,
        AgentRuntimeContext(
            query_understanding_service=StaticQueryUnderstandingService(result)
        ),
    )

    assert result.intent == "compare"
    assert state.compare_product_ids == ["p1", "p2"]
    assert state.compare_context is not None
    assert state.compare_context.product_ids == ["p1", "p2"]
    assert _route_name_for_state(state) == "compare"
