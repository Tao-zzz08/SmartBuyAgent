from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

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
from app.chat.product_comparison import CompareContext


class _WorkflowState(TypedDict):
    state: AgentState


RouteName = Literal[
    "compare",
    "clarification",
    "shopping_guide",
    "product_knowledge",
    "chitchat",
    "response_compose",
]


class AgentWorkflow:
    def __init__(self, context: AgentRuntimeContext) -> None:
        self.context = context
        self._graph = self._build_graph().compile()

    def run(
        self,
        query: str,
        session_id: str | None = None,
        compare_context: CompareContext | None = None,
    ) -> AgentState:
        initial_state = create_initial_agent_state(query=query, session_id=session_id)
        initial_state.compare_context = compare_context
        result = self._graph.invoke({"state": initial_state})
        return result["state"]

    def _build_graph(self):
        graph = StateGraph(_WorkflowState)

        graph.add_node("load_context", self._node(load_context_node))
        graph.add_node("follow_up_rewrite", self._node(follow_up_rewrite_node))
        graph.add_node("intent_router", self._node(intent_router_node))
        graph.add_node("route_by_intent", self._route_by_intent_node)
        graph.add_node("shopping_guide", self._node(shopping_guide_node))
        graph.add_node("product_knowledge", self._node(product_knowledge_node))
        graph.add_node("compare", self._node(compare_node))
        graph.add_node("clarification", self._node(clarification_node))
        graph.add_node("chitchat", self._node(chitchat_node))
        graph.add_node("response_compose", self._node(response_compose_node))
        graph.add_node("save_trace", self._node(save_trace_node))

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "follow_up_rewrite")
        graph.add_edge("follow_up_rewrite", "intent_router")
        graph.add_edge("intent_router", "route_by_intent")
        graph.add_conditional_edges(
            "route_by_intent",
            self._route_name,
            {
                "compare": "compare",
                "clarification": "clarification",
                "shopping_guide": "shopping_guide",
                "product_knowledge": "product_knowledge",
                "chitchat": "chitchat",
                "response_compose": "response_compose",
            },
        )
        graph.add_edge("compare", "response_compose")
        graph.add_edge("shopping_guide", "response_compose")
        graph.add_edge("product_knowledge", "response_compose")
        graph.add_edge("response_compose", "save_trace")
        graph.add_edge("clarification", "save_trace")
        graph.add_edge("chitchat", "save_trace")
        graph.add_edge("save_trace", END)
        return graph

    def _node(self, node_func):
        def run_node(workflow_state: _WorkflowState) -> _WorkflowState:
            return {"state": node_func(workflow_state["state"], self.context)}

        return run_node

    @staticmethod
    def _route_by_intent_node(workflow_state: _WorkflowState) -> _WorkflowState:
        state = workflow_state["state"]
        append_trace(
            state,
            "route_by_intent",
            status="routed",
            route=_route_name_for_state(state),
        )
        return {"state": state}

    @staticmethod
    def _route_name(workflow_state: _WorkflowState) -> RouteName:
        return _route_name_for_state(workflow_state["state"])


def _route_name_for_state(state: AgentState) -> RouteName:
    if _has_compare_product_ids(state.compare_context):
        return "compare"

    if state.need_clarification or state.intent == "clarification":
        return "clarification"

    if state.intent == "shopping_guide":
        return "shopping_guide"

    if state.intent == "product_knowledge":
        return "product_knowledge"

    if state.intent == "compare":
        return "clarification"

    if state.intent == "chitchat":
        return "chitchat"

    return "response_compose"


def _has_compare_product_ids(compare_context) -> bool:
    if isinstance(compare_context, CompareContext):
        return bool(compare_context.product_ids)
    if isinstance(compare_context, dict):
        return bool(compare_context.get("product_ids"))
    return False
