from __future__ import annotations

from typing import Any

from app.agent.state import AgentState


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


def load_context_node(state: AgentState) -> AgentState:
    return append_trace(state, "load_context")


def follow_up_rewrite_node(state: AgentState) -> AgentState:
    return append_trace(state, "follow_up_rewrite")


def intent_router_node(state: AgentState) -> AgentState:
    return append_trace(state, "intent_router")


def shopping_guide_node(state: AgentState) -> AgentState:
    return append_trace(state, "shopping_guide")


def product_knowledge_node(state: AgentState) -> AgentState:
    return append_trace(state, "product_knowledge")


def compare_node(state: AgentState) -> AgentState:
    return append_trace(state, "compare")


def clarification_node(state: AgentState) -> AgentState:
    return append_trace(state, "clarification")


def chitchat_node(state: AgentState) -> AgentState:
    return append_trace(state, "chitchat")


def response_compose_node(state: AgentState) -> AgentState:
    return append_trace(state, "response_compose")


def save_trace_node(state: AgentState) -> AgentState:
    return append_trace(state, "save_trace")
