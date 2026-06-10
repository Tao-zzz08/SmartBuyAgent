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


NODE_CASES = [
    (load_context_node, "load_context"),
    (follow_up_rewrite_node, "follow_up_rewrite"),
    (intent_router_node, "intent_router"),
    (shopping_guide_node, "shopping_guide"),
    (product_knowledge_node, "product_knowledge"),
    (compare_node, "compare"),
    (clarification_node, "clarification"),
    (chitchat_node, "chitchat"),
    (response_compose_node, "response_compose"),
    (save_trace_node, "save_trace"),
]


def test_agent_nodes_return_state_and_append_stub_trace() -> None:
    for node_func, node_name in NODE_CASES:
        state = create_initial_agent_state("推荐一款手机", session_id="session_test")

        result = node_func(state)

        assert isinstance(result, AgentState)
        assert result is state
        assert result.trace[-1] == {
            "step": "agent_node",
            "node": node_name,
            "status": "stub",
        }


def test_agent_nodes_do_not_destroy_existing_fields() -> None:
    state = create_initial_agent_state("原始问题", session_id="session_keep")
    state.effective_query = "改写问题"
    state.intent = "shopping_guide"
    state.category_id = "cat_phone"
    state.preferences = ["拍照"]

    for node_func, _node_name in NODE_CASES:
        node_func(state)

    assert state.original_query == "原始问题"
    assert state.effective_query == "改写问题"
    assert state.session_id == "session_keep"
    assert state.intent == "shopping_guide"
    assert state.category_id == "cat_phone"
    assert state.preferences == ["拍照"]
    assert len(state.trace) == len(NODE_CASES)


def test_append_trace_supports_extra_fields() -> None:
    state = create_initial_agent_state("query")

    append_trace(state, "custom", status="ready", route="shopping_guide")

    assert state.trace == [
        {
            "step": "agent_node",
            "node": "custom",
            "status": "ready",
            "route": "shopping_guide",
        }
    ]
