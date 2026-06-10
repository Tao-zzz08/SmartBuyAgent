from app.agent.state import AgentState, create_initial_agent_state


def test_create_initial_agent_state() -> None:
    state = create_initial_agent_state(
        query="预算3000，推荐一款拍照好的手机",
        session_id="session_test",
    )

    assert isinstance(state, AgentState)
    assert state.original_query == "预算3000，推荐一款拍照好的手机"
    assert state.effective_query == state.original_query
    assert state.session_id == "session_test"
    assert state.intent is None
    assert state.category_id is None
    assert state.category_path is None
    assert state.budget_min is None
    assert state.budget_max is None
    assert state.preferences == []
    assert state.need_clarification is False
    assert state.clarification_question is None
    assert state.product_candidates == []
    assert state.citations == []
    assert state.product_cards == []
    assert state.answer is None
    assert state.recent_turns == []
    assert state.query_result is None
    assert state.rewrite_result is None
    assert state.compare_context is None
    assert state.trace == []
    assert state.errors == []


def test_create_initial_agent_state_uses_independent_lists() -> None:
    first = create_initial_agent_state("query one")
    second = create_initial_agent_state("query two")

    first.trace.append({"step": "test"})
    first.errors.append("error")
    first.preferences.append("拍照")

    assert second.trace == []
    assert second.errors == []
    assert second.preferences == []
