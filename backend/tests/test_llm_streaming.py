from app.agent.context import AgentRuntimeContext
from app.agent.stream_runner import AgentStreamRunner


def test_agent_stream_runner_emits_token_events_for_composed_answer() -> None:
    runner = AgentStreamRunner(AgentRuntimeContext())

    events = list(
        runner.stream(
            "\u63a8\u8350\u4e00\u4e0b",
            request_id="req_stream_unit",
            event_session_id="session_stream_unit",
        )
    )
    event_names = [event.event for event in events]
    token_text = "".join(
        event.data.get("delta", "") for event in events if event.event == "token"
    )

    assert "node_start" in event_names
    assert "node_end" in event_names
    assert "trace" in event_names
    assert "token" in event_names
    assert token_text
    assert all(
        isinstance(event.data.get("duration_ms"), int)
        for event in events
        if event.event == "node_end"
    )

