import json

from app.streaming.event_emitter import StreamEventEmitter
from app.streaming.events import sse_event


def test_sse_event_serializes_utf8_json() -> None:
    event_text = sse_event("trace", {"message": "节点开始"})

    assert event_text.startswith("event: trace\n")
    assert event_text.endswith("\n\n")
    assert "节点开始" in event_text
    payload = json.loads(event_text.split("data: ", 1)[1])
    assert payload["message"] == "节点开始"


def test_stream_event_emitter_adds_request_and_session_ids() -> None:
    emitter = StreamEventEmitter(request_id="req_1", session_id="sess_1")

    emitter.emit("node_start", {"node": "intent_router"})
    events = emitter.drain()

    assert len(events) == 1
    assert events[0].event == "node_start"
    assert events[0].data["request_id"] == "req_1"
    assert events[0].data["session_id"] == "sess_1"
    assert events[0].data["node"] == "intent_router"
    assert emitter.drain() == []

