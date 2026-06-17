from __future__ import annotations

from typing import Any

from app.streaming.events import StreamEvent


class StreamEventEmitter:
    """Small synchronous event buffer used by node-level stream runners."""

    def __init__(self, *, request_id: str, session_id: str | None = None) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self._events: list[StreamEvent] = []

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {
            "request_id": self.request_id,
            **({"session_id": self.session_id} if self.session_id else {}),
            **data,
        }
        self._events.append(StreamEvent(event=event_type, data=payload))

    def drain(self) -> list[StreamEvent]:
        events = list(self._events)
        self._events.clear()
        return events

