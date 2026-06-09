from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.chat.response_composer import ChatResponse
from app.models import ChatSession, ChatTurn


class ConversationMemoryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_session(self, session_id: str) -> ChatSession:
        session = self.db.get(ChatSession, session_id)
        if session is not None:
            return session

        session = ChatSession(session_id=session_id)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_next_turn_index(self, session_id: str) -> int:
        max_index = self.db.scalar(
            select(func.max(ChatTurn.turn_index)).where(
                ChatTurn.session_id == session_id
            )
        )
        return int(max_index or 0) + 1

    def save_turn(
        self,
        session_id: str,
        user_query: str,
        chat_response: ChatResponse,
    ) -> ChatTurn:
        try:
            session = self.ensure_session(session_id)
            query_trace = _query_understanding_trace(chat_response.trace)
            product_ids = [card.product_id for card in chat_response.product_cards]
            citation_chunk_ids = [
                citation.chunk_id for citation in chat_response.citations
            ]

            turn = ChatTurn(
                session_id=session_id,
                turn_index=self.get_next_turn_index(session_id),
                user_query=user_query,
                assistant_answer=chat_response.answer,
                intent=_str_or_none(query_trace.get("intent")),
                category_id=_str_or_none(query_trace.get("category_id")),
                category_path=_str_or_none(query_trace.get("category_path")),
                budget_min=_int_or_none(query_trace.get("budget_min")),
                budget_max=_int_or_none(query_trace.get("budget_max")),
                preferences_json=_json_dumps(
                    _list_or_empty(query_trace.get("preferences"))
                ),
                product_ids_json=_json_dumps(product_ids),
                citation_chunk_ids_json=_json_dumps(citation_chunk_ids),
            )
            session.updated_at = datetime.utcnow()
            self.db.add(turn)
            self.db.commit()
            self.db.refresh(turn)
            return turn
        except Exception:
            self.db.rollback()
            raise

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
    ) -> list[ChatTurn]:
        if limit <= 0:
            return []

        turns = (
            self.db.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == session_id)
                .order_by(desc(ChatTurn.turn_index))
                .limit(limit)
            )
            .all()
        )
        return list(reversed(turns))


def _query_understanding_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    for step in trace:
        if isinstance(step, dict) and step.get("step") == "query_understanding":
            return step
    return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
