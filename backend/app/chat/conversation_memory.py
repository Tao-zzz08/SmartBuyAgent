from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.cache.cache_service import CacheService
from app.core.config import settings
from app.chat.response_composer import ChatResponse
from app.models import ChatSession, ChatTurn


SESSION_RECENT_TURNS_KEY = "smartbuy:session:{session_id}:recent_turns"
SESSION_LAST_CANDIDATES_KEY = "smartbuy:session:{session_id}:last_candidates"


@dataclass
class CachedChatTurn:
    id: int | None
    session_id: str
    turn_index: int
    user_query: str
    assistant_answer: str
    intent: str | None
    category_id: str | None
    category_path: str | None
    budget_min: int | None
    budget_max: int | None
    preferences_json: str
    product_ids_json: str
    citation_chunk_ids_json: str
    created_at: str | None = None


class ConversationMemoryService:
    def __init__(self, db: Session, cache_service: CacheService | None = None) -> None:
        self.db = db
        self.cache_service = cache_service
        self.last_cache_status = "disabled" if cache_service is None else "miss"

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
            self._refresh_session_cache(session_id, product_ids)
            return turn
        except Exception:
            self.db.rollback()
            raise

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
    ) -> list[ChatTurn | CachedChatTurn]:
        if limit <= 0:
            return []

        cached_turns = self._get_cached_recent_turns(session_id, limit)
        if cached_turns is not None:
            return cached_turns

        turns = (
            self.db.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == session_id)
                .order_by(desc(ChatTurn.turn_index))
                .limit(limit)
            )
            .all()
        )
        ordered_turns = list(reversed(turns))
        self._set_cached_recent_turns(session_id, ordered_turns)
        return ordered_turns

    def _get_cached_recent_turns(
        self,
        session_id: str,
        limit: int,
    ) -> list[CachedChatTurn] | None:
        if self.cache_service is None:
            self.last_cache_status = "disabled"
            return None
        try:
            cached = self.cache_service.get_json(_recent_turns_key(session_id))
        except Exception:
            self.last_cache_status = "failed"
            return None
        if not isinstance(cached, list):
            self.last_cache_status = "miss"
            return None

        turns = [
            _turn_from_cache(item)
            for item in cached
            if isinstance(item, dict)
        ]
        self.last_cache_status = "hit"
        return turns[-limit:]

    def _set_cached_recent_turns(
        self,
        session_id: str,
        turns: list[ChatTurn],
    ) -> None:
        if self.cache_service is None:
            return
        try:
            self.cache_service.set_json(
                _recent_turns_key(session_id),
                [_turn_to_cache(turn) for turn in turns[-5:]],
                ttl_seconds=settings.SESSION_CACHE_TTL_SECONDS,
            )
        except Exception:
            return

    def _refresh_session_cache(
        self,
        session_id: str,
        product_ids: list[str],
    ) -> None:
        if self.cache_service is None:
            return
        try:
            self.cache_service.delete(_recent_turns_key(session_id))
            self.cache_service.set_json(
                _last_candidates_key(session_id),
                product_ids,
                ttl_seconds=settings.SESSION_CACHE_TTL_SECONDS,
            )
        except Exception:
            return


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


def _recent_turns_key(session_id: str) -> str:
    return SESSION_RECENT_TURNS_KEY.format(session_id=session_id)


def _last_candidates_key(session_id: str) -> str:
    return SESSION_LAST_CANDIDATES_KEY.format(session_id=session_id)


def _turn_to_cache(turn: ChatTurn) -> dict[str, Any]:
    return {
        "id": turn.id,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "user_query": turn.user_query,
        "assistant_answer": turn.assistant_answer,
        "intent": turn.intent,
        "category_id": turn.category_id,
        "category_path": turn.category_path,
        "budget_min": turn.budget_min,
        "budget_max": turn.budget_max,
        "preferences_json": turn.preferences_json,
        "product_ids_json": turn.product_ids_json,
        "citation_chunk_ids_json": turn.citation_chunk_ids_json,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
    }


def _turn_from_cache(value: dict[str, Any]) -> CachedChatTurn:
    return CachedChatTurn(
        id=_int_or_none(value.get("id")),
        session_id=str(value.get("session_id") or ""),
        turn_index=_int_or_none(value.get("turn_index")) or 0,
        user_query=str(value.get("user_query") or ""),
        assistant_answer=str(value.get("assistant_answer") or ""),
        intent=_str_or_none(value.get("intent")),
        category_id=_str_or_none(value.get("category_id")),
        category_path=_str_or_none(value.get("category_path")),
        budget_min=_int_or_none(value.get("budget_min")),
        budget_max=_int_or_none(value.get("budget_max")),
        preferences_json=str(value.get("preferences_json") or "[]"),
        product_ids_json=str(value.get("product_ids_json") or "[]"),
        citation_chunk_ids_json=str(value.get("citation_chunk_ids_json") or "[]"),
        created_at=_str_or_none(value.get("created_at")),
    )
