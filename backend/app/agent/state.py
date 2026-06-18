from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    original_query: str
    effective_query: str
    session_id: str | None = None
    query_understanding: dict[str, Any] = field(default_factory=dict)
    intent: str | None = None
    category: str | None = None
    category_id: str | None = None
    category_path: str | None = None
    budget_min: int | None = None
    budget_max: int | None = None
    preferences: list[str] = field(default_factory=list)
    negative_preferences: list[str] = field(default_factory=list)
    compare_product_ids: list[str] = field(default_factory=list)
    referenced_product_indices: list[int] = field(default_factory=list)
    need_clarification: bool = False
    clarification_question: str | None = None
    shopping_memory: Any | None = None
    product_candidates: list[Any] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    product_cards: list[Any] = field(default_factory=list)
    answer: str | None = None
    recent_turns: list[Any] = field(default_factory=list)
    query_result: Any | None = None
    rewrite_result: Any | None = None
    compare_context: Any | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def create_initial_agent_state(
    query: str,
    session_id: str | None = None,
) -> AgentState:
    return AgentState(
        original_query=query,
        effective_query=query,
        session_id=session_id,
    )
