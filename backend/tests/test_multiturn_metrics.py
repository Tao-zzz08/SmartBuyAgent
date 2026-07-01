from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multiturn_metrics import (  # noqa: E402
    aggregate_multiturn_metrics,
    evaluate_multiturn_session,
)


def test_budget_followup_context_carryover_success() -> None:
    case = {
        "id": "budget_followup",
        "task_type": "budget_followup",
        "turns": [
            {"user": "first", "expect": {"route": "shopping_guide"}},
            {
                "user": "second",
                "expect": {
                    "route": "shopping_guide",
                    "budget_max": 4000,
                    "context_carryover": {
                        "category": True,
                        "preferences": ["camera"],
                        "budget_updated": True,
                    },
                },
            },
        ],
    }

    result = evaluate_multiturn_session(
        case,
        [
            _turn(category="phone", budget_max=3000, preferences=["camera"]),
            _turn(category="phone", budget_max=4000, preferences=["camera"]),
        ],
    )

    assert result["session_success"] is True
    assert result["checks"]["context_carryover"]["passed"] == 3
    assert result["checks"]["context_carryover"]["total"] == 3


def test_category_switch_success() -> None:
    case = {
        "id": "category_switch",
        "task_type": "category_switch",
        "turns": [
            {"user": "phone", "expect": {"route": "shopping_guide"}},
            {
                "user": "shoes",
                "expect": {
                    "route": "shopping_guide",
                    "category": "shoes",
                    "forbidden_preferences": ["camera"],
                    "forbidden_categories": ["phone"],
                },
            },
        ],
    }

    result = evaluate_multiturn_session(
        case,
        [
            _turn(
                category="phone",
                preferences=["camera"],
                cards=[{"product_id": "phone_001", "title": "camera phone"}],
            ),
            _turn(
                category="shoes",
                preferences=[],
                cards=[{"product_id": "shoes_001", "title": "commute shoes"}],
            ),
        ],
    )

    assert result["session_success"] is True
    assert result["checks"]["category_switch"]["total"] == 3


def test_compare_followup_resolves_previous_product_indices() -> None:
    case = {
        "id": "compare_followup",
        "task_type": "compare_followup",
        "turns": [
            {"user": "recommend", "expect": {"route": "shopping_guide"}},
            {
                "user": "compare",
                "expect": {
                    "route": "compare",
                    "should_compare": True,
                    "compare_indices": [1, 2],
                    "compare_product_ids_count": 2,
                    "compare_product_ids_from_previous_turn": True,
                },
            },
        ],
    }

    result = evaluate_multiturn_session(
        case,
        [
            _turn(cards=[{"product_id": "p1"}, {"product_id": "p2"}]),
            _turn(
                intent="compare",
                route="compare",
                compare_product_ids=["p1", "p2"],
                referenced_product_indices=[1, 2],
                cards=[{"product_id": "p1"}, {"product_id": "p2"}],
            ),
        ],
    )

    assert result["session_success"] is True
    assert result["checks"]["compare_resolution"]["passed"] == 4


def test_clarification_without_previous_products_success() -> None:
    case = {
        "id": "clarify",
        "task_type": "clarification_without_context",
        "turns": [
            {
                "user": "compare without context",
                "expect": {
                    "route": "clarification",
                    "should_clarify": True,
                    "should_compare": False,
                },
            }
        ],
    }

    result = evaluate_multiturn_session(
        case,
        [
            _turn(
                intent="compare",
                route="clarification",
                need_clarification=True,
                referenced_product_indices=[1, 2],
                cards=[],
            )
        ],
    )

    assert result["session_success"] is True
    assert result["checks"]["clarification"]["passed"] == 2
    assert result["checks"]["compare_resolution"]["passed"] == 1


def test_aggregate_multiturn_metrics_handles_zero_totals() -> None:
    metrics = aggregate_multiturn_metrics(
        [
            {
                "session_success": True,
                "checks": {
                    "context_carryover": {"passed": 0, "total": 0, "failures": []},
                    "category_switch": {"passed": 0, "total": 0, "failures": []},
                    "compare_resolution": {"passed": 0, "total": 0, "failures": []},
                    "clarification": {"passed": 0, "total": 0, "failures": []},
                    "route_stability": {"passed": 0, "total": 0, "failures": []},
                },
            }
        ]
    )

    assert metrics["session_success_rate"] == 1.0
    assert metrics["context_carryover_accuracy"] is None
    assert metrics["evaluated_sessions"] == 1
    assert metrics["failed_sessions"] == 0


def _turn(
    *,
    intent: str = "shopping_guide",
    route: str = "shopping_guide",
    category: str = "phone",
    budget_max: int | None = None,
    preferences: list[str] | None = None,
    negative_preferences: list[str] | None = None,
    compare_product_ids: list[str] | None = None,
    referenced_product_indices: list[int] | None = None,
    need_clarification: bool = False,
    cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query_understanding = {
        "intent": intent,
        "category": category,
        "budget": {"max": budget_max},
        "budget_max": budget_max,
        "preferences": preferences or [],
        "negative_preferences": negative_preferences or [],
        "compare_product_ids": compare_product_ids or [],
        "referenced_product_indices": referenced_product_indices or [],
        "need_clarification": need_clarification,
    }
    trace = [
        {"step": "query_understanding", **query_understanding},
        {"step": "route_by_intent", "route": route},
    ]
    if route == "compare":
        trace.append(
            {
                "step": "product_comparison",
                "compare_product_ids": compare_product_ids or [],
                "referenced_product_indices": referenced_product_indices or [],
            }
        )
    return {
        "turn_index": 1,
        "response": {"trace": trace, "product_cards": cards or []},
        "query_understanding": query_understanding,
        "failure_reasons": [],
    }
