from __future__ import annotations

from typing import Any


CHECK_NAMES = [
    "context_carryover",
    "category_switch",
    "compare_resolution",
    "clarification",
    "route_stability",
]

CHECK_METRICS = {
    "context_carryover": "context_carryover_accuracy",
    "category_switch": "category_switch_accuracy",
    "compare_resolution": "compare_resolution_accuracy",
    "clarification": "clarification_accuracy",
    "route_stability": "route_stability_rate",
}

CATEGORY_PRODUCT_ID_HINTS = {
    "phone": ["phone", "mobile", "iphone"],
    "shoes": ["shoe", "shoes", "boot", "sneaker"],
    "skincare": ["skin", "skincare", "cream", "serum", "cleanser"],
}


def evaluate_multiturn_session(
    case: dict[str, Any],
    turn_results: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = _empty_checks()
    task_type = str(case.get("task_type") or "generic")
    turns = case.get("turns") or []

    for index, turn_result in enumerate(turn_results):
        turn = turns[index] if index < len(turns) and isinstance(turns[index], dict) else {}
        expected = turn.get("expect") or {}
        previous_turn = turn_results[index - 1] if index > 0 else None

        _evaluate_route_stability(checks, index + 1, turn_result, expected)
        _evaluate_context_carryover(
            checks,
            index + 1,
            turn_result,
            previous_turn,
            expected,
        )
        _evaluate_category_switch(
            checks,
            index + 1,
            turn_result,
            previous_turn,
            expected,
        )
        _evaluate_compare_resolution(
            checks,
            index + 1,
            turn_result,
            previous_turn,
            expected,
        )
        _evaluate_clarification(
            checks,
            index + 1,
            turn_result,
            expected,
            force=task_type == "clarification_without_context",
        )

    failure_reasons = _session_failure_reasons(checks)
    for turn_result in turn_results:
        for reason in turn_result.get("failure_reasons") or []:
            failure_reasons.append(f"turn_{turn_result.get('turn_index')}: {reason}")

    session_success = not failure_reasons
    expected_success = (case.get("session_expect") or {}).get("success")
    if expected_success is False:
        session_success = not session_success

    return {
        "session_success": session_success,
        "task_type": task_type,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def aggregate_multiturn_metrics(
    session_results: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluated_sessions = len(session_results)
    successful_sessions = sum(
        1 for result in session_results if result.get("session_success") is True
    )
    metrics: dict[str, Any] = {
        "session_success_rate": _round_rate(successful_sessions, evaluated_sessions),
        "evaluated_sessions": evaluated_sessions,
        "failed_sessions": evaluated_sessions - successful_sessions,
    }

    for check_name, metric_name in CHECK_METRICS.items():
        passed = 0
        total = 0
        for result in session_results:
            check = (result.get("checks") or {}).get(check_name) or {}
            passed += int(check.get("passed") or 0)
            total += int(check.get("total") or 0)
        metrics[metric_name] = _round_rate(passed, total) if total else None

    return metrics


def _evaluate_route_stability(
    checks: dict[str, dict[str, Any]],
    turn_index: int,
    turn_result: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if "route" not in expected:
        return
    actual_route = _route(turn_result)
    expected_route = expected["route"]
    _record(
        checks,
        "route_stability",
        actual_route == expected_route,
        turn_index=turn_index,
        reason=f"route mismatch: expected {expected_route!r}, got {actual_route!r}",
        details={"expected_route": expected_route, "actual_route": actual_route},
    )


def _evaluate_context_carryover(
    checks: dict[str, dict[str, Any]],
    turn_index: int,
    turn_result: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    context = expected.get("context_carryover")
    if not isinstance(context, dict):
        return

    actual_qu = _query_understanding(turn_result)
    previous_qu = _query_understanding(previous_turn) if previous_turn else {}

    if context.get("category"):
        previous_category = previous_qu.get("category")
        actual_category = actual_qu.get("category")
        _record(
            checks,
            "context_carryover",
            previous_category is not None and actual_category == previous_category,
            turn_index=turn_index,
            reason=(
                "category carryover mismatch: "
                f"expected previous {previous_category!r}, got {actual_category!r}"
            ),
            details={
                "previous_category": previous_category,
                "actual_category": actual_category,
            },
        )

    for field, actual_values in [
        ("preferences", _list(actual_qu.get("preferences"))),
        ("negative_preferences", _list(actual_qu.get("negative_preferences"))),
    ]:
        expected_values = _list(context.get(field))
        if not expected_values:
            continue
        missing = [value for value in expected_values if value not in actual_values]
        _record(
            checks,
            "context_carryover",
            not missing,
            turn_index=turn_index,
            reason=f"{field} carryover missing values: {missing!r}",
            details={
                "expected_values": expected_values,
                "actual_values": actual_values,
            },
        )

    if context.get("budget_updated"):
        expected_budget = expected.get("budget_max")
        actual_budget = _budget_max(actual_qu)
        _record(
            checks,
            "context_carryover",
            expected_budget is not None and actual_budget == expected_budget,
            turn_index=turn_index,
            reason=(
                "budget update mismatch: "
                f"expected {expected_budget!r}, got {actual_budget!r}"
            ),
            details={"expected_budget": expected_budget, "actual_budget": actual_budget},
        )


def _evaluate_category_switch(
    checks: dict[str, dict[str, Any]],
    turn_index: int,
    turn_result: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    actual_qu = _query_understanding(turn_result)
    previous_qu = _query_understanding(previous_turn) if previous_turn else {}
    expected_category = expected.get("category")
    previous_category = previous_qu.get("category")

    if (
        previous_category
        and expected_category
        and expected_category != previous_category
    ):
        actual_category = actual_qu.get("category")
        _record(
            checks,
            "category_switch",
            actual_category == expected_category,
            turn_index=turn_index,
            reason=(
                "category switch mismatch: "
                f"expected {expected_category!r}, got {actual_category!r}"
            ),
            details={
                "previous_category": previous_category,
                "expected_category": expected_category,
                "actual_category": actual_category,
            },
        )

    forbidden_preferences = _list(expected.get("forbidden_preferences"))
    if forbidden_preferences:
        actual_preferences = _list(actual_qu.get("preferences"))
        present = [value for value in forbidden_preferences if value in actual_preferences]
        _record(
            checks,
            "category_switch",
            not present,
            turn_index=turn_index,
            reason=f"forbidden preferences carried over: {present!r}",
            details={
                "forbidden_preferences": forbidden_preferences,
                "actual_preferences": actual_preferences,
            },
        )

    forbidden_categories = _list(expected.get("forbidden_categories"))
    if forbidden_categories:
        cards = _product_cards(turn_result)
        mismatched = [
            card.get("product_id")
            for card in cards
            for category in forbidden_categories
            if _card_matches_category(card, str(category))
        ]
        actual_category = actual_qu.get("category")
        current_forbidden = actual_category in forbidden_categories
        _record(
            checks,
            "category_switch",
            not current_forbidden and not mismatched,
            turn_index=turn_index,
            reason=(
                "forbidden category present after switch: "
                f"category={actual_category!r}, cards={mismatched!r}"
            ),
            details={
                "forbidden_categories": forbidden_categories,
                "actual_category": actual_category,
                "mismatched_product_ids": mismatched,
            },
        )


def _evaluate_compare_resolution(
    checks: dict[str, dict[str, Any]],
    turn_index: int,
    turn_result: dict[str, Any],
    previous_turn: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    should_check = any(
        key in expected
        for key in [
            "should_compare",
            "compare_indices",
            "referenced_product_indices",
            "compare_product_ids_count",
            "compare_product_ids_from_previous_turn",
        ]
    )
    if not should_check:
        return

    actual_qu = _query_understanding(turn_result)
    actual_route = _route(turn_result)
    compare_ids = _compare_product_ids(turn_result)
    referenced_indices = _list(actual_qu.get("referenced_product_indices"))

    if expected.get("should_compare") is True:
        _record(
            checks,
            "compare_resolution",
            actual_route == "compare" and actual_qu.get("intent") == "compare",
            turn_index=turn_index,
            reason=(
                "compare route mismatch: "
                f"route={actual_route!r}, intent={actual_qu.get('intent')!r}"
            ),
            details={
                "actual_route": actual_route,
                "actual_intent": actual_qu.get("intent"),
            },
        )
    elif expected.get("should_compare") is False:
        _record(
            checks,
            "compare_resolution",
            actual_route != "compare" and not compare_ids,
            turn_index=turn_index,
            reason=(
                "unexpected compare action: "
                f"route={actual_route!r}, ids={compare_ids!r}, indices={referenced_indices!r}"
            ),
            details={
                "actual_route": actual_route,
                "compare_product_ids": compare_ids,
                "referenced_product_indices": referenced_indices,
            },
        )

    expected_indices = expected.get("compare_indices")
    if expected_indices is None:
        expected_indices = expected.get("referenced_product_indices")
    if expected_indices is not None:
        _record(
            checks,
            "compare_resolution",
            referenced_indices == list(expected_indices),
            turn_index=turn_index,
            reason=(
                "compare indices mismatch: "
                f"expected {list(expected_indices)!r}, got {referenced_indices!r}"
            ),
            details={
                "expected_indices": list(expected_indices),
                "actual_indices": referenced_indices,
            },
        )

    expected_count = expected.get("compare_product_ids_count")
    if expected_count is not None:
        _record(
            checks,
            "compare_resolution",
            len(compare_ids) == int(expected_count),
            turn_index=turn_index,
            reason=(
                "compare id count mismatch: "
                f"expected {expected_count!r}, got {len(compare_ids)!r}"
            ),
            details={"expected_count": expected_count, "actual_ids": compare_ids},
        )

    if expected.get("compare_product_ids_from_previous_turn"):
        previous_ids = set(_product_ids(previous_turn) if previous_turn else [])
        _record(
            checks,
            "compare_resolution",
            bool(compare_ids) and set(compare_ids).issubset(previous_ids),
            turn_index=turn_index,
            reason="compare ids were not resolved from previous product cards",
            details={
                "compare_product_ids": compare_ids,
                "previous_product_ids": sorted(previous_ids),
            },
        )


def _evaluate_clarification(
    checks: dict[str, dict[str, Any]],
    turn_index: int,
    turn_result: dict[str, Any],
    expected: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    if not force and "should_clarify" not in expected:
        return

    actual_qu = _query_understanding(turn_result)
    actual_route = _route(turn_result)
    compare_ids = _compare_product_ids(turn_result)
    cards = _product_cards(turn_result)
    should_clarify = bool(expected.get("should_clarify", True))

    if should_clarify:
        clarified = (
            actual_route == "clarification"
            or actual_qu.get("intent") == "clarification"
            or bool(actual_qu.get("need_clarification"))
        )
        _record(
            checks,
            "clarification",
            clarified,
            turn_index=turn_index,
            reason=(
                "clarification mismatch: "
                f"route={actual_route!r}, intent={actual_qu.get('intent')!r}, "
                f"need_clarification={actual_qu.get('need_clarification')!r}"
            ),
            details={
                "actual_route": actual_route,
                "actual_intent": actual_qu.get("intent"),
                "need_clarification": actual_qu.get("need_clarification"),
            },
        )
        _record(
            checks,
            "clarification",
            not compare_ids and not cards,
            turn_index=turn_index,
            reason=(
                "clarification generated comparison or product cards: "
                f"ids={compare_ids!r}, cards={len(cards)!r}"
            ),
            details={
                "compare_product_ids": compare_ids,
                "product_card_count": len(cards),
            },
        )


def _empty_checks() -> dict[str, dict[str, Any]]:
    return {
        name: {"passed": 0, "total": 0, "failures": []}
        for name in CHECK_NAMES
    }


def _record(
    checks: dict[str, dict[str, Any]],
    check_name: str,
    passed: bool,
    *,
    turn_index: int,
    reason: str,
    details: dict[str, Any] | None = None,
) -> None:
    check = checks[check_name]
    check["total"] += 1
    if passed:
        check["passed"] += 1
        return
    check["failures"].append(
        {
            "turn_index": turn_index,
            "reason": reason,
            "details": details or {},
        }
    )


def _session_failure_reasons(checks: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for check_name, check in checks.items():
        for failure in check.get("failures") or []:
            reasons.append(f"{check_name}: {failure.get('reason')}")
    return reasons


def _query_understanding(turn_result: dict[str, Any] | None) -> dict[str, Any]:
    if not turn_result:
        return {}
    value = turn_result.get("query_understanding")
    if isinstance(value, dict) and value:
        return value
    response = turn_result.get("response") or {}
    if isinstance(response, dict):
        value = response.get("query_understanding")
        if isinstance(value, dict):
            return value
        for step in response.get("trace") or []:
            if isinstance(step, dict) and step.get("step") == "query_understanding":
                return step
    return {}


def _route(turn_result: dict[str, Any]) -> str | None:
    response = turn_result.get("response") or {}
    if isinstance(response, dict):
        if response.get("route"):
            return str(response["route"])
        for step in response.get("trace") or []:
            if isinstance(step, dict) and step.get("step") == "route_by_intent":
                route = step.get("route")
                return str(route) if route is not None else None
    return _query_understanding(turn_result).get("intent")


def _budget_max(query_understanding: dict[str, Any]) -> Any:
    if query_understanding.get("budget_max") is not None:
        return query_understanding.get("budget_max")
    budget = query_understanding.get("budget") or {}
    if isinstance(budget, dict):
        return budget.get("max")
    return None


def _product_cards(turn_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not turn_result:
        return []
    response = turn_result.get("response") or {}
    cards = response.get("product_cards") if isinstance(response, dict) else []
    return [card for card in (cards or []) if isinstance(card, dict)]


def _product_ids(turn_result: dict[str, Any] | None) -> list[str]:
    return [
        str(card.get("product_id"))
        for card in _product_cards(turn_result)
        if card.get("product_id")
    ]


def _compare_product_ids(turn_result: dict[str, Any]) -> list[str]:
    query_understanding = _query_understanding(turn_result)
    ids = query_understanding.get("compare_product_ids") or []
    if ids:
        return [str(value) for value in ids]

    response = turn_result.get("response") or {}
    if isinstance(response, dict):
        for step in response.get("trace") or []:
            if not isinstance(step, dict) or step.get("step") != "product_comparison":
                continue
            for key in ["compare_product_ids", "requested_product_ids", "returned_product_ids"]:
                value = step.get(key) or []
                if value:
                    return [str(product_id) for product_id in value]
    return []


def _card_matches_category(card: dict[str, Any], category: str) -> bool:
    product_id = str(card.get("product_id") or "").lower()
    title = str(card.get("title") or "").lower()
    raw_category = str(card.get("category") or "").lower()
    if raw_category == category:
        return True
    hints = CATEGORY_PRODUCT_ID_HINTS.get(category, [])
    return any(hint in product_id or hint in title for hint in hints)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _round_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)
