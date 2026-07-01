from __future__ import annotations

from typing import Any


FIELD_METRICS = [
    "intent_accuracy",
    "intent_in_accuracy",
    "category_accuracy",
    "budget_min_accuracy",
    "budget_max_accuracy",
    "followup_accuracy",
    "clarification_accuracy",
    "referenced_indices_accuracy",
    "compare_product_ids_accuracy",
    "dialog_state_accuracy",
    "next_dialog_state_accuracy",
    "dialog_state_in_accuracy",
    "forbidden_preference_violation_rate",
]


def evaluate_intent_case(
    *,
    case: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    expected = case.get("expected") or {}
    normalized_actual = _normalize_actual(actual)
    checks: dict[str, bool] = {}
    failure_reasons: list[str] = []

    if "intent" in expected:
        checks["intent"] = normalized_actual.get("intent") == expected.get("intent")
        if not checks["intent"]:
            failure_reasons.append(
                f"intent_mismatch: expected {expected.get('intent')}, got {normalized_actual.get('intent')}"
            )
    elif "intent_in" in expected:
        allowed_intents = [str(value) for value in expected.get("intent_in") or []]
        checks["intent_in"] = normalized_actual.get("intent") in allowed_intents
        if not checks["intent_in"]:
            failure_reasons.append(
                f"intent_not_allowed: expected one of {allowed_intents}, got {normalized_actual.get('intent')}"
            )

    if "category" in expected:
        checks["category"] = normalized_actual.get("category") == expected.get("category")
        if not checks["category"]:
            failure_reasons.append(
                f"category_mismatch: expected {expected.get('category')}, got {normalized_actual.get('category')}"
            )

    if "dialog_state" in expected:
        checks["dialog_state"] = normalized_actual.get("dialog_state") == expected.get("dialog_state")
        if not checks["dialog_state"]:
            failure_reasons.append(
                f"dialog_state_mismatch: expected {expected.get('dialog_state')}, got {normalized_actual.get('dialog_state')}"
            )
    elif "dialog_state_in" in expected:
        allowed_states = [str(value) for value in expected.get("dialog_state_in") or []]
        checks["dialog_state_in"] = normalized_actual.get("dialog_state") in allowed_states
        if not checks["dialog_state_in"]:
            failure_reasons.append(
                f"dialog_state_not_allowed: expected one of {allowed_states}, got {normalized_actual.get('dialog_state')}"
            )

    if "next_dialog_state" in expected:
        checks["next_dialog_state"] = normalized_actual.get("next_dialog_state") == expected.get("next_dialog_state")
        if not checks["next_dialog_state"]:
            failure_reasons.append(
                f"next_dialog_state_mismatch: expected {expected.get('next_dialog_state')}, got {normalized_actual.get('next_dialog_state')}"
            )

    for field_name in ["budget_min", "budget_max"]:
        if field_name in expected:
            checks[field_name] = normalized_actual.get(field_name) == expected.get(field_name)
            if not checks[field_name]:
                failure_reasons.append(
                    f"{field_name}_mismatch: expected {expected.get(field_name)}, got {normalized_actual.get(field_name)}"
                )

    _check_contains(
        checks,
        failure_reasons,
        check_name="preferences_contains",
        actual_terms=normalized_actual["preferences"],
        expected_terms=expected.get("preferences_contains"),
        reason_prefix="preference_missing",
    )
    _check_not_contains(
        checks,
        failure_reasons,
        check_name="preferences_not_contains",
        actual_terms=normalized_actual["preferences"],
        forbidden_terms=expected.get("preferences_not_contains"),
        reason_prefix="forbidden_preference_present",
    )
    _check_contains(
        checks,
        failure_reasons,
        check_name="negative_preferences_contains",
        actual_terms=normalized_actual["negative_preferences"],
        expected_terms=expected.get("negative_preferences_contains"),
        reason_prefix="negative_preference_missing",
    )
    _check_not_contains(
        checks,
        failure_reasons,
        check_name="negative_preferences_not_contains",
        actual_terms=normalized_actual["negative_preferences"],
        forbidden_terms=expected.get("negative_preferences_not_contains"),
        reason_prefix="forbidden_negative_preference_present",
    )

    if "is_follow_up" in expected:
        checks["is_follow_up"] = normalized_actual.get("is_follow_up") is expected.get("is_follow_up")
        if not checks["is_follow_up"]:
            failure_reasons.append(
                f"followup_mismatch: expected {expected.get('is_follow_up')}, got {normalized_actual.get('is_follow_up')}"
            )

    if "need_clarification" in expected:
        checks["need_clarification"] = normalized_actual.get("need_clarification") is expected.get("need_clarification")
        if not checks["need_clarification"]:
            failure_reasons.append(
                f"clarification_mismatch: expected {expected.get('need_clarification')}, got {normalized_actual.get('need_clarification')}"
            )

    if "referenced_product_indices" in expected:
        checks["referenced_product_indices"] = normalized_actual["referenced_product_indices"] == list(expected.get("referenced_product_indices") or [])
        if not checks["referenced_product_indices"]:
            failure_reasons.append(
                f"referenced_indices_mismatch: expected {expected.get('referenced_product_indices')}, got {normalized_actual['referenced_product_indices']}"
            )

    if "compare_product_ids" in expected:
        checks["compare_product_ids"] = normalized_actual["compare_product_ids"] == list(expected.get("compare_product_ids") or [])
        if not checks["compare_product_ids"]:
            failure_reasons.append(
                f"compare_product_ids_mismatch: expected {expected.get('compare_product_ids')}, got {normalized_actual['compare_product_ids']}"
            )

    diagnostic_checks = _evaluate_diagnostics(case.get("diagnostic") or {}, normalized_actual)
    passed = all(checks.values()) if checks else True

    return {
        "id": case.get("id"),
        "case_group": case.get("case_group") or "ungrouped",
        "passed": passed,
        "query": case.get("query"),
        "expected": expected,
        "actual": normalized_actual,
        "checks": checks,
        "diagnostic_checks": diagnostic_checks,
        "failure_reasons": failure_reasons,
    }


def aggregate_intent_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "intent_accuracy": _accuracy(results, "intent"),
        "intent_in_accuracy": _accuracy(results, "intent_in"),
        "category_accuracy": _accuracy(results, "category"),
        "budget_min_accuracy": _accuracy(results, "budget_min"),
        "budget_max_accuracy": _accuracy(results, "budget_max"),
        "followup_accuracy": _accuracy(results, "is_follow_up"),
        "clarification_accuracy": _accuracy(results, "need_clarification"),
        "referenced_indices_accuracy": _accuracy(results, "referenced_product_indices"),
        "compare_product_ids_accuracy": _accuracy(results, "compare_product_ids"),
        "dialog_state_accuracy": _accuracy(results, "dialog_state"),
        "next_dialog_state_accuracy": _accuracy(results, "next_dialog_state"),
        "dialog_state_in_accuracy": _accuracy(results, "dialog_state_in"),
        "forbidden_preference_violation_rate": _violation_rate(
            results,
            ["preferences_not_contains", "negative_preferences_not_contains"],
        ),
    }
    metrics.update(_micro_term_metrics(results, positive_key="preferences_contains", actual_key="preferences", prefix="preference"))
    metrics.update(
        _micro_term_metrics(
            results,
            positive_key="negative_preferences_contains",
            actual_key="negative_preferences",
            prefix="negative_preference",
        )
    )
    return metrics


def aggregate_diagnostic_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_checks = [
        (result.get("diagnostic_checks") or {}).get("should_call_llm_fallback")
        for result in results
        if (result.get("diagnostic_checks") or {}).get("should_call_llm_fallback")
    ]
    fallback_accuracy = _rate(
        sum(1 for check in fallback_checks if check.get("passed")),
        len(fallback_checks),
    ) if fallback_checks else None

    multi_intent_cases = 0
    secondary_supported = 0
    secondary_gap = 0
    knowledge_supported = 0
    knowledge_gap = 0
    long_tail_cases = 0

    for result in results:
        checks = result.get("diagnostic_checks") or {}
        if "secondary_intents" in checks:
            multi_intent_cases += 1
            if checks["secondary_intents"].get("passed"):
                secondary_supported += 1
            else:
                secondary_gap += 1
        if "knowledge_questions" in checks:
            if checks["knowledge_questions"].get("passed"):
                knowledge_supported += 1
            else:
                knowledge_gap += 1
        if checks.get("long_tail_first_turn", {}).get("actual"):
            long_tail_cases += 1

    return {
        "llm_fallback_trigger_observed_accuracy": fallback_accuracy,
        "multi_intent_case_count": multi_intent_cases,
        "secondary_intent_supported_cases": secondary_supported,
        "secondary_intent_capability_gap": secondary_gap,
        "knowledge_question_supported_cases": knowledge_supported,
        "knowledge_question_capability_gap": knowledge_gap,
        "long_tail_first_turn_cases": long_tail_cases,
    }


def failure_type_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for reason in result.get("failure_reasons") or []:
            failure_type = str(reason).split(":", 1)[0]
            counts[failure_type] = counts.get(failure_type, 0) + 1
    return dict(sorted(counts.items()))


def case_group_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        group = str(result.get("case_group") or "ungrouped")
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def _normalize_actual(actual: dict[str, Any]) -> dict[str, Any]:
    budget = actual.get("budget") if isinstance(actual.get("budget"), dict) else {}
    return {
        "intent": actual.get("intent"),
        "category": actual.get("category"),
        "budget_min": actual.get("budget_min", budget.get("min")),
        "budget_max": actual.get("budget_max", budget.get("max")),
        "preferences": _list_of_str(actual.get("preferences")),
        "negative_preferences": _list_of_str(actual.get("negative_preferences")),
        "is_follow_up": bool(actual.get("is_follow_up")),
        "need_clarification": bool(actual.get("need_clarification")),
        "compare_product_ids": _list_of_str(actual.get("compare_product_ids")),
        "referenced_product_indices": _list_of_int(actual.get("referenced_product_indices")),
        "llm_fallback_attempted": bool(actual.get("llm_fallback_attempted")),
        "llm_fallback_should_call": actual.get("llm_fallback_should_call"),
        "llm_fallback_trigger_reasons": _list_of_str(actual.get("llm_fallback_trigger_reasons")),
        "dialog_state": actual.get("dialog_state"),
        "next_dialog_state": actual.get("next_dialog_state"),
        "dialog_state_reason": actual.get("dialog_state_reason"),
        "secondary_intents": _list_of_str(actual.get("secondary_intents")),
        "knowledge_questions": _list_of_str(actual.get("knowledge_questions")),
        "source": actual.get("source"),
        "reason": actual.get("reason"),
    }


def _check_contains(
    checks: dict[str, bool],
    failure_reasons: list[str],
    *,
    check_name: str,
    actual_terms: list[str],
    expected_terms: Any,
    reason_prefix: str,
) -> None:
    if expected_terms is None:
        return
    missing = [
        term for term in _list_of_str(expected_terms)
        if not _contains_term(actual_terms, term)
    ]
    checks[check_name] = not missing
    for term in missing:
        failure_reasons.append(f"{reason_prefix}: {term}")


def _check_not_contains(
    checks: dict[str, bool],
    failure_reasons: list[str],
    *,
    check_name: str,
    actual_terms: list[str],
    forbidden_terms: Any,
    reason_prefix: str,
) -> None:
    if forbidden_terms is None:
        return
    present = [
        term for term in _list_of_str(forbidden_terms)
        if _contains_term(actual_terms, term)
    ]
    checks[check_name] = not present
    for term in present:
        failure_reasons.append(f"{reason_prefix}: {term}")


def _evaluate_diagnostics(diagnostic: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if "should_call_llm_fallback" in diagnostic:
        expected = bool(diagnostic.get("should_call_llm_fallback"))
        actual_value = bool(
            actual.get("llm_fallback_should_call")
            if actual.get("llm_fallback_should_call") is not None
            else actual.get("llm_fallback_attempted")
        )
        checks["should_call_llm_fallback"] = {
            "expected": expected,
            "actual": actual_value,
            "passed": expected == actual_value,
            "affects_case_pass": False,
        }
    if "expected_secondary_intents" in diagnostic:
        expected = _list_of_str(diagnostic.get("expected_secondary_intents"))
        actual_values = _list_of_str(actual.get("secondary_intents"))
        checks["secondary_intents"] = {
            "expected": expected,
            "actual": actual_values,
            "passed": all(_contains_term(actual_values, item) for item in expected),
            "affects_case_pass": False,
        }
    if "knowledge_questions_contains" in diagnostic:
        expected = _list_of_str(diagnostic.get("knowledge_questions_contains"))
        actual_values = _list_of_str(actual.get("knowledge_questions"))
        checks["knowledge_questions"] = {
            "expected": expected,
            "actual": actual_values,
            "passed": all(_contains_term(actual_values, item) for item in expected),
            "affects_case_pass": False,
        }
    if "long_tail_first_turn" in diagnostic:
        checks["long_tail_first_turn"] = {
            "expected": bool(diagnostic.get("long_tail_first_turn")),
            "actual": bool(diagnostic.get("long_tail_first_turn")),
            "passed": True,
            "affects_case_pass": False,
        }
    return checks


def _accuracy(results: list[dict[str, Any]], check_name: str) -> float | None:
    applicable = [
        result for result in results if check_name in (result.get("checks") or {})
    ]
    if not applicable:
        return None
    return _rate(
        sum(1 for result in applicable if (result.get("checks") or {}).get(check_name) is True),
        len(applicable),
    )


def _violation_rate(results: list[dict[str, Any]], check_names: list[str]) -> float | None:
    applicable = [
        result
        for result in results
        for check_name in check_names
        if check_name in (result.get("checks") or {})
    ]
    if not applicable:
        return None
    failed = 0
    total = 0
    for result in results:
        checks = result.get("checks") or {}
        for check_name in check_names:
            if check_name in checks:
                total += 1
                if checks[check_name] is False:
                    failed += 1
    return _rate(failed, total)


def _micro_term_metrics(
    results: list[dict[str, Any]],
    *,
    positive_key: str,
    actual_key: str,
    prefix: str,
) -> dict[str, float | None]:
    true_positive = 0
    false_negative = 0
    false_positive = 0
    applicable = False
    for result in results:
        expected_terms = _list_of_str((result.get("expected") or {}).get(positive_key))
        if not expected_terms:
            continue
        applicable = True
        actual_terms = _list_of_str((result.get("actual") or {}).get(actual_key))
        for term in expected_terms:
            if _contains_term(actual_terms, term):
                true_positive += 1
            else:
                false_negative += 1
        for term in actual_terms:
            if not _contains_term(expected_terms, term):
                false_positive += 1

    if not applicable:
        return {
            f"{prefix}_precision": None,
            f"{prefix}_recall": None,
            f"{prefix}_f1": None,
        }

    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    f1 = None if precision is None or recall is None or precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def _contains_term(values: list[str], term: str) -> bool:
    needle = str(term).lower()
    return any(needle == value.lower() or needle in value.lower() for value in values)


def _list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _list_of_int(value: Any) -> list[int]:
    items = value if isinstance(value, list) else ([] if value is None else [value])
    output: list[int] = []
    for item in items:
        try:
            output.append(int(item))
        except (TypeError, ValueError):
            continue
    return output


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)
