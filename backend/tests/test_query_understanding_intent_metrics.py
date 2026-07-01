from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from query_understanding_intent_metrics import (  # noqa: E402
    aggregate_diagnostic_metrics,
    aggregate_intent_metrics,
    evaluate_intent_case,
)


def test_evaluate_intent_case_passes_when_fields_match() -> None:
    result = evaluate_intent_case(
        case={
            "id": "ok",
            "case_group": "explicit_shopping",
            "query": "phone",
            "expected": {
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": 4000,
                "preferences_contains": ["拍照"],
            },
        },
        actual={
            "intent": "shopping_guide",
            "category": "phone",
            "budget": {"max": 4000},
            "preferences": ["拍照"],
        },
    )

    assert result["passed"] is True
    assert result["checks"]["intent"] is True
    assert result["checks"]["category"] is True
    assert result["checks"]["budget_max"] is True


def test_preferences_contains_missing_fails_with_reason() -> None:
    result = evaluate_intent_case(
        case={
            "id": "missing_preference",
            "expected": {"preferences_contains": ["续航"]},
        },
        actual={"preferences": ["拍照"]},
    )

    assert result["passed"] is False
    assert "preference_missing: 续航" in result["failure_reasons"]


def test_preferences_not_contains_fails_when_forbidden_present() -> None:
    result = evaluate_intent_case(
        case={
            "id": "forbidden_preference",
            "expected": {"preferences_not_contains": ["治疗"]},
        },
        actual={"preferences": ["治疗", "保湿"]},
    )

    assert result["passed"] is False
    assert "forbidden_preference_present: 治疗" in result["failure_reasons"]


def test_negative_preferences_contains_missing_fails() -> None:
    result = evaluate_intent_case(
        case={
            "id": "negative_missing",
            "expected": {"negative_preferences_contains": ["苹果"]},
        },
        actual={"negative_preferences": []},
    )

    assert result["passed"] is False
    assert "negative_preference_missing: 苹果" in result["failure_reasons"]


def test_intent_in_allows_multiple_valid_intents() -> None:
    result = evaluate_intent_case(
        case={
            "id": "intent_in",
            "expected": {"intent_in": ["shopping_guide", "clarification"]},
        },
        actual={"intent": "clarification"},
    )

    assert result["passed"] is True
    assert result["checks"]["intent_in"] is True


def test_compare_reference_fields_require_exact_match() -> None:
    result = evaluate_intent_case(
        case={
            "id": "compare",
            "expected": {
                "referenced_product_indices": [1, 2],
                "compare_product_ids": ["p1", "p2"],
            },
        },
        actual={
            "referenced_product_indices": [1, 2],
            "compare_product_ids": ["p1", "p2"],
        },
    )

    assert result["passed"] is True
    assert result["checks"]["referenced_product_indices"] is True
    assert result["checks"]["compare_product_ids"] is True


def test_dialog_state_fields_can_pass() -> None:
    result = evaluate_intent_case(
        case={
            "id": "dialog_state_ok",
            "expected": {
                "dialog_state": "showing_products",
                "next_dialog_state": "comparing_products",
            },
        },
        actual={
            "dialog_state": "showing_products",
            "next_dialog_state": "comparing_products",
        },
    )

    assert result["passed"] is True
    assert result["checks"]["dialog_state"] is True
    assert result["checks"]["next_dialog_state"] is True


def test_dialog_state_mismatch_fails_with_reason() -> None:
    result = evaluate_intent_case(
        case={
            "id": "dialog_state_bad",
            "expected": {
                "dialog_state": "showing_products",
                "next_dialog_state": "comparing_products",
            },
        },
        actual={
            "dialog_state": "idle",
            "next_dialog_state": "showing_products",
        },
    )

    assert result["passed"] is False
    assert any("dialog_state_mismatch" in reason for reason in result["failure_reasons"])
    assert any("next_dialog_state_mismatch" in reason for reason in result["failure_reasons"])


def test_dialog_state_in_allows_any_listed_state() -> None:
    result = evaluate_intent_case(
        case={
            "id": "dialog_state_in",
            "expected": {
                "dialog_state_in": ["showing_products", "comparing_products"],
            },
        },
        actual={"dialog_state": "comparing_products"},
    )

    assert result["passed"] is True
    assert result["checks"]["dialog_state_in"] is True


def test_diagnostic_checks_do_not_affect_passed() -> None:
    result = evaluate_intent_case(
        case={
            "id": "diagnostic_only",
            "expected": {"intent": "shopping_guide"},
            "diagnostic": {
                "should_call_llm_fallback": True,
                "expected_secondary_intents": ["product_knowledge"],
            },
        },
        actual={
            "intent": "shopping_guide",
            "llm_fallback_attempted": False,
            "secondary_intents": [],
        },
    )

    assert result["passed"] is True
    assert result["diagnostic_checks"]["should_call_llm_fallback"]["passed"] is False
    assert result["diagnostic_checks"]["should_call_llm_fallback"]["affects_case_pass"] is False


def test_diagnostic_prefers_theoretical_fallback_should_call() -> None:
    result = evaluate_intent_case(
        case={
            "id": "diagnostic_should_call",
            "expected": {"intent": "shopping_guide"},
            "diagnostic": {"should_call_llm_fallback": True},
        },
        actual={
            "intent": "shopping_guide",
            "llm_fallback_attempted": False,
            "llm_fallback_should_call": True,
        },
    )

    assert result["passed"] is True
    assert result["diagnostic_checks"]["should_call_llm_fallback"]["actual"] is True
    assert result["diagnostic_checks"]["should_call_llm_fallback"]["passed"] is True


def test_aggregate_metrics_compute_field_accuracy_and_preference_f1() -> None:
    results = [
        evaluate_intent_case(
            case={
                "id": "one",
                "expected": {
                    "intent": "shopping_guide",
                    "category": "phone",
                    "budget_max": 4000,
                    "preferences_contains": ["拍照", "续航"],
                },
            },
            actual={
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": 4000,
                "preferences": ["拍照", "性能"],
            },
        ),
        evaluate_intent_case(
            case={
                "id": "two",
                "expected": {
                    "intent": "compare",
                    "category": "phone",
                    "budget_max": 5000,
                    "preferences_contains": ["拍照"],
                },
            },
            actual={
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": 5000,
                "preferences": ["拍照"],
            },
        ),
    ]

    metrics = aggregate_intent_metrics(results)

    assert metrics["intent_accuracy"] == 0.5
    assert metrics["category_accuracy"] == 1.0
    assert metrics["budget_max_accuracy"] == 1.0
    assert metrics["dialog_state_accuracy"] is None
    assert metrics["next_dialog_state_accuracy"] is None
    assert metrics["preference_precision"] == 0.6667
    assert metrics["preference_recall"] == 0.6667
    assert metrics["preference_f1"] == 0.6667


def test_aggregate_dialog_state_metrics() -> None:
    results = [
        evaluate_intent_case(
            case={
                "id": "one",
                "expected": {
                    "dialog_state": "showing_products",
                    "next_dialog_state": "comparing_products",
                },
            },
            actual={
                "dialog_state": "showing_products",
                "next_dialog_state": "comparing_products",
            },
        ),
        evaluate_intent_case(
            case={
                "id": "two",
                "expected": {
                    "dialog_state": "showing_products",
                    "next_dialog_state": "showing_products",
                },
            },
            actual={
                "dialog_state": "idle",
                "next_dialog_state": "showing_products",
            },
        ),
    ]

    metrics = aggregate_intent_metrics(results)

    assert metrics["dialog_state_accuracy"] == 0.5
    assert metrics["next_dialog_state_accuracy"] == 1.0


def test_aggregate_metrics_return_none_when_no_applicable_fields() -> None:
    result = evaluate_intent_case(
        case={"id": "empty", "expected": {}},
        actual={},
    )

    metrics = aggregate_intent_metrics([result])

    assert metrics["intent_accuracy"] is None
    assert metrics["category_accuracy"] is None
    assert metrics["budget_max_accuracy"] is None
    assert metrics["dialog_state_accuracy"] is None
    assert metrics["next_dialog_state_accuracy"] is None
    assert metrics["preference_precision"] is None
    assert metrics["negative_preference_precision"] is None


def test_aggregate_diagnostic_metrics_count_capability_gaps() -> None:
    result = evaluate_intent_case(
        case={
            "id": "multi",
            "diagnostic": {
                "should_call_llm_fallback": True,
                "expected_secondary_intents": ["product_knowledge"],
                "knowledge_questions_contains": ["为什么像素高不一定拍照好"],
                "long_tail_first_turn": True,
            },
        },
        actual={
            "llm_fallback_attempted": False,
            "secondary_intents": [],
            "knowledge_questions": [],
        },
    )

    metrics = aggregate_diagnostic_metrics([result])

    assert metrics["llm_fallback_trigger_observed_accuracy"] == 0.0
    assert metrics["multi_intent_case_count"] == 1
    assert metrics["secondary_intent_capability_gap"] == 1
    assert metrics["knowledge_question_capability_gap"] == 1
    assert metrics["long_tail_first_turn_cases"] == 1
