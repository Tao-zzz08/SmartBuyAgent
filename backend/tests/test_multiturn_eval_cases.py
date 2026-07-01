from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "backend" / "tests"

for path in [SCRIPTS_DIR, TESTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_query_understanding_eval import load_suite_cases, run_case, run_eval  # noqa: E402
from test_query_understanding_regression_eval import RegressionFakeClient  # noqa: E402


def test_multiturn_eval_cases_are_query_understanding_2_style() -> None:
    cases = load_suite_cases("multiturn")

    assert {
        "phone_budget_ladder",
        "compare_then_return_to_guide",
        "category_switch_chain",
        "compare_without_context_clarifies",
    } <= {case["id"] for case in cases}
    for case in cases:
        assert case.get("type") == "multiturn"
        assert case.get("task_type")
        assert isinstance(case.get("session_expect"), dict)
        for turn in case["turns"]:
            expect = turn["expect"]
            assert "route" in expect or "required_trace_steps" in expect


def test_core_multiturn_eval_cases_pass_with_fake_client() -> None:
    cases = [
        case
        for case in load_suite_cases("multiturn")
        if case["id"]
        in {
            "phone_budget_ladder",
            "compare_then_return_to_guide",
            "category_switch_chain",
            "compare_without_context_clarifies",
        }
    ]

    results = [run_case(RegressionFakeClient(), case) for case in cases]

    assert all(result["passed"] for result in results)
    assert all("session_metrics" in result for result in results)
    assert all(result["session_metrics"]["session_success"] is True for result in results)


def test_multiturn_eval_summary_includes_session_metrics() -> None:
    cases = load_suite_cases("multiturn")

    output = run_eval(cases, client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 0
    assert "metrics" in output["summary"]
    metrics = output["summary"]["metrics"]
    assert metrics["session_success_rate"] == 1.0
    assert metrics["evaluated_sessions"] == len(cases)
    assert metrics["context_carryover_accuracy"] is not None
    assert metrics["category_switch_accuracy"] is not None
    assert metrics["compare_resolution_accuracy"] is not None
    assert metrics["clarification_accuracy"] is not None
    assert metrics["route_stability_rate"] is not None
    assert all("session_metrics" in result for result in output["results"])


def test_legacy_multiturn_case_without_session_expect_still_runs() -> None:
    source_case = next(
        case for case in load_suite_cases("multiturn") if case["id"] == "phone_budget_ladder"
    )
    source_turn = source_case["turns"][0]
    case = {
        "id": "legacy_multiturn_without_session_expect",
        "description": "Legacy multiturn case should still run turn-level eval.",
        "turns": [
            {
                "user": source_turn["user"],
                "expect": {
                    "intent": "shopping_guide",
                    "route": "shopping_guide",
                    "category": "phone",
                    "budget_max": 3000,
                    "preferences_contains": source_turn["expect"]["preferences_contains"],
                    "product_cards_category": "phone",
                },
            }
        ],
    }

    output = run_eval([case], client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 0
    assert output["summary"]["passed_cases"] == 1
    assert output["results"][0]["passed"] is True
    assert "session_metrics" not in output["results"][0]
    assert "metrics" not in output["summary"]
