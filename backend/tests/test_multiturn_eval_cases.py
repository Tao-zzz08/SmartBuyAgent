from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "backend" / "tests"

for path in [SCRIPTS_DIR, TESTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_query_understanding_eval import load_suite_cases, run_case  # noqa: E402
from test_query_understanding_regression_eval import RegressionFakeClient  # noqa: E402


def test_multiturn_eval_cases_are_query_understanding_2_style() -> None:
    cases = load_suite_cases("multiturn")

    assert {
        "phone_budget_ladder",
        "compare_then_return_to_guide",
        "category_switch_chain",
    } <= {case["id"] for case in cases}
    for case in cases:
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
        }
    ]

    results = [run_case(RegressionFakeClient(), case) for case in cases]

    assert all(result["passed"] for result in results)
