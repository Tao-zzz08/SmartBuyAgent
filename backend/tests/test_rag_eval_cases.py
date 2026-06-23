from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "backend" / "tests"

for path in [SCRIPTS_DIR, TESTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_query_understanding_eval import load_suite_cases, run_eval  # noqa: E402
from test_query_understanding_regression_eval import RegressionFakeClient  # noqa: E402


def test_rag_eval_cases_include_groundedness_and_safety_cases() -> None:
    cases = load_suite_cases("rag")

    assert len(cases) >= 9
    assert {
        "phone_camera_knowledge_grounded_answer",
        "skincare_acne_safety_rag",
        "citation_must_come_from_knowledge_chunks",
        "unknown_knowledge_should_not_fabricate",
    } <= {case["id"] for case in cases}


def test_rag_eval_cases_have_meaningful_expectations() -> None:
    cases = load_suite_cases("rag")

    for case in cases:
        expect = case["expect"]
        assert "answer_forbidden" in expect, case["id"]
        assert (
            "citation_must_have_fields" in expect
            or expect.get("allow_no_citations") is True
        ), case["id"]
        assert (
            "answer_must_include_any" in expect
            or "citation_keywords_any" in expect
            or expect.get("must_not_fabricate") is True
        ), case["id"]


def test_core_rag_eval_cases_pass_with_fake_client() -> None:
    cases = [
        case
        for case in load_suite_cases("rag")
        if case["id"]
        in {
            "phone_camera_knowledge_grounded_answer",
            "skincare_acne_safety_rag",
            "citation_must_come_from_knowledge_chunks",
            "unknown_knowledge_should_not_fabricate",
        }
    ]

    output = run_eval(cases, client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 0
    assert output["summary"]["passed_cases"] == len(cases)
