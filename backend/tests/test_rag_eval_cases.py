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


def test_rag_eval_summary_includes_claim_support_metrics() -> None:
    cases = [
        case
        for case in load_suite_cases("rag")
        if case["id"]
        in {
            "phone_camera_knowledge_grounded_answer",
            "phone_battery_knowledge_grounded_answer",
            "shoes_size_grounded_answer",
        }
    ]

    output = run_eval(cases, client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 0
    assert "metrics" in output["summary"]
    metrics = output["summary"]["metrics"]
    assert metrics["claim_support_rate"] == 1.0
    assert metrics["grounded_answer_rate"] == 1.0
    assert metrics["evaluated_claim_cases"] == len(cases)
    assert any(
        "claim_metrics" in turn
        for result in output["results"]
        for turn in result["turns"]
    )
    assert all("claim_metrics" in result for result in output["results"])


def test_rag_eval_fails_when_answer_claim_lacks_citation_support() -> None:
    source_case = next(
        case
        for case in load_suite_cases("rag")
        if case["id"] == "phone_camera_knowledge_grounded_answer"
    )
    case = {
        "id": "unsupported_claim_case",
        "description": "Answer claim should fail when citations do not support it.",
        "query": source_case["query"],
        "expected_category": "phone",
        "expect": {
            "intent": "product_knowledge",
            "category": "phone",
            "min_citations": 1,
            "expected_claims": [
                {
                    "id": "unsupported_camera_claim",
                    "answer_terms_any": source_case["expect"]["answer_must_include_any"],
                    "citation_terms_any": ["definitely_not_in_fixture_citations"],
                }
            ],
        },
    }

    output = run_eval([case], client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 1
    assert output["results"][0]["passed"] is False
    assert output["results"][0]["turns"][0]["claim_metrics"]["grounded"] is False
    reasons = output["results"][0]["failure_reasons"][0]["reasons"]
    assert any("citation_support_missing" in reason for reason in reasons)
