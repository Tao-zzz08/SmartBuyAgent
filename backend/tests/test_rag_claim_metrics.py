from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag_claim_metrics import (  # noqa: E402
    aggregate_rag_claim_metrics,
    evaluate_claim_support,
)


def test_supported_claim_is_grounded() -> None:
    result = evaluate_claim_support(
        answer="手机拍照主要看传感器和防抖。",
        citations=[{"content_preview": "拍照需要关注传感器尺寸、防抖和影像算法。"}],
        expected_claims=[
            {
                "id": "sensor",
                "answer_terms_any": ["传感器"],
                "citation_terms_any": ["传感器"],
            }
        ],
    )

    assert result["grounded"] is True
    assert result["claim_support_rate"] == 1.0
    assert result["supported_claims"] == 1


def test_unsupported_claim_is_not_grounded() -> None:
    result = evaluate_claim_support(
        answer="手机拍照主要看传感器和独家AI夜视芯片。",
        citations=[{"content_preview": "拍照需要关注传感器尺寸和防抖。"}],
        expected_claims=[
            {
                "id": "ai_chip",
                "answer_terms_any": ["独家AI夜视芯片"],
                "citation_terms_any": ["独家AI夜视芯片"],
            }
        ],
    )

    assert result["grounded"] is False
    assert result["unsupported_claims"] == 1
    assert result["unsupported_claim_rate"] == 1.0
    assert result["claim_results"][0]["failure_reason"] == "citation_support_missing"


def test_required_missing_claim_fails_grounding() -> None:
    result = evaluate_claim_support(
        answer="手机拍照可以关注传感器。",
        citations=[{"content_preview": "拍照需要关注传感器尺寸、防抖和影像算法。"}],
        expected_claims=[
            {
                "id": "stabilization",
                "answer_terms_any": ["防抖"],
                "citation_terms_any": ["防抖"],
                "required": True,
            }
        ],
    )

    assert result["grounded"] is False
    assert result["missing_required_claims"] == 1
    assert result["triggered_claims"] == 0


def test_citation_required_for_terms_records_violation() -> None:
    result = evaluate_claim_support(
        answer="这款手机支持快充。",
        citations=[{"content_preview": "手机续航可以关注电池容量。"}],
        citation_required_for_terms=["快充"],
    )

    assert result["grounded"] is False
    assert result["hallucination_violation_count"] == 1
    assert result["violations"][0]["type"] == "citation_required_term_missing"


def test_unsupported_answer_terms_records_violation() -> None:
    result = evaluate_claim_support(
        answer="这款手机值得立即购买。",
        citations=[{"content_preview": "手机拍照需要关注传感器。"}],
        unsupported_answer_terms=["立即购买"],
    )

    assert result["grounded"] is False
    assert result["hallucination_violation_count"] == 1
    assert result["violations"][0]["type"] == "unsupported_answer_term"


def test_aggregate_rag_claim_metrics() -> None:
    metrics = aggregate_rag_claim_metrics(
        [
            {
                "grounded": True,
                "triggered_claims": 2,
                "supported_claims": 2,
                "unsupported_claims": 0,
                "missing_required_claims": 0,
                "hallucination_violation_count": 0,
            },
            {
                "grounded": False,
                "triggered_claims": 2,
                "supported_claims": 1,
                "unsupported_claims": 1,
                "missing_required_claims": 1,
                "hallucination_violation_count": 1,
            },
        ]
    )

    assert metrics["claim_support_rate"] == 0.75
    assert metrics["citation_coverage_rate"] == 0.75
    assert metrics["unsupported_claim_rate"] == 0.25
    assert metrics["grounded_answer_rate"] == 0.5
    assert metrics["evaluated_claim_cases"] == 2
    assert metrics["missing_required_claims"] == 1
