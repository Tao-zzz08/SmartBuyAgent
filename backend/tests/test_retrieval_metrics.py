from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    compute_filter_compliance,
    compute_retrieval_metrics,
)


def test_compute_retrieval_metrics_recall_at_k() -> None:
    metrics = compute_retrieval_metrics(
        ["a", "b", "c"],
        {"b": 3, "d": 2},
        k=3,
    )

    assert metrics["recall_at_3"] == 0.5


def test_compute_retrieval_metrics_mrr_at_k() -> None:
    metrics = compute_retrieval_metrics(
        ["a", "b", "c"],
        {"b": 3},
        k=3,
    )

    assert metrics["mrr_at_3"] == 0.5


def test_compute_retrieval_metrics_ndcg_at_k() -> None:
    ideal = compute_retrieval_metrics(
        ["a", "b"],
        {"a": 3, "b": 2},
        k=2,
    )
    imperfect = compute_retrieval_metrics(
        ["b", "a"],
        {"a": 3, "b": 2},
        k=2,
    )

    assert ideal["ndcg_at_2"] == 1.0
    assert imperfect["ndcg_at_2"] is not None
    assert 0 < imperfect["ndcg_at_2"] < 1


def test_compute_retrieval_metrics_without_gold_returns_none() -> None:
    metrics = compute_retrieval_metrics(["a", "b"], {}, k=5)

    assert metrics["recall_at_5"] is None
    assert metrics["ndcg_at_5"] is None
    assert metrics["mrr_at_5"] is None


def test_compute_filter_compliance_detects_excluded_brand() -> None:
    products = [
        {
            "id": "phone_xiaomi_14",
            "category": "phone",
            "brand": "小米",
            "price": 3999,
        },
        {
            "id": "phone_iphone_15",
            "category": "phone",
            "brand": "Apple",
            "price": 4999,
        },
    ]

    result = compute_filter_compliance(
        products,
        {
            "category": "phone",
            "price_lte": 5000,
            "exclude_brands": ["Apple"],
        },
    )

    assert result["filter_compliance"] is False
    assert result["filter_violation_count"] == 1
    assert result["filter_violations"][0]["type"] == "exclude_brands"


def test_aggregate_retrieval_metrics_skips_null_ranking_values() -> None:
    summary = aggregate_retrieval_metrics(
        [
            {
                "type": "product_retrieval",
                "result_count": 2,
                "negative_preference_violations": 0,
                "metrics": {
                    "recall_at_5": 1.0,
                    "ndcg_at_5": 0.8,
                    "mrr_at_5": 1.0,
                    "filter_compliance": True,
                    "latency_ms": 20,
                },
            },
            {
                "type": "product_retrieval",
                "result_count": 0,
                "negative_preference_violations": 1,
                "metrics": {
                    "recall_at_5": None,
                    "ndcg_at_5": None,
                    "mrr_at_5": None,
                    "filter_compliance": False,
                    "latency_ms": 40,
                },
            },
        ]
    )

    assert summary["recall_at_5"] == 1.0
    assert summary["ndcg_at_5"] == 0.8
    assert summary["mrr_at_5"] == 1.0
    assert summary["filter_compliance_rate"] == 0.5
    assert summary["negative_preference_violation_rate"] == 0.5
    assert summary["empty_rate"] == 0.5
    assert summary["latency_p50_ms"] == 30
    assert summary["evaluated_ranking_cases"] == 1
