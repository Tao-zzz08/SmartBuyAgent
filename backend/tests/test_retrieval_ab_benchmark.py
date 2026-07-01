from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_ab_benchmark import (  # noqa: E402
    render_markdown,
    run_benchmark,
    write_outputs,
)
from retrieval_ab_strategies import SUPPORTED_STRATEGIES  # noqa: E402


def test_retrieval_ab_benchmark_runs_all_strategies_and_summarizes() -> None:
    report = run_benchmark(
        cases=_cases(),
        products=_products(),
        strategies=list(SUPPORTED_STRATEGIES),
        top_k=5,
        baseline="structured_filter_only",
    )

    summary = report["summary"]
    assert set(summary["strategies"]) == set(SUPPORTED_STRATEGIES)
    assert "best_by_metric" in summary
    assert "deltas_vs_baseline" in summary
    assert "strategy_win_counts" in summary
    assert "category_breakdown" in summary
    assert "phone" in summary["category_breakdown"]
    assert "shoes" in summary["category_breakdown"]
    assert summary["strategies"]["hybrid_plus_rerank"]["evaluated_cases"] == len(_cases())
    assert summary["strategies"]["hybrid_plus_rerank"]["evaluated_ranking_cases"] >= 1
    assert len(report["results"]) == len(_cases()) * len(SUPPORTED_STRATEGIES)

    json.dumps(report, ensure_ascii=False)


def test_retrieval_ab_markdown_contains_expected_sections() -> None:
    report = run_benchmark(
        cases=_cases(),
        products=_products(),
        strategies=list(SUPPORTED_STRATEGIES),
        top_k=5,
        baseline="structured_filter_only",
    )

    markdown = render_markdown(report)

    assert "# Retrieval A/B Strategy Benchmark" in markdown
    assert "## Strategy Summary" in markdown
    assert "## Best By Metric" in markdown
    assert "## Deltas vs Baseline" in markdown
    assert "## Category Breakdown" in markdown
    assert "## Failed/Empty Cases" in markdown
    assert "hybrid_plus_rerank" in markdown


def test_retrieval_ab_write_outputs_creates_files(tmp_path: Path) -> None:
    report = run_benchmark(
        cases=_cases(),
        products=_products(),
        strategies=["structured_filter_only", "hybrid_plus_rerank"],
        top_k=5,
        baseline="structured_filter_only",
    )
    markdown_path = tmp_path / "reports" / "retrieval-ab-report.md"
    details_path = tmp_path / "reports" / "retrieval-ab-details.json"

    write_outputs(report, output_path=markdown_path, details_path=details_path)

    assert "Strategy Summary" in markdown_path.read_text(encoding="utf-8")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert "summary" in details
    assert "strategies" in details["summary"]
    assert details["summary"]["deltas_vs_baseline"]


def _cases() -> list[dict]:
    return [
        {
            "id": "phone_camera_under_5000",
            "type": "product_retrieval",
            "query": "camera phone under 5000",
            "structured_filters": {
                "category": "phone",
                "budget_max": 5000,
                "preferences": ["camera"],
                "negative_preferences": [],
            },
            "gold_relevance": {
                "phone_camera": 3,
                "phone_generic": 1,
            },
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
            },
        },
        {
            "id": "phone_camera_without_apple",
            "type": "product_retrieval",
            "query": "camera phone no apple",
            "structured_filters": {
                "category": "phone",
                "budget_max": 5000,
                "preferences": ["camera"],
                "negative_preferences": ["Apple"],
            },
            "gold_relevance": {
                "phone_camera": 3,
            },
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
                "exclude_brands": ["Apple"],
            },
        },
        {
            "id": "shoes_commute_no_high_heel",
            "type": "product_retrieval",
            "query": "commute lightweight shoes",
            "structured_filters": {
                "category": "shoes",
                "budget_max": 800,
                "preferences": ["commute", "lightweight"],
                "negative_preferences": ["high heel"],
            },
            "gold_relevance": {
                "shoes_commute": 3,
            },
            "hard_filters": {
                "category": "shoes",
                "price_lte": 800,
                "forbidden_terms": ["high heel"],
            },
        },
    ]


def _products() -> list[dict]:
    return [
        {
            "id": "phone_camera",
            "category": "phone",
            "title": "Xiaomi camera phone",
            "brand": "Xiaomi",
            "price": 3999,
            "tags": ["camera", "image"],
            "description": "A phone for camera and daily usage.",
            "rating": 4.7,
        },
        {
            "id": "phone_generic",
            "category": "phone",
            "title": "Balanced phone",
            "brand": "Honor",
            "price": 4299,
            "tags": ["battery"],
            "description": "A balanced phone.",
            "rating": 4.5,
        },
        {
            "id": "phone_apple",
            "category": "phone",
            "title": "Apple camera phone",
            "brand": "Apple",
            "price": 4999,
            "tags": ["camera"],
            "description": "Apple brand phone for exclusion tests.",
        },
        {
            "id": "phone_over_budget",
            "category": "phone",
            "title": "Premium camera phone",
            "brand": "Huawei",
            "price": 6999,
            "tags": ["camera"],
            "description": "Strong camera but above budget.",
        },
        {
            "id": "shoes_commute",
            "category": "shoes",
            "title": "Commute lightweight shoes",
            "brand": "Anta",
            "price": 499,
            "tags": ["commute", "lightweight"],
            "description": "Daily commute shoes.",
        },
        {
            "id": "shoes_high_heel",
            "category": "shoes",
            "title": "High heel commute shoes",
            "brand": "Demo",
            "price": 599,
            "tags": ["commute", "high heel"],
            "description": "Contains high heel term for negative filtering.",
        },
    ]
