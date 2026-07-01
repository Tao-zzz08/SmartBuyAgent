from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_ab_strategies import run_strategy  # noqa: E402


def test_structured_filter_only_filters_category_price_and_excluded_brand() -> None:
    results = run_strategy(
        "structured_filter_only",
        _products(),
        {
            "query": "camera phone",
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
                "exclude_brands": ["Apple"],
            },
        },
        top_k=5,
    )

    ids = [product["id"] for product in results]
    assert "phone_camera" in ids
    assert "phone_over_budget" not in ids
    assert "phone_apple" not in ids
    assert "shoes_commute" not in ids


def test_lexical_keyword_ranks_keyword_match_first() -> None:
    results = run_strategy(
        "lexical_keyword",
        _products(),
        {
            "query": "camera phone",
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
            },
        },
        top_k=5,
    )

    assert [product["id"] for product in results][:2] == [
        "phone_camera",
        "phone_apple",
    ]


def test_hybrid_filter_keyword_filters_before_ranking() -> None:
    products = [
        *_products(),
        {
            "id": "shoes_camera_word",
            "category": "shoes",
            "title": "Camera keyword shoes",
            "brand": "Demo",
            "price": 499,
            "tags": ["camera", "commute"],
            "description": "Contains many camera terms but wrong category.",
        },
    ]

    results = run_strategy(
        "hybrid_filter_keyword",
        products,
        {
            "query": "camera phone",
            "structured_filters": {
                "category": "phone",
                "preferences": ["camera"],
            },
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
            },
        },
        top_k=5,
    )

    ids = [product["id"] for product in results]
    assert "phone_camera" in ids
    assert "shoes_camera_word" not in ids


def test_hybrid_plus_rerank_prioritizes_preference_match() -> None:
    results = run_strategy(
        "hybrid_plus_rerank",
        _products(),
        {
            "query": "recommend phone",
            "structured_filters": {
                "category": "phone",
                "preferences": ["camera"],
            },
            "hard_filters": {
                "category": "phone",
                "price_lte": 5000,
            },
        },
        top_k=5,
    )

    assert results[0]["id"] == "phone_camera"


def test_forbidden_terms_are_removed_from_results() -> None:
    results = run_strategy(
        "structured_filter_only",
        _products(),
        {
            "query": "commute shoes",
            "hard_filters": {
                "category": "shoes",
                "price_lte": 800,
                "forbidden_terms": ["high heel"],
            },
        },
        top_k=5,
    )

    ids = [product["id"] for product in results]
    assert "shoes_commute" in ids
    assert "shoes_high_heel" not in ids


def test_strategy_handles_empty_catalog() -> None:
    assert run_strategy(
        "hybrid_plus_rerank",
        [],
        {"query": "camera phone"},
        top_k=5,
    ) == []


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
            "title": "Generic phone",
            "brand": "Honor",
            "price": 4299,
            "tags": ["battery"],
            "description": "A balanced phone.",
            "rating": 4.5,
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
            "id": "phone_apple",
            "category": "phone",
            "title": "Apple camera phone",
            "brand": "Apple",
            "price": 4999,
            "tags": ["camera"],
            "description": "Apple brand phone for exclusion tests.",
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
