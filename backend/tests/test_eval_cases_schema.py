from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def test_all_eval_case_files_have_valid_schema() -> None:
    files = [
        "query_understanding_regression_cases.json",
        "retrieval_eval_cases.json",
        "rag_eval_cases.json",
        "multiturn_eval_cases.json",
        "grounding_guard_eval_cases.json",
    ]

    for filename in files:
        cases = json.loads((EVAL_DIR / filename).read_text(encoding="utf-8"))
        assert isinstance(cases, list), filename
        assert cases, filename
        for case in cases:
            assert case.get("id"), filename
            assert case.get("description") or case.get("query"), case.get("id")
            if case.get("turns"):
                for turn in case["turns"]:
                    assert turn.get("user"), case["id"]
                    assert isinstance(turn.get("expect", {}), dict), case["id"]
            else:
                assert case.get("query") or case.get("answer") or case.get("context"), case["id"]
                assert isinstance(case.get("expect", {}), dict), case["id"]
            if "gold_relevance" in case:
                assert isinstance(case["gold_relevance"], dict), case["id"]
                assert case["gold_relevance"], case["id"]
                for product_id, relevance in case["gold_relevance"].items():
                    assert product_id, case["id"]
                    assert isinstance(relevance, (int, float)), case["id"]
                    assert relevance > 0, case["id"]
            if "hard_filters" in case:
                assert isinstance(case["hard_filters"], dict), case["id"]
                allowed_hard_filters = {
                    "category",
                    "price_lte",
                    "price_gte",
                    "exclude_brands",
                    "exclude_product_ids",
                    "required_in_stock",
                    "forbidden_terms",
                }
                assert set(case["hard_filters"]) <= allowed_hard_filters, case["id"]


def test_eval_case_ids_are_unique_per_file() -> None:
    for path in EVAL_DIR.glob("*_eval_cases.json"):
        cases = json.loads(path.read_text(encoding="utf-8"))
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)), path.name
