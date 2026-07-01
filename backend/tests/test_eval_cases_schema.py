from __future__ import annotations

from pathlib import Path
import hashlib
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
            _assert_no_placeholder_tokens(case, case["id"])
            assert case.get("description") or case.get("query"), case.get("id")
            if "task_type" in case:
                assert isinstance(case["task_type"], str), case["id"]
            if "session_expect" in case:
                assert isinstance(case["session_expect"], dict), case["id"]
                checks = case["session_expect"].get("checks")
                if checks is not None:
                    assert isinstance(checks, list), case["id"]
                    assert all(isinstance(check, str) for check in checks), case["id"]
            if case.get("turns"):
                for turn in case["turns"]:
                    assert turn.get("user"), case["id"]
                    assert isinstance(turn.get("expect", {}), dict), case["id"]
                    _assert_valid_turn_expect(turn.get("expect", {}), case["id"])
            else:
                assert case.get("query") or case.get("answer") or case.get("context"), case["id"]
                assert isinstance(case.get("expect", {}), dict), case["id"]
                _assert_valid_turn_expect(case.get("expect", {}), case["id"])
            if "gold_relevance" in case:
                assert isinstance(case["gold_relevance"], dict), case["id"]
                assert case["gold_relevance"], case["id"]
                for product_id, relevance in case["gold_relevance"].items():
                    assert product_id, case["id"]
                    if filename == "retrieval_eval_cases.json":
                        assert not product_id.startswith("fixture_"), case["id"]
                        assert product_id in _real_product_ids_from_processed_data(), case["id"]
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
                if filename == "retrieval_eval_cases.json":
                    for product_id in case["hard_filters"].get("exclude_product_ids") or []:
                        assert not str(product_id).startswith("fixture_"), case["id"]
                        assert str(product_id) in _real_product_ids_from_processed_data(), case["id"]


def test_eval_case_ids_are_unique_per_file() -> None:
    for path in EVAL_DIR.glob("*_eval_cases.json"):
        cases = json.loads(path.read_text(encoding="utf-8"))
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)), path.name


def _real_product_ids_from_processed_data() -> set[str]:
    path = PROJECT_ROOT / "data" / "processed" / "products" / "all_products_900.jsonl"
    product_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw = "|".join(
            [
                str(record.get("category") or ""),
                str(record.get("source_platform") or "other"),
                str(record.get("source_product_id") or ""),
                str(record.get("brand") or ""),
                str(record.get("title") or ""),
            ]
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        product_ids.add(f"real_{record.get('category')}_{digest}"[:64])
    return product_ids


def _assert_no_placeholder_tokens(value: object, case_id: str) -> None:
    text = json.dumps(value, ensure_ascii=False)
    assert "??" not in text, case_id


def _assert_valid_turn_expect(expect: dict, case_id: str) -> None:
    if "context_carryover" in expect:
        assert isinstance(expect["context_carryover"], dict), case_id

    for field in ["should_clarify", "should_compare"]:
        if field in expect:
            assert isinstance(expect[field], bool), case_id

    if "compare_indices" in expect:
        assert isinstance(expect["compare_indices"], list), case_id
        assert all(isinstance(index, int) for index in expect["compare_indices"]), case_id

    list_fields = [
        "forbidden_categories",
        "forbidden_preferences",
        "negative_preferences_include",
        "preferences_include",
        "negative_preferences_contains",
        "preferences_contains",
        "citation_required_for_terms",
        "unsupported_answer_terms",
    ]
    for field in list_fields:
        if field in expect:
            assert isinstance(expect[field], list), case_id

    if "expected_claims" in expect:
        assert isinstance(expect["expected_claims"], list), case_id
        for claim in expect["expected_claims"]:
            assert isinstance(claim, dict), case_id
            assert claim.get("id"), case_id
            for field in [
                "answer_terms_any",
                "answer_terms_all",
                "citation_terms_any",
                "citation_terms_all",
            ]:
                if field in claim:
                    assert isinstance(claim[field], list), case_id
            if "required" in claim:
                assert isinstance(claim["required"], bool), case_id
