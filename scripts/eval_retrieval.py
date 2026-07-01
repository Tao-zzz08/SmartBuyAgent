from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import json
import sys
import time

from retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_filter_compliance,
    compute_retrieval_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json"


EvalCase = Dict[str, Any]
EvalResult = Dict[str, Any]


def load_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    with Path(path).open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Eval cases file must contain a JSON list")
    return cases


def keyword_hit(texts: list[str], keywords: list[str] | None) -> bool:
    if not keywords:
        return True

    haystack = "\n".join(text for text in texts if text).lower()
    return any(keyword.lower() in haystack for keyword in keywords if keyword)


def evaluate_product_case(
    case: EvalCase,
    product_service: Any,
    knowledge_service: Any,
) -> EvalResult:
    started = time.perf_counter()
    filters_payload = case.get("structured_filters") or {}
    expect = case.get("expect") or _legacy_product_expect(case)
    top_k = int(expect.get("top_k") or case.get("top_k") or 5)
    category_id = filters_payload.get("category_id") or case.get("category_id")

    filters = _product_filters(
        category_id=category_id,
        budget_min=filters_payload.get("budget_min") or case.get("budget_min"),
        budget_max=filters_payload.get("budget_max") or case.get("budget_max"),
        preferences=filters_payload.get("preferences") or [],
        negative_preferences=filters_payload.get("negative_preferences") or [],
    )
    products = product_service.search_products(
        query=case["query"],
        filters=filters,
        top_k=top_k,
    )
    citations = knowledge_service.search_knowledge(
        query=case["query"],
        category_id=category_id,
        top_k=int(case.get("citation_top_k") or top_k),
    )

    failure_reasons: list[str] = []
    _check_product_results(products, expect, failure_reasons)
    _check_knowledge_results(citations, _product_case_knowledge_expect(case), failure_reasons)

    latency_ms = _elapsed_ms(started)
    actual_product_ids = [
        str(product_id)
        for product_id in [_field(product, "product_id") for product in products]
        if product_id
    ]
    product_dicts = [_product_to_dict(product) for product in products]
    filter_metrics = compute_filter_compliance(
        product_dicts,
        _hard_filters_for_product_case(case, expect, filters),
    )
    if filter_metrics["filter_violation_count"]:
        failure_reasons.append("hard filter violation")
    ranking_metrics = compute_retrieval_metrics(
        actual_product_ids,
        case.get("gold_relevance"),
        k=top_k,
    )
    metrics = {
        **ranking_metrics,
        "filter_compliance": filter_metrics["filter_compliance"],
        "filter_violation_count": filter_metrics["filter_violation_count"],
        "latency_ms": latency_ms,
    }

    return {
        "id": case["id"],
        "type": "product_retrieval",
        "query": case["query"],
        "passed": not failure_reasons,
        "actual_product_ids": actual_product_ids,
        "retrieved_product_ids": actual_product_ids,
        "result_count": len(products),
        "category_ok": not any(reason == "category mismatch" for reason in failure_reasons),
        "budget_ok": not any(reason == "budget constraint violated" for reason in failure_reasons),
        "negative_preference_violations": _forbidden_violation_count(
            _product_texts(products),
            expect.get("forbidden_terms") or [],
        ),
        "metrics": metrics,
        "filter_violations": filter_metrics["filter_violations"],
        "citation_keyword_hit": keyword_hit(
            _citation_texts(citations),
            _product_case_knowledge_expect(case).get("must_contain_any") or [],
        ),
        "citation_count": len(citations),
        "failure_reasons": failure_reasons,
    }


def evaluate_knowledge_case(
    case: EvalCase,
    knowledge_service: Any,
) -> EvalResult:
    started = time.perf_counter()
    expect = case.get("expect") or _legacy_knowledge_expect(case)
    top_k = int(expect.get("top_k") or case.get("top_k") or 5)
    citations = knowledge_service.search_knowledge(
        query=case["query"],
        category_id=case.get("category_id"),
        top_k=top_k,
    )

    failure_reasons: list[str] = []
    _check_knowledge_results(citations, expect, failure_reasons)
    actual_citation_sources = [
        _field(citation, "source_file")
        for citation in citations
        if _field(citation, "source_file")
    ]
    citation_keyword_hit = keyword_hit(
        _citation_texts(citations),
        expect.get("must_contain_any") or [],
    )

    return {
        "id": case["id"],
        "type": "knowledge_retrieval",
        "query": case["query"],
        "passed": not failure_reasons,
        "actual_citation_sources": actual_citation_sources,
        "citation_keyword_hit": citation_keyword_hit,
        "citation_count": len(citations),
        "forbidden_term_violations": _forbidden_violation_count(
            _citation_texts(citations),
            expect.get("forbidden_terms") or [],
        ),
        "metrics": {"latency_ms": _elapsed_ms(started)},
        "failure_reasons": failure_reasons,
    }


def run_eval(
    cases: list[EvalCase],
    product_service: Any,
    knowledge_service: Any,
) -> dict[str, Any]:
    results: list[EvalResult] = []
    for case in cases:
        case_type = case.get("type")
        if case_type in {"product", "product_retrieval"}:
            results.append(evaluate_product_case(case, product_service, knowledge_service))
        elif case_type in {"knowledge", "knowledge_retrieval"}:
            results.append(evaluate_knowledge_case(case, knowledge_service))
        else:
            raise ValueError(f"Unsupported eval case type: {case_type}")

    return {
        "results": results,
        "summary": summarize_results(results),
    }


def run_default_eval(cases: list[EvalCase] | None = None) -> dict[str, Any]:
    product_service, knowledge_service, cleanup = _default_services()
    try:
        return run_eval(cases or load_eval_cases(DEFAULT_CASES_PATH), product_service, knowledge_service)
    finally:
        cleanup()


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["passed"])
    product_results = [
        result for result in results if result["type"] == "product_retrieval"
    ]
    knowledge_results = [
        result for result in results if result["type"] == "knowledge_retrieval"
    ]
    failed_results = [result for result in results if not result["passed"]]

    product_category_passes = sum(1 for result in product_results if result["category_ok"])
    product_budget_passes = sum(1 for result in product_results if result["budget_ok"])
    min_result_passes = sum(
        1 for result in results if "not enough results" not in result["failure_reasons"]
    )
    knowledge_hits = sum(
        1 for result in knowledge_results if result.get("citation_keyword_hit")
    )
    forbidden_violations = sum(
        int(result.get("negative_preference_violations", 0))
        + int(result.get("forbidden_term_violations", 0))
        for result in results
    )

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "failed_case_ids": [result["id"] for result in failed_results],
        "failure_reason_counts": _failure_reason_counts(failed_results),
        "product_category_compliance": _rate(product_category_passes, len(product_results)),
        "budget_compliance": _rate(product_budget_passes, len(product_results)),
        "negative_preference_violation_count": sum(
            int(result.get("negative_preference_violations", 0))
            for result in product_results
        ),
        "min_result_pass_rate": _rate(min_result_passes, total_cases),
        "knowledge_chunk_hit_rate": _rate(knowledge_hits, len(knowledge_results)),
        "forbidden_term_violation_count": forbidden_violations,
        "metrics": aggregate_retrieval_metrics(results),
    }


def print_report(eval_output: dict[str, Any]) -> None:
    for result in eval_output["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        print(f"query: {result['query']}")

        if result["type"] == "product_retrieval":
            print(f"actual_products: {', '.join(result['actual_product_ids']) or '-'}")
            print(f"category_ok: {_bool_text(result['category_ok'])}")
            print(f"budget_ok: {_bool_text(result['budget_ok'])}")
            print(
                "negative_preference_violations: "
                f"{result['negative_preference_violations']}"
            )
            print(f"citation_keyword_hit: {_bool_text(result['citation_keyword_hit'])}")
            print(f"metrics: {result.get('metrics', {})}")
        else:
            print(
                "actual_citation_sources: "
                f"{', '.join(result['actual_citation_sources']) or '-'}"
            )
            print(f"citation_count: {result['citation_count']}")
            print(f"citation_keyword_hit: {_bool_text(result['citation_keyword_hit'])}")
            print(f"metrics: {result.get('metrics', {})}")

        if not result["passed"]:
            print("failure_reasons:")
            for reason in result.get("failure_reasons", []):
                print(f"- {reason}")

        print()

    print("summary:")
    for key, value in eval_output["summary"].items():
        print(f"{key}: {value}")


def main() -> None:
    eval_output = run_default_eval(load_eval_cases(DEFAULT_CASES_PATH))
    print_report(eval_output)
    if eval_output["summary"]["failed_cases"]:
        raise SystemExit(1)


def _product_filters(
    *,
    category_id: str | None,
    budget_min: int | None,
    budget_max: int | None,
    preferences: list[str],
    negative_preferences: list[str],
) -> Any:
    try:
        if str(BACKEND_DIR) not in sys.path:
            sys.path.insert(0, str(BACKEND_DIR))
        from app.retrieval.retrieval_service import ProductSearchFilters

        return ProductSearchFilters(
            category_id=category_id,
            budget_min=budget_min,
            budget_max=budget_max,
            stock_only=True,
            brand_exclude=list(negative_preferences),
            preferences=list(preferences),
        )
    except Exception:
        return SimpleNamespace(
            category_id=category_id,
            budget_min=budget_min,
            budget_max=budget_max,
            stock_only=True,
            brand_exclude=list(negative_preferences),
            preferences=list(preferences),
        )


def _check_product_results(
    products: list[Any],
    expect: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    min_results = int(expect.get("min_results") or 0)
    if len(products) < min_results:
        failure_reasons.append("not enough results")

    expected_category = expect.get("all_category")
    if expected_category:
        expected_category_id = _category_id(expected_category)
        if any(_field(product, "category_id") != expected_category_id for product in products):
            failure_reasons.append("category mismatch")

    max_price = expect.get("max_price")
    if max_price is not None and any(
        _numeric(_field(product, "price")) is not None
        and _numeric(_field(product, "price")) > float(max_price)
        for product in products
    ):
        failure_reasons.append("budget constraint violated")

    if not keyword_hit(_product_texts(products), expect.get("must_match_any") or []):
        failure_reasons.append("product preference keywords not found")

    if _forbidden_violation_count(
        _product_texts(products),
        expect.get("forbidden_terms") or [],
    ):
        failure_reasons.append("forbidden product terms found")

    expected_product_ids = expect.get("expected_product_ids") or []
    if expected_product_ids:
        actual_ids = [_field(product, "product_id") for product in products]
        if not any(product_id in actual_ids for product_id in expected_product_ids):
            failure_reasons.append("expected product ids not found")


def _check_knowledge_results(
    citations: list[Any],
    expect: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    min_chunks = int(expect.get("min_chunks") or 0)
    if len(citations) < min_chunks:
        failure_reasons.append("not enough results")
    if len(citations) == 0 and min_chunks > 0:
        failure_reasons.append("no citations returned")

    if not keyword_hit(_citation_texts(citations), expect.get("must_contain_any") or []):
        failure_reasons.append("citation keywords not found")

    if _forbidden_violation_count(
        _citation_texts(citations),
        expect.get("forbidden_terms") or [],
    ):
        failure_reasons.append("forbidden citation terms found")


def _legacy_product_expect(case: EvalCase) -> dict[str, Any]:
    return {
        "min_results": 1,
        "all_category": _category_from_id(case.get("category_id")),
        "max_price": case.get("budget_max"),
        "expected_product_ids": case.get("expected_product_ids") or [],
        "top_k": case.get("top_k", 3),
    }


def _legacy_knowledge_expect(case: EvalCase) -> dict[str, Any]:
    return {
        "min_chunks": 1,
        "must_contain_any": case.get("expected_doc_keywords") or [],
        "top_k": case.get("top_k", 5),
    }


def _product_case_knowledge_expect(case: EvalCase) -> dict[str, Any]:
    expect = case.get("knowledge_expect")
    if isinstance(expect, dict):
        return expect
    return {
        "must_contain_any": case.get("expected_doc_keywords") or [],
        "forbidden_terms": (case.get("expect") or {}).get("forbidden_terms") or [],
    }


def _hard_filters_for_product_case(
    case: EvalCase,
    expect: dict[str, Any],
    filters: Any,
) -> dict[str, Any]:
    hard_filters = dict(case.get("hard_filters") or {})
    if "category" not in hard_filters:
        hard_filters["category"] = (
            expect.get("all_category")
            or _category_from_id(_field(filters, "category_id"))
            or (case.get("structured_filters") or {}).get("category")
        )
    if "price_lte" not in hard_filters:
        hard_filters["price_lte"] = expect.get("max_price") or _field(filters, "budget_max")
    if "price_gte" not in hard_filters:
        hard_filters["price_gte"] = _field(filters, "budget_min")

    exclude_brands = list(hard_filters.get("exclude_brands") or [])
    for term in _field(filters, "brand_exclude", []) or []:
        if term not in exclude_brands:
            exclude_brands.append(term)
    hard_filters["exclude_brands"] = exclude_brands

    forbidden_terms = list(hard_filters.get("forbidden_terms") or [])
    for term in expect.get("forbidden_terms") or []:
        if term not in forbidden_terms:
            forbidden_terms.append(term)
    hard_filters["forbidden_terms"] = forbidden_terms
    return {
        key: value
        for key, value in hard_filters.items()
        if value is not None and value != []
    }


def _product_to_dict(product: Any) -> dict[str, Any]:
    category_id = _field(product, "category_id")
    return {
        "product_id": _field(product, "product_id"),
        "id": _field(product, "product_id"),
        "category": _category_from_id(category_id) or _field(product, "category"),
        "category_id": category_id,
        "title": _field(product, "title"),
        "brand": _field(product, "brand"),
        "price": _field(product, "price"),
        "tags": list(_field(product, "tags", []) or []),
        "description": _field(product, "description"),
        "product_text": _field(product, "product_text"),
        "in_stock": _field(product, "in_stock", _field(product, "stock")),
    }


def _default_services() -> tuple[Any, Any, Any]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    from sqlalchemy import func, select
    from app.core.db import Base, SessionLocal, engine
    from app.models import DocumentChunk, Product
    from app.retrieval.chroma_indexer import (
        KNOWLEDGE_COLLECTION,
        PRODUCT_COLLECTION,
        get_chroma_client,
    )
    from app.retrieval.retrieval_service import (
        KnowledgeRetrievalService,
        ProductRetrievalService,
    )
    from app.services.embedding import get_embedding_service
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    product_count = db.scalar(select(func.count()).select_from(Product)) or 0
    if product_count == 0:
        db.close()
        raise SystemExit("No products found. Please run import_products first.")

    chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    if chunk_count == 0:
        db.close()
        raise SystemExit("No document chunks found. Please run import_docs first.")

    chroma_client = get_chroma_client()
    if (
        _collection_count(chroma_client, PRODUCT_COLLECTION) == 0
        or _collection_count(chroma_client, KNOWLEDGE_COLLECTION) == 0
    ):
        db.close()
        raise SystemExit("No Chroma index found. Please run rebuild_index.py first.")

    embedding_service = get_embedding_service()
    product_service = ProductRetrievalService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )
    knowledge_service = KnowledgeRetrievalService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )
    return product_service, knowledge_service, db.close


def _collection_count(chroma_client, collection_name: str) -> int:
    try:
        return chroma_client.get_collection(collection_name).count()
    except Exception:
        return 0


def _citation_texts(citations: list[Any]) -> list[str]:
    texts: list[str] = []
    for citation in citations:
        for field_name in (
            "title",
            "section",
            "section_path",
            "source_file",
            "source",
            "text",
            "content_preview",
        ):
            value = _field(citation, field_name)
            if value:
                texts.append(str(value))
    return texts


def _product_texts(products: list[Any]) -> list[str]:
    texts: list[str] = []
    for product in products:
        parts: list[str] = []
        for field_name in (
            "product_id",
            "title",
            "brand",
            "category_id",
            "description",
            "product_text",
        ):
            value = _field(product, field_name)
            if value:
                parts.append(str(value))
        for value in _field(product, "tags", []) or []:
            parts.append(str(value))
        texts.append(" ".join(parts))
    return texts


def _forbidden_violation_count(texts: list[str] | str, terms: list[str]) -> int:
    haystack = "\n".join(texts if isinstance(texts, list) else [texts]).lower()
    return sum(1 for term in terms if term and term.lower() in haystack)


def _failure_reason_counts(results: list[EvalResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for reason in result.get("failure_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _numeric(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _category_id(category: str | None) -> str | None:
    return {
        "phone": "cat_phone",
        "shoes": "cat_shoes",
        "skincare": "cat_skincare",
    }.get(category or "")


def _category_from_id(category_id: str | None) -> str | None:
    return {
        "cat_phone": "phone",
        "cat_shoes": "shoes",
        "cat_skincare": "skincare",
    }.get(category_id or "")


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    main()
