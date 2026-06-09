from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import func, select  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import DocumentChunk, Product  # noqa: E402
from app.retrieval.chroma_indexer import (  # noqa: E402
    KNOWLEDGE_COLLECTION,
    PRODUCT_COLLECTION,
    get_chroma_client,
)
from app.retrieval.retrieval_service import (  # noqa: E402
    KnowledgeRetrievalService,
    ProductRetrievalService,
    ProductSearchFilters,
)
from app.services.embedding import get_embedding_service  # noqa: E402
import app.models  # noqa: E402,F401


EvalCase = dict[str, Any]
EvalResult = dict[str, Any]


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
    product_service: ProductRetrievalService,
    knowledge_service: KnowledgeRetrievalService,
) -> EvalResult:
    top_k = int(case.get("top_k", 3))
    category_id = case.get("category_id")
    budget_max = case.get("budget_max")

    products = product_service.search_products(
        query=case["query"],
        filters=ProductSearchFilters(
            category_id=category_id,
            budget_min=case.get("budget_min"),
            budget_max=budget_max,
            stock_only=True,
        ),
        top_k=top_k,
    )
    citations = knowledge_service.search_knowledge(
        query=case["query"],
        category_id=category_id,
        top_k=int(case.get("citation_top_k", top_k)),
    )

    actual_product_ids = [_field(product, "product_id") for product in products]
    expected_product_ids = case.get("expected_product_ids") or []
    product_hit = (
        any(product_id in actual_product_ids for product_id in expected_product_ids)
        if expected_product_ids
        else True
    )
    category_ok = (
        all(_field(product, "category_id") == category_id for product in products)
        if category_id
        else True
    )
    budget_ok = (
        all(_field(product, "price", 0) <= budget_max for product in products)
        if budget_max is not None
        else True
    )
    citation_keyword_hit = keyword_hit(
        _citation_texts(citations),
        case.get("expected_doc_keywords") or [],
    )
    passed = product_hit and category_ok and budget_ok and citation_keyword_hit

    return {
        "id": case["id"],
        "type": "product",
        "query": case["query"],
        "passed": passed,
        "actual_product_ids": actual_product_ids,
        "product_hit": product_hit,
        "category_ok": category_ok,
        "budget_ok": budget_ok,
        "citation_keyword_hit": citation_keyword_hit,
        "citation_count": len(citations),
    }


def evaluate_knowledge_case(
    case: EvalCase,
    knowledge_service: KnowledgeRetrievalService,
) -> EvalResult:
    citations = knowledge_service.search_knowledge(
        query=case["query"],
        category_id=case.get("category_id"),
        top_k=int(case.get("top_k", 5)),
    )
    citation_keyword_hit = keyword_hit(
        _citation_texts(citations),
        case.get("expected_doc_keywords") or [],
    )
    actual_citation_sources = [
        _field(citation, "source_file")
        for citation in citations
        if _field(citation, "source_file")
    ]
    passed = bool(citations) and citation_keyword_hit

    return {
        "id": case["id"],
        "type": "knowledge",
        "query": case["query"],
        "passed": passed,
        "actual_citation_sources": actual_citation_sources,
        "citation_keyword_hit": citation_keyword_hit,
        "citation_count": len(citations),
    }


def run_eval(
    cases: list[EvalCase],
    product_service: ProductRetrievalService,
    knowledge_service: KnowledgeRetrievalService,
) -> dict[str, Any]:
    results: list[EvalResult] = []
    for case in cases:
        case_type = case.get("type")
        if case_type == "product":
            results.append(evaluate_product_case(case, product_service, knowledge_service))
        elif case_type == "knowledge":
            results.append(evaluate_knowledge_case(case, knowledge_service))
        else:
            raise ValueError(f"Unsupported eval case type: {case_type}")

    return {
        "results": results,
        "summary": summarize_results(results),
    }


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["passed"])
    product_results = [result for result in results if result["type"] == "product"]
    citation_results = [
        result for result in results if "citation_keyword_hit" in result
    ]

    product_hits = sum(1 for result in product_results if result.get("product_hit"))
    citation_hits = sum(
        1 for result in citation_results if result.get("citation_keyword_hit")
    )

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": total_cases - passed_cases,
        "product_hit_rate": _rate(product_hits, len(product_results)),
        "citation_keyword_hit_rate": _rate(citation_hits, len(citation_results)),
    }


def print_report(eval_output: dict[str, Any]) -> None:
    for result in eval_output["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        print(f"query: {result['query']}")

        if result["type"] == "product":
            print(f"actual_products: {', '.join(result['actual_product_ids']) or '-'}")
            print(f"product_hit: {_bool_text(result['product_hit'])}")
            print(f"category_ok: {_bool_text(result['category_ok'])}")
            print(f"budget_ok: {_bool_text(result['budget_ok'])}")
            print(f"citation_keyword_hit: {_bool_text(result['citation_keyword_hit'])}")
        else:
            print(
                "actual_citation_sources: "
                f"{', '.join(result['actual_citation_sources']) or '-'}"
            )
            print(f"citation_count: {result['citation_count']}")
            print(f"citation_keyword_hit: {_bool_text(result['citation_keyword_hit'])}")

        print()

    print("summary:")
    for key, value in eval_output["summary"].items():
        print(f"{key}: {value}")


def main() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        _check_prerequisites(db)
        cases = load_eval_cases(DEFAULT_CASES_PATH)
        embedding_service = get_embedding_service()
        chroma_client = get_chroma_client()
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
        eval_output = run_eval(cases, product_service, knowledge_service)
    finally:
        db.close()

    print_report(eval_output)


def _check_prerequisites(db) -> None:
    product_count = db.scalar(select(func.count()).select_from(Product)) or 0
    if product_count == 0:
        raise SystemExit("No products found. Please run import_products first.")

    chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    if chunk_count == 0:
        raise SystemExit("No document chunks found. Please run import_docs first.")

    chroma_client = get_chroma_client()
    product_index_count = _collection_count(chroma_client, PRODUCT_COLLECTION)
    knowledge_index_count = _collection_count(chroma_client, KNOWLEDGE_COLLECTION)
    if product_index_count == 0 or knowledge_index_count == 0:
        raise SystemExit("No Chroma index found. Please run rebuild_index.py first.")


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
            "content_preview",
        ):
            value = _field(citation, field_name)
            if value:
                texts.append(str(value))
    return texts


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    main()
