from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "multiturn_eval_cases.json"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import DocumentChunk, Product  # noqa: E402
from app.retrieval.chroma_indexer import (  # noqa: E402
    KNOWLEDGE_COLLECTION,
    PRODUCT_COLLECTION,
    get_chroma_client,
)
import app.models  # noqa: E402,F401


EvalCase = dict[str, Any]


def load_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    with Path(path).open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("Multiturn eval cases file must contain a JSON list")
    return cases


def run_case(client: TestClient, case: EvalCase) -> dict[str, Any]:
    session_id: str | None = None
    responses: list[dict[str, Any]] = []

    for query in case["turns"]:
        payload: dict[str, Any] = {"query": query, "debug": True}
        if session_id:
            payload["session_id"] = session_id
        response = client.post("/api/chat", json=payload)
        body = response.json()
        session_id = body.get("session_id") or session_id
        responses.append(body)

    failure_reasons = _evaluate_expectations(case, responses)
    return {
        "id": case["id"],
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
        "turn_count": len(responses),
    }


def _evaluate_expectations(
    case: EvalCase,
    responses: list[dict[str, Any]],
) -> list[str]:
    expectations = case.get("expectations", {})
    failure_reasons: list[str] = []
    first_response = responses[0]
    last_response = responses[-1]

    for step in expectations.get("second_turn_trace_steps", []):
        if len(responses) < 2 or _trace_step(responses[1], step) is None:
            failure_reasons.append(f"missing trace step on second turn: {step}")

    rewrite_step = _trace_step(responses[1], "follow_up_rewrite") if len(responses) > 1 else None
    if rewrite_step and expectations.get("rewritten_query_contains"):
        rewritten_query = rewrite_step.get("rewritten_query") or ""
        for keyword in expectations["rewritten_query_contains"]:
            if keyword not in rewritten_query:
                failure_reasons.append(f"rewritten_query missing keyword: {keyword}")

    comparison_step = _trace_step(last_response, "product_comparison")
    expected_source = expectations.get("comparison_source")
    if expected_source and (comparison_step or {}).get("source") != expected_source:
        failure_reasons.append(f"comparison source mismatch: {expected_source}")

    if expectations.get("product_cards_subset_of_first_turn"):
        first_ids = _product_ids(first_response)
        second_ids = _product_ids(last_response)
        if not second_ids or not set(second_ids).issubset(set(first_ids)):
            failure_reasons.append("product cards are not subset of first turn")

    if expectations.get("product_cards_match_resolved_product_ids"):
        resolved_ids = (rewrite_step or {}).get("resolved_product_ids") or []
        if _product_ids(last_response) != resolved_ids:
            failure_reasons.append("product cards do not match resolved product ids")

    for step in expectations.get("trace_steps_absent", []):
        if _trace_step(last_response, step) is not None:
            failure_reasons.append(f"unexpected trace step: {step}")

    min_product_cards = expectations.get("min_product_cards")
    if min_product_cards is not None and len(_product_ids(last_response)) < int(min_product_cards):
        failure_reasons.append("not enough product cards")

    return failure_reasons


def run_eval(cases: list[EvalCase]) -> dict[str, Any]:
    client = TestClient(app)
    results = [run_case(client, case) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    return {
        "results": results,
        "summary": {
            "total_cases": len(results),
            "passed_cases": passed,
            "failed_cases": len(results) - passed,
            "failed_case_ids": [
                result["id"] for result in results if not result["passed"]
            ],
        },
    }


def print_report(output: dict[str, Any]) -> None:
    for result in output["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        if result["failure_reasons"]:
            for reason in result["failure_reasons"]:
                print(f"- {reason}")
        print()

    print("summary:")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        _check_prerequisites(db)
    finally:
        db.close()

    output = run_eval(load_eval_cases(DEFAULT_CASES_PATH))
    print_report(output)


def _check_prerequisites(db) -> None:
    product_count = db.scalar(select(func.count()).select_from(Product)) or 0
    if product_count == 0:
        raise SystemExit("No products found. Please run import_products first.")

    chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    if chunk_count == 0:
        raise SystemExit("No document chunks found. Please run import_docs first.")

    chroma_client = get_chroma_client()
    if (
        _collection_count(chroma_client, PRODUCT_COLLECTION) == 0
        or _collection_count(chroma_client, KNOWLEDGE_COLLECTION) == 0
    ):
        raise SystemExit("No Chroma index found. Please run rebuild_index.py first.")


def _collection_count(chroma_client, collection_name: str) -> int:
    try:
        return chroma_client.get_collection(collection_name).count()
    except Exception:
        return 0


def _trace_step(response: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    for step in response.get("trace", []):
        if step.get("step") == step_name:
            return step
    return None


def _product_ids(response: dict[str, Any]) -> list[str]:
    return [
        product.get("product_id")
        for product in response.get("product_cards", [])
        if product.get("product_id")
    ]


if __name__ == "__main__":
    main()
