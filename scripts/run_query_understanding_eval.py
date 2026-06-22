from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "data" / "eval" / "query_understanding_regression_cases.json"
)


EvalCase = Dict[str, Any]
EvalResult = Dict[str, Any]

MEDICAL_CLAIM_TERMS = [
    "治疗",
    "治愈",
    "药效",
    "处方",
    "医学修复",
    "修复疾病",
]

PURCHASE_BOUNDARY_TERMS = [
    "立即购买",
    "下单",
    "支付",
    "购物车",
    "订单",
    "购买链接",
]

GENERIC_FALLBACK_PHRASES = [
    "你好，我可以帮你挑选手机、鞋靴和护肤品",
    "你可以告诉我预算、用途和偏好",
]

CATEGORY_PRODUCT_ID_HINTS = {
    "phone": ["phone", "mobile", "iphone"],
    "shoes": ["shoe", "shoes", "boot", "sneaker"],
    "skincare": ["skin", "skincare", "cream", "serum", "cleanser"],
}


def load_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    with Path(path).open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("Query understanding eval cases file must contain a JSON list")
    return cases


def run_eval(
    cases: list[EvalCase],
    *,
    client: Any | None = None,
    case_id: str | None = None,
    mode: str = "chat",
) -> dict[str, Any]:
    selected_cases = [
        case for case in cases if case_id is None or case.get("id") == case_id
    ]
    if case_id is not None and not selected_cases:
        raise ValueError(f"Unknown eval case id: {case_id}")

    eval_client = client or _default_test_client()
    results = [
        run_case(eval_client, case, mode=mode)
        for case in selected_cases
    ]
    return {
        "results": results,
        "summary": summarize_results(results),
    }


def run_case(client: Any, case: EvalCase, *, mode: str = "chat") -> EvalResult:
    session_id: str | None = None
    turn_results: list[dict[str, Any]] = []

    for turn_index, turn in enumerate(case.get("turns", []), start=1):
        query = str(turn["user"])
        turn_mode = str(turn.get("mode") or mode)
        response = _post_chat_turn(
            client,
            query=query,
            session_id=session_id,
            stream=turn_mode == "stream",
        )
        session_id = response.get("session_id") or session_id

        turn_result = {
            "turn_index": turn_index,
            "user": query,
            "mode": turn_mode,
            "response": response,
            "query_understanding": query_understanding_from_response(response),
            "failure_reasons": [],
        }
        turn_result["failure_reasons"] = evaluate_turn(
            case=case,
            turn=turn,
            turn_result=turn_result,
            previous_turn_result=turn_results[-1] if turn_results else None,
        )
        turn_results.append(turn_result)

    failure_reasons = [
        {
            "turn_index": turn_result["turn_index"],
            "user": turn_result["user"],
            "reasons": turn_result["failure_reasons"],
            "expected": (case["turns"][turn_result["turn_index"] - 1]).get(
                "expect",
                {},
            ),
            "actual_query_understanding": turn_result["query_understanding"],
            "actual_product_cards": product_cards_summary(
                turn_result["response"].get("product_cards", [])
            ),
        }
        for turn_result in turn_results
        if turn_result["failure_reasons"]
    ]

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "passed": not failure_reasons,
        "turn_count": len(turn_results),
        "turns": turn_results,
        "failure_reasons": failure_reasons,
    }


def evaluate_turn(
    *,
    case: EvalCase,
    turn: dict[str, Any],
    turn_result: dict[str, Any],
    previous_turn_result: dict[str, Any] | None,
) -> list[str]:
    del case
    response = turn_result["response"]
    expected = turn.get("expect") or {}
    query_understanding = turn_result["query_understanding"]
    failure_reasons: list[str] = []

    if response.get("_http_status", 200) >= 400:
        failure_reasons.append(f"HTTP status {response['_http_status']}")
        return failure_reasons

    _check_query_understanding(
        query_understanding,
        expected,
        failure_reasons,
    )
    _check_product_cards(
        response.get("product_cards", []),
        expected,
        failure_reasons,
        previous_turn_result,
    )
    _check_citations(response.get("citations", []), expected, failure_reasons)
    _check_comparison(response, query_understanding, expected, failure_reasons, previous_turn_result)
    _check_safety_boundaries(response, query_understanding, expected, failure_reasons)
    return failure_reasons


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total_cases = len(results)
    failed_results = [result for result in results if not result["passed"]]
    total_turns = sum(result["turn_count"] for result in results)
    failed_turns = sum(len(result["failure_reasons"]) for result in results)
    return {
        "total_cases": total_cases,
        "passed_cases": total_cases - len(failed_results),
        "failed_cases": len(failed_results),
        "failed_case_ids": [result["id"] for result in failed_results],
        "total_turns": total_turns,
        "passed_turns": total_turns - failed_turns,
        "failed_turns": failed_turns,
        "failure_reason_counts": _failure_reason_counts(results),
    }


def print_report(output: dict[str, Any]) -> None:
    print("QueryUnderstanding Regression Eval")
    print("==================================")
    summary = output["summary"]
    print(f"Cases: {summary['total_cases']}")
    print(f"Turns: {summary['total_turns']}")
    print(f"Passed: {summary['passed_turns']}")
    print(f"Failed: {summary['failed_turns']}")
    print()

    for result in output["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        for failure in result.get("failure_reasons", []):
            print(f"  turn {failure['turn_index']}: {failure['user']}")
            print(f"  expected: {json.dumps(failure['expected'], ensure_ascii=False)}")
            print(
                "  actual_query_understanding: "
                f"{json.dumps(failure['actual_query_understanding'], ensure_ascii=False)}"
            )
            print(
                "  actual_product_cards: "
                f"{json.dumps(failure['actual_product_cards'], ensure_ascii=False)}"
            )
            for reason in failure["reasons"]:
                print(f"  - {reason}")
        print()

    print("summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def query_understanding_from_response(response: dict[str, Any]) -> dict[str, Any]:
    step = trace_step(response, "query_understanding")
    return step or {}


def trace_step(response: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    for step in response.get("trace", []) or []:
        if step.get("step") == step_name or step.get("node") == step_name:
            return step
    return None


def product_cards_summary(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "product_id": card.get("product_id"),
            "title": card.get("title"),
            "brand": card.get("brand"),
            "price": card.get("price"),
        }
        for card in cards
    ]


def _check_query_understanding(
    actual: dict[str, Any],
    expected: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    scalar_fields = ["intent", "category", "source", "is_follow_up"]
    for field in scalar_fields:
        if field in expected and actual.get(field) != expected[field]:
            failure_reasons.append(
                f"{field} mismatch: expected {expected[field]!r}, got {actual.get(field)!r}"
            )

    if "budget_max" in expected and _budget_max(actual) != expected["budget_max"]:
        failure_reasons.append(
            f"budget_max mismatch: expected {expected['budget_max']!r}, got {_budget_max(actual)!r}"
        )

    _expect_list_contains(
        actual.get("preferences") or [],
        expected.get("preferences_contains") or [],
        "preferences",
        failure_reasons,
    )
    _expect_list_contains(
        actual.get("negative_preferences") or [],
        expected.get("negative_preferences_contains") or [],
        "negative_preferences",
        failure_reasons,
    )
    for value in expected.get("preferences_not_contains") or []:
        if value in (actual.get("preferences") or []):
            failure_reasons.append(f"preferences unexpectedly contains {value!r}")

    if "referenced_product_indices" in expected:
        actual_indices = actual.get("referenced_product_indices") or []
        if actual_indices != expected["referenced_product_indices"]:
            failure_reasons.append(
                "referenced_product_indices mismatch: "
                f"expected {expected['referenced_product_indices']!r}, got {actual_indices!r}"
            )


def _check_product_cards(
    cards: list[dict[str, Any]],
    expected: dict[str, Any],
    failure_reasons: list[str],
    previous_turn_result: dict[str, Any] | None,
) -> None:
    min_product_cards = expected.get("min_product_cards")
    if min_product_cards is not None and len(cards) < int(min_product_cards):
        failure_reasons.append(
            f"not enough product_cards: expected at least {min_product_cards}, got {len(cards)}"
        )

    expected_category = expected.get("product_cards_category")
    if expected_category and cards:
        mismatched = [
            card.get("product_id")
            for card in cards
            if not _card_matches_category(card, expected_category)
        ]
        if mismatched:
            failure_reasons.append(
                f"product_cards category mismatch for ids: {mismatched}"
            )

    for term in expected.get("product_cards_exclude_terms") or []:
        if term and term.lower() in _cards_text(cards).lower():
            failure_reasons.append(f"product_cards contain excluded term: {term}")

    if expected.get("product_cards_subset_of_previous_turn") and previous_turn_result:
        previous_ids = set(_product_ids(previous_turn_result["response"]))
        current_ids = set(_product_ids_from_cards(cards))
        if current_ids and not current_ids.issubset(previous_ids):
            failure_reasons.append("product_cards are not a subset of previous turn")

    if expected.get("product_cards_have_required_fields"):
        for card in cards:
            missing = [
                field
                for field in ["product_id", "title", "price", "recommend_reason"]
                if card.get(field) in {None, ""}
            ]
            if missing:
                failure_reasons.append(
                    f"product_card {card.get('product_id')} missing fields: {missing}"
                )


def _check_citations(
    citations: list[dict[str, Any]],
    expected: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    min_citations = expected.get("require_citations_min")
    if min_citations is not None and len(citations) < int(min_citations):
        failure_reasons.append(
            f"not enough citations: expected at least {min_citations}, got {len(citations)}"
        )

    if expected.get("citations_have_required_fields"):
        for citation in citations:
            missing = [
                field
                for field in ["chunk_id", "content_preview", "score"]
                if citation.get(field) in {None, ""}
            ]
            if missing:
                failure_reasons.append(
                    f"citation {citation.get('chunk_id')} missing fields: {missing}"
                )


def _check_comparison(
    response: dict[str, Any],
    query_understanding: dict[str, Any],
    expected: dict[str, Any],
    failure_reasons: list[str],
    previous_turn_result: dict[str, Any] | None,
) -> None:
    if "compare_product_ids_count" not in expected and not expected.get(
        "compare_product_ids_from_previous_turn"
    ):
        return

    compare_ids = _compare_product_ids(response, query_understanding)
    expected_count = expected.get("compare_product_ids_count")
    if expected_count is not None and len(compare_ids) != int(expected_count):
        failure_reasons.append(
            f"compare_product_ids count mismatch: expected {expected_count}, got {len(compare_ids)}"
        )

    if expected.get("compare_product_ids_from_previous_turn"):
        previous_ids = (
            set(_product_ids(previous_turn_result["response"]))
            if previous_turn_result
            else set()
        )
        if not compare_ids or not set(compare_ids).issubset(previous_ids):
            failure_reasons.append(
                "compare_product_ids are not resolved from previous product_cards"
            )


def _check_safety_boundaries(
    response: dict[str, Any],
    query_understanding: dict[str, Any],
    expected: dict[str, Any],
    failure_reasons: list[str],
) -> None:
    answer = str(response.get("answer") or "")
    effective_query = str(query_understanding.get("effective_query") or "")
    knowledge_query = str((trace_step(response, "knowledge_retrieval") or {}).get("query") or "")

    if expected.get("no_generic_fallback"):
        for phrase in GENERIC_FALLBACK_PHRASES:
            if phrase in answer:
                failure_reasons.append("answer fell back to generic welcome message")

    _expect_no_terms(answer, expected.get("forbidden_answer_terms") or [], "answer", failure_reasons)
    _expect_no_terms(
        effective_query,
        expected.get("forbidden_effective_query_terms") or [],
        "effective_query",
        failure_reasons,
    )
    _expect_no_terms(
        knowledge_query,
        expected.get("forbidden_knowledge_query_terms") or [],
        "knowledge_query",
        failure_reasons,
    )

    safe_terms = expected.get("safe_terms_any") or []
    if safe_terms:
        safe_haystack = "\n".join([answer, effective_query, knowledge_query])
        if not any(term in safe_haystack for term in safe_terms):
            failure_reasons.append(f"safe terms not found: expected any of {safe_terms}")

    if expected.get("no_purchase_boundary_terms", True):
        payload_text = json.dumps(response, ensure_ascii=False)
        for term in PURCHASE_BOUNDARY_TERMS:
            if term in payload_text:
                failure_reasons.append(f"purchase boundary term found: {term}")


def _post_chat_turn(
    client: Any,
    *,
    query: str,
    session_id: str | None,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "debug": True}
    if session_id:
        payload["session_id"] = session_id

    path = "/api/chat/stream" if stream else "/api/chat"
    raw_response = client.post(path, json=payload)
    status_code = int(getattr(raw_response, "status_code", 200))

    if stream:
        events = parse_sse_events(getattr(raw_response, "text", ""))
        result = dict(_event_data(events, "result") or {})
        result["_sse_events"] = events
    else:
        result = raw_response.json()

    if not isinstance(result, dict):
        result = {}
    result["_http_status"] = status_code
    return result


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is None:
            data_lines = []
            return
        data_text = "\n".join(data_lines).strip()
        try:
            data = json.loads(data_text) if data_text else {}
        except json.JSONDecodeError:
            data = {"raw": data_text}
        events.append({"event": event_name, "data": data})
        event_name = None
        data_lines = []

    for line in text.splitlines():
        if not line:
            flush()
            continue
        if line.startswith("event:"):
            event_name = line.partition(":")[2].strip()
        elif line.startswith("data:"):
            data_lines.append(line.partition(":")[2].strip())
    flush()
    return events


def _event_data(events: list[dict[str, Any]], event_name: str) -> dict[str, Any] | None:
    for event in events:
        if event.get("event") == event_name and isinstance(event.get("data"), dict):
            return event["data"]
    return None


def _default_test_client() -> Any:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _budget_max(query_understanding: dict[str, Any]) -> int | None:
    if query_understanding.get("budget_max") is not None:
        return int(query_understanding["budget_max"])
    budget = query_understanding.get("budget") or {}
    if isinstance(budget, dict) and budget.get("max") is not None:
        return int(budget["max"])
    return None


def _expect_list_contains(
    actual: list[Any],
    expected_values: list[str],
    label: str,
    failure_reasons: list[str],
) -> None:
    actual_values = [str(value) for value in actual]
    for value in expected_values:
        if value not in actual_values:
            failure_reasons.append(f"{label} missing expected value: {value}")


def _expect_no_terms(
    text: str,
    terms: list[str],
    label: str,
    failure_reasons: list[str],
) -> None:
    for term in terms:
        if term and term in text:
            failure_reasons.append(f"{label} contains forbidden term: {term}")


def _card_matches_category(card: dict[str, Any], category: str) -> bool:
    product_id = str(card.get("product_id") or "").lower()
    title = str(card.get("title") or "").lower()
    hints = CATEGORY_PRODUCT_ID_HINTS.get(category, [])
    return any(hint in product_id or hint in title for hint in hints)


def _cards_text(cards: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(card, ensure_ascii=False) for card in cards)


def _product_ids(response: dict[str, Any]) -> list[str]:
    return _product_ids_from_cards(response.get("product_cards", []) or [])


def _product_ids_from_cards(cards: list[dict[str, Any]]) -> list[str]:
    return [
        str(card.get("product_id"))
        for card in cards
        if card.get("product_id")
    ]


def _compare_product_ids(
    response: dict[str, Any],
    query_understanding: dict[str, Any],
) -> list[str]:
    ids = query_understanding.get("compare_product_ids") or []
    if ids:
        return [str(product_id) for product_id in ids]

    comparison = trace_step(response, "product_comparison") or {}
    for key in ["compare_product_ids", "requested_product_ids", "returned_product_ids"]:
        value = comparison.get(key) or []
        if value:
            return [str(product_id) for product_id in value]
    return []


def _failure_reason_counts(results: list[EvalResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for failure in result.get("failure_reasons", []):
            for reason in failure.get("reasons", []):
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SmartBuyAgent query-understanding regression eval."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--case", default=None)
    parser.add_argument("--mode", choices=["chat", "stream"], default="chat")
    args = parser.parse_args()

    output = run_eval(
        load_eval_cases(args.cases),
        case_id=args.case,
        mode=args.mode,
    )
    print_report(output)
    if output["summary"]["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
