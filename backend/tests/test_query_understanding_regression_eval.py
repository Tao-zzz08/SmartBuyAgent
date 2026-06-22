from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_query_understanding_eval import (  # noqa: E402
    DEFAULT_CASES_PATH,
    load_eval_cases,
    query_understanding_from_response,
    run_case,
    run_eval,
)


PHOTO = "\u62cd\u7167"
BATTERY = "\u7eed\u822a"
COMMUTE = "\u901a\u52e4"
LIGHTWEIGHT = "\u8f7b\u4fbf"
APPLE = "\u82f9\u679c"
OIL_CONTROL = "\u63a7\u6cb9"
GENTLE_CARE = "\u6e29\u548c\u62a4\u7406"


class FakeResponse:
    def __init__(self, body: dict[str, Any], *, stream: bool = False) -> None:
        self.status_code = 200
        self._body = body
        self.text = _sse_text(body) if stream else json.dumps(body, ensure_ascii=False)

    def json(self) -> dict[str, Any]:
        return self._body


class RegressionFakeClient:
    def __init__(self) -> None:
        self._session_counter = 0
        self._sessions: dict[str, dict[str, Any]] = {}

    def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
        session_id = json.get("session_id") or self._new_session_id()
        session = self._sessions.setdefault(
            session_id,
            {"product_ids": [], "category": None, "budget_max": None, "preferences": []},
        )
        body = _build_response(
            query=str(json["query"]),
            session_id=session_id,
            session=session,
        )
        return FakeResponse(body, stream=path.endswith("/stream"))

    def _new_session_id(self) -> str:
        self._session_counter += 1
        return f"session_{self._session_counter}"


def test_load_query_understanding_regression_cases() -> None:
    cases = load_eval_cases(DEFAULT_CASES_PATH)

    assert len(cases) >= 8
    assert {
        "phone_budget_three_turns",
        "phone_numeric_budget_follow_up",
        "category_switch_phone_to_shoes",
        "preference_negative_update",
        "llm_fallback_ambiguous_shoes",
        "compare_first_second",
        "skincare_medical_safety",
        "boundary_product_cards_citations",
    } <= {case["id"] for case in cases}
    assert all(case.get("turns") for case in cases)


def test_query_understanding_regression_cases_pass_with_fake_client() -> None:
    cases = load_eval_cases(DEFAULT_CASES_PATH)

    output = run_eval(cases, client=RegressionFakeClient())

    assert output["summary"]["failed_cases"] == 0
    assert output["summary"]["failed_turns"] == 0
    assert output["summary"]["passed_cases"] == len(cases)


def test_regression_eval_reports_actionable_failure_details() -> None:
    cases = [
        {
            "id": "failing_budget",
            "description": "Intentional failure for report shape.",
            "turns": [
                {
                    "user": "5000\u5462",
                    "expect": {
                        "category": "phone",
                        "budget_max": 5000,
                        "preferences_contains": [PHOTO],
                    },
                }
            ],
        }
    ]
    client = RegressionFakeClient()

    output = run_eval(cases, client=client)
    failure = output["results"][0]["failure_reasons"][0]

    assert output["summary"]["failed_cases"] == 1
    assert failure["turn_index"] == 1
    assert failure["expected"]["budget_max"] == 5000
    assert any("budget_max mismatch" in reason for reason in failure["reasons"])
    assert "actual_query_understanding" in failure
    assert "actual_product_cards" in failure


def test_stream_and_non_stream_consistency_case() -> None:
    cases = load_eval_cases(DEFAULT_CASES_PATH)
    case = next(case for case in cases if case["id"] == "stream_non_stream_consistency")

    chat_result = run_case(RegressionFakeClient(), case, mode="chat")
    stream_result = run_case(RegressionFakeClient(), case, mode="stream")

    assert chat_result["passed"] is True
    assert stream_result["passed"] is True
    chat_qu = chat_result["turns"][-1]["query_understanding"]
    stream_qu = stream_result["turns"][-1]["query_understanding"]
    assert _qu_signature(chat_qu) == _qu_signature(stream_qu)


def test_query_understanding_from_stream_result_trace() -> None:
    client = RegressionFakeClient()
    response = client.post(
        "/api/chat/stream",
        json={"query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"},
    )
    events = response.text

    assert "event: trace" in events
    assert "event: result" in events
    assert query_understanding_from_response(response.json())["category"] == "phone"


def _build_response(
    *,
    query: str,
    session_id: str,
    session: dict[str, Any],
) -> dict[str, Any]:
    query_understanding = _query_understanding(query, session)
    product_cards = _product_cards(query_understanding, session)
    citations = _citations(query_understanding)
    trace = [
        {"step": "query_understanding", **query_understanding},
        {
            "step": "product_retrieval",
            "query": query_understanding["effective_query"],
            "category": query_understanding["category"],
            "structured_filters": {
                "category": query_understanding["category"],
                "budget_max": query_understanding.get("budget_max"),
                "preferences": query_understanding.get("preferences", []),
                "negative_preferences": query_understanding.get(
                    "negative_preferences",
                    [],
                ),
            },
            "candidate_count": len(product_cards),
            "product_ids": [card["product_id"] for card in product_cards],
        },
        {
            "step": "knowledge_retrieval",
            "query": _knowledge_query(query_understanding),
            "category": query_understanding["category"],
            "citation_count": len(citations),
        },
    ]
    if query_understanding["intent"] == "compare":
        trace.append(
            {
                "step": "product_comparison",
                "status": "compared",
                "compare_product_ids": query_understanding["compare_product_ids"],
                "requested_product_ids": query_understanding["compare_product_ids"],
                "returned_product_ids": query_understanding["compare_product_ids"],
                "referenced_product_indices": query_understanding[
                    "referenced_product_indices"
                ],
                "resolved_from_last_products": True,
                "comparison_product_count": len(product_cards),
            }
        )

    session["category"] = query_understanding["category"]
    session["budget_max"] = query_understanding.get("budget_max")
    session["preferences"] = query_understanding.get("preferences", [])
    if product_cards:
        session["product_ids"] = [card["product_id"] for card in product_cards]

    return {
        "answer": _answer(query_understanding),
        "product_cards": product_cards,
        "citations": citations,
        "trace": trace,
        "session_id": session_id,
    }


def _query_understanding(query: str, session: dict[str, Any]) -> dict[str, Any]:
    category = session.get("category")
    budget_max = session.get("budget_max")
    preferences = list(session.get("preferences") or [])
    negative_preferences: list[str] = []
    intent = "shopping_guide"
    source = "rule"
    is_follow_up = bool(category)
    reason = "rule"
    compare_product_ids: list[str] = []
    referenced_product_indices: list[int] = []

    if "鞋" in query:
        category = "shoes"
        if COMMUTE not in preferences:
            preferences = [COMMUTE] if "通勤" in query else []
        if "换成" in query:
            preferences = []
        budget_max = _number_budget(query) or budget_max
    elif "护肤" in query:
        category = "skincare"
        budget_max = _number_budget(query) or 300
        preferences = [OIL_CONTROL, GENTLE_CARE]
    elif "手机" in query:
        category = "phone"
        budget_max = _number_budget(query) or budget_max
        preferences = [PHOTO]

    if "4000" in query:
        budget_max = 4000
        is_follow_up = True
    if "5000" in query and category == "phone":
        budget_max = 5000
        is_follow_up = True
    if "续航" in query:
        preferences = _dedupe([*preferences, BATTERY])
    if "不考虑苹果" in query:
        negative_preferences = [APPLE]
    if "贵一点" in query:
        category = "shoes"
        budget_max = 1200
        preferences = _dedupe([COMMUTE, LIGHTWEIGHT])
        source = "mixed"
        is_follow_up = True
    if "第一个" in query and "第二个" in query:
        intent = "compare"
        referenced_product_indices = [1, 2]
        compare_product_ids = list(session.get("product_ids") or [])[:2]
        is_follow_up = True

    effective_query = _effective_query(category, budget_max, preferences, negative_preferences)
    if category is None:
        effective_query = query

    return {
        "original_query": query,
        "effective_query": effective_query,
        "is_follow_up": is_follow_up,
        "intent": intent,
        "category": category,
        "category_id": f"cat_{category}" if category else None,
        "budget": {"min": None, "max": budget_max, "currency": "CNY"},
        "budget_min": None,
        "budget_max": budget_max,
        "preferences": preferences,
        "negative_preferences": negative_preferences,
        "compare_product_ids": compare_product_ids,
        "referenced_product_indices": referenced_product_indices,
        "source": source,
        "confidence": 0.9 if source == "rule" else 0.82,
        "reason": reason,
    }


def _product_cards(
    query_understanding: dict[str, Any],
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    category = query_understanding["category"]
    if query_understanding["intent"] == "compare":
        cards = []
        previous_ids = list(session.get("product_ids") or [])
        for product_id in previous_ids[:2]:
            cards.append(_card(product_id, category or "phone"))
        return cards
    if category == "phone":
        return [_card("phone_001", "phone"), _card("phone_002", "phone"), _card("phone_003", "phone")]
    if category == "shoes":
        return [_card("shoes_001", "shoes"), _card("shoes_002", "shoes")]
    if category == "skincare":
        return [_card("skincare_001", "skincare")]
    return []


def _card(product_id: str, category: str) -> dict[str, Any]:
    title_by_category = {
        "phone": f"{PHOTO} phone",
        "shoes": "commute shoes",
        "skincare": "skincare lotion",
    }
    return {
        "product_id": product_id,
        "title": title_by_category[category],
        "brand": "Demo",
        "price": 2999 if category == "phone" else 699,
        "image_url": None,
        "tags": [PHOTO] if category == "phone" else [COMMUTE],
        "attributes": {},
        "source_url": None,
        "compare_url": None,
        "recommend_reason": "retrieval grounded candidate",
    }


def _citations(query_understanding: dict[str, Any]) -> list[dict[str, Any]]:
    category = query_understanding["category"] or "general"
    return [
        {
            "chunk_id": f"chunk_{category}_001",
            "title": f"{category} guide",
            "section": "guide",
            "section_path": f"{category}/guide",
            "source_file": f"data/knowledge_docs/{category}/guide.md",
            "content_preview": _knowledge_query(query_understanding),
            "score": 0.9,
        }
    ]


def _knowledge_query(query_understanding: dict[str, Any]) -> str:
    if query_understanding["category"] == "skincare":
        return f"skincare {OIL_CONTROL} {GENTLE_CARE}"
    return " ".join(
        [
            str(query_understanding.get("category") or ""),
            *query_understanding.get("preferences", []),
            str(query_understanding.get("budget_max") or ""),
        ]
    )


def _answer(query_understanding: dict[str, Any]) -> str:
    if query_understanding["intent"] == "compare":
        return "下面只基于上一轮推荐的商品做比较。"
    if query_understanding["category"] == "skincare":
        return f"可以优先看{OIL_CONTROL}、{GENTLE_CARE}方向的日常护理产品。"
    return "已根据结构化导购条件整理候选商品。"


def _effective_query(
    category: str | None,
    budget_max: int | None,
    preferences: list[str],
    negative_preferences: list[str],
) -> str:
    if not category:
        return ""
    category_label = {"phone": "手机", "shoes": "鞋靴", "skincare": "护肤品"}[category]
    parts = []
    if budget_max:
        parts.append(f"预算{budget_max}元以内")
    if preferences:
        parts.append("推荐" + "、".join(preferences) + "的" + category_label)
    else:
        parts.append("推荐" + category_label)
    if negative_preferences:
        parts.append("不考虑" + "、".join(negative_preferences))
    return "，".join(parts)


def _number_budget(query: str) -> int | None:
    for value in [5000, 4000, 3000, 1200, 800, 300]:
        if str(value) in query:
            return value
    return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _sse_text(body: dict[str, Any]) -> str:
    events = [
        ("session", {"session_id": body["session_id"], "request_id": "req_test"}),
        ("trace", body["trace"][0]),
        ("result", body),
        ("done", {"status": "ok"}),
    ]
    return "".join(
        f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        for event, data in events
    )


def _qu_signature(query_understanding: dict[str, Any]) -> tuple[Any, ...]:
    return (
        query_understanding.get("category"),
        query_understanding.get("budget_max"),
        tuple(query_understanding.get("preferences") or []),
        tuple(query_understanding.get("negative_preferences") or []),
    )
