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
    SUITE_CASES_PATHS,
    load_eval_cases,
    load_suite_cases,
    query_understanding_from_response,
    run_case,
    run_eval,
)


PHOTO = "拍照"
BATTERY = "续航"
COMMUTE = "通勤"
LIGHTWEIGHT = "轻便"
APPLE = "苹果"
OIL_CONTROL = "控油"
GENTLE_CARE = "温和护理"
MOISTURIZING = "保湿"


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
            {
                "product_ids": [],
                "category": None,
                "budget_max": None,
                "preferences": [],
                "negative_preferences": [],
            },
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

    assert len(cases) >= 10
    assert {
        "phone_budget_three_turns",
        "phone_numeric_budget_follow_up",
        "category_switch_phone_to_shoes",
        "preference_negative_update",
        "llm_fallback_ambiguous_shoes",
        "compare_first_second",
        "budget_follow_up_should_not_route_to_compare",
        "compare_without_previous_products_clarifies",
    } <= {case["id"] for case in cases}
    assert all(case.get("turns") for case in cases)


def test_load_suite_cases_for_multiturn_and_rag() -> None:
    assert load_suite_cases("multiturn")
    assert load_suite_cases("rag")
    assert SUITE_CASES_PATHS["rag"].name == "rag_eval_cases.json"


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
                    "user": "5000呢",
                        "expect": {
                            "category": "phone",
                            "budget_max": 4000,
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
    assert failure["expected"]["budget_max"] == 4000
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
        json={"query": "预算3000，推荐一款拍照好的手机"},
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
    route = _route(query_understanding, session)
    product_cards = _product_cards(query_understanding, session, route)
    citations = _citations(query_understanding)
    trace = [
        {"step": "query_understanding", **query_understanding},
        {"step": "route_by_intent", "route": route, "status": "routed"},
    ]
    if route == "shopping_guide":
        trace.extend(
            [
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
        )
    elif route == "product_knowledge":
        trace.append(
            {
                "step": "knowledge_retrieval",
                "query": _knowledge_query(query_understanding),
                "category": query_understanding["category"],
                "citation_count": len(citations),
            }
        )
    elif route == "compare":
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

    if query_understanding["category"]:
        session["category"] = query_understanding["category"]
    session["budget_max"] = query_understanding.get("budget_max")
    session["preferences"] = query_understanding.get("preferences", [])
    session["negative_preferences"] = query_understanding.get("negative_preferences", [])
    if product_cards:
        session["product_ids"] = [card["product_id"] for card in product_cards]

    return {
        "answer": _answer(query_understanding, route),
        "product_cards": product_cards,
        "citations": citations,
        "trace": trace,
        "session_id": session_id,
    }


def _query_understanding(query: str, session: dict[str, Any]) -> dict[str, Any]:
    category = session.get("category")
    budget_max = session.get("budget_max")
    preferences = list(session.get("preferences") or [])
    negative_preferences = list(session.get("negative_preferences") or [])
    intent = "shopping_guide"
    source = "rule"
    is_follow_up = bool(category)
    need_clarification = False
    compare_product_ids: list[str] = []
    referenced_product_indices: list[int] = []

    if _is_knowledge_query(query):
        intent = "product_knowledge"
        category = _category_from_query(query) or category
        preferences = _knowledge_preferences(query, category)
    elif "第一个" in query and "第二个" in query:
        intent = "compare"
        referenced_product_indices = [1, 2]
        compare_product_ids = list(session.get("product_ids") or [])[:2]
        is_follow_up = True
        if not compare_product_ids:
            need_clarification = True
    else:
        query_category = _category_from_query(query)
        if query_category:
            category = query_category
            if "换成" in query or "看看" in query:
                preferences = []
        budget_max = _number_budget(query) or budget_max
        if "拍照" in query:
            preferences = _dedupe([*preferences, PHOTO])
        if "续航" in query:
            preferences = _dedupe([*preferences, BATTERY])
        if "通勤" in query:
            preferences = _dedupe([*preferences, COMMUTE])
        if "轻便" in query or "别太重" in query:
            preferences = _dedupe([*preferences, LIGHTWEIGHT])
        if "不考虑苹果" in query or "不要苹果" in query:
            negative_preferences = _dedupe([*negative_preferences, APPLE])
        if "贵一点" in query:
            category = "shoes"
            budget_max = 1200
            preferences = _dedupe([COMMUTE, LIGHTWEIGHT])
            source = "mixed"
            is_follow_up = True
        if category == "skincare":
            preferences = _safe_skincare_preferences(query, preferences)

    effective_query = _effective_query(category, budget_max, preferences, negative_preferences)
    if intent == "product_knowledge":
        effective_query = _knowledge_query(
            {
                "category": category,
                "budget_max": budget_max,
                "preferences": preferences,
            }
        )
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
        "reason": "rule" if source == "rule" else "llm_slot_fallback",
        "need_clarification": need_clarification,
    }


def _route(query_understanding: dict[str, Any], session: dict[str, Any]) -> str:
    if query_understanding.get("need_clarification"):
        return "clarification"
    intent = query_understanding["intent"]
    if intent == "compare" and not query_understanding.get("compare_product_ids"):
        return "clarification"
    if intent == "product_knowledge":
        return "product_knowledge"
    return intent


def _product_cards(
    query_understanding: dict[str, Any],
    session: dict[str, Any],
    route: str,
) -> list[dict[str, Any]]:
    category = query_understanding["category"]
    if route == "compare":
        return [_card(product_id, category or "phone") for product_id in query_understanding["compare_product_ids"]]
    if route != "shopping_guide":
        return []
    if query_understanding["original_query"].startswith("请告诉我某某不存在成分"):
        return []
    if category == "phone":
        cards = [_card("phone_001", "phone"), _card("phone_002", "phone"), _card("phone_003", "phone")]
    elif category == "shoes":
        cards = [_card("shoes_001", "shoes"), _card("shoes_002", "shoes")]
    elif category == "skincare":
        cards = [_card("skincare_001", "skincare")]
    else:
        cards = []
    if APPLE in query_understanding.get("negative_preferences", []):
        cards = [card for card in cards if "Apple" not in json.dumps(card, ensure_ascii=False)]
    return cards


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
    text = _citation_text(category, query_understanding)
    if query_understanding["original_query"].startswith("请告诉我某某不存在成分"):
        return []
    return [
        {
            "chunk_id": f"chunk_{category}_001",
            "title": f"{category} guide",
            "section": "guide",
            "section_path": f"{category}/guide",
            "source_file": f"data/knowledge_docs/{category}/guide.md",
            "content_preview": text,
            "score": 0.9,
        }
    ]


def _citation_text(category: str, query_understanding: dict[str, Any]) -> str:
    if category == "phone" and BATTERY in query_understanding.get("preferences", []):
        return "手机续航需要看电池容量、快充、功耗和系统调度。"
    if category == "phone":
        return "手机拍照主要看传感器、光圈、防抖、影像算法和夜景表现。"
    if category == "shoes" and "防滑" in query_understanding["original_query"]:
        return "通勤鞋防滑主要看鞋底材质、纹路深度和抓地表现。"
    if category == "shoes":
        return "鞋子尺码需要结合脚长、脚宽、鞋楦和试穿反馈。"
    if category == "skincare":
        return "敏感肌日常护理应关注温和、保湿、低刺激，清爽控油也要避免过度清洁。"
    return "通用导购知识。"


def _knowledge_query(query_understanding: dict[str, Any]) -> str:
    if query_understanding["category"] == "skincare":
        return "护肤 清爽 控油 温和护理 保湿 敏感肌"
    return " ".join(
        [
            str(query_understanding.get("category") or ""),
            *query_understanding.get("preferences", []),
            str(query_understanding.get("budget_max") or ""),
        ]
    )


def _answer(query_understanding: dict[str, Any], route: str) -> str:
    if route == "clarification":
        return "我还没有上一轮可引用的商品，请先让我推荐几款商品后再比较。"
    if route == "compare":
        return "下面只基于上一轮推荐的真实商品做比较。"
    if query_understanding["original_query"].startswith("请告诉我某某不存在成分"):
        return "这个问题缺少可靠商品知识依据，我不能编造临床结论。建议改问日常护理注意事项。"
    category = query_understanding["category"]
    if category == "phone" and BATTERY in query_understanding.get("preferences", []):
        return "手机续航主要看电池容量、快充、功耗和系统调度。"
    if category == "phone":
        return "拍照表现可以看传感器、防抖和影像算法，也会结合真实商品候选说明。"
    if category == "shoes" and "防滑" in query_understanding["original_query"]:
        return "通勤鞋防滑主要看鞋底纹路和抓地表现。"
    if category == "shoes":
        return "鞋子尺码建议结合脚长、脚宽、鞋楦和试穿反馈。"
    if category == "skincare":
        return "敏感肌日常护理建议关注清爽、控油、温和和保湿方向。"
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


def _category_from_query(query: str) -> str | None:
    if "手机" in query:
        return "phone"
    if "鞋" in query:
        return "shoes"
    if "护肤" in query or "敏感肌" in query or "皮肤" in query or "成分" in query:
        return "skincare"
    return None


def _is_knowledge_query(query: str) -> bool:
    return any(token in query for token in ["主要看", "怎么选", "应该注意", "尺码"])


def _knowledge_preferences(query: str, category: str | None) -> list[str]:
    if category == "phone" and "续航" in query:
        return [BATTERY]
    if category == "phone":
        return [PHOTO]
    if category == "skincare":
        return ["敏感肌", MOISTURIZING]
    if category == "shoes" and "防滑" in query:
        return ["防滑", COMMUTE]
    if category == "shoes":
        return []
    return []


def _safe_skincare_preferences(query: str, preferences: list[str]) -> list[str]:
    values = list(preferences)
    if any(term in query for term in ["治疗", "痘痘", "药效"]):
        values.extend([OIL_CONTROL, GENTLE_CARE])
    if "敏感肌" in query:
        values.extend(["敏感肌", MOISTURIZING])
    return _dedupe([value for value in values if value not in {"治疗", "治愈", "药效"}])


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
