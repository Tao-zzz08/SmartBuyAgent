from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_query_understanding_intent as intent_eval  # noqa: E402


def test_runner_loads_cases_and_runs_with_previous_memory(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class FakeService:
        def __init__(self, *, llm_enabled: bool) -> None:
            seen["llm_enabled"] = llm_enabled
            self.llm_confidence_threshold = 0.75

        def understand(self, *, query: str, previous_memory=None):
            if query == "增加到5000呢":
                seen["previous_memory_category"] = previous_memory.category
                seen["previous_memory_dialog_state"] = previous_memory.dialog_state
                return FakeResult(
                    {
                        "intent": "shopping_guide",
                        "category": "phone",
                        "budget_max": 5000,
                        "preferences": ["拍照"],
                        "is_follow_up": True,
                        "need_clarification": False,
                        "llm_fallback_attempted": False,
                        "dialog_state": "awaiting_budget",
                        "next_dialog_state": "showing_products",
                    }
                )
            return FakeResult(
                {
                    "intent": "shopping_guide",
                    "category": "phone",
                    "budget_max": 4000,
                    "preferences": ["拍照"],
                    "negative_preferences": ["苹果"],
                    "is_follow_up": False,
                    "need_clarification": False,
                    "llm_fallback_attempted": False,
                    "secondary_intents": ["product_knowledge"],
                    "knowledge_questions": ["为什么像素高不一定拍照好"],
                }
            )

    monkeypatch.setattr(intent_eval, "QueryUnderstandingService", FakeService)
    monkeypatch.setattr(
        intent_eval,
        "decide_llm_fallback",
        lambda **kwargs: SimpleDecision(
            should_call=kwargs["query"] == "增加到5000呢",
            reasons=["ambiguous_follow_up"] if kwargs["query"] == "增加到5000呢" else ["strong_rule"],
        ),
    )
    cases = [
        {
            "id": "explicit",
            "case_group": "explicit_shopping",
            "query": "预算4000，推荐拍照好的手机，不要苹果",
            "previous_memory": None,
            "expected": {
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": 4000,
                "preferences_contains": ["拍照"],
                "negative_preferences_contains": ["苹果"],
            },
            "diagnostic": {
                "expected_secondary_intents": ["product_knowledge"],
                "knowledge_questions_contains": ["为什么像素高不一定拍照好"],
            },
        },
        {
            "id": "followup",
            "case_group": "followup_budget_preference",
            "query": "增加到5000呢",
            "previous_memory": {
                "category": "phone",
                "budget": {"min": None, "max": 4000, "currency": "CNY"},
                "preferences": ["拍照"],
                "negative_preferences": [],
                "last_product_ids": ["p1", "p2"],
                "last_intent": "shopping_guide",
                "dialog_state": "awaiting_budget",
            },
            "expected": {
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": 5000,
                "preferences_contains": ["拍照"],
                "is_follow_up": True,
                "dialog_state": "awaiting_budget",
                "next_dialog_state": "showing_products",
            },
        },
    ]
    case_path = tmp_path / "cases.json"
    case_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    loaded = intent_eval.load_cases(case_path)
    output = intent_eval.run_eval(loaded)

    assert seen["llm_enabled"] is False
    assert seen["previous_memory_category"] == "phone"
    assert seen["previous_memory_dialog_state"] == "awaiting_budget"
    assert output["summary"]["total_cases"] == 2
    assert output["summary"]["passed_cases"] == 2
    assert output["summary"]["metrics"]["intent_accuracy"] == 1.0
    assert output["summary"]["metrics"]["dialog_state_accuracy"] == 1.0
    assert output["summary"]["metrics"]["next_dialog_state_accuracy"] == 1.0
    assert output["results"][1]["actual"]["dialog_state"] == "awaiting_budget"
    assert output["results"][1]["actual"]["next_dialog_state"] == "showing_products"
    assert output["results"][1]["actual"]["llm_fallback_should_call"] is True
    assert output["results"][1]["actual"]["llm_fallback_trigger_reasons"] == ["ambiguous_follow_up"]
    assert output["results"][0]["actual"]["secondary_intents"] == ["product_knowledge"]
    assert output["results"][0]["actual"]["knowledge_questions"] == ["为什么像素高不一定拍照好"]
    assert output["summary"]["diagnostic_metrics"]["multi_intent_case_count"] == 1
    assert output["summary"]["diagnostic_metrics"]["secondary_intent_supported_cases"] == 1
    assert output["summary"]["diagnostic_metrics"]["knowledge_question_supported_cases"] == 1
    json.dumps(output, ensure_ascii=False)


def test_runner_writes_markdown_and_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(intent_eval, "QueryUnderstandingService", PassingFakeService)
    case_path = tmp_path / "cases.json"
    markdown_path = tmp_path / "query-understanding-intent-report.md"
    details_path = tmp_path / "query-understanding-intent-details.json"
    case_path.write_text(
        json.dumps(
            [
                {
                    "id": "case_one",
                    "case_group": "explicit_shopping",
                    "query": "phone",
                    "expected": {"intent": "shopping_guide", "category": "phone"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = intent_eval.main(
        [
            "--cases",
            str(case_path),
            "--output",
            str(markdown_path),
            "--details",
            str(details_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert "# QueryUnderstanding Intent Eval" in markdown
    assert "## Field Metrics" in markdown
    assert details["suite"] == "query_understanding_intent"
    assert details["summary"]["metrics"]["intent_accuracy"] == 1.0


def test_fail_on_regression_returns_nonzero_for_failed_cases(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(intent_eval, "QueryUnderstandingService", PassingFakeService)
    case_path = tmp_path / "cases.json"
    case_path.write_text(
        json.dumps(
            [
                {
                    "id": "failed",
                    "query": "phone",
                    "expected": {"intent": "compare"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = intent_eval.main(
        [
            "--cases",
            str(case_path),
            "--fail-on-regression",
            "--quiet",
        ]
    )

    assert exit_code == 1


class FakeResult:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_trace_dict(self) -> dict:
        return {
            "budget": {},
            "preferences": [],
            "negative_preferences": [],
            "compare_product_ids": [],
            "referenced_product_indices": [],
            **self.payload,
        }


class PassingFakeService:
    def __init__(self, *, llm_enabled: bool) -> None:
        self.llm_enabled = llm_enabled
        self.llm_confidence_threshold = 0.75

    def understand(self, *, query: str, previous_memory=None):
        del query, previous_memory
        return FakeResult(
            {
                "intent": "shopping_guide",
                "category": "phone",
                "budget_max": None,
                "preferences": [],
                "negative_preferences": [],
                "is_follow_up": False,
                "need_clarification": False,
                "llm_fallback_attempted": False,
            }
        )


class SimpleDecision:
    def __init__(self, *, should_call: bool, reasons: list[str]) -> None:
        self.should_call = should_call
        self.reasons = reasons
