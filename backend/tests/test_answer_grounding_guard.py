from __future__ import annotations

import json
from pathlib import Path

from app.services.answer_grounding_guard import (
    AnswerGroundingContext,
    AnswerGroundingGuard,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_answer_grounding_guard_cases() -> None:
    guard = AnswerGroundingGuard()

    for case in _load_cases():
        result = guard.check(AnswerGroundingContext(**case["context"]))
        expect = case["expect"]
        assert result.action == expect["action"], case["id"]
        assert result.passed is (expect["action"] == "pass"), case["id"]

        actual_types = {violation.type for violation in result.violations}
        for violation_type in expect.get("violation_types") or []:
            assert violation_type in actual_types, case["id"]

        if expect.get("fallback_forbidden"):
            assert result.fallback_answer
            for term in expect["fallback_forbidden"]:
                assert term not in result.fallback_answer, case["id"]


def _load_cases() -> list[dict]:
    return json.loads(
        (FIXTURES_DIR / "answer_grounding_guard_cases.json").read_text(
            encoding="utf-8"
        )
    )
