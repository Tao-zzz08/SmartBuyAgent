from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_eval_all  # noqa: E402


def test_suite_result_from_output_normalizes_failures() -> None:
    output = {
        "results": [
            {
                "id": "phone_budget_case",
                "passed": False,
                "turn_count": 1,
                "failure_reasons": [
                    {
                        "turn_index": 1,
                        "user": "增加到5000呢",
                        "reasons": [
                            "budget_max mismatch: expected 5000, got null",
                            "route mismatch: expected shopping_guide, got compare",
                        ],
                    }
                ],
            }
        ],
        "summary": {
            "total_cases": 1,
            "passed_cases": 0,
            "failed_cases": 1,
            "total_turns": 1,
            "passed_turns": 0,
            "failed_turns": 1,
        },
    }

    result = run_eval_all.suite_result_from_output(
        "query_understanding",
        output,
        duration_ms=12,
    )

    assert result.total_cases == 1
    assert result.failed_cases == 1
    assert result.total_turns == 1
    assert result.failure_reason_counts == {
        "budget_mismatch": 1,
        "route_mismatch": 1,
    }
    assert [failure.reason for failure in result.failures] == [
        "budget_mismatch",
        "route_mismatch",
    ]


def test_write_report_files_creates_markdown_and_json(tmp_path: Path) -> None:
    report = run_eval_all.EvalReport(
        generated_at="2026-07-01T00:00:00+00:00",
        total_suites=2,
        completed_suites=1,
        skipped_suites=1,
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        pass_rate=0.5,
        suites=[
            run_eval_all.EvalSuiteResult(
                suite="query_understanding",
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
                duration_ms=10,
            ),
            run_eval_all.EvalSuiteResult(
                suite="retrieval",
                total_cases=1,
                passed_cases=0,
                failed_cases=1,
                metrics={
                    "recall_at_5": 0.75,
                    "ndcg_at_5": 0.8,
                    "mrr_at_5": 1.0,
                    "filter_compliance_rate": 0.5,
                },
                failure_reason_counts={"insufficient_results": 1},
                failures=[
                    run_eval_all.EvalFailure(
                        case_id="retrieval_phone",
                        suite="retrieval",
                        reason="insufficient_results",
                        details={"min_results": 1, "actual_results": 0},
                    )
                ],
                duration_ms=20,
            ),
        ],
    )
    markdown_path = tmp_path / "nested" / "eval-report.md"
    json_path = tmp_path / "nested" / "eval-details.json"

    run_eval_all.write_report_files(
        report,
        output_path=markdown_path,
        details_path=json_path,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    details = json.loads(json_path.read_text(encoding="utf-8"))
    assert "# SmartBuyAgent Eval Report" in markdown
    assert "## Summary" in markdown
    assert "## Suite Summary" in markdown
    assert "## Suite Metrics" in markdown
    assert "## Failed Cases" in markdown
    assert "Completed Suites" in markdown
    assert "Skipped Suites" in markdown
    assert "insufficient_results" in markdown
    assert "recall_at_5" in markdown
    assert "ndcg_at_5" in markdown
    assert details["completed_suites"] == 1
    assert details["skipped_suites"] == 1
    assert details["total_cases"] == 2
    assert details["failed_cases"] == 1
    assert details["pass_rate"] == 0.5
    assert details["suites"][1]["pass_rate"] == 0.0
    assert details["suites"][1]["metrics"]["recall_at_5"] == 0.75
    assert details["suites"][1]["failures"][0]["case_id"] == "retrieval_phone"


def test_build_eval_report_respects_suite_filter(monkeypatch) -> None:
    seen: list[str] = []

    def fake_run_suite_result(suite: str) -> run_eval_all.EvalSuiteResult:
        seen.append(suite)
        return run_eval_all.EvalSuiteResult(
            suite=suite,
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
        )

    monkeypatch.setattr(run_eval_all, "run_suite_result", fake_run_suite_result)

    report = run_eval_all.build_eval_report(["query_understanding"])

    assert seen == ["query_understanding"]
    assert report.total_suites == 1
    assert report.completed_suites == 1
    assert report.skipped_suites == 0
    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0


def test_fail_on_regression_passes_without_failed_or_skipped_suites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run_suite_result(suite: str) -> run_eval_all.EvalSuiteResult:
        return run_eval_all.EvalSuiteResult(
            suite=suite,
            total_cases=3,
            passed_cases=3,
            failed_cases=0,
            duration_ms=5,
        )

    monkeypatch.setattr(run_eval_all, "run_suite_result", fake_run_suite_result)
    markdown_path = tmp_path / "reports" / "eval-report.md"
    json_path = tmp_path / "reports" / "eval-details.json"

    exit_code = run_eval_all.main(
        [
            "--suite",
            "query_understanding",
            "--output",
            str(markdown_path),
            "--details",
            str(json_path),
            "--fail-on-regression",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert markdown_path.exists()
    assert json_path.exists()
    details = json.loads(json_path.read_text(encoding="utf-8"))
    assert [suite["suite"] for suite in details["suites"]] == ["query_understanding"]
    assert details["completed_suites"] == 1
    assert details["skipped_suites"] == 0
    assert details["passed_cases"] == 3


def test_fail_on_regression_fails_when_suite_skipped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run_suite_result(suite: str) -> run_eval_all.EvalSuiteResult:
        return run_eval_all.EvalSuiteResult(
            suite=suite,
            status="skipped",
            reason="existing standalone runner unavailable: fake missing dependency",
        )

    monkeypatch.setattr(run_eval_all, "run_suite_result", fake_run_suite_result)
    markdown_path = tmp_path / "reports" / "eval-report.md"
    json_path = tmp_path / "reports" / "eval-details.json"

    exit_code = run_eval_all.main(
        [
            "--suite",
            "query_understanding",
            "--output",
            str(markdown_path),
            "--details",
            str(json_path),
            "--fail-on-regression",
            "--quiet",
        ]
    )

    assert exit_code == 1
    details = json.loads(json_path.read_text(encoding="utf-8"))
    assert details["completed_suites"] == 0
    assert details["skipped_suites"] == 1
    assert details["failed_cases"] == 0


def test_fail_on_regression_fails_when_cases_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run_suite_result(suite: str) -> run_eval_all.EvalSuiteResult:
        return run_eval_all.EvalSuiteResult(
            suite=suite,
            total_cases=3,
            passed_cases=2,
            failed_cases=1,
        )

    monkeypatch.setattr(run_eval_all, "run_suite_result", fake_run_suite_result)
    markdown_path = tmp_path / "reports" / "eval-report.md"
    json_path = tmp_path / "reports" / "eval-details.json"

    exit_code = run_eval_all.main(
        [
            "--suite",
            "query_understanding",
            "--output",
            str(markdown_path),
            "--details",
            str(json_path),
            "--fail-on-regression",
            "--quiet",
        ]
    )

    assert exit_code == 1


def test_markdown_handles_no_failed_cases() -> None:
    report = run_eval_all.EvalReport(
        generated_at="2026-07-01T00:00:00+00:00",
        total_suites=1,
        completed_suites=1,
        skipped_suites=0,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        suites=[
            run_eval_all.EvalSuiteResult(
                suite="grounding_guard",
                total_cases=1,
                passed_cases=1,
                failed_cases=0,
            )
        ],
    )

    markdown = run_eval_all.render_markdown(report)

    assert "No failures were recorded." in markdown
    assert "No failed cases." in markdown
    assert "No suite metrics were recorded." in markdown
    assert "grounding_guard" in markdown


def test_suite_result_from_output_preserves_metrics() -> None:
    output = {
        "results": [],
        "summary": {
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "metrics": {
                "recall_at_5": 0.84,
                "ndcg_at_5": 0.76,
                "mrr_at_5": 0.91,
            },
        },
    }

    result = run_eval_all.suite_result_from_output("retrieval", output)

    assert result.metrics == {
        "recall_at_5": 0.84,
        "ndcg_at_5": 0.76,
        "mrr_at_5": 0.91,
    }


def test_eval_report_renders_multiturn_session_metrics() -> None:
    output = {
        "results": [],
        "summary": {
            "total_cases": 2,
            "passed_cases": 2,
            "failed_cases": 0,
            "metrics": {
                "session_success_rate": 1.0,
                "context_carryover_accuracy": 1.0,
                "category_switch_accuracy": 0.5,
                "compare_resolution_accuracy": 1.0,
                "clarification_accuracy": 1.0,
                "route_stability_rate": 1.0,
                "evaluated_sessions": 2,
                "failed_sessions": 0,
            },
        },
    }

    suite = run_eval_all.suite_result_from_output("multiturn", output)
    report = run_eval_all.EvalReport(
        generated_at="2026-07-01T00:00:00+00:00",
        total_suites=1,
        completed_suites=1,
        skipped_suites=0,
        total_cases=2,
        passed_cases=2,
        failed_cases=0,
        pass_rate=1.0,
        suites=[suite],
    )

    markdown = run_eval_all.render_markdown(report)
    details = run_eval_all.report_to_json(report)

    assert suite.metrics["session_success_rate"] == 1.0
    assert "### multiturn" in markdown
    assert "session_success_rate" in markdown
    assert details["suites"][0]["metrics"]["category_switch_accuracy"] == 0.5
