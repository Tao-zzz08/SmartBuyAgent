from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


DEFAULT_SUITES = [
    "query_understanding",
    "retrieval",
    "rag",
    "multiturn",
    "grounding_guard",
]
SUITE_CHOICES = [*DEFAULT_SUITES, "all"]


@dataclass
class EvalFailure:
    case_id: str
    suite: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSuiteResult:
    suite: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    total_turns: int | None = None
    passed_turns: int | None = None
    failed_turns: int | None = None
    failure_reason_counts: dict[str, int] = field(default_factory=dict)
    failures: list[EvalFailure] = field(default_factory=list)
    duration_ms: int | None = None
    status: str = "completed"
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return _rate(self.passed_cases, self.total_cases)

    @property
    def turn_pass_rate(self) -> float | None:
        if self.total_turns is None or self.passed_turns is None:
            return None
        return _rate(self.passed_turns, self.total_turns)


@dataclass
class EvalReport:
    generated_at: str
    total_suites: int
    completed_suites: int
    skipped_suites: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    suites: list[EvalSuiteResult]


def build_eval_report(suites: list[str] | None = None) -> EvalReport:
    selected = DEFAULT_SUITES if not suites or suites == ["all"] else suites
    suite_results = [run_suite_result(suite) for suite in selected]
    total_cases = sum(result.total_cases for result in suite_results)
    passed_cases = sum(result.passed_cases for result in suite_results)
    failed_cases = sum(result.failed_cases for result in suite_results)
    completed_suites = sum(1 for result in suite_results if result.status == "completed")
    skipped_suites = sum(1 for result in suite_results if result.status == "skipped")
    return EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_suites=len(suite_results),
        completed_suites=completed_suites,
        skipped_suites=skipped_suites,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=_rate(passed_cases, total_cases),
        suites=suite_results,
    )


def run_suite_result(suite: str) -> EvalSuiteResult:
    started = time.perf_counter()
    try:
        from run_query_understanding_eval import run_suite

        output = run_suite(suite)
        duration_ms = _elapsed_ms(started)
        return suite_result_from_output(suite, output, duration_ms=duration_ms)
    except Exception as exc:  # pragma: no cover - exercised by CLI environments.
        return EvalSuiteResult(
            suite=suite,
            status="skipped",
            reason=f"existing standalone runner unavailable: {exc}",
            duration_ms=_elapsed_ms(started),
        )


def suite_result_from_output(
    suite: str,
    output: dict[str, Any],
    *,
    duration_ms: int | None = None,
) -> EvalSuiteResult:
    summary = output.get("summary") or {}
    results = output.get("results") or []
    failures = _extract_failures(suite, results)
    if not failures and int(summary.get("failed_cases", 0) or 0):
        failures = [
            EvalFailure(
                case_id=str(case_id),
                suite=suite,
                reason="unknown_failure",
                details={"failed_case_id": case_id},
            )
            for case_id in summary.get("failed_case_ids", [])
        ]

    reason_counts = _count_reasons(failures)
    if not reason_counts:
        reason_counts = {
            _normalize_reason(str(reason)): int(count)
            for reason, count in (summary.get("failure_reason_counts") or {}).items()
        }

    return EvalSuiteResult(
        suite=suite,
        total_cases=int(summary.get("total_cases", len(results)) or 0),
        passed_cases=int(summary.get("passed_cases", 0) or 0),
        failed_cases=int(summary.get("failed_cases", 0) or 0),
        total_turns=_optional_int(summary.get("total_turns")),
        passed_turns=_optional_int(summary.get("passed_turns")),
        failed_turns=_optional_int(summary.get("failed_turns")),
        failure_reason_counts=reason_counts,
        failures=failures,
        duration_ms=duration_ms,
        status="completed",
        metrics=summary.get("metrics") or output.get("metrics") or {},
    )


def write_report_files(
    report: EvalReport,
    *,
    output_path: str | Path,
    details_path: str | Path,
) -> None:
    output = Path(output_path)
    details = Path(details_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    details.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    details.write_text(
        json.dumps(report_to_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# SmartBuyAgent Eval Report",
        "",
        f"Generated at: {report.generated_at}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Suites | {report.total_suites} |",
        f"| Completed Suites | {report.completed_suites} |",
        f"| Skipped Suites | {report.skipped_suites} |",
        f"| Total Cases | {report.total_cases} |",
        f"| Passed Cases | {report.passed_cases} |",
        f"| Failed Cases | {report.failed_cases} |",
        f"| Pass Rate | {_percent(report.pass_rate)} |",
        "",
        "## Suite Summary",
        "",
        "| Suite | Status | Total | Passed | Failed | Pass Rate | Duration |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for suite in report.suites:
        lines.append(
            "| {suite} | {status} | {total} | {passed} | {failed} | {rate} | {duration} |".format(
                suite=suite.suite,
                status=suite.status,
                total=suite.total_cases,
                passed=suite.passed_cases,
                failed=suite.failed_cases,
                rate=_percent(suite.pass_rate),
                duration=_format_duration(suite.duration_ms),
            )
        )

    lines.extend(["", "## Failure Reason Counts", ""])
    any_counts = False
    for suite in report.suites:
        if not suite.failure_reason_counts:
            continue
        any_counts = True
        lines.extend(
            [
                f"### {suite.suite}",
                "",
                "| Reason | Count |",
                "|---|---:|",
            ]
        )
        for reason, count in sorted(suite.failure_reason_counts.items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    if not any_counts:
        lines.extend(["No failures were recorded.", ""])

    lines.extend(["## Suite Metrics", ""])
    any_metrics = False
    for suite in report.suites:
        if not suite.metrics:
            continue
        any_metrics = True
        lines.extend(
            [
                f"### {suite.suite}",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for key, value in sorted(suite.metrics.items()):
            lines.append(f"| {key} | {_format_metric(value)} |")
        lines.append("")
    if not any_metrics:
        lines.extend(["No suite metrics were recorded.", ""])

    lines.extend(["## Failed Cases", ""])
    any_failures = False
    for suite in report.suites:
        if suite.status == "skipped":
            any_failures = True
            lines.extend(
                [
                    f"### {suite.suite}",
                    "",
                    f"- skipped: {suite.reason or 'suite runner unavailable'}",
                    "",
                ]
            )
            continue
        if not suite.failures:
            continue
        any_failures = True
        lines.extend([f"### {suite.suite}", ""])
        for failure in suite.failures:
            details = _compact_json(failure.details)
            lines.extend(
                [
                    f"- `{failure.case_id}`",
                    f"  - reason: {failure.reason}",
                    f"  - details: {details}",
                ]
            )
        lines.append("")

    if not any_failures:
        lines.extend(["No failed cases.", ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- This report is deterministic.",
            "- No external LLM API is required by the report layer.",
            "- No external network is required by the report layer.",
            "- Skipped suites indicate that their existing standalone runner could not be loaded in the current environment.",
            "",
        ]
    )
    return "\n".join(lines)


def report_to_json(report: EvalReport) -> dict[str, Any]:
    data = asdict(report)
    data["pass_rate"] = report.pass_rate
    data["suites"] = []
    for suite in report.suites:
        suite_data = asdict(suite)
        suite_data["pass_rate"] = suite.pass_rate
        suite_data["turn_pass_rate"] = suite.turn_pass_rate
        data["suites"].append(suite_data)
    return data


def _extract_failures(suite: str, results: list[dict[str, Any]]) -> list[EvalFailure]:
    failures: list[EvalFailure] = []
    for result in results:
        if result.get("passed") is True:
            continue
        case_id = str(result.get("id") or result.get("case_id") or "unknown_case")
        raw_reasons = result.get("failure_reasons") or result.get("failures") or []
        if not raw_reasons:
            failures.append(
                EvalFailure(
                    case_id=case_id,
                    suite=suite,
                    reason="unknown_failure",
                    details=_summarize_details(result),
                )
            )
            continue

        for raw_reason in raw_reasons:
            if isinstance(raw_reason, dict):
                reasons = raw_reason.get("reasons") or raw_reason.get("failure_reasons")
                if isinstance(reasons, list) and reasons:
                    for reason in reasons:
                        failures.append(
                            EvalFailure(
                                case_id=case_id,
                                suite=suite,
                                reason=_normalize_reason(str(reason)),
                                details=_summarize_details(raw_reason),
                            )
                        )
                else:
                    reason = raw_reason.get("reason") or raw_reason.get("type") or "unknown_failure"
                    failures.append(
                        EvalFailure(
                            case_id=case_id,
                            suite=suite,
                            reason=_normalize_reason(str(reason)),
                            details=_summarize_details(raw_reason),
                        )
                    )
            else:
                failures.append(
                    EvalFailure(
                        case_id=case_id,
                        suite=suite,
                        reason=_normalize_reason(str(raw_reason)),
                        details={"message": _truncate(str(raw_reason))},
                    )
                )
    return failures


def _normalize_reason(reason: str) -> str:
    text = reason.lower()
    mappings = [
        ("intent", "intent_mismatch"),
        ("category", "category_mismatch"),
        ("budget", "budget_mismatch"),
        ("negative", "negative_preference_violation"),
        ("preference", "preference_mismatch"),
        ("route", "route_mismatch"),
        ("not enough", "insufficient_results"),
        ("insufficient", "insufficient_results"),
        ("citation", "citation_missing"),
        ("purchase", "purchase_boundary_violation"),
        ("buy", "purchase_boundary_violation"),
        ("forbidden", "forbidden_term_violation"),
        ("medical", "skincare_medical_claim"),
        ("skincare", "skincare_medical_claim"),
        ("grounding", "grounding_guard_miss"),
        ("unsupported", "unsupported_claim"),
    ]
    for needle, normalized in mappings:
        if needle in text:
            return normalized
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized[:80] or "unknown_failure"


def _count_reasons(failures: list[EvalFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.reason] = counts.get(failure.reason, 0) + 1
    return counts


def _summarize_details(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return _truncate(str(value))
    if isinstance(value, dict):
        items = list(value.items())[:12]
        return {str(key): _summarize_details(val, depth=depth + 1) for key, val in items}
    if isinstance(value, list):
        return [_summarize_details(item, depth=depth + 1) for item in value[:5]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate(value) if isinstance(value, str) else value
    return _truncate(str(value))


def _compact_json(value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _truncate(text, limit=240)


def _truncate(value: str, *, limit: int = 300) -> str:
    return value if len(value) <= limit else f"{value[:limit]}..."


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _percent(rate: float | None) -> str:
    return "-" if rate is None else f"{rate * 100:.2f}%"


def _format_duration(duration_ms: int | None) -> str:
    return "-" if duration_ms is None else f"{duration_ms}ms"


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _suite_names(value: str) -> list[str]:
    return DEFAULT_SUITES if value == "all" else [value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run SmartBuyAgent deterministic eval suites and write reports."
    )
    parser.add_argument("--suite", choices=SUITE_CHOICES, default="all")
    parser.add_argument("--output", default="results/eval-report.md")
    parser.add_argument("--details", default="results/eval-details.json")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = build_eval_report(_suite_names(args.suite))
    write_report_files(report, output_path=args.output, details_path=args.details)

    if not args.quiet:
        print(f"Markdown report: {args.output}")
        print(f"JSON details: {args.details}")
        print(
            f"Suites: {report.total_suites}, "
            f"Cases: {report.passed_cases}/{report.total_cases} passed, "
            f"Pass rate: {_percent(report.pass_rate)}"
        )

    has_skipped = any(suite.status == "skipped" for suite in report.suites)
    if args.fail_on_regression and (report.failed_cases > 0 or has_skipped):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
