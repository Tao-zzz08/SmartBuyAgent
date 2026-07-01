from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "query_understanding_intent_eval_cases.json"

for path in [BACKEND_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from query_understanding_intent_metrics import (  # noqa: E402
    aggregate_diagnostic_metrics,
    aggregate_intent_metrics,
    case_group_counts,
    evaluate_intent_case,
    failure_type_counts,
)


QueryUnderstandingService = None
shopping_memory_from_dict = None
decide_llm_fallback = None


def load_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("QueryUnderstanding intent eval cases must be a JSON list")
    return cases


def run_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    service = _create_service()
    results: list[dict[str, Any]] = []
    for case in cases:
        previous_memory = _previous_memory(case.get("previous_memory"))
        query = str(case.get("query") or "")
        actual_result = service.understand(query=query, previous_memory=previous_memory)
        actual = _actual_from_result(actual_result)
        _add_theoretical_fallback_decision(
            actual,
            actual_result=actual_result,
            query=query,
            previous_memory=previous_memory,
            service=service,
        )
        results.append(evaluate_intent_case(case=case, actual=actual))

    passed_cases = sum(1 for result in results if result.get("passed"))
    failed_cases = len(results) - passed_cases
    return {
        "suite": "query_understanding_intent",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": len(results),
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": _rate(passed_cases, len(results)),
            "case_group_counts": case_group_counts(results),
            "failure_type_counts": failure_type_counts(results),
            "metrics": aggregate_intent_metrics(results),
            "diagnostic_metrics": aggregate_diagnostic_metrics(results),
        },
        "results": results,
    }


def render_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "# QueryUnderstanding Intent Eval",
        "",
        f"Generated at: {output.get('generated_at')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_cases | {summary['total_cases']} |",
        f"| passed_cases | {summary['passed_cases']} |",
        f"| failed_cases | {summary['failed_cases']} |",
        f"| pass_rate | {_format_metric(summary['pass_rate'])} |",
        "",
        "## Field Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in sorted((summary.get("metrics") or {}).items()):
        lines.append(f"| {key} | {_format_metric(value)} |")

    lines.extend(
        [
            "",
            "## Diagnostic Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, value in sorted((summary.get("diagnostic_metrics") or {}).items()):
        lines.append(f"| {key} | {_format_metric(value)} |")

    lines.extend(["", "## Case Groups", "", "| Group | Count |", "|---|---:|"])
    for group, count in (summary.get("case_group_counts") or {}).items():
        lines.append(f"| {group} | {count} |")

    lines.extend(["", "## Failure Type Counts", "", "| Failure Type | Count |", "|---|---:|"])
    if summary.get("failure_type_counts"):
        for failure_type, count in summary["failure_type_counts"].items():
            lines.append(f"| {failure_type} | {count} |")
    else:
        lines.append("| - | 0 |")

    lines.extend(["", "## Failed Cases", ""])
    failed_results = [result for result in output["results"] if not result.get("passed")]
    if not failed_results:
        lines.append("No failed cases.")
        lines.append("")
        return "\n".join(lines)

    for result in failed_results:
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- Group: {result.get('case_group')}",
                f"- Query: {result.get('query')}",
                f"- Failure reasons: {', '.join(result.get('failure_reasons') or [])}",
                f"- Expected: `{_compact_json(result.get('expected') or {})}`",
                f"- Actual: `{_compact_json(result.get('actual') or {})}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    output: dict[str, Any],
    *,
    markdown_path: str | Path | None,
    details_path: str | Path | None,
) -> None:
    if markdown_path:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(output), encoding="utf-8")
    if details_path:
        path = Path(details_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QueryUnderstanding field-level intent eval.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output")
    parser.add_argument("--details")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    output = run_eval(load_cases(args.cases))
    write_outputs(output, markdown_path=args.output, details_path=args.details)

    if not args.quiet:
        if args.output:
            print(f"Markdown report: {args.output}")
        if args.details:
            print(f"JSON details: {args.details}")
        if not args.output and not args.details:
            print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
        else:
            summary = output["summary"]
            print(
                f"Cases: {summary['passed_cases']}/{summary['total_cases']} passed, "
                f"pass rate: {_format_metric(summary['pass_rate'])}"
            )

    if args.fail_on_regression and output["summary"]["failed_cases"] > 0:
        return 1
    return 0


def _previous_memory(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("previous_memory must be an object or null")
    converter = _shopping_memory_converter()
    if converter is not None:
        return converter(value)
    budget = value.get("budget") or {}
    return SimpleNamespace(
        category=value.get("category"),
        budget=SimpleNamespace(
            min=budget.get("min"),
            max=budget.get("max"),
            currency=budget.get("currency") or "CNY",
        ),
        preferences=list(value.get("preferences") or []),
        negative_preferences=list(value.get("negative_preferences") or []),
        last_product_ids=list(value.get("last_product_ids") or []),
        last_intent=value.get("last_intent"),
    )


def _create_service() -> Any:
    global QueryUnderstandingService
    if QueryUnderstandingService is None:
        from app.chat.query_understanding import QueryUnderstandingService as service_cls

        QueryUnderstandingService = service_cls
    return QueryUnderstandingService(llm_enabled=False)


def _add_theoretical_fallback_decision(
    actual: dict[str, Any],
    *,
    actual_result: Any,
    query: str,
    previous_memory: Any,
    service: Any,
) -> None:
    decision_fn = _fallback_decision_fn()
    if decision_fn is None:
        return
    try:
        decision = decision_fn(
            rule_result=actual_result,
            query=query,
            previous_memory=previous_memory,
            confidence_threshold=getattr(service, "llm_confidence_threshold", 0.75),
            enabled=True,
        )
    except Exception:
        return
    actual["llm_fallback_should_call"] = bool(getattr(decision, "should_call", False))
    actual["llm_fallback_trigger_reasons"] = list(getattr(decision, "reasons", []) or [])


def _fallback_decision_fn() -> Any:
    global decide_llm_fallback
    if decide_llm_fallback is not None:
        return decide_llm_fallback
    try:
        from app.chat.query_understanding import decide_llm_fallback as decision_fn
    except Exception:
        return None
    decide_llm_fallback = decision_fn
    return decision_fn


def _shopping_memory_converter() -> Any:
    global shopping_memory_from_dict
    if shopping_memory_from_dict is not None:
        return shopping_memory_from_dict
    try:
        from app.chat.shopping_memory import shopping_memory_from_dict as converter
    except Exception:
        return None
    shopping_memory_from_dict = converter
    return converter


def _actual_from_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_trace_dict"):
        return result.to_trace_dict()
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    raise TypeError(f"Unsupported QueryUnderstanding result: {type(result)!r}")


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _compact_json(value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 320 else f"{text[:320]}..."


if __name__ == "__main__":
    raise SystemExit(main())
