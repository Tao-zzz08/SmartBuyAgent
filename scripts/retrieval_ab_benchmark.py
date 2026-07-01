from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrieval_ab_strategies import (  # noqa: E402
    SUPPORTED_STRATEGIES,
    build_hard_filters,
    normalize_product,
    run_strategy,
)
from retrieval_metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    compute_filter_compliance,
    compute_retrieval_metrics,
)


DEFAULT_STRATEGIES = ",".join(SUPPORTED_STRATEGIES)


def run_benchmark(
    *,
    cases: list[dict[str, Any]],
    products: list[dict[str, Any]],
    strategies: list[str] | None = None,
    top_k: int = 5,
    baseline: str = "structured_filter_only",
) -> dict[str, Any]:
    selected_strategies = strategies or list(SUPPORTED_STRATEGIES)
    product_cases = [
        case for case in cases if case.get("type") in {None, "product_retrieval"}
    ]
    normalized_products = [normalize_product(product) for product in products]

    results: list[dict[str, Any]] = []
    for case in product_cases:
        for strategy in selected_strategies:
            started = time.perf_counter()
            retrieved_products = run_strategy(
                strategy,
                normalized_products,
                case,
                top_k=top_k,
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 4)
            retrieved_product_ids = [
                str(product.get("product_id") or product.get("id"))
                for product in retrieved_products
            ]

            hard_filters = build_hard_filters(case)
            ranking_metrics = compute_retrieval_metrics(
                retrieved_product_ids,
                case.get("gold_relevance"),
                k=top_k,
            )
            filter_result = compute_filter_compliance(retrieved_products, hard_filters)
            negative_violations = _negative_preference_violations(
                filter_result.get("filter_violations") or []
            )

            metrics = {
                **ranking_metrics,
                "filter_compliance": filter_result["filter_compliance"],
                "filter_violation_count": filter_result["filter_violation_count"],
                "latency_ms": latency_ms,
            }
            results.append(
                {
                    "case_id": case.get("id"),
                    "type": "product_retrieval",
                    "strategy": strategy,
                    "category": hard_filters.get("category")
                    or (case.get("structured_filters") or {}).get("category"),
                    "query": case.get("query"),
                    "retrieved_product_ids": retrieved_product_ids,
                    "result_count": len(retrieved_products),
                    "metrics": metrics,
                    "filter_violations": filter_result.get("filter_violations") or [],
                    "negative_preference_violations": negative_violations,
                }
            )

    summary = build_summary(
        results,
        strategies=selected_strategies,
        top_k=top_k,
        baseline=baseline,
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": top_k,
        "baseline": baseline,
        "strategies": selected_strategies,
        "summary": summary,
        "results": results,
    }


def build_summary(
    results: list[dict[str, Any]],
    *,
    strategies: list[str],
    top_k: int,
    baseline: str,
) -> dict[str, Any]:
    strategy_summary = {
        strategy: _aggregate_strategy_results(
            [result for result in results if result.get("strategy") == strategy],
            top_k=top_k,
        )
        for strategy in strategies
    }
    return {
        "strategies": strategy_summary,
        "best_by_metric": _best_by_metric(strategy_summary, top_k=top_k),
        "deltas_vs_baseline": _deltas_vs_baseline(
            strategy_summary,
            baseline=baseline,
            top_k=top_k,
        ),
        "strategy_win_counts": _strategy_win_counts(
            results,
            strategies=strategies,
            top_k=top_k,
        ),
        "category_breakdown": _category_breakdown(
            results,
            strategies=strategies,
            top_k=top_k,
        ),
        "empty_cases": [
            {
                "case_id": result.get("case_id"),
                "strategy": result.get("strategy"),
                "category": result.get("category"),
            }
            for result in results
            if int(result.get("result_count") or 0) == 0
        ],
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_products(path: str | Path) -> list[dict[str, Any]]:
    product_path = Path(path)
    products: list[dict[str, Any]] = []
    with product_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                products.append(json.loads(stripped))
    return products


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    top_k = int(report["top_k"])
    metric_keys = _ranking_metric_keys(top_k) + ["filter_compliance_rate", "empty_rate", "latency_p50_ms", "latency_p95_ms", "evaluated_cases"]
    lines = [
        "# Retrieval A/B Strategy Benchmark",
        "",
        f"Generated at: {report['generated_at']}",
        f"Top K: {top_k}",
        f"Baseline: `{report['baseline']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | " + " | ".join(metric_keys) + " |",
        "|---" + "|---:" * len(metric_keys) + "|",
    ]
    for strategy, metrics in summary["strategies"].items():
        values = [_format_metric(metrics.get(key)) for key in metric_keys]
        lines.append(f"| {strategy} | " + " | ".join(values) + " |")

    lines.extend(["", "## Best By Metric", "", "| Metric | Strategy |", "|---|---|"])
    for metric, strategy in summary["best_by_metric"].items():
        lines.append(f"| {metric} | {strategy} |")

    lines.extend(["", "## Deltas vs Baseline", ""])
    if summary["deltas_vs_baseline"]:
        delta_keys = sorted(
            {
                key
                for deltas in summary["deltas_vs_baseline"].values()
                for key in deltas
            }
        )
        lines.extend(
            [
                "| Strategy | " + " | ".join(delta_keys) + " |",
                "|---" + "|---:" * len(delta_keys) + "|",
            ]
        )
        for strategy, deltas in summary["deltas_vs_baseline"].items():
            values = [_format_metric(deltas.get(key)) for key in delta_keys]
            lines.append(f"| {strategy} | " + " | ".join(values) + " |")
    else:
        lines.append("No baseline deltas were computed.")

    lines.extend(["", "## Category Breakdown", ""])
    for category, by_strategy in summary["category_breakdown"].items():
        lines.extend(
            [
                f"### {category}",
                "",
                "| Strategy | " + " | ".join(metric_keys) + " |",
                "|---" + "|---:" * len(metric_keys) + "|",
            ]
        )
        for strategy, metrics in by_strategy.items():
            values = [_format_metric(metrics.get(key)) for key in metric_keys]
            lines.append(f"| {strategy} | " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(["## Failed/Empty Cases", ""])
    if summary["empty_cases"]:
        for item in summary["empty_cases"]:
            lines.append(
                f"- `{item['case_id']}` via `{item['strategy']}`"
                f" ({item.get('category') or 'unknown'})"
            )
    else:
        lines.append("No empty cases were recorded.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: dict[str, Any],
    *,
    output_path: str | Path | None,
    details_path: str | Path | None,
) -> None:
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(report), encoding="utf-8")
    if details_path:
        details = Path(details_path)
        details.parent.mkdir(parents=True, exist_ok=True)
        details.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic retrieval A/B benchmark.")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json"))
    parser.add_argument("--products", default=str(PROJECT_ROOT / "data" / "processed" / "products" / "all_products_900.jsonl"))
    parser.add_argument("--strategies", default=DEFAULT_STRATEGIES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--baseline", default="structured_filter_only")
    parser.add_argument("--output")
    parser.add_argument("--details")
    args = parser.parse_args(argv)

    strategies = [strategy.strip() for strategy in args.strategies.split(",") if strategy.strip()]
    cases = load_cases(args.cases)
    products = load_products(args.products)
    report = run_benchmark(
        cases=cases,
        products=products,
        strategies=strategies,
        top_k=args.top_k,
        baseline=args.baseline,
    )
    write_outputs(report, output_path=args.output, details_path=args.details)
    if not args.output and not args.details:
        print(render_markdown(report))
    return 0


def _aggregate_strategy_results(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    summary = aggregate_retrieval_metrics(results)
    summary["evaluated_cases"] = len(results)
    if top_k != 5:
        for key in _ranking_metric_keys(top_k):
            summary[key] = _mean_metric([result.get("metrics") or {} for result in results], key)
    return summary


def _best_by_metric(
    strategy_summary: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, str]:
    best: dict[str, str] = {}
    for metric in _ranking_metric_keys(top_k) + ["filter_compliance_rate", "mrr_at_5", "ndcg_at_5", "recall_at_5"]:
        candidates = {
            strategy: values.get(metric)
            for strategy, values in strategy_summary.items()
            if values.get(metric) is not None
        }
        if not candidates:
            continue
        best[metric] = max(candidates.items(), key=lambda item: float(item[1]))[0]
    return best


def _deltas_vs_baseline(
    strategy_summary: dict[str, dict[str, Any]],
    *,
    baseline: str,
    top_k: int,
) -> dict[str, dict[str, float]]:
    baseline_metrics = strategy_summary.get(baseline) or {}
    delta_metrics = _ranking_metric_keys(top_k) + ["filter_compliance_rate"]
    deltas: dict[str, dict[str, float]] = {}
    for strategy, metrics in strategy_summary.items():
        if strategy == baseline:
            continue
        strategy_deltas: dict[str, float] = {}
        for metric in delta_metrics:
            base_value = baseline_metrics.get(metric)
            value = metrics.get(metric)
            if base_value is None or value is None:
                continue
            strategy_deltas[f"{metric}_delta"] = round(float(value) - float(base_value), 4)
        deltas[strategy] = strategy_deltas
    return deltas


def _strategy_win_counts(
    results: list[dict[str, Any]],
    *,
    strategies: list[str],
    top_k: int,
) -> dict[str, dict[str, int]]:
    metrics = _ranking_metric_keys(top_k)
    counts = {metric: {strategy: 0 for strategy in strategies} for metric in metrics}
    by_case: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_case.setdefault(str(result.get("case_id")), []).append(result)

    for case_results in by_case.values():
        for metric in metrics:
            candidates = [
                (result["strategy"], (result.get("metrics") or {}).get(metric))
                for result in case_results
                if (result.get("metrics") or {}).get(metric) is not None
            ]
            if not candidates:
                continue
            best_value = max(float(value) for _, value in candidates)
            for strategy, value in candidates:
                if float(value) == best_value:
                    counts[metric][strategy] += 1
    return counts


def _category_breakdown(
    results: list[dict[str, Any]],
    *,
    strategies: list[str],
    top_k: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    categories = sorted({str(result.get("category") or "unknown") for result in results})
    breakdown: dict[str, dict[str, dict[str, Any]]] = {}
    for category in categories:
        breakdown[category] = {}
        for strategy in strategies:
            strategy_results = [
                result
                for result in results
                if str(result.get("category") or "unknown") == category
                and result.get("strategy") == strategy
            ]
            breakdown[category][strategy] = _aggregate_strategy_results(
                strategy_results,
                top_k=top_k,
            )
    return breakdown


def _negative_preference_violations(violations: list[dict[str, Any]]) -> int:
    return sum(
        1
        for violation in violations
        if violation.get("type") in {"exclude_brands", "exclude_product_ids", "forbidden_terms"}
    )


def _ranking_metric_keys(top_k: int) -> list[str]:
    return [f"recall_at_{top_k}", f"ndcg_at_{top_k}", f"mrr_at_{top_k}"]


def _mean_metric(metrics_by_case: list[dict[str, Any]], key: str) -> float | None:
    values = [float(metrics[key]) for metrics in metrics_by_case if metrics.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
