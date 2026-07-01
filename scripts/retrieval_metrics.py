from __future__ import annotations

import math
from typing import Any


CATEGORY_IDS = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}


def compute_retrieval_metrics(
    retrieved_product_ids: list[str],
    gold_relevance: dict[str, int | float] | None,
    *,
    k: int = 5,
) -> dict[str, float | None]:
    relevant = {
        str(product_id): float(relevance)
        for product_id, relevance in (gold_relevance or {}).items()
        if _numeric(relevance) is not None and float(relevance) > 0
    }
    if not relevant:
        return {
            f"recall_at_{k}": None,
            f"ndcg_at_{k}": None,
            f"mrr_at_{k}": None,
        }

    top_ids = [str(product_id) for product_id in retrieved_product_ids[:k]]
    retrieved_relevant = {product_id for product_id in top_ids if product_id in relevant}
    recall = len(retrieved_relevant) / len(relevant)

    dcg = _dcg([relevant.get(product_id, 0.0) for product_id in top_ids])
    ideal_relevances = sorted(relevant.values(), reverse=True)[:k]
    idcg = _dcg(ideal_relevances)
    ndcg = None if idcg <= 0 else dcg / idcg

    mrr = 0.0
    for index, product_id in enumerate(top_ids, start=1):
        if product_id in relevant:
            mrr = 1 / index
            break

    return {
        f"recall_at_{k}": round(recall, 4),
        f"ndcg_at_{k}": None if ndcg is None else round(ndcg, 4),
        f"mrr_at_{k}": round(mrr, 4),
    }


def compute_filter_compliance(
    products: list[dict[str, Any]],
    hard_filters: dict[str, Any] | None,
) -> dict[str, Any]:
    filters = hard_filters or {}
    violations: list[dict[str, Any]] = []

    for product in products:
        product_id = str(_field(product, "product_id") or _field(product, "id") or "")
        category = _category(_field(product, "category") or _field(product, "category_id"))
        price = _numeric(_field(product, "price"))
        brand = str(_field(product, "brand") or "")
        product_text = _product_text(product)

        expected_category = filters.get("category")
        if expected_category and category != expected_category:
            violations.append(
                {
                    "type": "category",
                    "product_id": product_id,
                    "expected": expected_category,
                    "actual": category,
                }
            )

        price_lte = _numeric(filters.get("price_lte"))
        if price_lte is not None and price is not None and price > price_lte:
            violations.append(
                {
                    "type": "price_lte",
                    "product_id": product_id,
                    "expected": price_lte,
                    "actual": price,
                }
            )

        price_gte = _numeric(filters.get("price_gte"))
        if price_gte is not None and price is not None and price < price_gte:
            violations.append(
                {
                    "type": "price_gte",
                    "product_id": product_id,
                    "expected": price_gte,
                    "actual": price,
                }
            )

        excluded_product_ids = {str(value) for value in filters.get("exclude_product_ids") or []}
        if product_id and product_id in excluded_product_ids:
            violations.append({"type": "exclude_product_ids", "product_id": product_id})

        for excluded_brand in filters.get("exclude_brands") or []:
            if _contains(brand, str(excluded_brand)) or _contains(product_text, str(excluded_brand)):
                violations.append(
                    {
                        "type": "exclude_brands",
                        "product_id": product_id,
                        "matched": str(excluded_brand),
                    }
                )

        if filters.get("required_in_stock") is True:
            stock_value = _field(product, "in_stock", _field(product, "stock"))
            if stock_value in {False, 0, "0", "false", "False", "out_of_stock"}:
                violations.append({"type": "required_in_stock", "product_id": product_id})

        for term in filters.get("forbidden_terms") or []:
            if _contains(product_text, str(term)):
                violations.append(
                    {
                        "type": "forbidden_terms",
                        "product_id": product_id,
                        "matched": str(term),
                    }
                )

    return {
        "filter_compliance": not violations,
        "filter_violation_count": len(violations),
        "filter_violations": violations,
    }


def aggregate_retrieval_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(case_results)
    product_results = [
        result for result in case_results if result.get("type") == "product_retrieval"
    ]
    metrics_by_case = [result.get("metrics") or {} for result in product_results]

    return {
        "recall_at_5": _mean_metric(metrics_by_case, "recall_at_5"),
        "ndcg_at_5": _mean_metric(metrics_by_case, "ndcg_at_5"),
        "mrr_at_5": _mean_metric(metrics_by_case, "mrr_at_5"),
        "filter_compliance_rate": _filter_compliance_rate(metrics_by_case),
        "negative_preference_violation_rate": _negative_violation_rate(product_results),
        "empty_rate": _rate(
            sum(1 for result in case_results if int(result.get("result_count") or result.get("citation_count") or 0) == 0),
            total_cases,
        ),
        "latency_p50_ms": _percentile(
            [float((result.get("metrics") or {}).get("latency_ms")) for result in case_results if (result.get("metrics") or {}).get("latency_ms") is not None],
            50,
        ),
        "latency_p95_ms": _percentile(
            [float((result.get("metrics") or {}).get("latency_ms")) for result in case_results if (result.get("metrics") or {}).get("latency_ms") is not None],
            95,
        ),
        "evaluated_ranking_cases": sum(
            1 for metrics in metrics_by_case if metrics.get("recall_at_5") is not None
        ),
    }


def _dcg(relevances: list[float]) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances, start=1):
        total += (2**relevance - 1) / math.log2(rank + 1)
    return total


def _mean_metric(metrics_by_case: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(metrics[key])
        for metrics in metrics_by_case
        if metrics.get(key) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _filter_compliance_rate(metrics_by_case: list[dict[str, Any]]) -> float | None:
    applicable = [
        metrics
        for metrics in metrics_by_case
        if metrics.get("filter_compliance") is not None
    ]
    if not applicable:
        return None
    return _rate(
        sum(1 for metrics in applicable if metrics.get("filter_compliance") is True),
        len(applicable),
    )


def _negative_violation_rate(product_results: list[dict[str, Any]]) -> float | None:
    if not product_results:
        return None
    violating_cases = sum(
        1 for result in product_results if int(result.get("negative_preference_violations") or 0) > 0
    )
    return _rate(violating_cases, len(product_results))


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * (percentile / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[int(position)], 4)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in CATEGORY_IDS:
        return text
    for category, category_id in CATEGORY_IDS.items():
        if text == category_id:
            return category
    return text


def _product_text(product: dict[str, Any]) -> str:
    parts: list[str] = []
    for field_name in [
        "product_id",
        "id",
        "title",
        "name",
        "brand",
        "category",
        "category_id",
        "description",
        "product_text",
    ]:
        value = _field(product, field_name)
        if value:
            parts.append(str(value))
    for tag in _field(product, "tags", []) or []:
        parts.append(str(tag))
    return " ".join(parts)


def _field(obj: dict[str, Any], name: str, default: Any = None) -> Any:
    return obj.get(name, default)


def _numeric(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains(text: str, term: str) -> bool:
    return bool(term) and term.lower() in text.lower()
