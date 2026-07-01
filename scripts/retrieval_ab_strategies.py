from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable


CATEGORY_IDS = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}
CATEGORY_BY_ID = {value: key for key, value in CATEGORY_IDS.items()}

SUPPORTED_STRATEGIES = [
    "structured_filter_only",
    "lexical_keyword",
    "hybrid_filter_keyword",
    "hybrid_plus_rerank",
]

KNOWN_QUERY_TERMS = [
    "phone",
    "camera",
    "battery",
    "shoes",
    "skincare",
    "Apple",
    "iPhone",
    "拍照",
    "影像",
    "传感器",
    "防抖",
    "续航",
    "电池",
    "快充",
    "通勤",
    "轻便",
    "防滑",
    "高跟",
    "敏感肌",
    "保湿",
    "温和",
    "美白",
    "苹果",
]


def run_strategy(
    strategy: str,
    products: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported retrieval A/B strategy: {strategy}")

    normalized_products = [normalize_product(product) for product in products]
    filters = build_hard_filters(case)
    terms = extract_case_terms(case)

    if strategy == "structured_filter_only":
        return _rank(
            _apply_hard_filters(normalized_products, filters),
            lambda product: _structured_score(product, terms, filters),
            top_k=top_k,
        )

    if strategy == "lexical_keyword":
        candidates = _apply_hard_filters(normalized_products, filters)
        return _rank(candidates, lambda product: _keyword_score(product, terms), top_k=top_k)

    if strategy == "hybrid_filter_keyword":
        candidates = _apply_hard_filters(normalized_products, filters)
        return _rank(
            candidates,
            lambda product: _keyword_score(product, terms)
            + _preference_score(product, case) * 2
            + _budget_score(product, filters),
            top_k=top_k,
        )

    candidates = _apply_hard_filters(normalized_products, filters)
    return _rank(
        candidates,
        lambda product: _rerank_score(product, case, terms, filters),
        top_k=top_k,
    )


def normalize_product(product: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(product)
    product_id = (
        normalized.get("product_id")
        or normalized.get("id")
        or _generated_product_id(normalized)
    )
    category = _category(normalized.get("category") or normalized.get("category_id"))
    normalized["product_id"] = str(product_id)
    normalized["id"] = str(product_id)
    normalized["category"] = category
    if category:
        normalized["category_id"] = CATEGORY_IDS.get(category, normalized.get("category_id"))
    normalized["price"] = _numeric(normalized.get("price"))
    normalized["tags"] = list(normalized.get("tags") or [])
    return normalized


def build_hard_filters(case: dict[str, Any]) -> dict[str, Any]:
    filters = dict(case.get("hard_filters") or {})
    structured = case.get("structured_filters") or {}
    expect = case.get("expect") or {}

    category = filters.get("category") or structured.get("category") or structured.get("category_id")
    if category:
        filters["category"] = _category(category)

    if filters.get("price_lte") is None:
        budget_max = structured.get("budget_max") or expect.get("max_price")
        if _numeric(budget_max) is not None:
            filters["price_lte"] = _numeric(budget_max)

    if filters.get("price_gte") is None and _numeric(structured.get("budget_min")) is not None:
        filters["price_gte"] = _numeric(structured.get("budget_min"))

    exclude_brands = list(filters.get("exclude_brands") or [])
    for term in structured.get("negative_preferences") or []:
        if term and term not in exclude_brands:
            exclude_brands.append(term)
    if exclude_brands:
        filters["exclude_brands"] = exclude_brands

    forbidden_terms = list(filters.get("forbidden_terms") or [])
    for term in expect.get("forbidden_terms") or []:
        if term and term not in forbidden_terms:
            forbidden_terms.append(term)
    if forbidden_terms:
        filters["forbidden_terms"] = forbidden_terms

    return filters


def extract_case_terms(case: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    query = str(case.get("query") or "")
    terms.extend(_query_tokens(query))

    structured = case.get("structured_filters") or {}
    expect = case.get("expect") or {}
    for field_name in ["preferences", "negative_preferences"]:
        terms.extend(str(term) for term in structured.get(field_name) or [] if term)
    for field_name in ["must_match_any", "forbidden_terms"]:
        terms.extend(str(term) for term in expect.get(field_name) or [] if term)

    for known in KNOWN_QUERY_TERMS:
        if known and known.lower() in query.lower():
            terms.append(known)

    return _dedupe_terms(terms)


def _apply_hard_filters(
    products: list[dict[str, Any]],
    filters: dict[str, Any],
) -> list[dict[str, Any]]:
    return [product for product in products if _matches_hard_filters(product, filters)]


def _matches_hard_filters(product: dict[str, Any], filters: dict[str, Any]) -> bool:
    category = filters.get("category")
    if category and _category(product.get("category") or product.get("category_id")) != category:
        return False

    price = _numeric(product.get("price"))
    price_lte = _numeric(filters.get("price_lte"))
    if price is not None and price_lte is not None and price > price_lte:
        return False

    price_gte = _numeric(filters.get("price_gte"))
    if price is not None and price_gte is not None and price < price_gte:
        return False

    product_id = str(product.get("product_id") or product.get("id") or "")
    if product_id in {str(value) for value in filters.get("exclude_product_ids") or []}:
        return False

    text = _product_text(product)
    brand = str(product.get("brand") or "")
    for excluded_brand in filters.get("exclude_brands") or []:
        if _contains(brand, excluded_brand) or _contains(text, excluded_brand):
            return False

    if filters.get("required_in_stock") is True:
        stock_value = product.get("in_stock", product.get("stock"))
        if stock_value in {False, 0, "0", "false", "False", "out_of_stock"}:
            return False

    for term in filters.get("forbidden_terms") or []:
        if _contains(text, term):
            return False

    return True


def _rank(
    products: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        products,
        key=lambda product: (
            -score_fn(product),
            _numeric(product.get("price")) if _numeric(product.get("price")) is not None else float("inf"),
            str(product.get("product_id") or product.get("id") or ""),
        ),
    )
    return ranked[:top_k]


def _structured_score(
    product: dict[str, Any],
    terms: list[str],
    filters: dict[str, Any],
) -> float:
    return _keyword_score(product, terms) + _budget_score(product, filters)


def _keyword_score(product: dict[str, Any], terms: list[str]) -> float:
    text = _product_text(product)
    score = 0.0
    for term in terms:
        if not term:
            continue
        if _contains(text, term):
            score += 1.0
    return score


def _preference_score(product: dict[str, Any], case: dict[str, Any]) -> float:
    preferences = (case.get("structured_filters") or {}).get("preferences") or []
    text = _product_text(product)
    return sum(1.0 for term in preferences if term and _contains(text, term))


def _budget_score(product: dict[str, Any], filters: dict[str, Any]) -> float:
    price = _numeric(product.get("price"))
    price_lte = _numeric(filters.get("price_lte"))
    if price is None or price_lte is None or price_lte <= 0:
        return 0.0
    if price > price_lte:
        return -100.0
    return round(max(0.0, 1 - abs(price_lte - price) / price_lte), 4)


def _rerank_score(
    product: dict[str, Any],
    case: dict[str, Any],
    terms: list[str],
    filters: dict[str, Any],
) -> float:
    score = _keyword_score(product, terms)
    score += _preference_score(product, case) * 3
    score += _budget_score(product, filters) * 2
    if filters.get("category") and product.get("category") == filters.get("category"):
        score += 1.0
    rating = _numeric(product.get("rating"))
    if rating is not None:
        score += min(rating, 5.0) / 10
    sales = _numeric(product.get("sales") or product.get("sales_count"))
    if sales is not None:
        score += min(sales, 10000.0) / 100000
    return score


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query)
    return [token for token in tokens if len(token.strip()) >= 2]


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _generated_product_id(record: dict[str, Any]) -> str:
    raw = "|".join(
        [
            str(record.get("category") or ""),
            str(record.get("source_platform") or "other"),
            str(record.get("source_product_id") or ""),
            str(record.get("brand") or ""),
            str(record.get("title") or ""),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"real_{record.get('category')}_{digest}"[:64]


def _product_text(product: dict[str, Any]) -> str:
    parts: list[str] = []
    for field_name in [
        "product_id",
        "id",
        "source_product_id",
        "title",
        "name",
        "brand",
        "category",
        "category_id",
        "description",
        "product_text",
    ]:
        value = product.get(field_name)
        if value:
            parts.append(str(value))
    for tag in product.get("tags") or []:
        parts.append(str(tag))
    attributes = product.get("attributes")
    if attributes:
        parts.append(json.dumps(attributes, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def _category(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in CATEGORY_IDS:
        return text
    return CATEGORY_BY_ID.get(text, text)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _contains(text: str, term: Any) -> bool:
    if term is None:
        return False
    needle = str(term).strip().lower()
    if not needle:
        return False
    return needle in str(text).lower()
