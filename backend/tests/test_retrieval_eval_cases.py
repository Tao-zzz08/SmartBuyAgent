from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_retrieval import load_eval_cases, run_eval  # noqa: E402


CATEGORY_IDS = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}
CATEGORY_BY_ID = {value: key for key, value in CATEGORY_IDS.items()}


class RoutingFakeProductService:
    def __init__(self, products: list[dict]) -> None:
        self._products = products

    def search_products(self, query, filters, top_k):
        filtered = [
            product
            for product in self._products
            if _matches_category(product, filters.category_id)
            and _matches_budget(product, filters)
        ]
        positive = [
            product
            for product in filtered
            if not _contains_any(_product_text(product), filters.brand_exclude)
        ]
        if positive:
            filtered = positive

        ranked = sorted(
            filtered,
            key=lambda product: (
                -_preference_score(product, filters.preferences, query),
                int(product["price"]),
                product["id"],
            ),
        )
        return [_product_namespace(product) for product in ranked[:top_k]]


class RoutingFakeKnowledgeService:
    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks

    def search_knowledge(self, query, category_id=None, top_k=5):
        category = CATEGORY_BY_ID.get(category_id or "")
        filtered = [
            chunk
            for chunk in self._chunks
            if category is None or chunk["category"] == category
        ]
        ranked = sorted(
            filtered,
            key=lambda chunk: (
                -_keyword_score(chunk["content"], query),
                chunk["id"],
            ),
        )
        return [_citation_namespace(chunk) for chunk in ranked[:top_k]]


def test_core_retrieval_eval_cases_pass_with_fake_services() -> None:
    catalog = load_fixture_catalog()
    wanted = {
        "phone_camera_under_5000",
        "phone_without_apple",
        "skincare_sensitive_moisturizing_under_300",
    }
    cases = [
        case
        for case in load_eval_cases(PROJECT_ROOT / "data" / "eval" / "retrieval_eval_cases.json")
        if case["id"] in wanted
    ]

    output = run_eval(
        cases,
        RoutingFakeProductService(catalog["products"]),
        RoutingFakeKnowledgeService(catalog["knowledge_chunks"]),
    )

    assert output["summary"]["failed_cases"] == 0
    assert output["summary"]["passed_cases"] == len(cases)


def load_fixture_catalog() -> dict:
    return json.loads(
        (FIXTURES_DIR / "retrieval_eval_fixture_catalog.json").read_text(
            encoding="utf-8"
        )
    )


def _matches_category(product: dict, category_id: str | None) -> bool:
    return category_id is None or _category_id(product["category"]) == category_id


def _matches_budget(product: dict, filters) -> bool:
    if filters.budget_min is not None and product["price"] < filters.budget_min:
        return False
    if filters.budget_max is not None and product["price"] > filters.budget_max:
        return False
    return True


def _preference_score(product: dict, preferences: list[str], query: str) -> int:
    text = _product_text(product)
    return sum(1 for term in list(preferences or []) + [query] if term and term in text)


def _keyword_score(text: str, query: str) -> int:
    return sum(1 for token in _query_tokens(query) if token in text)


def _query_tokens(query: str) -> list[str]:
    return [token for token in query.replace("，", " ").replace("？", " ").split() if token]


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in terms or [])


def _product_text(product: dict) -> str:
    return " ".join(
        [
            product["id"],
            product["title"],
            product["brand"],
            product["description"],
            " ".join(product.get("tags") or []),
        ]
    )


def _product_namespace(product: dict):
    return SimpleNamespace(
        product_id=product["id"],
        category_id=_category_id(product["category"]),
        price=product["price"],
        title=product["title"],
        brand=product["brand"],
        tags=list(product.get("tags") or []),
        description=product["description"],
    )


def _citation_namespace(chunk: dict):
    return SimpleNamespace(
        chunk_id=chunk["id"],
        source_file=chunk["source_file"],
        content_preview=chunk["content"],
        text=chunk["content"],
    )


def _category_id(category: str) -> str:
    return CATEGORY_IDS[category]
