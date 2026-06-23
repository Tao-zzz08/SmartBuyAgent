from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.chat.query_understanding import QueryUnderstandingService
    from app.core.db import Base
    from app.models import Product, ProductTag
    from app.retrieval.retrieval_service import (
        ProductRetrievalService,
        ProductSearchFilters,
    )
    from app.services.embedding import MockEmbeddingService
except ModuleNotFoundError as exc:
    REAL_SMOKE_IMPORT_ERROR = exc
else:
    REAL_SMOKE_IMPORT_ERROR = None


FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATEGORY_IDS = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}


class EmptyCollection:
    def count(self) -> int:
        return 0


class EmptyChromaClient:
    def get_collection(self, name: str):
        del name
        return EmptyCollection()


def test_real_query_understanding_three_turn_budget_smoke() -> None:
    _skip_without_real_smoke_dependencies()
    case = load_product_smoke_fixture(
        "query_understanding_cases",
        "phone_budget_three_turns_smoke",
    )
    service = QueryUnderstandingService(llm_enabled=False)

    previous_memory = None
    result = None
    for query in case["turns"]:
        result = service.understand(query, previous_memory=previous_memory)
        previous_memory = result.to_shopping_memory()

    assert result is not None
    assert_query_understanding_expectations(result, case["expect_last_turn"])


def test_real_product_retrieval_consumes_structured_filters_smoke(tmp_path) -> None:
    _skip_without_real_smoke_dependencies()
    case = load_product_smoke_fixture(
        "product_retrieval_cases",
        "structured_phone_camera_without_apple_smoke",
    )
    engine, db = _test_session(tmp_path)
    try:
        for product in case["catalog"]:
            _add_product(db, product)
        db.commit()

        service = ProductRetrievalService(
            db,
            embedding_service=MockEmbeddingService(),
            chroma_client=EmptyChromaClient(),
        )
        filters_payload = case["filters"]
        results = service.search_products(
            query=case["query"],
            filters=ProductSearchFilters(
                category_id=_category_id(filters_payload["category"]),
                budget_max=filters_payload.get("budget_max"),
                preferences=list(filters_payload.get("preferences") or []),
                brand_exclude=list(filters_payload.get("negative_preferences") or []),
            ),
            top_k=5,
        )

        assert_product_retrieval_expectations(results, service, case["expect"])
    finally:
        db.close()
        engine.dispose()


def load_product_smoke_fixture(section: str, case_id: str) -> dict:
    payload = json.loads(
        (FIXTURES_DIR / "product_retrieval_smoke_fixtures.json").read_text(
            encoding="utf-8"
        )
    )
    for case in payload[section]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"missing product smoke fixture: {section}/{case_id}")


def assert_query_understanding_expectations(result, expect: dict) -> None:
    assert result.intent == expect["intent"]
    assert result.category == expect["category"]
    assert result.budget_max == expect["budget_max"]
    assert result.is_follow_up is expect["is_follow_up"]
    assert result.source == expect["source"]
    assert result.llm_fallback_attempted is expect["llm_fallback_attempted"]
    for preference in expect.get("preferences_contains") or []:
        assert preference in result.preferences
    for term in expect.get("effective_query_must_include") or []:
        assert term in result.effective_query


def assert_product_retrieval_expectations(results, service, expect: dict) -> None:
    ids = [candidate.product_id for candidate in results]
    for product_id in expect.get("must_include_product_ids") or []:
        assert product_id in ids
    for product_id in expect.get("must_exclude_product_ids") or []:
        assert product_id not in ids

    preferred_first = expect.get("preferred_first_product_id")
    for product_id in expect.get("negative_brand_product_ids") or []:
        if product_id in ids:
            assert preferred_first in ids
            assert ids.index(preferred_first) < ids.index(product_id)
        else:
            assert service.last_negative_filtered_count >= 1

    structured = expect["structured_filters"]
    assert service.last_structured_filters["category_id"] == _category_id(
        structured["category"]
    )
    assert service.last_structured_filters["budget_max"] == structured["budget_max"]
    for preference in structured.get("preferences") or []:
        assert preference in service.last_structured_filters["preferences"]
    for preference in structured.get("negative_preferences") or []:
        assert preference in service.last_structured_filters["negative_preferences"]


def _skip_without_real_smoke_dependencies() -> None:
    if REAL_SMOKE_IMPORT_ERROR is not None:
        pytest.skip(
            "real query/retrieval smoke requires backend dependencies: "
            f"{REAL_SMOKE_IMPORT_ERROR}"
        )


def _test_session(tmp_path):
    db_path = tmp_path / "real_query_understanding_smoke.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal()


def _add_product(db, product: dict) -> None:
    db.add(
        Product(
            id=product["id"],
            category_id=_category_id(product["category"]),
            title=product["title"],
            brand=product["brand"],
            price=product["price"],
            stock=10,
            description=product["description"],
            status="active",
        )
    )
    for index, tag in enumerate(product.get("tags") or []):
        db.add(
            ProductTag(
                id=f"{product['id']}_tag_{index}",
                product_id=product["id"],
                tag_type="tag",
                value=tag,
            )
        )


def _category_id(category: str) -> str:
    return CATEGORY_IDS[category]
