from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent.context import AgentRuntimeContext
from app.agent.nodes import intent_router_node
from app.agent.state import create_initial_agent_state
from app.chat.query_understanding import QueryUnderstandingResult
from app.core.db import Base
from app.models import Product, ProductTag
from app.retrieval.retrieval_service import (
    ProductRetrievalService,
    ProductSearchFilters,
    build_knowledge_retrieval_query,
)
from app.services.embedding import MockEmbeddingService


PHONE_CN = "\u624b\u673a"
PHOTO_CN = "\u62cd\u7167"
BATTERY_CN = "\u7eed\u822a"
APPLE_CN = "\u82f9\u679c"
OIL_CONTROL_CN = "\u63a7\u6cb9"
GENTLE_CARE_CN = "\u6e29\u548c\u62a4\u7406"
TREAT_CN = "\u6cbb\u7597"
CURE_CN = "\u6cbb\u6108"
DRUG_EFFECT_CN = "\u836f\u6548"
PRESCRIPTION_CN = "\u5904\u65b9"
MEDICAL_REPAIR_CN = "\u533b\u5b66\u4fee\u590d"


class EmptyCollection:
    def count(self) -> int:
        return 0


class EmptyChromaClient:
    def get_collection(self, name: str):
        return EmptyCollection()


class StaticQueryUnderstandingService:
    def __init__(self, result: QueryUnderstandingResult) -> None:
        self.result = result

    def understand(self, *args, **kwargs) -> QueryUnderstandingResult:
        return self.result


def _session(tmp_path):
    db_path = tmp_path / "structured_retrieval.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal()


def _add_product(
    db,
    *,
    product_id: str,
    category_id: str,
    price: int,
    title: str,
    brand: str | None,
    tags: list[str],
) -> None:
    db.add(
        Product(
            id=product_id,
            category_id=category_id,
            title=title,
            brand=brand,
            price=price,
            stock=5,
            description=title,
            status="active",
        )
    )
    for index, tag in enumerate(tags):
        db.add(
            ProductTag(
                id=f"{product_id}_tag_{index}",
                product_id=product_id,
                tag_type="tag",
                value=tag,
            )
        )


def test_product_retrieval_uses_category_and_budget_filters(tmp_path) -> None:
    engine, db = _session(tmp_path)
    try:
        _add_product(
            db,
            product_id="p1",
            category_id="cat_phone",
            price=3999,
            title=f"{PHOTO_CN} {PHONE_CN} A",
            brand="Xiaomi",
            tags=[PHOTO_CN],
        )
        _add_product(
            db,
            product_id="p2",
            category_id="cat_phone",
            price=5999,
            title=f"{PHOTO_CN} {PHONE_CN} B",
            brand="Xiaomi",
            tags=[PHOTO_CN],
        )
        _add_product(
            db,
            product_id="s1",
            category_id="cat_shoes",
            price=499,
            title="commute shoes",
            brand="Demo",
            tags=["commute"],
        )
        db.commit()

        service = ProductRetrievalService(
            db,
            embedding_service=MockEmbeddingService(),
            chroma_client=EmptyChromaClient(),
        )
        results = service.search_products(
            query=f"budget 5000 {PHOTO_CN} {PHONE_CN}",
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=5000,
                preferences=[PHOTO_CN],
            ),
            top_k=5,
        )

        assert [candidate.product_id for candidate in results] == ["p1"]
        assert service.last_structured_filters["category_id"] == "cat_phone"
        assert service.last_structured_filters["budget_max"] == 5000
        assert service.last_filtered_count == 1
    finally:
        db.close()
        engine.dispose()


def test_product_retrieval_filters_negative_preferences(tmp_path) -> None:
    engine, db = _session(tmp_path)
    try:
        _add_product(
            db,
            product_id="p1",
            category_id="cat_phone",
            price=3999,
            title=f"{APPLE_CN} {PHOTO_CN} {PHONE_CN}",
            brand="Apple",
            tags=[PHOTO_CN],
        )
        _add_product(
            db,
            product_id="p2",
            category_id="cat_phone",
            price=3999,
            title=f"Xiaomi {PHOTO_CN} {PHONE_CN}",
            brand="Xiaomi",
            tags=[PHOTO_CN],
        )
        db.commit()

        service = ProductRetrievalService(
            db,
            embedding_service=MockEmbeddingService(),
            chroma_client=EmptyChromaClient(),
        )
        results = service.search_products(
            query=f"budget 5000 {PHOTO_CN} {PHONE_CN}, not Apple",
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=5000,
                preferences=[PHOTO_CN],
                brand_exclude=[APPLE_CN],
            ),
            top_k=5,
        )

        assert [candidate.product_id for candidate in results] == ["p2"]
        assert service.last_negative_filtered_count == 1
        assert service.last_negative_filter_fallback is False
    finally:
        db.close()
        engine.dispose()


def test_knowledge_retrieval_query_uses_category_and_preferences() -> None:
    query = build_knowledge_retrieval_query(
        f"budget 5000 {PHOTO_CN} {BATTERY_CN} {PHONE_CN}",
        category_id="cat_phone",
        preferences=[PHOTO_CN, BATTERY_CN],
    )

    assert PHONE_CN in query
    assert PHOTO_CN in query
    assert BATTERY_CN in query


def test_skincare_knowledge_retrieval_query_removes_medical_claims() -> None:
    query = build_knowledge_retrieval_query(
        f"budget 300 {TREAT_CN} acne skincare",
        category_id="cat_skincare",
        preferences=[OIL_CONTROL_CN],
    )

    for term in [
        TREAT_CN,
        CURE_CN,
        DRUG_EFFECT_CN,
        PRESCRIPTION_CN,
        MEDICAL_REPAIR_CN,
    ]:
        assert term not in query
    assert OIL_CONTROL_CN in query
    assert GENTLE_CARE_CN in query


def test_product_comparison_resolves_referenced_indices_from_last_products() -> None:
    state = create_initial_agent_state("compare first and second", session_id="s1")
    result = QueryUnderstandingResult(
        original_query=state.original_query,
        effective_query=state.original_query,
        intent="compare",
        category="phone",
        referenced_product_indices=[1, 2],
        shopping_memory={
            "category": "phone",
            "budget": {"min": None, "max": 5000, "currency": "CNY"},
            "preferences": [PHOTO_CN],
            "negative_preferences": [],
            "last_product_ids": ["p1", "p2", "p3"],
            "last_intent": "shopping_guide",
        },
    )
    context = AgentRuntimeContext(
        query_understanding_service=StaticQueryUnderstandingService(result)
    )

    intent_router_node(state, context)

    assert state.compare_context is not None
    assert state.compare_context.product_ids == ["p1", "p2"]
    assert state.compare_context.referenced_product_indices == [1, 2]
    assert state.compare_context.resolved_from_last_products is True


def test_product_comparison_without_last_products_needs_clarification() -> None:
    state = create_initial_agent_state("compare first and second", session_id="s1")
    result = QueryUnderstandingResult(
        original_query=state.original_query,
        effective_query=state.original_query,
        intent="compare",
        category="phone",
        referenced_product_indices=[1, 2],
        shopping_memory={
            "category": "phone",
            "budget": {"min": None, "max": 5000, "currency": "CNY"},
            "preferences": [PHOTO_CN],
            "negative_preferences": [],
            "last_product_ids": [],
            "last_intent": "shopping_guide",
        },
    )
    context = AgentRuntimeContext(
        query_understanding_service=StaticQueryUnderstandingService(result)
    )

    intent_router_node(state, context)

    assert state.compare_context is None
    assert state.need_clarification is True
    comparison_trace = next(
        step for step in state.trace if step.get("step") == "product_comparison"
    )
    assert comparison_trace["reason"] == "missing_last_products"
    assert comparison_trace["comparison_product_count"] == 0
