from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.query_understanding import QueryUnderstandingService
from app.core.db import Base
from app.models import Product, ProductTag
from app.retrieval.retrieval_service import ProductRetrievalService, ProductSearchFilters
from app.services.embedding import MockEmbeddingService


PHONE = "\u624b\u673a"
PHOTO = "\u62cd\u7167"
IMAGE = "\u5f71\u50cf"
SHOES = "\u978b"
COMMUTE = "\u901a\u52e4"
APPLE_CN = "\u82f9\u679c"


class EmptyCollection:
    def count(self) -> int:
        return 0


class EmptyChromaClient:
    def get_collection(self, name: str):
        return EmptyCollection()


def test_real_query_understanding_three_turn_budget_smoke() -> None:
    service = QueryUnderstandingService(llm_enabled=False)

    first = service.understand(f"\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e{PHOTO}\u597d\u7684{PHONE}")
    second = service.understand(
        "\u6211\u7684\u9884\u7b97\u589e\u52a0\u52304000\u5462",
        previous_memory=first.to_shopping_memory(),
    )
    third = service.understand(
        "\u589e\u52a0\u52305000\u5462",
        previous_memory=second.to_shopping_memory(),
    )

    assert third.intent == "shopping_guide"
    assert third.category == "phone"
    assert third.budget_max == 5000
    assert PHOTO in third.preferences
    assert third.is_follow_up is True
    assert third.source == "rule"
    assert third.llm_fallback_attempted is False
    assert "5000" in third.effective_query
    assert PHONE in third.effective_query
    assert PHOTO in third.effective_query


def test_real_product_retrieval_consumes_structured_filters_smoke(tmp_path) -> None:
    engine, db = _test_session(tmp_path)
    try:
        _add_product(
            db,
            product_id="phone_ok",
            category_id="cat_phone",
            price=3999,
            brand="Xiaomi",
            title=f"Xiaomi {PHOTO} {PHONE}",
            tags=[PHOTO, IMAGE],
            description=f"\u9002\u5408{PHOTO}\u548c\u65e5\u5e38\u4f7f\u7528",
        )
        _add_product(
            db,
            product_id="phone_over_budget",
            category_id="cat_phone",
            price=5999,
            brand="Xiaomi",
            title=f"premium {PHOTO} {PHONE}",
            tags=[PHOTO],
            description="\u4ef7\u683c\u8d85\u8fc7\u9884\u7b97",
        )
        _add_product(
            db,
            product_id="shoe_wrong_category",
            category_id="cat_shoes",
            price=499,
            brand="Nike",
            title=f"{COMMUTE} {SHOES}",
            tags=[COMMUTE],
            description=f"\u4e0d\u662f{PHONE}\u54c1\u7c7b",
        )
        _add_product(
            db,
            product_id="phone_negative_brand",
            category_id="cat_phone",
            price=3999,
            brand="Apple",
            title=f"{APPLE_CN} {PHOTO} {PHONE}",
            tags=[PHOTO],
            description="\u8d1f\u5411\u504f\u597d\u5e94\u8be5\u8fc7\u6ee4\u6216\u964d\u6743",
        )
        db.commit()

        service = ProductRetrievalService(
            db,
            embedding_service=MockEmbeddingService(),
            chroma_client=EmptyChromaClient(),
        )
        results = service.search_products(
            query=f"\u9884\u7b975000\u5143\u4ee5\u5185\uff0c\u63a8\u8350{PHOTO}\u597d\u7684{PHONE}\uff0c\u4e0d\u8003\u8651{APPLE_CN}",
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=5000,
                preferences=[PHOTO],
                brand_exclude=[APPLE_CN],
            ),
            top_k=5,
        )

        ids = [candidate.product_id for candidate in results]
        assert "phone_ok" in ids
        assert "shoe_wrong_category" not in ids
        assert "phone_over_budget" not in ids
        if "phone_negative_brand" in ids:
            assert ids.index("phone_ok") < ids.index("phone_negative_brand")
        else:
            assert service.last_negative_filtered_count >= 1

        assert service.last_structured_filters["category_id"] == "cat_phone"
        assert service.last_structured_filters["budget_max"] == 5000
        assert PHOTO in service.last_structured_filters["preferences"]
        assert APPLE_CN in service.last_structured_filters["negative_preferences"]
    finally:
        db.close()
        engine.dispose()


def _test_session(tmp_path):
    db_path = tmp_path / "real_query_understanding_smoke.db"
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
    brand: str,
    title: str,
    tags: list[str],
    description: str,
) -> None:
    db.add(
        Product(
            id=product_id,
            category_id=category_id,
            title=title,
            brand=brand,
            price=price,
            stock=10,
            description=description,
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
