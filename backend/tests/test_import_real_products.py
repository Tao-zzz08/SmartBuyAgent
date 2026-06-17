from pathlib import Path
import json
import sys

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models import Product, ProductAttribute, ProductTag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import import_seed_data  # noqa: E402
from import_real_products import import_real_products  # noqa: E402


def _create_test_session(db_name: str):
    db_path = PROJECT_ROOT / "data" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestingSessionLocal(), db_path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def _product_record(price: int = 54999, title: str = "OnePlus 11 5G") -> dict:
    return {
        "source_product_id": "phone_real_001",
        "category": "phone",
        "title": title,
        "brand": "OnePlus",
        "price": price,
        "currency": "INR",
        "description": "OnePlus 11 5G test product",
        "image_url": None,
        "source_url": None,
        "source_platform": "kaggle",
        "rating": 4.5,
        "tags": ["5g", "快充"],
        "attributes": {
            "network_type": "5g",
            "chipset": "Snapdragon 8 Gen2",
            "ram_gb": 12,
            "battery_mah": 5000,
        },
        "data_quality": {"missing_fields": [], "warnings": []},
    }


def test_import_real_products_dry_run_does_not_write_db(tmp_path: Path) -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_real_dry_run_test.db")
    input_path = tmp_path / "products.jsonl"
    _write_jsonl(input_path, [_product_record()])
    try:
        import_seed_data(db, PROJECT_ROOT)

        stats = import_real_products(
            db,
            input_path=input_path,
            category="phone",
            upsert=True,
            dry_run=True,
        )

        assert stats["inserted_count"] == 1
        assert db.scalar(select(func.count()).select_from(Product)) == 0
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_import_real_products_upsert_updates_existing_product(tmp_path: Path) -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_real_upsert_test.db")
    input_path = tmp_path / "products.jsonl"
    updated_path = tmp_path / "products_updated.jsonl"
    _write_jsonl(input_path, [_product_record()])
    _write_jsonl(updated_path, [_product_record(price=52999, title="OnePlus 11 5G Updated")])
    try:
        import_seed_data(db, PROJECT_ROOT)

        first_stats = import_real_products(
            db,
            input_path=input_path,
            category="phone",
            upsert=True,
        )
        second_stats = import_real_products(
            db,
            input_path=updated_path,
            category="phone",
            upsert=True,
        )

        products = db.scalars(select(Product)).all()
        product = products[0]
        tag_count = db.scalar(
            select(func.count()).select_from(ProductTag).where(ProductTag.product_id == product.id)
        )
        attribute_count = db.scalar(
            select(func.count())
            .select_from(ProductAttribute)
            .where(ProductAttribute.product_id == product.id)
        )

        assert first_stats["inserted_count"] == 1
        assert second_stats["updated_count"] == 1
        assert len(products) == 1
        assert product.title == "OnePlus 11 5G Updated"
        assert product.price == 52999
        assert product.currency == "INR"
        assert product.external_product_id == "phone_real_001"
        assert tag_count == 2
        assert attribute_count >= 5
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
