from pathlib import Path
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
from import_products import (  # noqa: E402
    MISSING_CATEGORIES_MESSAGE,
    ProductImportError,
    import_products,
    load_dataset_files,
    load_product_csv,
)


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


def _count(db, model, *where_clauses) -> int:
    statement = select(func.count()).select_from(model)
    if where_clauses:
        statement = statement.where(*where_clauses)
    return db.scalar(statement) or 0


def test_mini_product_csv_files_can_be_loaded() -> None:
    files = load_dataset_files(PROJECT_ROOT, "mini")
    rows_by_file = {path.name: load_product_csv(path) for path in files}

    assert set(rows_by_file) == {
        "products_phone.csv",
        "products_shoes.csv",
        "products_skincare.csv",
    }
    assert all(len(rows) == 10 for rows in rows_by_file.values())


def test_import_products_requires_categories() -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_products_empty_test.db")
    try:
        try:
            import_products(db, PROJECT_ROOT, dataset="mini")
        except ProductImportError as exc:
            assert MISSING_CATEGORIES_MESSAGE in str(exc)
        else:
            raise AssertionError("ProductImportError was not raised")
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_import_products_is_idempotent() -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_products_test.db")
    try:
        import_seed_data(db, PROJECT_ROOT)

        first_stats = import_products(db, PROJECT_ROOT, dataset="mini")
        first_tag_count = _count(db, ProductTag)
        first_attribute_count = _count(db, ProductAttribute)

        assert first_stats["products_inserted"] == 30
        assert _count(db, Product) == 30
        assert _count(db, Product, Product.category_id == "cat_phone") == 10
        assert _count(db, Product, Product.category_id == "cat_shoes") == 10
        assert _count(db, Product, Product.category_id == "cat_skincare") == 10

        phone = db.get(Product, "phone_001")
        assert phone is not None
        assert phone.category_id == "cat_phone"
        assert _count(db, ProductTag, ProductTag.product_id == "phone_001") >= 2
        assert (
            _count(
                db,
                ProductAttribute,
                ProductAttribute.product_id == "phone_001",
            )
            >= 5
        )

        second_stats = import_products(db, PROJECT_ROOT, dataset="mini")

        assert second_stats["products_inserted"] == 0
        assert second_stats["products_updated"] == 30
        assert _count(db, Product) == 30
        assert _count(db, ProductTag) == first_tag_count
        assert _count(db, ProductAttribute) == first_attribute_count
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
