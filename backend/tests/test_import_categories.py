from pathlib import Path
import json
import sys

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models import Category, CategoryAttributeDef, CategoryProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import (  # noqa: E402
    import_seed_data,
    load_attribute_defs,
    load_categories,
    load_category_profiles,
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


def test_seed_json_files_can_be_loaded() -> None:
    categories = load_categories(PROJECT_ROOT / "data" / "seed" / "categories.json")
    attribute_defs = load_attribute_defs(
        PROJECT_ROOT / "data" / "seed" / "category_attribute_defs.json"
    )
    profiles = load_category_profiles(PROJECT_ROOT / "data" / "category_profiles")

    assert len(categories) == 6
    assert {category["id"] for category in categories} == {
        "cat_digital",
        "cat_phone",
        "cat_fashion",
        "cat_shoes",
        "cat_beauty",
        "cat_skincare",
    }
    assert len(attribute_defs) >= 20
    assert {profile["category_id"] for profile in profiles} == {
        "cat_phone",
        "cat_shoes",
        "cat_skincare",
    }


def test_import_categories_is_idempotent() -> None:
    engine, db, db_path = _create_test_session("smartbuy_import_categories_test.db")
    try:
        first_stats = import_seed_data(db, PROJECT_ROOT)
        second_stats = import_seed_data(db, PROJECT_ROOT)

        assert first_stats["categories"]["inserted"] == 6
        assert second_stats["categories"]["inserted"] == 0
        assert second_stats["categories"]["updated"] == 6

        category_ids = {
            row[0] for row in db.execute(select(Category.id)).all()
        }
        assert {
            "cat_digital",
            "cat_phone",
            "cat_fashion",
            "cat_shoes",
            "cat_beauty",
            "cat_skincare",
        }.issubset(category_ids)

        cat_phone = db.get(Category, "cat_phone")
        assert cat_phone is not None
        assert cat_phone.parent_id == "cat_digital"

        phone_attr_count = db.scalar(
            select(func.count())
            .select_from(CategoryAttributeDef)
            .where(CategoryAttributeDef.category_id == "cat_phone")
        )
        assert phone_attr_count >= 5

        profile = db.get(CategoryProfile, "profile_cat_phone")
        assert profile is not None
        profile_json = json.loads(profile.profile_json)
        assert "decision_factors" in profile_json
        assert "card_fields" in profile_json

        assert db.scalar(select(func.count()).select_from(Category)) == 6
        assert (
            db.scalar(select(func.count()).select_from(CategoryProfile))
            == 3
        )
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
