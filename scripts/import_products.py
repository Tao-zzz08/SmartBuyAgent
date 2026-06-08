from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.orm import Session as DbSession  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Category, Product, ProductAttribute, ProductTag  # noqa: E402
import app.models  # noqa: E402,F401


CATEGORY_PATH_MAP = {
    "数码/手机": "cat_phone",
    "服装/鞋靴": "cat_shoes",
    "美妆/护肤": "cat_skincare",
}
REQUIRED_CATEGORY_IDS = set(CATEGORY_PATH_MAP.values())
MISSING_CATEGORIES_MESSAGE = "请先运行 python ../scripts/import_categories.py"

ProductRecord = dict[str, Any]
ImportStats = dict[str, int]


class ProductImportError(RuntimeError):
    pass


def load_product_csv(path: str | Path) -> list[ProductRecord]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_dataset_files(
    root_dir: str | Path,
    dataset: str = "mini",
) -> list[Path]:
    if dataset != "mini":
        raise ValueError("Only the mini dataset is available in this task.")

    seed_dir = Path(root_dir) / "data" / "seed" / dataset
    files = [
        seed_dir / "products_phone.csv",
        seed_dir / "products_shoes.csv",
        seed_dir / "products_skincare.csv",
    ]
    missing_files = [path for path in files if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing product seed files: {missing}")
    return files


def resolve_category_id(category_path: str) -> str:
    try:
        return CATEGORY_PATH_MAP[category_path.strip()]
    except KeyError as exc:
        raise ValueError(f"Unsupported category_path: {category_path}") from exc


def parse_tags(tags_text: str | None) -> list[str]:
    if not tags_text:
        return []

    tags: list[str] = []
    seen: set[str] = set()
    for tag in re.split(r"[,，]", tags_text):
        value = tag.strip()
        if value and value not in seen:
            tags.append(value)
            seen.add(value)
    return tags


def parse_attributes_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}

    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("attributes_json must be a JSON object")
    return value


def extract_number(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _ensure_required_categories(db: DbSession) -> None:
    existing_ids = {
        row[0]
        for row in db.execute(
            select(Category.id).where(Category.id.in_(REQUIRED_CATEGORY_IDS))
        ).all()
    }
    if existing_ids != REQUIRED_CATEGORY_IDS:
        raise ProductImportError(MISSING_CATEGORIES_MESSAGE)


def _product_record_from_row(row: ProductRecord) -> ProductRecord:
    category_id = resolve_category_id(row["category_path"])
    return {
        "id": row["id"].strip(),
        "category_id": category_id,
        "title": row["title"].strip(),
        "brand": row.get("brand") or None,
        "price": int(row["price"]),
        "stock": int(row.get("stock") or 0),
        "description": row.get("description") or None,
        "image_url": row.get("image_url") or None,
        "source_platform": row.get("source_platform") or None,
        "source_url": row.get("source_url") or None,
        "compare_url": row.get("compare_url") or None,
        "external_links_json": "[]",
        "status": "active",
    }


def _upsert_product(db: DbSession, record: ProductRecord) -> str:
    existing = db.get(Product, record["id"])
    if existing is None:
        db.add(Product(**record))
        return "inserted"

    for key, value in record.items():
        setattr(existing, key, value)
    return "updated"


def _replace_tags(
    db: DbSession,
    product_id: str,
    tags: list[str],
) -> int:
    db.execute(delete(ProductTag).where(ProductTag.product_id == product_id))
    for index, tag in enumerate(tags, start=1):
        db.add(
            ProductTag(
                id=f"{product_id}_tag_{index:02d}",
                product_id=product_id,
                tag_type="tag",
                value=tag,
            )
        )
    return len(tags)


def _replace_attributes(
    db: DbSession,
    product_id: str,
    attributes: dict[str, Any],
) -> int:
    db.execute(
        delete(ProductAttribute).where(ProductAttribute.product_id == product_id)
    )
    for index, (name, value) in enumerate(attributes.items(), start=1):
        db.add(
            ProductAttribute(
                id=f"{product_id}_attr_{index:02d}",
                product_id=product_id,
                attr_name=name,
                attr_value=str(value),
                attr_value_number=extract_number(value),
            )
        )
    return len(attributes)


def import_products(
    db: DbSession,
    root_dir: str | Path = PROJECT_ROOT,
    dataset: str = "mini",
) -> ImportStats:
    _ensure_required_categories(db)

    stats: ImportStats = {
        "products_inserted": 0,
        "products_updated": 0,
        "product_tags_inserted": 0,
        "product_attributes_inserted": 0,
        "total_products": 0,
    }

    for path in load_dataset_files(root_dir, dataset):
        for row in load_product_csv(path):
            record = _product_record_from_row(row)
            result = _upsert_product(db, record)
            stats[f"products_{result}"] += 1
            stats["product_tags_inserted"] += _replace_tags(
                db,
                record["id"],
                parse_tags(row.get("tags")),
            )
            stats["product_attributes_inserted"] += _replace_attributes(
                db,
                record["id"],
                parse_attributes_json(row.get("attributes_json")),
            )

    db.commit()
    stats["total_products"] = db.scalar(select(func.count()).select_from(Product)) or 0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SmartBuyAgent product seeds.")
    parser.add_argument("--dataset", default="mini", help="Seed dataset name. Default: mini")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        stats = import_products(db, PROJECT_ROOT, dataset=args.dataset)
    except ProductImportError as exc:
        db.rollback()
        raise SystemExit(str(exc)) from exc
    finally:
        db.close()

    print(
        "products inserted: "
        f"{stats['products_inserted']}, updated: {stats['products_updated']}"
    )
    print(f"product_tags inserted: {stats['product_tags_inserted']}")
    print(f"product_attributes inserted: {stats['product_attributes_inserted']}")
    print(f"total products: {stats['total_products']}")


if __name__ == "__main__":
    main()
