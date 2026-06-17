from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import delete, func, select  # noqa: E402
from sqlalchemy.orm import Session as DbSession  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Category, Product, ProductAttribute, ProductTag  # noqa: E402
import app.models  # noqa: E402,F401
from validate_product_dataset import load_jsonl, validate_record  # noqa: E402


CATEGORY_ID_MAP = {
    "phone": "cat_phone",
    "shoes": "cat_shoes",
    "skincare": "cat_skincare",
}


def import_real_products(
    db: DbSession,
    input_path: str | Path,
    category: str | None = None,
    upsert: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    records = load_jsonl(input_path)
    stats = {
        "inserted_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "total_products": db.scalar(select(func.count()).select_from(Product)) or 0,
    }

    try:
        for record in records:
            effective_category = category or record.get("category")
            category_id = CATEGORY_ID_MAP.get(str(effective_category))
            if category_id is None:
                stats["skipped_count"] += 1
                stats["error_count"] += 1
                continue
            _ensure_category_exists(db, category_id)

            errors = validate_record(record, expected_category=str(effective_category))
            if errors:
                stats["skipped_count"] += 1
                stats["error_count"] += 1
                continue

            existing = _find_existing_product(db, record, category_id)
            if existing is not None and not upsert:
                stats["skipped_count"] += 1
                continue

            action = "updated" if existing is not None else "inserted"
            if dry_run:
                stats[f"{action}_count"] += 1
                continue

            product_id = existing.id if existing is not None else _product_id(record)
            product_data = _product_data(record, product_id=product_id, category_id=category_id)
            if existing is None:
                db.add(Product(**product_data))
            else:
                for key, value in product_data.items():
                    setattr(existing, key, value)

            _replace_tags(db, product_id, record.get("tags") or [])
            _replace_attributes(db, product_id, _attributes_for_import(record))
            stats[f"{action}_count"] += 1

        if dry_run:
            db.rollback()
        else:
            db.commit()
            stats["total_products"] = (
                db.scalar(select(func.count()).select_from(Product)) or 0
            )
        return stats
    except Exception:
        db.rollback()
        raise


def _ensure_category_exists(db: DbSession, category_id: str) -> None:
    if db.get(Category, category_id) is None:
        raise RuntimeError(
            f"Category {category_id} is missing. Run scripts/import_categories.py first."
        )


def _find_existing_product(
    db: DbSession,
    record: dict[str, Any],
    category_id: str,
) -> Product | None:
    source_platform = str(record.get("source_platform") or "other")
    source_product_id = str(record.get("source_product_id") or "")
    if source_product_id:
        existing = db.scalar(
            select(Product).where(
                Product.category_id == category_id,
                Product.source_platform == source_platform,
                Product.external_product_id == source_product_id,
            )
        )
        if existing is not None:
            return existing

    brand = str(record.get("brand") or "")
    title = str(record.get("title") or "")
    if brand and title:
        return db.scalar(
            select(Product).where(
                Product.category_id == category_id,
                Product.brand == brand,
                Product.title == title,
            )
        )
    return None


def _product_id(record: dict[str, Any]) -> str:
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


def _product_data(
    record: dict[str, Any],
    product_id: str,
    category_id: str,
) -> dict[str, Any]:
    return {
        "id": product_id,
        "category_id": category_id,
        "title": str(record["title"]).strip(),
        "brand": str(record.get("brand") or "").strip() or None,
        "price": int(record["price"]),
        "currency": str(record.get("currency") or "UNKNOWN"),
        "stock": int(record.get("stock") or 100),
        "description": record.get("description") or None,
        "image_url": record.get("image_url") or None,
        "rating": float(record.get("rating") or 0.0),
        "sales": int(record.get("sales") or 0),
        "source_platform": str(record.get("source_platform") or "other"),
        "external_product_id": str(record.get("source_product_id") or ""),
        "source_url": record.get("source_url") or None,
        "compare_url": record.get("compare_url") or None,
        "external_links_json": "[]",
        "status": "active",
    }


def _replace_tags(db: DbSession, product_id: str, tags: list[str]) -> int:
    db.execute(delete(ProductTag).where(ProductTag.product_id == product_id))
    for index, tag in enumerate(_unique_tags(tags), start=1):
        db.add(
            ProductTag(
                id=f"{product_id}_tag_{index:02d}"[:64],
                product_id=product_id,
                tag_type="tag",
                value=str(tag),
            )
        )
    return len(tags)


def _replace_attributes(
    db: DbSession,
    product_id: str,
    attributes: dict[str, Any],
) -> int:
    db.execute(delete(ProductAttribute).where(ProductAttribute.product_id == product_id))
    for index, (name, value) in enumerate(attributes.items(), start=1):
        db.add(
            ProductAttribute(
                id=f"{product_id}_attr_{index:02d}"[:64],
                product_id=product_id,
                attr_name=str(name),
                attr_value=_attribute_value(value),
                attr_value_number=_extract_number(value),
            )
        )
    return len(attributes)


def _attributes_for_import(record: dict[str, Any]) -> dict[str, Any]:
    attributes = dict(record.get("attributes") or {})
    attributes.update(
        {
            "currency": record.get("currency"),
            "source_product_id": record.get("source_product_id"),
            "source_platform": record.get("source_platform"),
            "data_quality": record.get("data_quality") or {},
            "rag_product_text": build_rag_product_text(record),
        }
    )
    return {key: value for key, value in attributes.items() if value is not None}


def build_rag_product_text(record: dict[str, Any]) -> str:
    attributes = record.get("attributes") or {}
    specs: list[str] = []
    for key in (
        "chipset",
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "fast_charge_w",
        "refresh_rate_hz",
        "rear_camera_max_mp",
    ):
        value = attributes.get(key)
        if value not in (None, ""):
            specs.append(f"{key}: {value}")
    return "\n".join(
        [
            f"商品：{record.get('title')}",
            f"品牌：{record.get('brand')}",
            f"价格：{record.get('price')} {record.get('currency')}",
            f"标签：{'、'.join(record.get('tags') or [])}",
            f"参数：{'，'.join(specs)}",
            f"描述：{record.get('description') or ''}",
        ]
    ).strip()


def _attribute_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_number(value: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def _unique_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = str(tag).strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Import normalized real product JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--category", choices=sorted(CATEGORY_ID_MAP), default=None)
    parser.add_argument("--upsert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        stats = import_real_products(
            db,
            input_path=args.input,
            category=args.category,
            upsert=args.upsert,
            dry_run=args.dry_run,
        )
    finally:
        db.close()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
