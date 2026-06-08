from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy.orm import Session as DbSession  # noqa: E402

from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import Category, CategoryAttributeDef, CategoryProfile  # noqa: E402
import app.models  # noqa: E402,F401


SeedRecord = dict[str, Any]
ImportStats = dict[str, dict[str, int]]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_categories(path: str | Path) -> list[SeedRecord]:
    return _read_json(Path(path))


def load_attribute_defs(path: str | Path) -> list[SeedRecord]:
    return _read_json(Path(path))


def load_category_profiles(profile_dir: str | Path) -> list[SeedRecord]:
    profiles: list[SeedRecord] = []
    for path in sorted(Path(profile_dir).glob("*.json")):
        profiles.append(_read_json(path))
    return profiles


def _upsert_records(
    db: DbSession,
    model: type,
    records: list[SeedRecord],
) -> dict[str, int]:
    stats = {"inserted": 0, "updated": 0}

    for record in records:
        existing = db.get(model, record["id"])
        if existing is None:
            db.add(model(**record))
            stats["inserted"] += 1
            continue

        for key, value in record.items():
            setattr(existing, key, value)
        stats["updated"] += 1

    return stats


def _profile_to_record(profile: SeedRecord) -> SeedRecord:
    category_id = profile["category_id"]
    return {
        "id": f"profile_{category_id}",
        "category_id": category_id,
        "profile_json": json.dumps(profile, ensure_ascii=False, indent=2),
    }


def import_seed_data(db: DbSession, root_dir: str | Path = PROJECT_ROOT) -> ImportStats:
    root = Path(root_dir)
    categories = load_categories(root / "data" / "seed" / "categories.json")
    attribute_defs = load_attribute_defs(
        root / "data" / "seed" / "category_attribute_defs.json"
    )
    profiles = load_category_profiles(root / "data" / "category_profiles")
    profile_records = [_profile_to_record(profile) for profile in profiles]

    stats: ImportStats = {
        "categories": _upsert_records(db, Category, categories),
        "category_attribute_defs": _upsert_records(
            db, CategoryAttributeDef, attribute_defs
        ),
        "category_profiles": _upsert_records(db, CategoryProfile, profile_records),
    }
    db.commit()
    return stats


def main() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        stats = import_seed_data(db, PROJECT_ROOT)
    finally:
        db.close()

    print(
        "categories imported: "
        f"{stats['categories']['inserted']}, updated: {stats['categories']['updated']}"
    )
    print(
        "category_attribute_defs imported: "
        f"{stats['category_attribute_defs']['inserted']}, "
        f"updated: {stats['category_attribute_defs']['updated']}"
    )
    print(
        "category_profiles imported: "
        f"{stats['category_profiles']['inserted']}, "
        f"updated: {stats['category_profiles']['updated']}"
    )


if __name__ == "__main__":
    main()
