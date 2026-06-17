from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_product_dataset import validate_dataset  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def test_validate_product_dataset_accepts_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    _write_jsonl(
        path,
        [
            {
                "source_product_id": "phone_001",
                "category": "phone",
                "title": "OnePlus 11 5G",
                "brand": "OnePlus",
                "price": 54999,
                "currency": "INR",
                "description": "A test phone",
                "image_url": None,
                "source_url": None,
                "source_platform": "kaggle",
                "rating": 4.5,
                "tags": ["5g"],
                "attributes": {"network_type": "5g"},
                "data_quality": {"missing_fields": [], "warnings": []},
            }
        ],
    )

    is_valid, report = validate_dataset(path, category="phone", min_count=1)

    assert is_valid is True
    assert report["valid_count"] == 1
    assert report["image_url_coverage"] == 0.0
    assert report["attributes_coverage"] == 1.0


def test_validate_product_dataset_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    _write_jsonl(
        path,
        [
            {
                "category": "phone",
                "brand": "OnePlus",
                "price": 54999,
                "currency": "INR",
                "tags": [],
                "attributes": {},
            }
        ],
    )

    is_valid, report = validate_dataset(path, category="phone")

    assert is_valid is False
    assert report["skipped_count"] == 1
    assert any("missing required field: title" in error for error in report["errors"])


def test_validate_product_dataset_min_count_failure(tmp_path: Path) -> None:
    path = tmp_path / "valid.jsonl"
    _write_jsonl(
        path,
        [
            {
                "category": "phone",
                "title": "Phone",
                "brand": "Brand",
                "price": 1000,
                "currency": "INR",
                "tags": [],
                "attributes": {},
            }
        ],
    )

    is_valid, report = validate_dataset(path, category="phone", min_count=2)

    assert is_valid is False
    assert "valid_count 1 is lower than min_count 2" in report["errors"]


def test_validate_product_dataset_rejects_skincare_banned_terms(tmp_path: Path) -> None:
    path = tmp_path / "skincare.jsonl"
    _write_jsonl(
        path,
        [
            {
                "category": "skincare",
                "title": "Sensitive Cream",
                "brand": "Care",
                "price": 99,
                "currency": "CNY",
                "description": "可以治疗湿疹",
                "tags": ["修护"],
                "attributes": {},
            }
        ],
    )

    is_valid, report = validate_dataset(path, category="skincare")

    assert is_valid is False
    assert any("banned medical claim" in error for error in report["errors"])
