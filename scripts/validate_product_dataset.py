from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


SUPPORTED_CATEGORIES = {"phone", "shoes", "skincare"}
REQUIRED_FIELDS = ["title", "category", "price", "brand", "currency"]
SKINCARE_BANNED_TERMS = ["治疗", "治愈", "药效", "祛病", "医学修复", "药用", "处方"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            records.append(value)
    return records


def validate_dataset(
    input_path: str | Path,
    category: str | None = None,
    min_count: int = 0,
) -> tuple[bool, dict[str, Any]]:
    records = load_jsonl(input_path)
    missing_field_stats: Counter[str] = Counter()
    duplicate_counter: Counter[tuple[str, str]] = Counter()
    warnings: list[str] = []
    errors: list[str] = []
    valid_count = 0
    image_url_count = 0
    source_url_count = 0
    description_count = 0
    attributes_count = 0

    for index, record in enumerate(records, start=1):
        row_errors = validate_record(record, expected_category=category)
        for field in REQUIRED_FIELDS:
            if not record.get(field):
                missing_field_stats[field] += 1
        if record.get("image_url"):
            image_url_count += 1
        if record.get("source_url"):
            source_url_count += 1
        if record.get("description"):
            description_count += 1
        if isinstance(record.get("attributes"), dict) and record["attributes"]:
            attributes_count += 1
        brand_title = (
            str(record.get("brand") or "").strip().lower(),
            str(record.get("title") or "").strip().lower(),
        )
        if all(brand_title):
            duplicate_counter[brand_title] += 1

        if row_errors:
            errors.extend(f"line {index}: {error}" for error in row_errors)
            continue
        valid_count += 1

    duplicate_count = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    if records and duplicate_count / len(records) > 0.1:
        warnings.append("duplicate brand+title ratio is higher than 10%")
    if valid_count < min_count:
        errors.append(f"valid_count {valid_count} is lower than min_count {min_count}")

    total_count = len(records)
    report = {
        "total_count": total_count,
        "valid_count": valid_count,
        "skipped_count": total_count - valid_count,
        "missing_field_stats": dict(missing_field_stats),
        "image_url_coverage": _ratio(image_url_count, total_count),
        "source_url_coverage": _ratio(source_url_count, total_count),
        "description_coverage": _ratio(description_count, total_count),
        "attributes_coverage": _ratio(attributes_count, total_count),
        "duplicate_count": duplicate_count,
        "warnings": warnings,
        "errors": errors,
    }
    return not errors, report


def validate_record(
    record: dict[str, Any],
    expected_category: str | None = None,
) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if record.get(field) in (None, "", []):
            errors.append(f"missing required field: {field}")

    category = record.get("category")
    if category not in SUPPORTED_CATEGORIES:
        errors.append(f"unsupported category: {category}")
    if expected_category and category != expected_category:
        errors.append(f"category mismatch: expected {expected_category}, got {category}")

    price = record.get("price")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
        errors.append("price must be a positive number")

    if not isinstance(record.get("tags"), list):
        errors.append("tags must be a list")
    if not isinstance(record.get("attributes"), dict):
        errors.append("attributes must be a dict")

    if category == "skincare" and _contains_banned_skincare_term(record):
        errors.append("skincare record contains banned medical claim terms")

    return errors


def _contains_banned_skincare_term(record: dict[str, Any]) -> bool:
    text = json.dumps(record, ensure_ascii=False)
    return any(term in text for term in SKINCARE_BANNED_TERMS)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized product JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--category", choices=sorted(SUPPORTED_CATEGORIES), default=None)
    parser.add_argument("--min-count", type=int, default=0)
    args = parser.parse_args()

    is_valid, report = validate_dataset(
        input_path=args.input,
        category=args.category,
        min_count=args.min_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
