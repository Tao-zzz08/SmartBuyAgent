from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CATEGORIES = {"phone", "shoes", "skincare"}
SKINCARE_BANNED_TERMS = ["治疗", "治愈", "药效", "祛病", "医学修复", "药用", "处方"]

PHONE_BRAND_MAP = {
    "oneplus": "OnePlus",
    "samsung": "Samsung",
    "xiaomi": "Xiaomi",
    "redmi": "Redmi",
    "poco": "Poco",
    "vivo": "Vivo",
    "iqoo": "iQOO",
    "apple": "Apple",
    "realme": "Realme",
    "oppo": "Oppo",
    "motorola": "Motorola",
}


def read_input_records(path: str | Path, encoding: str = "utf-8-sig") -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding=encoding, newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError(f"CSV file has no header: {input_path}")
            return [dict(row) for row in reader]

    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open("r", encoding=encoding) as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                records.append(value)
        return records

    if suffix == ".json":
        with input_path.open("r", encoding=encoding) as file:
            value = json.load(file)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("products", "items", "data"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [value]
        raise ValueError("JSON input must be an object or array")

    raise ValueError(f"Unsupported input file type: {input_path.suffix}")


def normalize_file(
    input_path: str | Path,
    output_path: str | Path,
    category: str,
    source_platform: str,
    limit: int | None = None,
    dry_run: bool = False,
    encoding: str = "utf-8-sig",
    currency: str | None = None,
) -> dict[str, Any]:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")

    records = read_input_records(input_path, encoding=encoding)
    if Path(input_path).suffix.lower() == ".csv":
        _validate_csv_columns(records, category=category, input_path=Path(input_path))
    normalized: list[dict[str, Any]] = []
    skipped_count = 0
    warnings: list[str] = []

    for row_index, row in enumerate(records, start=1):
        if limit is not None and len(normalized) >= limit:
            break
        try:
            product = normalize_record(
                row,
                category=category,
                source_platform=source_platform,
                currency=currency,
                row_index=row_index,
            )
        except Exception as exc:
            skipped_count += 1
            warnings.append(f"row {row_index}: {exc}")
            continue

        if product is None:
            skipped_count += 1
            continue
        normalized.append(product)

    if not dry_run:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as file:
            for item in normalized:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "input_count": len(records),
        "normalized_count": len(normalized),
        "skipped_count": skipped_count,
        "warnings": warnings,
        "dry_run": dry_run,
    }


def normalize_record(
    row: dict[str, Any],
    category: str,
    source_platform: str,
    currency: str | None = None,
    row_index: int = 0,
) -> dict[str, Any] | None:
    if category == "phone":
        return normalize_phone_record(row, source_platform, currency, row_index)
    if category == "shoes":
        return normalize_shoes_record(row, source_platform, currency, row_index)
    if category == "skincare":
        return normalize_skincare_record(row, source_platform, currency, row_index)
    raise ValueError(f"Unsupported category: {category}")


def _validate_csv_columns(
    records: list[dict[str, Any]],
    category: str,
    input_path: Path,
) -> None:
    columns = set(records[0].keys()) if records else set()
    missing: list[str] = []
    if category == "phone":
        missing = [field for field in ["model", "price"] if field not in columns]
    elif category == "shoes":
        if not {"title", "product_name", "name"} & columns:
            missing.append("title/product_name/name")
        if "price" not in columns:
            missing.append("price")
    elif category == "skincare":
        if not {"product_name", "title", "name"} & columns:
            missing.append("product_name/title/name")
        if "price" not in columns:
            missing.append("price")
    if missing:
        raise ValueError(
            f"CSV file {input_path} is missing required columns: {', '.join(missing)}"
        )


def normalize_phone_record(
    row: dict[str, Any],
    source_platform: str,
    currency: str | None,
    row_index: int,
) -> dict[str, Any] | None:
    title = _text(_first(row, "model", "title", "name", "product_name"))
    if not title:
        return None

    price = _parse_price(row.get("price"))
    if price is None or price <= 0:
        return None

    warnings: list[str] = []
    missing_fields = _missing(row, ["model", "price"])
    brand = _text(row.get("brand")) or _phone_brand_from_model(title)
    ram_gb = _parse_int(row.get("ram_gb"))
    if ram_gb is not None and ram_gb > 32:
        return None

    storage_gb = _parse_int(row.get("storage_gb"))
    if storage_gb is not None and ram_gb is not None and storage_gb == ram_gb and storage_gb <= 32:
        warnings.append("storage_gb equals ram_gb and was set to null")
        storage_gb = None

    battery_mah = _parse_int(row.get("battery_mah"))
    if battery_mah is not None and (battery_mah < 1000 or battery_mah > 10000):
        warnings.append("battery_mah looks unusual")

    attributes = _dict_or_empty(row.get("attributes"))
    attributes.update(
        {
            "os": _value_or_existing(_text(row.get("os")) or None, attributes, "os"),
            "sim_type": _value_or_existing(_text(row.get("sim_type")) or None, attributes, "sim_type"),
            "network_type": _value_or_existing(_text(row.get("network_type")) or None, attributes, "network_type"),
            "chipset": _value_or_existing(_text(row.get("chipset")) or None, attributes, "chipset"),
            "core_type": _value_or_existing(_text(row.get("core_type")) or None, attributes, "core_type"),
            "clock_ghz": _value_or_existing(_parse_float(row.get("clock_ghz")), attributes, "clock_ghz"),
            "ram_gb": _value_or_existing(ram_gb, attributes, "ram_gb"),
            "storage_gb": _value_or_existing(storage_gb, attributes, "storage_gb"),
            "battery_mah": _value_or_existing(battery_mah, attributes, "battery_mah"),
            "fast_charge_w": _value_or_existing(_parse_int(row.get("fast_charge_w")), attributes, "fast_charge_w"),
            "screen_size_in": _value_or_existing(_parse_float(row.get("screen_size_in")), attributes, "screen_size_in"),
            "resolution": _value_or_existing(_text(row.get("resolution")) or None, attributes, "resolution"),
            "refresh_rate_hz": _value_or_existing(_parse_int(row.get("refresh_rate_hz")), attributes, "refresh_rate_hz"),
            "display_type": _value_or_existing(_text(row.get("display_type")) or None, attributes, "display_type"),
            "rear_camera_mp_list": _value_or_existing(_text(row.get("rear_camera_mp_list")) or None, attributes, "rear_camera_mp_list"),
            "rear_camera_count": _value_or_existing(_parse_int(row.get("rear_camera_count")), attributes, "rear_camera_count"),
            "rear_camera_max_mp": _value_or_existing(_parse_float(row.get("rear_camera_max_mp")), attributes, "rear_camera_max_mp"),
            "front_camera_mp": _value_or_existing(_parse_float(row.get("front_camera_mp")), attributes, "front_camera_mp"),
            "memory_card_supported": _value_or_existing(_parse_bool(row.get("memory_card_supported")), attributes, "memory_card_supported"),
            "memory_card_type": _value_or_existing(_text(row.get("memory_card_type")) or None, attributes, "memory_card_type"),
            "memory_card_max_gb": _value_or_existing(_parse_int(row.get("memory_card_max_gb")), attributes, "memory_card_max_gb"),
            "volte": _value_or_existing(_parse_bool(row.get("VoLTE")), attributes, "volte"),
            "nfc": _value_or_existing(_parse_bool(row.get("NFC")), attributes, "nfc"),
            "ir_blaster": _value_or_existing(_parse_bool(row.get("ir_blaster")), attributes, "ir_blaster"),
        }
    )
    tags = _unique([*_list_from_any(row.get("tags")), *_derive_phone_tags(attributes)])

    product_currency = currency or _text(row.get("currency")) or "INR"
    return _standard_product(
        row=row,
        category="phone",
        title=title,
        brand=brand,
        price=price,
        currency=product_currency,
        description=_text(row.get("description")) or _phone_description(title, attributes),
        image_url=_text(_first(row, "image_url", "image")),
        source_url=_text(row.get("source_url")),
        source_platform=source_platform,
        rating=_parse_float(row.get("rating")),
        tags=tags,
        attributes=attributes,
        missing_fields=missing_fields,
        warnings=warnings,
        row_index=row_index,
    )


def normalize_shoes_record(
    row: dict[str, Any],
    source_platform: str,
    currency: str | None,
    row_index: int,
) -> dict[str, Any] | None:
    title = _text(_first(row, "title", "product_name", "name"))
    price = _parse_price(row.get("price"))
    if not title or price is None or price <= 0:
        return None
    brand = _text(row.get("brand"))
    missing_fields = _missing_from_values({"title": title, "brand": brand, "price": price})
    text = " ".join(str(value) for value in row.values() if value is not None).lower()
    attributes = _dict_or_empty(row.get("attributes"))
    attributes.update(
        {
            "gender": _text(row.get("gender")) or None,
            "color": _text(row.get("color")) or None,
            "sizes": _sizes_from_any(_first(row, "sizes", "size")),
            "upper_material": _text(_first(row, "upper_material", "material")) or None,
            "sole_material": _text(row.get("sole_material")) or None,
            "season": _text(row.get("season")) or None,
            "anti_slip": _contains_any(text, ["anti slip", "anti-slip", "防滑"]),
            "breathable": _contains_any(text, ["breathable", "mesh", "透气"]),
        }
    )
    attributes = {key: value for key, value in attributes.items() if value not in (None, [], "")}
    tags = _unique([*_list_from_any(row.get("tags")), *_derive_shoes_tags(text)])
    return _standard_product(
        row=row,
        category="shoes",
        title=title,
        brand=brand,
        price=price,
        currency=currency or _text(row.get("currency")) or "UNKNOWN",
        description=_text(row.get("description")) or title,
        image_url=_text(row.get("image_url")),
        source_url=_text(row.get("source_url")),
        source_platform=source_platform,
        rating=_parse_float(row.get("rating")),
        tags=tags,
        attributes=attributes,
        missing_fields=missing_fields,
        warnings=["currency missing"] if not (currency or _text(row.get("currency"))) else [],
        row_index=row_index,
    )


def normalize_skincare_record(
    row: dict[str, Any],
    source_platform: str,
    currency: str | None,
    row_index: int,
) -> dict[str, Any] | None:
    title = _text(_first(row, "product_name", "title", "name"))
    price = _parse_price(row.get("price"))
    if not title or price is None or price <= 0:
        return None
    combined = " ".join(str(value) for value in row.values() if value is not None)
    if _contains_banned_skincare_term(combined):
        return None
    brand = _text(_first(row, "brands", "brand"))
    ingredients_text = _text(row.get("ingredients_text"))
    categories = _text(row.get("categories"))
    attributes = _dict_or_empty(row.get("attributes"))
    ingredients = _split_terms(ingredients_text)
    attributes.update(
        {
            "skin_type": _skin_types_from_text(combined),
            "texture": _texture_from_text(combined),
            "ingredients": ingredients,
            "contains_fragrance": _contains_any(ingredients_text.lower(), ["fragrance", "parfum", "香精"]),
            "contains_alcohol": _contains_any(ingredients_text.lower(), ["alcohol", "酒精"]),
            "routine_step": _routine_step_from_text(combined),
        }
    )
    attributes = {key: value for key, value in attributes.items() if value not in (None, [], "")}
    description = _text(row.get("description")) or _skincare_description(title, categories, ingredients)
    return _standard_product(
        row=row,
        category="skincare",
        title=title,
        brand=brand,
        price=price,
        currency=currency or _text(row.get("currency")) or "UNKNOWN",
        description=description,
        image_url=_text(_first(row, "image_url", "image_front_url")),
        source_url=_text(row.get("source_url")),
        source_platform=source_platform,
        rating=_parse_float(row.get("rating")),
        tags=_unique([*_list_from_any(row.get("tags")), *_derive_skincare_tags(combined)]),
        attributes=attributes,
        missing_fields=_missing_from_values({"title": title, "brand": brand, "price": price}),
        warnings=["currency missing"] if not (currency or _text(row.get("currency"))) else [],
        row_index=row_index,
    )


def _standard_product(
    row: dict[str, Any],
    category: str,
    title: str,
    brand: str,
    price: int,
    currency: str,
    description: str,
    image_url: str | None,
    source_url: str | None,
    source_platform: str,
    rating: float | None,
    tags: list[str],
    attributes: dict[str, Any],
    missing_fields: list[str],
    warnings: list[str],
    row_index: int,
) -> dict[str, Any]:
    source_product_id = _text(_first(row, "source_product_id", "id", "product_id"))
    if not source_product_id:
        source_product_id = _stable_source_id(category, source_platform, title, brand, row_index)
    return {
        "source_product_id": source_product_id,
        "category": category,
        "title": title,
        "brand": brand,
        "price": price,
        "currency": currency,
        "description": description,
        "image_url": image_url or None,
        "source_url": source_url or None,
        "source_platform": source_platform,
        "rating": rating,
        "tags": tags,
        "attributes": attributes,
        "data_quality": {
            "missing_fields": missing_fields,
            "warnings": warnings,
        },
    }


def _derive_phone_tags(attributes: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    network_type = str(attributes.get("network_type") or "").lower()
    chipset = str(attributes.get("chipset") or "").lower()
    if "5g" in network_type:
        tags.append("5g")
    if _num(attributes.get("refresh_rate_hz")) >= 120:
        tags.append("高刷屏")
    if _num(attributes.get("battery_mah")) >= 5000:
        tags.append("长续航")
    if _num(attributes.get("fast_charge_w")) >= 67:
        tags.append("快充")
    if _num(attributes.get("rear_camera_max_mp")) >= 50:
        tags.append("拍照")
    if _num(attributes.get("rear_camera_max_mp")) >= 100:
        tags.append("高像素")
    if any(token in chipset for token in ["snapdragon 8", "dimensity 9000", "a15", "a16", "a17"]):
        tags.append("性能")
    if attributes.get("nfc") is True:
        tags.append("NFC")
    if attributes.get("ir_blaster") is True:
        tags.append("红外")
    return tags


def _derive_shoes_tags(text: str) -> list[str]:
    mapping = [
        ("通勤", ["commute", "office", "通勤"]),
        ("运动", ["sport", "running", "运动", "跑步"]),
        ("户外", ["outdoor", "hiking", "户外"]),
        ("防滑", ["anti slip", "anti-slip", "防滑"]),
        ("透气", ["breathable", "mesh", "透气"]),
        ("冬季", ["winter", "冬"]),
        ("休闲", ["casual", "休闲"]),
        ("轻便", ["lightweight", "轻便"]),
        ("耐磨", ["durable", "耐磨", "耐穿"]),
    ]
    return [tag for tag, keywords in mapping if any(keyword in text for keyword in keywords)]


def _derive_skincare_tags(text: str) -> list[str]:
    lower = text.lower()
    mapping = [
        ("保湿", ["hyaluronic", "moistur", "保湿", "透明质酸"]),
        ("清爽", ["lightweight", "fresh", "清爽"]),
        ("舒缓", ["soothing", "calm", "舒缓"]),
        ("修护", ["ceramide", "panthenol", "repair", "修护"]),
        ("控油", ["oil control", "控油", "oily"]),
        ("日常护理", ["daily", "日常"]),
        ("屏障护理", ["barrier", "屏障"]),
        ("温和", ["gentle", "mild", "温和"]),
        ("无香精", ["fragrance-free", "no fragrance", "无香精"]),
        ("无酒精", ["alcohol-free", "无酒精"]),
    ]
    return [tag for tag, keywords in mapping if any(keyword in lower for keyword in keywords)]


def _phone_description(title: str, attributes: dict[str, Any]) -> str:
    parts = [title]
    if attributes.get("chipset"):
        parts.append(f"搭载 {attributes['chipset']}")
    if attributes.get("ram_gb"):
        parts.append(f"{attributes['ram_gb']}GB RAM")
    if attributes.get("battery_mah"):
        parts.append(f"{attributes['battery_mah']}mAh 电池")
    if attributes.get("fast_charge_w"):
        parts.append(f"{attributes['fast_charge_w']}W 快充")
    if attributes.get("refresh_rate_hz"):
        parts.append(f"{attributes['refresh_rate_hz']}Hz 屏幕")
    if attributes.get("rear_camera_max_mp"):
        parts.append(f"后置最高 {attributes['rear_camera_max_mp']}MP 摄像头")
    return "，".join(parts) + "。"


def _skincare_description(title: str, categories: str, ingredients: list[str]) -> str:
    parts = [title]
    if categories:
        parts.append(f"分类：{categories}")
    if ingredients:
        parts.append(f"包含成分：{', '.join(ingredients[:5])}")
    return "，".join(parts) + "。"


def _phone_brand_from_model(model: str) -> str:
    first = model.strip().split()[0].lower() if model.strip() else ""
    return PHONE_BRAND_MAP.get(first, first.title() if first else "")


def _stable_source_id(category: str, source_platform: str, title: str, brand: str, row_index: int) -> str:
    raw = f"{category}|{source_platform}|{brand}|{title}|{row_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{category}_{digest}"


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_price(value: Any) -> int | None:
    number = _parse_float(value)
    if number is None:
        return None
    return int(round(number))


def _parse_int(value: Any) -> int | None:
    number = _parse_float(value)
    if number is None:
        return None
    return int(round(number))


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "支持", "是"}:
        return True
    if text in {"0", "false", "no", "n", "不支持", "否"}:
        return False
    return None


def _value_or_existing(value: Any, attributes: dict[str, Any], key: str) -> Any:
    if value is not None:
        return value
    return attributes.get(key)


def _num(value: Any) -> float:
    return _parse_float(value) or 0.0


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _list_from_any(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，|/]", value) if item.strip()]
    return []


def _sizes_from_any(value: Any) -> list[str]:
    return _list_from_any(value)


def _split_terms(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;，]", value or "") if item.strip()]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _missing(row: dict[str, Any], fields: list[str]) -> list[str]:
    return [field for field in fields if not _text(row.get(field))]


def _missing_from_values(values: dict[str, Any]) -> list[str]:
    return [key for key, value in values.items() if value in (None, "", [])]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text.lower() for keyword in keywords)


def _contains_banned_skincare_term(text: str) -> bool:
    return any(term in text for term in SKINCARE_BANNED_TERMS)


def _skin_types_from_text(text: str) -> list[str]:
    lower = text.lower()
    result: list[str] = []
    for label, keys in {
        "oily": ["oily", "油皮", "控油"],
        "dry": ["dry", "干皮"],
        "sensitive": ["sensitive", "敏感"],
        "combination": ["combination", "混合"],
    }.items():
        if any(key in lower for key in keys):
            result.append(label)
    return result


def _texture_from_text(text: str) -> str | None:
    lower = text.lower()
    for label, keys in {
        "lotion": ["lotion", "乳液"],
        "cream": ["cream", "面霜"],
        "serum": ["serum", "精华"],
        "cleanser": ["cleanser", "洁面"],
        "toner": ["toner", "爽肤水"],
    }.items():
        if any(key in lower for key in keys):
            return label
    return None


def _routine_step_from_text(text: str) -> str | None:
    lower = text.lower()
    for label, keys in {
        "cleanser": ["cleanser", "洁面"],
        "toner": ["toner", "爽肤水"],
        "serum": ["serum", "精华"],
        "moisturizer": ["moisturizer", "cream", "lotion", "面霜", "乳液"],
        "sunscreen": ["sunscreen", "spf", "防晒"],
    }.items():
        if any(key in lower for key in keys):
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize CSV/JSON/JSONL products.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category", required=True, choices=sorted(SUPPORTED_CATEGORIES))
    parser.add_argument("--source-platform", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--currency", default=None)
    args = parser.parse_args()

    stats = normalize_file(
        input_path=args.input,
        output_path=args.output,
        category=args.category,
        source_platform=args.source_platform,
        limit=args.limit,
        dry_run=args.dry_run,
        encoding=args.encoding,
        currency=args.currency,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
