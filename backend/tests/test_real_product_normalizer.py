from pathlib import Path
import json
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from normalize_real_products import normalize_file  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phone_csv_normalize_derives_schema_tags_and_warnings(tmp_path: Path) -> None:
    input_path = tmp_path / "phones.csv"
    output_path = tmp_path / "phones.jsonl"
    input_path.write_text(
        "\n".join(
            [
                "model,price,rating,os,network_type,NFC,ir_blaster,chipset,ram_gb,storage_gb,battery_mah,fast_charge_w,screen_size_in,refresh_rate_hz,rear_camera_max_mp,front_camera_mp",
                "OnePlus 11 5G,54999,4.5,Android v13,5g,true,false,Snapdragon 8 Gen2,12,12,5000,100,6.7,120,50,16",
                "Samsung Budget 4G,15999,4.0,Android v12,4g,false,false,Exynos 9611,6,128,5000,25,6.4,60,48,12",
            ]
        ),
        encoding="utf-8",
    )

    stats = normalize_file(
        input_path=input_path,
        output_path=output_path,
        category="phone",
        source_platform="kaggle",
        limit=10,
    )
    products = _read_jsonl(output_path)

    assert stats["normalized_count"] == 2
    assert products[0]["title"] == "OnePlus 11 5G"
    assert products[0]["brand"] == "OnePlus"
    assert products[0]["price"] == 54999
    assert products[0]["currency"] == "INR"
    assert {"5g", "高刷屏", "长续航", "快充", "拍照", "性能", "NFC"} <= set(
        products[0]["tags"]
    )
    assert products[0]["attributes"]["storage_gb"] is None
    assert products[0]["data_quality"]["warnings"]


def test_jsonl_normalize_preserves_standard_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "products.jsonl"
    output_path = tmp_path / "products_normalized.jsonl"
    original = {
        "source_product_id": "phone_json_001",
        "category": "phone",
        "title": "Apple iPhone Test",
        "brand": "Apple",
        "price": 89999,
        "currency": "INR",
        "description": "Test phone",
        "tags": ["5g"],
        "attributes": {"network_type": "5g", "nfc": True},
        "source_platform": "manual",
        "image_url": None,
        "source_url": None,
    }
    input_path.write_text(json.dumps(original, ensure_ascii=False) + "\n", encoding="utf-8")

    normalize_file(
        input_path=input_path,
        output_path=output_path,
        category="phone",
        source_platform="manual",
    )
    products = _read_jsonl(output_path)

    assert products[0]["source_product_id"] == "phone_json_001"
    assert products[0]["title"] == "Apple iPhone Test"
    assert products[0]["attributes"]["network_type"] == "5g"
    assert "5g" in products[0]["tags"]


def test_phone_csv_normalize_reports_missing_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "phones_missing.csv"
    output_path = tmp_path / "phones.jsonl"
    input_path.write_text("model,rating\nOnePlus 11,4.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns: price"):
        normalize_file(
            input_path=input_path,
            output_path=output_path,
            category="phone",
            source_platform="kaggle",
        )
