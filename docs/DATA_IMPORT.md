# Data Import Pipeline

Stage Data-1 adds a reusable local dataset import pipeline for CSV, JSON, and JSONL product files.

The pipeline does not crawl ecommerce sites, call external APIs, download third-party images, create purchase links, or change the AgentWorkflow runtime behavior.

## Directory Layout

```text
data/raw/products/
data/processed/products/
data/templates/
```

- `data/raw/products/`: local raw CSV/JSON/JSONL files from sources such as Kaggle exports, Open Beauty Facts exports, eBay API exports, or manual datasets.
- `data/processed/products/`: normalized JSONL files using the SmartBuyAgent product schema.
- `data/templates/product_schema.json`: example normalized product object.

## Flow

```text
data/raw/products/*.csv or *.json or *.jsonl
-> scripts/normalize_real_products.py
-> data/processed/products/*.jsonl
-> scripts/validate_product_dataset.py
-> scripts/import_real_products.py
-> MySQL / SQLite
-> scripts/rebuild_index.py
```

Seed data import remains available through `scripts/import_products.py` for tests and lightweight demos.

## Standard Product Schema

Each processed JSONL line is one product:

```json
{
  "source_product_id": "phone_000001",
  "category": "phone",
  "title": "OnePlus 11 5G",
  "brand": "OnePlus",
  "price": 54999,
  "currency": "INR",
  "description": "OnePlus 11 5G 搭载 Snapdragon 8 Gen2，12GB RAM，5000mAh 电池，100W 快充。",
  "image_url": null,
  "source_url": null,
  "source_platform": "kaggle",
  "rating": 4.5,
  "tags": ["5g", "高刷屏", "长续航", "快充", "拍照", "性能"],
  "attributes": {
    "os": "Android v13",
    "network_type": "5g",
    "chipset": "Snapdragon 8 Gen2",
    "ram_gb": 12,
    "storage_gb": null,
    "battery_mah": 5000
  },
  "data_quality": {
    "missing_fields": [],
    "warnings": []
  }
}
```

`source_url` is a data-source reference only. It is not presented as a guaranteed purchase or checkout link.

Images are represented by `image_url` when available. The project does not download or redistribute third-party image files.

## Normalize CSV

Phone CSV example:

```bash
python scripts/normalize_real_products.py \
  --input data/raw/products/phones_cleaned_data.csv \
  --output data/processed/products/phones_500.jsonl \
  --category phone \
  --source-platform kaggle \
  --limit 500
```

The phone normalizer supports fields such as:

```text
model, price, rating, os, sim_type, network_type, VoLTE, NFC, ir_blaster,
chipset, core_type, clock_ghz, ram_gb, storage_gb, battery_mah, fast_charge_w,
screen_size_in, resolution, refresh_rate_hz, display_type, rear_camera_mp_list,
rear_camera_count, rear_camera_max_mp, front_camera_mp, memory_card_supported,
memory_card_type, memory_card_max_gb
```

Phone-specific rules:

- `model` becomes `title`.
- Brand is inferred from the first token and normalized for common brands such as OnePlus, Samsung, Xiaomi, Redmi, Poco, Vivo, iQOO, Apple, Realme, Oppo, and Motorola.
- Phone price defaults to `INR` unless `--currency` is provided.
- `storage_gb` is set to `null` with a warning if it obviously duplicates `ram_gb`.
- Tags are derived from available fields: `5g`, `高刷屏`, `长续航`, `快充`, `拍照`, `高像素`, `性能`, `NFC`, `红外`.
- Rows with missing title, invalid price, or clearly invalid RAM are skipped.

## Normalize JSON / JSONL

```bash
python scripts/normalize_real_products.py \
  --input data/raw/products/products.jsonl \
  --output data/processed/products/products_normalized.jsonl \
  --category phone \
  --source-platform kaggle
```

JSON array, JSON object, and JSONL inputs are supported. If the input is already close to the standard schema, tags and attributes are preserved.

## Shoes Mapping

The shoes normalizer supports fields such as:

```text
title / product_name / name
brand
price
currency
description
image_url
source_url
gender
color
size / sizes
material
```

It can derive tags such as `通勤`, `运动`, `户外`, `防滑`, `透气`, `冬季`, `休闲`, `轻便`, and `耐磨` from available text. It does not invent missing fields.

## Skincare Mapping and Safety

The skincare normalizer supports fields such as:

```text
product_name / title / name
brands / brand
price
currency
ingredients_text
categories
image_url / image_front_url
source_url
```

Allowed tags are shopping-oriented and non-medical, such as `保湿`, `清爽`, `舒缓`, `修护`, `控油`, `日常护理`, `屏障护理`, `温和`, `无香精`, and `无酒精`.

The normalizer and validator reject or flag medical claim terms:

```text
治疗, 治愈, 药效, 祛病, 医学修复, 药用, 处方
```

Skincare descriptions must not promise treatment, cure, or drug effects.

## Validate Processed JSONL

```bash
python scripts/validate_product_dataset.py \
  --input data/processed/products/phones_500.jsonl \
  --category phone \
  --min-count 500
```

The validator reports:

- `total_count`
- `valid_count`
- `skipped_count`
- `missing_field_stats`
- `image_url_coverage`
- `source_url_coverage`
- `description_coverage`
- `attributes_coverage`
- `duplicate_count`
- `warnings`
- `errors`

If `--min-count` is not reached, the script exits with a non-zero code.

## Import Normalized Products

```bash
python scripts/import_real_products.py \
  --input data/processed/products/phones_500.jsonl \
  --category phone \
  --upsert
```

Dry run:

```bash
python scripts/import_real_products.py \
  --input data/processed/products/phones_500.jsonl \
  --category phone \
  --upsert \
  --dry-run
```

Import behavior:

- Products are written to `products`.
- Tags are written to `product_tags`.
- Dynamic attributes are written to `product_attributes`.
- Extra source and quality fields are stored as attributes when they do not fit top-level model fields.
- Upsert uses `category + source_platform + source_product_id`, falling back to `category + brand + title`.
- Re-running with `--upsert` updates records instead of duplicating them.

After importing real products, rebuild Chroma:

```bash
cd backend
python ../scripts/rebuild_index.py
```

## Database Compatibility

The import pipeline uses ordinary SQLAlchemy inserts, updates, deletes, and selects. It is compatible with SQLite and MySQL 5.7. It does not depend on MySQL 8-only features.
