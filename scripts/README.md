# Scripts

This directory contains project maintenance, seed import, indexing, and evaluation scripts.

- `init_db.py`: initializes database tables from SQLAlchemy models.
- `import_categories.py`: imports base categories, category attribute definitions, and category guide profiles.
- `import_products.py`: imports mini product seed data and splits `tags` plus `attributes_json` into product tag and product attribute tables.
- `normalize_real_products.py`: normalizes local CSV/JSON/JSONL product datasets into processed JSONL.
- `validate_product_dataset.py`: validates processed product JSONL and prints a quality report.
- `import_real_products.py`: imports processed real-product JSONL into products, product tags, and product attributes with optional upsert/dry-run.
- `import_docs.py`: imports Markdown knowledge documents and splits them into `document_chunks`.
- `rebuild_index.py`: rebuilds Chroma indexes for `product_text` and `knowledge_docs`.
- `eval_retrieval.py`: runs lightweight product and knowledge retrieval evaluation cases.
- `eval_multiturn.py`: runs lightweight multiturn rewrite and comparison evaluation cases.

Default local indexing uses mock embedding. OpenAI-compatible embedding can be enabled through `.env`, and indexes must be rebuilt after changing provider, model, or dimension.

These scripts do not create login, cart, order, payment, or purchase behavior.

The real-product pipeline is:

```text
data/raw/products/*.csv|*.json|*.jsonl
-> normalize_real_products.py
-> data/processed/products/*.jsonl
-> validate_product_dataset.py
-> import_real_products.py
-> rebuild_index.py
```

It stores third-party image references as `image_url` only and does not download image files.
