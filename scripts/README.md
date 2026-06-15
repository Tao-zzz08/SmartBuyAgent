# Scripts

This directory contains project maintenance, seed import, indexing, and evaluation scripts.

- `init_db.py`: initializes database tables from SQLAlchemy models.
- `import_categories.py`: imports base categories, category attribute definitions, and category guide profiles.
- `import_products.py`: imports mini product seed data and splits `tags` plus `attributes_json` into product tag and product attribute tables.
- `import_docs.py`: imports Markdown knowledge documents and splits them into `document_chunks`.
- `rebuild_index.py`: rebuilds Chroma indexes for `product_text` and `knowledge_docs`.
- `eval_retrieval.py`: runs lightweight product and knowledge retrieval evaluation cases.
- `eval_multiturn.py`: runs lightweight multiturn rewrite and comparison evaluation cases.

Default local indexing uses mock embedding. OpenAI-compatible embedding can be enabled through `.env`, and indexes must be rebuilt after changing provider, model, or dimension.

These scripts do not create login, cart, order, payment, or purchase behavior.
