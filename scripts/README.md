# Scripts

This directory contains project maintenance and seed import scripts.

- `init_db.py`: initializes database tables from SQLAlchemy models.
- `import_categories.py`: imports base categories, category attribute definitions, and category guide profiles.
- `import_products.py`: imports mini product seed data and splits `tags` plus `attributes_json` into product tag and product attribute tables.

Not implemented yet:

- document import
- index rebuild
