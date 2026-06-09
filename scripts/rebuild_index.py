from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import Base, SessionLocal, engine  # noqa: E402
from app.models import DocumentChunk, Product  # noqa: E402
from app.retrieval.chroma_indexer import rebuild_all_indexes  # noqa: E402
from app.services.embedding import get_embedding_service  # noqa: E402
import app.models  # noqa: E402,F401


def _resolve_chroma_dir() -> Path:
    chroma_dir = Path(settings.CHROMA_DIR)
    if chroma_dir.is_absolute():
        return chroma_dir
    return PROJECT_ROOT / chroma_dir


def main() -> None:
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        product_count = db.scalar(select(func.count()).select_from(Product)) or 0
        if product_count == 0:
            raise SystemExit(
                "No products found. Please run python ../scripts/import_products.py --dataset mini first."
            )

        chunk_count = db.scalar(select(func.count()).select_from(DocumentChunk)) or 0
        if chunk_count == 0:
            raise SystemExit(
                "No document chunks found. Please run python ../scripts/import_docs.py first."
            )

        try:
            embedding_service = get_embedding_service()
        except ValueError as exc:
            raise SystemExit(f"Embedding provider configuration error: {exc}") from exc

        stats = rebuild_all_indexes(
            db,
            embedding_service=embedding_service,
            reset=True,
        )
    finally:
        db.close()

    print(f"embedding_provider: {settings.EMBEDDING_PROVIDER}")
    print(f"embedding_dim: {settings.EMBEDDING_DIM}")
    print(f"embedding_model: {settings.EMBEDDING_MODEL or '-'}")
    print(f"indexed_products: {stats['product_text']['indexed_products']}")
    print(f"indexed_chunks: {stats['knowledge_docs']['indexed_chunks']}")
    print(f"chroma_dir: {_resolve_chroma_dir().as_posix()}")
    print(f"collections: {', '.join(stats['collections'])}")


if __name__ == "__main__":
    main()
