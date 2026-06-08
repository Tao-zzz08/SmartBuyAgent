from pathlib import Path
import shutil
import sys

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models import DocumentChunk
from app.retrieval.chroma_indexer import (
    KNOWLEDGE_COLLECTION,
    PRODUCT_COLLECTION,
    get_chroma_client,
    rebuild_all_indexes,
)
from app.services.embedding import MockEmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import import_seed_data  # noqa: E402
from import_docs import import_documents  # noqa: E402
from import_products import import_products  # noqa: E402


def _create_test_session(db_name: str):
    db_path = PROJECT_ROOT / "data" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestingSessionLocal(), db_path


def _count(db, model, *where_clauses) -> int:
    statement = select(func.count()).select_from(model)
    if where_clauses:
        statement = statement.where(*where_clauses)
    return db.scalar(statement) or 0


def test_rebuild_all_indexes_with_mock_embeddings() -> None:
    engine, db, db_path = _create_test_session("smartbuy_chroma_index_test.db")
    chroma_dir = PROJECT_ROOT / "data" / "chroma_test"
    shutil.rmtree(chroma_dir, ignore_errors=True)

    try:
        import_seed_data(db, PROJECT_ROOT)
        import_products(db, PROJECT_ROOT, dataset="mini")
        import_documents(db, PROJECT_ROOT)

        chunk_count = _count(db, DocumentChunk)
        embedding_service = MockEmbeddingService()
        client = get_chroma_client(chroma_dir)

        stats = rebuild_all_indexes(
            db,
            embedding_service=embedding_service,
            reset=True,
            client=client,
        )

        product_collection = client.get_collection(PRODUCT_COLLECTION)
        knowledge_collection = client.get_collection(KNOWLEDGE_COLLECTION)

        assert stats["product_text"]["indexed_products"] == 30
        assert product_collection.count() == 30
        assert knowledge_collection.count() == chunk_count
        assert (
            _count(db, DocumentChunk, DocumentChunk.vector_id.is_not(None))
            == chunk_count
        )

        query_result = product_collection.query(
            query_embeddings=[embedding_service.embed_text("拍照 手机")],
            n_results=3,
        )
        assert query_result["ids"]
        assert query_result["ids"][0]

        rebuild_all_indexes(
            db,
            embedding_service=embedding_service,
            reset=True,
            client=client,
        )

        product_collection = client.get_collection(PRODUCT_COLLECTION)
        knowledge_collection = client.get_collection(KNOWLEDGE_COLLECTION)
        assert product_collection.count() == 30
        assert knowledge_collection.count() == chunk_count
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
