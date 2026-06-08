from pathlib import Path
import shutil
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.retrieval.chroma_indexer import get_chroma_client, rebuild_all_indexes
from app.retrieval.retrieval_service import (
    KnowledgeRetrievalService,
    ProductRetrievalService,
    ProductSearchFilters,
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


def _prepare_retrieval_stack(db_name: str, chroma_dir_name: str):
    engine, db, db_path = _create_test_session(db_name)
    chroma_dir = PROJECT_ROOT / "data" / chroma_dir_name
    shutil.rmtree(chroma_dir, ignore_errors=True)

    import_seed_data(db, PROJECT_ROOT)
    import_products(db, PROJECT_ROOT, dataset="mini")
    import_documents(db, PROJECT_ROOT)

    embedding_service = MockEmbeddingService()
    chroma_client = get_chroma_client(chroma_dir)
    rebuild_all_indexes(
        db,
        embedding_service=embedding_service,
        reset=True,
        client=chroma_client,
    )
    return engine, db, db_path, chroma_dir, chroma_client, embedding_service


def test_product_retrieval_filters_and_returns_candidates() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_products_test.db",
            "chroma_retrieval_products_test",
        )
    )
    try:
        service = ProductRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        results = service.search_products(
            query="拍照好的手机",
            filters=ProductSearchFilters(category_id="cat_phone", budget_max=3000),
            top_k=3,
        )

        assert results
        assert len(results) <= 3
        assert all(candidate.category_id == "cat_phone" for candidate in results)
        assert all(candidate.price <= 3000 for candidate in results)
        assert all(candidate.stock > 0 for candidate in results)
        assert all(candidate.tags for candidate in results)
        assert all(candidate.attributes for candidate in results)
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_product_retrieval_returns_empty_for_unknown_category() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_empty_test.db",
            "chroma_retrieval_empty_test",
        )
    )
    try:
        service = ProductRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        results = service.search_products(
            query="不存在的品类",
            filters={"category_id": "cat_unknown"},
            top_k=3,
        )

        assert results == []
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_knowledge_retrieval_returns_citations() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_knowledge_test.db",
            "chroma_retrieval_knowledge_test",
        )
    )
    try:
        service = KnowledgeRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        citations = service.search_knowledge(
            query="为什么拍照不能只看像素",
            category_id="cat_phone",
            top_k=3,
        )

        assert citations
        assert len(citations) <= 3
        assert all(citation.category_id == "cat_phone" for citation in citations)
        assert all(citation.title for citation in citations)
        assert all(citation.section for citation in citations)
        assert all(citation.source_file for citation in citations)
        assert all(citation.content_preview for citation in citations)
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_knowledge_retrieval_filters_doc_type() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_doc_type_test.db",
            "chroma_retrieval_doc_type_test",
        )
    )
    try:
        service = KnowledgeRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
        )

        citations = service.search_knowledge(
            query="七天无理由",
            doc_type="after_sales",
            top_k=3,
        )

        assert citations
        assert all(citation.doc_type == "after_sales" for citation in citations)
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
