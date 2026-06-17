from pathlib import Path
import shutil
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.cache.cache_service import InMemoryCacheService
from app.core.db import Base
from app.retrieval.chroma_indexer import get_chroma_client, rebuild_all_indexes
from app.retrieval.retrieval_service import (
    Citation,
    KnowledgeRetrievalService,
    ProductCandidate,
    ProductRetrievalService,
    ProductSearchFilters,
    _rerank_citations,
    _rerank_product_candidates,
    _with_citation_rerank_score,
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


def test_product_rerank_prioritizes_preference_matches() -> None:
    generic = ProductCandidate(
        product_id="phone_generic",
        title="均衡手机",
        brand="Test",
        category_id="cat_phone",
        price=2500,
        stock=10,
        description="日常使用",
        image_url=None,
        tags=["性价比"],
        attributes={"处理器": "中端芯片"},
        source_url=None,
        compare_url=None,
        distance=0.1,
        score=0.9,
        product_text="均衡手机",
    )
    camera = ProductCandidate(
        product_id="phone_camera",
        title="影像手机",
        brand="Test",
        category_id="cat_phone",
        price=2600,
        stock=10,
        description="适合旅行拍摄",
        image_url=None,
        tags=["拍照好"],
        attributes={"拍照能力": "OIS 防抖 影像主摄"},
        source_url=None,
        compare_url=None,
        distance=0.9,
        score=0.1,
        product_text="影像手机",
    )

    ranked = _rerank_product_candidates(
        [generic, camera],
        query="推荐拍照好的手机",
        preferences=["拍照"],
    )

    assert ranked[0].product_id == "phone_camera"
    assert ranked[0].score > ranked[1].score


def test_citation_rerank_prioritizes_keyword_matches() -> None:
    irrelevant = Citation(
        chunk_id="chunk_battery",
        document_id="doc_battery",
        title="手机续航指南",
        section="电池容量",
        section_path="手机续航指南/电池容量",
        source_file="data/knowledge_docs/phone/phone_battery_guide.md",
        doc_type="guide",
        category_id="cat_phone",
        category_path="数码/手机",
        content_preview="电池容量和快充会影响重度使用体验。",
        distance=0.1,
        score=0.9,
    )
    relevant = Citation(
        chunk_id="chunk_camera",
        document_id="doc_camera",
        title="手机拍照选购指南",
        section="为什么不能只看像素",
        section_path="手机拍照选购指南/为什么不能只看像素",
        source_file="data/knowledge_docs/phone/phone_camera_guide.md",
        doc_type="guide",
        category_id="cat_phone",
        category_path="数码/手机",
        content_preview="拍照还要看影像、防抖和夜景表现。",
        distance=0.9,
        score=0.1,
    )

    reranked = _rerank_citations(
        [
            irrelevant,
            _with_citation_rerank_score(
                relevant,
                query_keywords=["像素", "防抖", "影像", "夜景"],
                extra_text="主摄和防抖会影响成片。",
            ),
        ]
    )

    assert reranked[0].chunk_id == "chunk_camera"


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
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=3000,
                preferences=["拍照"],
            ),
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


def test_product_retrieval_uses_cache_on_repeated_query() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_product_cache_test.db",
            "chroma_retrieval_product_cache_test",
        )
    )
    try:
        cache = InMemoryCacheService()
        service = ProductRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
            cache_service=cache,
        )

        first = service.search_products(
            query="鎷嶇収濂界殑鎵嬫満",
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=3000,
                preferences=["鎷嶇収"],
            ),
            top_k=3,
        )
        first_status = service.last_cache_status
        second = service.search_products(
            query="鎷嶇収濂界殑鎵嬫満",
            filters=ProductSearchFilters(
                category_id="cat_phone",
                budget_max=3000,
                preferences=["鎷嶇収"],
            ),
            top_k=3,
        )

        assert first
        assert first_status == "miss"
        assert service.last_cache_status == "hit"
        assert [item.product_id for item in second] == [
            item.product_id for item in first
        ]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_knowledge_retrieval_uses_cache_on_repeated_query() -> None:
    engine, db, db_path, chroma_dir, chroma_client, embedding_service = (
        _prepare_retrieval_stack(
            "smartbuy_retrieval_knowledge_cache_test.db",
            "chroma_retrieval_knowledge_cache_test",
        )
    )
    try:
        cache = InMemoryCacheService()
        service = KnowledgeRetrievalService(
            db,
            embedding_service=embedding_service,
            chroma_client=chroma_client,
            cache_service=cache,
        )

        first = service.search_knowledge(
            query="涓轰粈涔堟媿鐓т笉鑳藉彧鐪嬪儚绱?",
            category_id="cat_phone",
            top_k=3,
        )
        first_status = service.last_cache_status
        second = service.search_knowledge(
            query="涓轰粈涔堟媿鐓т笉鑳藉彧鐪嬪儚绱?",
            category_id="cat_phone",
            top_k=3,
        )

        assert first
        assert first_status == "miss"
        assert service.last_cache_status == "hit"
        assert [item.chunk_id for item in second] == [
            item.chunk_id for item in first
        ]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
