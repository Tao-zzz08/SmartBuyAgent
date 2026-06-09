from pathlib import Path
import shutil
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.chat.chat_service import ChatService
from app.chat.llm_answer_composer import SAFE_LLM_FALLBACK_ANSWER
from app.chat.query_understanding import QueryUnderstandingResult
from app.core.db import Base
from app.retrieval.chroma_indexer import get_chroma_client, rebuild_all_indexes
from app.retrieval.retrieval_service import Citation, ProductCandidate
from app.services.embedding import MockEmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import import_seed_data  # noqa: E402
from import_docs import import_documents  # noqa: E402
from import_products import import_products  # noqa: E402


class StaticQueryUnderstandingService:
    def __init__(self, result: QueryUnderstandingResult) -> None:
        self.result = result

    def understand(self, query: str) -> QueryUnderstandingResult:
        return self.result


class RecordingProductRetrievalService:
    def __init__(self) -> None:
        self.last_filters = None

    def search_products(self, query, filters, top_k):
        self.last_filters = filters
        return []


class EmptyKnowledgeRetrievalService:
    def search_knowledge(self, query, category_id=None, top_k=3):
        return []


class StaticProductRetrievalService:
    def __init__(self, candidates: list[ProductCandidate]) -> None:
        self.candidates = candidates

    def search_products(self, query, filters, top_k):
        return self.candidates[:top_k]


class StaticKnowledgeRetrievalService:
    def __init__(self, citations: list[Citation]) -> None:
        self.citations = citations

    def search_knowledge(self, query, category_id=None, top_k=3):
        return self.citations[:top_k]


class FakeLLMAnswerComposer:
    provider = "fake"

    def __init__(
        self,
        answer: str = "LLM generated shopping answer",
        should_raise: bool = False,
    ) -> None:
        self.answer = answer
        self.should_raise = should_raise
        self.calls = []

    def compose(
        self,
        query,
        query_result,
        product_candidates=None,
        citations=None,
    ) -> str:
        self.calls.append(
            {
                "query": query,
                "query_result": query_result,
                "product_candidates": product_candidates or [],
                "citations": citations or [],
            }
        )
        if self.should_raise:
            raise RuntimeError("fake llm error")
        return self.answer


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


def _prepare_chat_service(db_name: str, chroma_dir_name: str):
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

    chat_service = ChatService(
        db=db,
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )
    return engine, db, db_path, chroma_dir, chat_service


def _steps(response) -> set[str]:
    return {step["step"] for step in response.trace}


def _step(response, step_name: str) -> dict:
    return next(step for step in response.trace if step["step"] == step_name)


def _shopping_query_result() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        raw_query="budget 3000 camera phone",
        intent="shopping_guide",
        category_id="cat_phone",
        category_path="Digital/Phone",
        budget_min=None,
        budget_max=3000,
        preferences=["camera"],
        need_clarification=False,
        clarification_question=None,
    )


def _knowledge_query_result() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        raw_query="why phone camera is not only pixels",
        intent="product_knowledge",
        category_id="cat_phone",
        category_path="Digital/Phone",
        budget_min=None,
        budget_max=None,
        preferences=["camera"],
        need_clarification=False,
        clarification_question=None,
    )


def _sample_product_candidates() -> list[ProductCandidate]:
    return [
        ProductCandidate(
            product_id="phone_001",
            title="Camera Phone A",
            brand="DemoBrand",
            category_id="cat_phone",
            price=2599,
            stock=10,
            description="A new phone for camera-focused users.",
            image_url="https://example.com/phone_001.jpg",
            tags=["camera", "value"],
            attributes={"storage": "256GB", "battery": "5000mAh"},
            source_url="https://example.com/products/phone_001",
            compare_url="https://example.com/compare/phone_001",
            distance=0.1,
            score=0.9,
            product_text="Camera Phone A with 256GB storage.",
        )
    ]


def _sample_citations() -> list[Citation]:
    return [
        Citation(
            chunk_id="chunk_phone_camera",
            document_id="doc_phone_camera",
            title="Phone Camera Guide",
            section="Camera basics",
            section_path="Phone Camera Guide > Camera basics",
            source_file="data/knowledge_docs/phone/phone_camera_guide.md",
            doc_type="guide",
            category_id="cat_phone",
            category_path="Digital/Phone",
            content_preview="Do not judge phone cameras only by pixels.",
            distance=0.2,
            score=0.8,
        )
    ]


def _make_chat_service(
    query_result: QueryUnderstandingResult,
    product_candidates: list[ProductCandidate] | None = None,
    citations: list[Citation] | None = None,
    llm_answer_composer=None,
) -> ChatService:
    chat_service = ChatService(
        db=None,
        embedding_service=MockEmbeddingService(),
        chroma_client=object(),
        query_understanding_service=StaticQueryUnderstandingService(query_result),
        llm_answer_composer=llm_answer_composer,
    )
    chat_service.product_retrieval_service = StaticProductRetrievalService(
        product_candidates or []
    )
    chat_service.knowledge_retrieval_service = StaticKnowledgeRetrievalService(
        citations or []
    )
    return chat_service


def test_chat_service_passes_preferences_to_product_retrieval() -> None:
    query_result = QueryUnderstandingResult(
        raw_query="预算3000，推荐一款拍照好的手机",
        intent="shopping_guide",
        category_id="cat_phone",
        category_path="数码/手机",
        budget_min=None,
        budget_max=3000,
        preferences=["拍照"],
        need_clarification=False,
        clarification_question=None,
    )
    product_service = RecordingProductRetrievalService()
    chat_service = ChatService(
        db=None,
        embedding_service=MockEmbeddingService(),
        chroma_client=object(),
        query_understanding_service=StaticQueryUnderstandingService(query_result),
    )
    chat_service.product_retrieval_service = product_service
    chat_service.knowledge_retrieval_service = EmptyKnowledgeRetrievalService()

    response = chat_service.handle_message(query_result.raw_query)

    assert response.answer
    assert product_service.last_filters is not None
    assert product_service.last_filters.preferences == ["拍照"]
    assert product_service.last_filters.category_id == "cat_phone"
    assert product_service.last_filters.budget_max == 3000


def test_chat_service_uses_llm_answer_for_shopping_guide() -> None:
    fake_llm = FakeLLMAnswerComposer(answer="LLM generated shopping answer")
    products = _sample_product_candidates()
    citations = _sample_citations()
    chat_service = _make_chat_service(
        _shopping_query_result(),
        product_candidates=products,
        citations=citations,
        llm_answer_composer=fake_llm,
    )

    response = chat_service.handle_message("budget 3000 camera phone")

    assert response.answer == "LLM generated shopping answer"
    assert len(response.product_cards) == 1
    assert response.product_cards[0].product_id == "phone_001"
    assert len(response.citations) == 1
    assert _step(response, "llm_answer")["status"] == "success"
    assert fake_llm.calls[0]["product_candidates"] == products
    assert fake_llm.calls[0]["citations"] == citations


def test_chat_service_falls_back_to_template_when_llm_raises() -> None:
    fake_llm = FakeLLMAnswerComposer(should_raise=True)
    chat_service = _make_chat_service(
        _shopping_query_result(),
        product_candidates=_sample_product_candidates(),
        citations=_sample_citations(),
        llm_answer_composer=fake_llm,
    )

    response = chat_service.handle_message("budget 3000 camera phone")

    assert response.answer
    assert response.answer != fake_llm.answer
    assert len(response.product_cards) == 1
    assert len(response.citations) == 1
    assert _step(response, "llm_answer")["status"] == "fallback"


def test_chat_service_falls_back_to_template_when_llm_composer_returns_safe_fallback() -> None:
    fake_llm = FakeLLMAnswerComposer(answer=SAFE_LLM_FALLBACK_ANSWER)
    chat_service = _make_chat_service(
        _shopping_query_result(),
        product_candidates=_sample_product_candidates(),
        citations=_sample_citations(),
        llm_answer_composer=fake_llm,
    )

    response = chat_service.handle_message("budget 3000 camera phone")

    assert response.answer
    assert response.answer != SAFE_LLM_FALLBACK_ANSWER
    assert len(response.product_cards) == 1
    assert len(response.citations) == 1
    assert _step(response, "llm_answer")["status"] == "fallback"


def test_chat_service_without_llm_keeps_template_answer() -> None:
    chat_service = _make_chat_service(
        _shopping_query_result(),
        product_candidates=_sample_product_candidates(),
        citations=_sample_citations(),
    )

    response = chat_service.handle_message("budget 3000 camera phone")

    assert response.answer
    assert response.answer != "LLM generated shopping answer"
    assert len(response.product_cards) == 1
    assert len(response.citations) == 1
    assert _step(response, "llm_answer")["status"] == "disabled"


def test_chat_service_uses_llm_for_product_knowledge() -> None:
    fake_llm = FakeLLMAnswerComposer(answer="LLM generated knowledge answer")
    citations = _sample_citations()
    chat_service = _make_chat_service(
        _knowledge_query_result(),
        citations=citations,
        llm_answer_composer=fake_llm,
    )

    response = chat_service.handle_message("why phone camera is not only pixels")

    assert response.answer == "LLM generated knowledge answer"
    assert response.product_cards == []
    assert len(response.citations) == 1
    assert _step(response, "llm_answer")["status"] == "success"
    assert fake_llm.calls[0]["product_candidates"] == []
    assert fake_llm.calls[0]["citations"] == citations


def test_chat_service_shopping_guide_chain() -> None:
    engine, db, db_path, chroma_dir, chat_service = _prepare_chat_service(
        "smartbuy_chat_shopping_test.db",
        "chroma_chat_shopping_test",
    )
    try:
        response = chat_service.handle_message("预算3000，推荐一款拍照好的手机")
        steps = _steps(response)

        assert response.answer
        assert 0 < len(response.product_cards) <= 3
        assert len(response.citations) > 0
        assert "query_understanding" in steps
        assert "product_retrieval" in steps
        assert "knowledge_retrieval" in steps
        assert "response_composer" in steps
        assert all(card.price <= 3000 for card in response.product_cards)
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_chat_service_clarification_does_not_trigger_retrieval() -> None:
    engine, db, db_path, chroma_dir, chat_service = _prepare_chat_service(
        "smartbuy_chat_clarification_test.db",
        "chroma_chat_clarification_test",
    )
    try:
        response = chat_service.handle_message("推荐一下")
        steps = _steps(response)

        assert "你想看哪个品类" in response.answer
        assert response.product_cards == []
        assert response.citations == []
        assert "query_understanding" in steps
        assert "response_composer" in steps
        assert "product_retrieval" not in steps
        assert "knowledge_retrieval" not in steps
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_chat_service_product_knowledge_only_triggers_knowledge() -> None:
    engine, db, db_path, chroma_dir, chat_service = _prepare_chat_service(
        "smartbuy_chat_knowledge_test.db",
        "chroma_chat_knowledge_test",
    )
    try:
        response = chat_service.handle_message("为什么手机拍照不能只看像素")
        steps = _steps(response)

        assert response.answer
        assert response.product_cards == []
        assert len(response.citations) > 0
        assert "query_understanding" in steps
        assert "knowledge_retrieval" in steps
        assert "response_composer" in steps
        assert "product_retrieval" not in steps
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_chat_service_chitchat_does_not_trigger_retrieval() -> None:
    engine, db, db_path, chroma_dir, chat_service = _prepare_chat_service(
        "smartbuy_chat_chitchat_test.db",
        "chroma_chat_chitchat_test",
    )
    try:
        response = chat_service.handle_message("你好")
        steps = _steps(response)

        assert response.answer
        assert response.product_cards == []
        assert response.citations == []
        assert "query_understanding" in steps
        assert "response_composer" in steps
        assert "product_retrieval" not in steps
        assert "knowledge_retrieval" not in steps
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)


def test_chat_service_compare_not_implemented() -> None:
    engine, db, db_path, chroma_dir, chat_service = _prepare_chat_service(
        "smartbuy_chat_compare_test.db",
        "chroma_chat_compare_test",
    )
    try:
        response = chat_service.handle_message("phone_001 和 phone_002 哪个更值得买")
        steps = _steps(response)

        assert "当前阶段还没有实现对比服务" in response.answer
        assert response.product_cards == []
        assert "product_retrieval" not in steps
        assert "knowledge_retrieval" not in steps
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
        shutil.rmtree(chroma_dir, ignore_errors=True)
