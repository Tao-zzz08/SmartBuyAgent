from collections.abc import Generator
from pathlib import Path
import shutil
import sys

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.chat import (
    get_chat_chroma_client,
    get_chat_embedding_service,
    get_chat_llm_answer_composer,
)
from app.chat.conversation_memory import ConversationMemoryService
from app.core.db import Base, get_db
from app.main import app
from app.models import ChatSession, ChatTurn
from app.retrieval.chroma_indexer import get_chroma_client, rebuild_all_indexes
from app.services.embedding import BaseEmbeddingService, MockEmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from import_categories import import_seed_data  # noqa: E402
from import_docs import import_documents  # noqa: E402
from import_products import import_products  # noqa: E402


class CountingEmbeddingService(BaseEmbeddingService):
    def __init__(self, embedding_dim: int = 32) -> None:
        self.embedding_dim = embedding_dim
        self.calls = 0
        self._delegate = MockEmbeddingService(embedding_dim=embedding_dim)

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        return self._delegate.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return self._delegate.embed_texts(texts)


class FakeLLMAnswerComposer:
    provider = "fake"

    def __init__(self, answer: str = "API fake LLM answer") -> None:
        self.answer = answer
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


def _prepare_api_stack(db_name: str, chroma_dir_name: str):
    engine, db, db_path = _create_test_session(db_name)
    chroma_dir = PROJECT_ROOT / "data" / chroma_dir_name
    shutil.rmtree(chroma_dir, ignore_errors=True)

    import_seed_data(db, PROJECT_ROOT)
    import_products(db, PROJECT_ROOT, dataset="mini")
    import_documents(db, PROJECT_ROOT)

    chroma_client = get_chroma_client(chroma_dir)
    rebuild_all_indexes(
        db,
        embedding_service=MockEmbeddingService(),
        reset=True,
        client=chroma_client,
    )
    db.close()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestingSessionLocal, db_path, chroma_dir, chroma_client


def _override_dependencies(
    TestingSessionLocal,
    chroma_client,
    embedding_service: BaseEmbeddingService | None = None,
    llm_answer_composer=None,
) -> None:
    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chat_chroma_client] = lambda: chroma_client
    app.dependency_overrides[get_chat_embedding_service] = (
        lambda: embedding_service or MockEmbeddingService()
    )
    app.dependency_overrides[get_chat_llm_answer_composer] = (
        lambda: llm_answer_composer or FakeLLMAnswerComposer()
    )


def _cleanup(engine, db_path: Path, chroma_dir: Path) -> None:
    app.dependency_overrides.clear()
    engine.dispose()
    db_path.unlink(missing_ok=True)
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _steps(payload: dict) -> set[str]:
    return {step["step"] for step in payload["trace"]}


def test_chat_api_shopping_guide() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_shopping_test.db",
        "chroma_chat_api_shopping_test",
    )
    embedding_service = CountingEmbeddingService()
    llm_answer_composer = FakeLLMAnswerComposer(answer="API fake LLM answer")
    _override_dependencies(
        TestingSessionLocal,
        chroma_client,
        embedding_service,
        llm_answer_composer,
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        payload = response.json()
        steps = _steps(payload)

        assert response.status_code == 200
        assert payload["answer"] == "API fake LLM answer"
        assert 0 < len(payload["product_cards"]) <= 3
        assert len(payload["citations"]) > 0
        assert "query_understanding" in steps
        assert "product_retrieval" in steps
        assert "knowledge_retrieval" in steps
        assert "llm_answer" in steps
        assert "response_composer" in steps
        assert embedding_service.calls > 0
        assert len(llm_answer_composer.calls) == 1
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_debug_false_hides_trace() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_debug_test.db",
        "chroma_chat_api_debug_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": False},
        )

        assert response.status_code == 200
        assert response.json()["trace"] == []
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_clarification() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_clarification_test.db",
        "chroma_chat_api_clarification_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post("/api/chat", json={"query": "推荐一下"})
        payload = response.json()

        assert response.status_code == 200
        assert "你想看哪个品类" in payload["answer"]
        assert payload["product_cards"] == []
        assert payload["citations"] == []
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_generates_session_id_when_missing() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_session_generate_test.db",
        "chroma_chat_api_session_generate_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        payload = response.json()
        session_id = payload["session_id"]

        assert response.status_code == 200
        assert session_id

        db = TestingSessionLocal()
        try:
            assert db.get(ChatSession, session_id) is not None
            turns = db.scalars(
                select(ChatTurn).where(ChatTurn.session_id == session_id)
            ).all()
            assert len(turns) == 1
            assert turns[0].turn_index == 1
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_reuses_existing_session_id() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_session_reuse_test.db",
        "chroma_chat_api_session_reuse_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat",
            json={"query": "预算3000，推荐一款拍照好的手机"},
        )
        session_id = first_response.json()["session_id"]
        second_response = client.post(
            "/api/chat",
            json={
                "query": "为什么手机拍照不能只看像素",
                "session_id": session_id,
            },
        )

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert second_response.json()["session_id"] == session_id

        db = TestingSessionLocal()
        try:
            turns = db.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == session_id)
                .order_by(ChatTurn.turn_index)
            ).all()
            assert [turn.turn_index for turn in turns] == [1, 2]
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_memory_failure_does_not_break_chat(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_memory_failure_test.db",
        "chroma_chat_api_memory_failure_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)

    def fail_save_turn(self, session_id, user_query, chat_response):
        raise RuntimeError("memory is temporarily unavailable")

    monkeypatch.setattr(ConversationMemoryService, "save_turn", fail_save_turn)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"query": "预算3000，推荐一款拍照好的手机", "debug": True},
        )
        payload = response.json()
        memory_trace = [
            step
            for step in payload["trace"]
            if step.get("step") == "conversation_memory"
        ]

        assert response.status_code == 200
        assert payload["answer"]
        assert payload["product_cards"]
        assert payload["citations"]
        assert memory_trace[0]["status"] == "failed"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_rejects_blank_query() -> None:
    client = TestClient(app)

    response = client.post("/api/chat", json={"query": "   "})

    assert response.status_code == 422


def test_health_still_works() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
