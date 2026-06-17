from collections.abc import Generator
import json
from pathlib import Path
import shutil
import sys

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent.stream_runner import AgentStreamRunner
from app.api.chat import (
    get_chat_chroma_client,
    get_chat_cache_service,
    get_chat_embedding_service,
    get_chat_llm_answer_composer,
)
from app.cache.cache_service import InMemoryCacheService
from app.chat.chat_service import ChatService
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
    cache_service=None,
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
    if cache_service is not None:
        app.dependency_overrides[get_chat_cache_service] = lambda: cache_service


def _cleanup(engine, db_path: Path, chroma_dir: Path) -> None:
    app.dependency_overrides.clear()
    engine.dispose()
    db_path.unlink(missing_ok=True)
    shutil.rmtree(chroma_dir, ignore_errors=True)


def _steps(payload: dict) -> set[str]:
    return {step["step"] for step in payload["trace"]}


def _trace_step(payload: dict, step_name: str) -> dict:
    return next(step for step in payload["trace"] if step.get("step") == step_name)


def _sse_events(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue

        event_name = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))

        if event_name is not None and data is not None:
            events.append({"event": event_name, "data": data})
    return events


def _sse_event_data(events: list[dict], event_name: str) -> dict:
    return next(event["data"] for event in events if event["event"] == event_name)


def _sse_event_datas(events: list[dict], event_name: str) -> list[dict]:
    return [event["data"] for event in events if event["event"] == event_name]


def _seed_memory_turn(
    TestingSessionLocal,
    session_id: str,
    product_ids: list[str] | None = None,
) -> None:
    db = TestingSessionLocal()
    try:
        db.add(ChatSession(session_id=session_id))
        db.add(
            ChatTurn(
                session_id=session_id,
                turn_index=1,
                user_query="预算3000，推荐一款拍照好的手机",
                assistant_answer="answer",
                intent="shopping_guide",
                category_id="cat_phone",
                category_path="数码/手机",
                budget_min=None,
                budget_max=3000,
                preferences_json=json.dumps(["拍照", "续航"], ensure_ascii=False),
                product_ids_json=json.dumps(
                    product_ids or ["phone_001", "phone_002", "phone_003"],
                    ensure_ascii=False,
                ),
                citation_chunk_ids_json="[]",
            )
        )
        db.commit()
    finally:
        db.close()


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
        assert "agent_node" in steps
        assert "product_retrieval" in steps
        assert "knowledge_retrieval" in steps
        assert "llm_answer" in steps
        assert "response_composer" in steps
        assert "follow_up_rewrite" in steps
        assert _trace_step(payload, "follow_up_rewrite")["status"] == "skipped"
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


def test_chat_api_with_session_but_no_history_does_not_rewrite() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_no_history_rewrite_test.db",
        "chroma_chat_api_no_history_rewrite_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "预算提高到4000呢",
                "session_id": "empty_session",
                "debug": True,
            },
        )
        payload = response.json()
        rewrite_trace = _trace_step(payload, "follow_up_rewrite")

        assert response.status_code == 200
        assert rewrite_trace["status"] == "not_follow_up"
        assert rewrite_trace["reason"] == "no_recent_turns"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_rewrites_budget_follow_up_and_saves_original_query() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_budget_rewrite_test.db",
        "chroma_chat_api_budget_rewrite_test",
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
                "query": "预算提高到4000呢",
                "session_id": session_id,
                "debug": True,
            },
        )
        payload = second_response.json()
        rewrite_trace = _trace_step(payload, "follow_up_rewrite")

        assert second_response.status_code == 200
        assert rewrite_trace["status"] == "rewritten"
        assert rewrite_trace["reason"] == "budget_update"
        assert "4000" in rewrite_trace["rewritten_query"]
        assert "手机" in rewrite_trace["rewritten_query"]
        assert "拍照" in rewrite_trace["rewritten_query"]

        db = TestingSessionLocal()
        try:
            turns = db.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == session_id)
                .order_by(ChatTurn.turn_index)
            ).all()
            assert turns[-1].user_query == "预算提高到4000呢"
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_rewrites_vague_product_follow_up() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_vague_rewrite_test.db",
        "chroma_chat_api_vague_rewrite_test",
    )
    session_id = "session_vague_rewrite"
    _seed_memory_turn(
        TestingSessionLocal,
        session_id=session_id,
        product_ids=["phone_001", "phone_002"],
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "这几款哪个更适合拍照",
                "session_id": session_id,
                "debug": True,
            },
        )
        payload = response.json()
        rewrite_trace = _trace_step(payload, "follow_up_rewrite")
        comparison_trace = _trace_step(payload, "product_comparison")
        returned_product_ids = {
            product["product_id"] for product in payload["product_cards"]
        }

        assert response.status_code == 200
        assert rewrite_trace["status"] == "rewritten"
        assert rewrite_trace["reason"] == "vague_product_reference"
        assert rewrite_trace["referenced_product_ids"] == ["phone_001", "phone_002"]
        assert comparison_trace["status"] == "compared"
        assert comparison_trace["source"] == "referenced_product_ids"
        assert returned_product_ids <= {"phone_001", "phone_002"}
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_rewrites_ordinal_product_follow_up() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_ordinal_rewrite_test.db",
        "chroma_chat_api_ordinal_rewrite_test",
    )
    session_id = "session_ordinal_rewrite"
    _seed_memory_turn(
        TestingSessionLocal,
        session_id=session_id,
        product_ids=["phone_001", "phone_002", "phone_003"],
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "第一个和第二个有什么区别",
                "session_id": session_id,
                "debug": True,
            },
        )
        payload = response.json()
        rewrite_trace = _trace_step(payload, "follow_up_rewrite")
        comparison_trace = _trace_step(payload, "product_comparison")
        returned_product_ids = [
            product["product_id"] for product in payload["product_cards"]
        ]

        assert response.status_code == 200
        assert rewrite_trace["status"] == "rewritten"
        assert rewrite_trace["reason"] == "ordinal_reference"
        assert rewrite_trace["resolved_product_ids"] == ["phone_001", "phone_002"]
        assert comparison_trace["status"] == "compared"
        assert comparison_trace["source"] == "resolved_product_ids"
        assert returned_product_ids == ["phone_001", "phone_002"]

        db = TestingSessionLocal()
        try:
            turns = db.scalars(
                select(ChatTurn)
                .where(ChatTurn.session_id == session_id)
                .order_by(ChatTurn.turn_index)
            ).all()
            assert turns[-1].user_query == "第一个和第二个有什么区别"
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_comparison_handles_missing_product_id() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_compare_missing_test.db",
        "chroma_chat_api_compare_missing_test",
    )
    session_id = "session_compare_missing"
    _seed_memory_turn(
        TestingSessionLocal,
        session_id=session_id,
        product_ids=["phone_001", "phone_missing"],
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "这几款哪个更适合拍照",
                "session_id": session_id,
                "debug": True,
            },
        )
        payload = response.json()
        comparison_trace = _trace_step(payload, "product_comparison")

        assert response.status_code == 200
        assert comparison_trace["missing_product_ids"] == ["phone_missing"]
        assert comparison_trace["returned_product_ids"] == ["phone_001"]
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


def test_chat_api_follow_up_memory_read_failure_does_not_break_chat(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_rewrite_failure_test.db",
        "chroma_chat_api_rewrite_failure_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)

    def fail_get_recent_turns(self, session_id, limit=5):
        raise RuntimeError("memory read failed")

    monkeypatch.setattr(
        ConversationMemoryService,
        "get_recent_turns",
        fail_get_recent_turns,
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={
                "query": "预算提高到4000呢",
                "session_id": "session_read_failure",
                "debug": True,
            },
        )
        payload = response.json()
        rewrite_trace = _trace_step(payload, "follow_up_rewrite")

        assert response.status_code == 200
        assert payload["answer"]
        assert rewrite_trace["status"] == "failed"
        assert rewrite_trace["rewritten_query"] == "预算提高到4000呢"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_shopping_guide() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_shopping_test.db",
        "chroma_chat_stream_shopping_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a",
                "debug": True,
            },
        )
        events = _sse_events(response.text)
        event_names = [event["event"] for event in events]
        result = _sse_event_data(events, "result")

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "session" in event_names
        assert "node_start" in event_names
        assert "node_end" in event_names
        assert "retrieval" in event_names
        assert "token" in event_names
        assert "trace" in event_names
        assert "result" in event_names
        assert "done" in event_names
        assert result["answer"]
        assert result["product_cards"]
        assert result["citations"]
        assert result["session_id"]
        assert any(
            event["data"].get("node") == "shopping_guide"
            for event in events
            if event["event"] == "node_start"
        )
        assert any(
            isinstance(event["data"].get("duration_ms"), int)
            and event["data"]["duration_ms"] >= 0
            for event in events
            if event["event"] == "node_end"
        )
        token_text = "".join(
            event["data"].get("delta", "")
            for event in events
            if event["event"] == "token"
        )
        assert token_text
        assert token_text == result["answer"]
        assert _sse_event_data(events, "done")["status"] == "ok"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_api_rate_limit_returns_429(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_api_rate_limit_test.db",
        "chroma_chat_api_rate_limit_test",
    )
    cache = InMemoryCacheService()
    _override_dependencies(TestingSessionLocal, chroma_client, cache_service=cache)
    monkeypatch.setattr("app.api.chat.settings.RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr("app.api.chat.settings.RATE_LIMIT_WINDOW_SECONDS", 10)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat",
            json={"query": "你好", "session_id": "session_rate_limit"},
        )
        second_response = client.post(
            "/api/chat",
            json={"query": "你好", "session_id": "session_rate_limit"},
        )

        assert first_response.status_code == 200
        assert second_response.status_code == 429
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_rate_limit_returns_429(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_rate_limit_test.db",
        "chroma_chat_stream_rate_limit_test",
    )
    cache = InMemoryCacheService()
    _override_dependencies(TestingSessionLocal, chroma_client, cache_service=cache)
    monkeypatch.setattr("app.api.chat.settings.RATE_LIMIT_MAX_REQUESTS", 1)
    monkeypatch.setattr("app.api.chat.settings.RATE_LIMIT_WINDOW_SECONDS", 10)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat/stream",
            json={"query": "你好", "session_id": "session_stream_rate_limit"},
        )
        second_response = client.post(
            "/api/chat/stream",
            json={"query": "你好", "session_id": "session_stream_rate_limit"},
        )

        assert first_response.status_code == 200
        assert second_response.status_code == 429
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_writes_trace_events_to_cache() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_trace_cache_test.db",
        "chroma_chat_stream_trace_cache_test",
    )
    cache = InMemoryCacheService()
    _override_dependencies(TestingSessionLocal, chroma_client, cache_service=cache)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a",
                "debug": True,
            },
        )
        events = _sse_events(response.text)
        session_event = _sse_event_data(events, "session")
        cache_key = (
            f"smartbuy:sse:{session_event['session_id']}:"
            f"{session_event['request_id']}:trace"
        )
        cached_events = cache.get_json(cache_key)

        assert response.status_code == 200
        assert isinstance(cached_events, list)
        assert any(event["event"] == "trace" for event in cached_events)
        assert any(event["event"] == "result" for event in cached_events)
        assert cached_events[-1]["event"] == "done"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_generates_session_id() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_session_generate_test.db",
        "chroma_chat_stream_session_generate_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"
            },
        )
        events = _sse_events(response.text)
        session_id = _sse_event_data(events, "session")["session_id"]
        result = _sse_event_data(events, "result")

        assert response.status_code == 200
        assert session_id
        assert result["session_id"] == session_id
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_reuses_session_id() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_session_reuse_test.db",
        "chroma_chat_stream_session_reuse_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"
            },
        )
        session_id = first_response.json()["session_id"]
        stream_response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u4e3a\u4ec0\u4e48\u624b\u673a\u62cd\u7167\u4e0d\u80fd\u53ea\u770b\u50cf\u7d20",
                "session_id": session_id,
            },
        )
        events = _sse_events(stream_response.text)

        assert first_response.status_code == 200
        assert stream_response.status_code == 200
        assert _sse_event_data(events, "session")["session_id"] == session_id
        assert _sse_event_data(events, "result")["session_id"] == session_id
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_follow_up_budget_rewrite() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_budget_rewrite_test.db",
        "chroma_chat_stream_budget_rewrite_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"
            },
        )
        session_id = first_response.json()["session_id"]
        stream_response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b97\u63d0\u9ad8\u52304000\u5462",
                "session_id": session_id,
                "debug": True,
            },
        )
        events = _sse_events(stream_response.text)
        result = _sse_event_data(events, "result")
        rewrite_trace = _trace_step(result, "follow_up_rewrite")

        assert stream_response.status_code == 200
        assert "follow_up_rewrite" in stream_response.text
        assert rewrite_trace["status"] == "rewritten"
        assert "4000" in rewrite_trace["rewritten_query"]
        assert "\u624b\u673a" in rewrite_trace["rewritten_query"]
        assert "\u62cd\u7167" in rewrite_trace["rewritten_query"]
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_follow_up_compare() -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_compare_test.db",
        "chroma_chat_stream_compare_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)
    try:
        client = TestClient(app)
        first_response = client.post(
            "/api/chat",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"
            },
        )
        first_payload = first_response.json()
        session_id = first_payload["session_id"]
        first_two_ids = [
            product["product_id"] for product in first_payload["product_cards"][:2]
        ]

        stream_response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u7b2c\u4e00\u4e2a\u548c\u7b2c\u4e8c\u4e2a\u6709\u4ec0\u4e48\u533a\u522b",
                "session_id": session_id,
                "debug": True,
            },
        )
        events = _sse_events(stream_response.text)
        result = _sse_event_data(events, "result")
        comparison_trace = _trace_step(result, "product_comparison")
        returned_ids = [product["product_id"] for product in result["product_cards"]]

        assert stream_response.status_code == 200
        assert "product_comparison" in stream_response.text
        assert comparison_trace["status"] == "compared"
        assert returned_ids == first_two_ids
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_memory_failure_does_not_break_stream(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_memory_failure_test.db",
        "chroma_chat_stream_memory_failure_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)

    def fail_save_turn(self, session_id, user_query, chat_response):
        raise RuntimeError("memory is temporarily unavailable")

    monkeypatch.setattr(ConversationMemoryService, "save_turn", fail_save_turn)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a",
                "debug": True,
            },
        )
        events = _sse_events(response.text)
        result = _sse_event_data(events, "result")
        memory_trace = _trace_step(result, "conversation_memory")

        assert response.status_code == 200
        assert memory_trace["status"] == "failed"
        assert "conversation_memory" in response.text
        assert _sse_event_data(events, "done")["status"] == "ok"
    finally:
        _cleanup(engine, db_path, chroma_dir)


def test_chat_stream_api_emits_error_event_when_chat_service_raises(monkeypatch) -> None:
    engine, TestingSessionLocal, db_path, chroma_dir, chroma_client = _prepare_api_stack(
        "smartbuy_chat_stream_error_test.db",
        "chroma_chat_stream_error_test",
    )
    _override_dependencies(TestingSessionLocal, chroma_client)

    def fail_stream(self, query, request_id, session_id=None, **kwargs):
        if False:
            yield None
        raise RuntimeError("stream workflow failed")

    monkeypatch.setattr(AgentStreamRunner, "stream", fail_stream)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "\u9884\u7b973000\uff0c\u63a8\u8350\u4e00\u6b3e\u62cd\u7167\u597d\u7684\u624b\u673a"
            },
        )
        events = _sse_events(response.text)

        assert response.status_code == 200
        assert _sse_event_data(events, "error")["message"] == "chat stream failed"
        assert _sse_event_data(events, "done")["status"] == "error"
        assert _sse_event_datas(events, "result") == []
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
