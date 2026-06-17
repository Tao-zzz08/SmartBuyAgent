import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.cache.cache_service import InMemoryCacheService
from app.chat.conversation_memory import ConversationMemoryService
from app.chat.response_composer import (
    ChatResponse,
    CitationView,
    ProductCard,
)
from app.core.db import Base
from app.models import ChatSession, ChatTurn


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _create_test_db(db_name: str):
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


def _chat_response(
    answer: str = "Template answer",
    include_query_trace: bool = True,
) -> ChatResponse:
    trace = []
    if include_query_trace:
        trace.append(
            {
                "step": "query_understanding",
                "intent": "shopping_guide",
                "category_id": "cat_phone",
                "category_path": "数码/手机",
                "budget_min": None,
                "budget_max": 3000,
                "preferences": ["拍照", "续航"],
            }
        )

    return ChatResponse(
        answer=answer,
        product_cards=[
            ProductCard(
                product_id="phone_001",
                title="星曜 X1",
                brand="星曜",
                price=2599,
                image_url=None,
                tags=["拍照好"],
                attributes={"存储容量": "256GB"},
                source_url="https://example.com/products/phone_001",
                compare_url=None,
                recommend_reason="价格在你的预算范围内",
            )
        ],
        citations=[
            CitationView(
                chunk_id="chunk_001",
                title="手机拍照选购指南",
                section="为什么不能只看像素",
                section_path="手机拍照选购指南/为什么不能只看像素",
                source_file="data/knowledge_docs/phone/phone_camera_guide.md",
                content_preview="手机拍照不能只看像素。",
                score=0.9,
            )
        ],
        trace=trace,
    )


def test_ensure_session_creates_session() -> None:
    engine, db, db_path = _create_test_db("smartbuy_conversation_session_test.db")
    try:
        memory = ConversationMemoryService(db)

        session = memory.ensure_session("session_test")

        assert session.session_id == "session_test"
        assert db.get(ChatSession, "session_test") is not None
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_save_turn_persists_chat_response_summary() -> None:
    engine, db, db_path = _create_test_db("smartbuy_conversation_save_test.db")
    try:
        memory = ConversationMemoryService(db)

        turn = memory.save_turn(
            session_id="session_test",
            user_query="预算3000，推荐一款拍照好的手机",
            chat_response=_chat_response(),
        )

        saved_turn = db.get(ChatTurn, turn.id)
        assert saved_turn is not None
        assert saved_turn.user_query == "预算3000，推荐一款拍照好的手机"
        assert saved_turn.assistant_answer == "Template answer"
        assert saved_turn.intent == "shopping_guide"
        assert saved_turn.category_id == "cat_phone"
        assert saved_turn.category_path == "数码/手机"
        assert saved_turn.budget_max == 3000
        assert json.loads(saved_turn.preferences_json) == ["拍照", "续航"]
        assert json.loads(saved_turn.product_ids_json) == ["phone_001"]
        assert json.loads(saved_turn.citation_chunk_ids_json) == ["chunk_001"]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_get_recent_turns_returns_latest_turns() -> None:
    engine, db, db_path = _create_test_db("smartbuy_conversation_recent_test.db")
    try:
        memory = ConversationMemoryService(db)
        for index in range(1, 5):
            memory.save_turn(
                session_id="session_test",
                user_query=f"query {index}",
                chat_response=_chat_response(answer=f"answer {index}"),
            )

        turns = memory.get_recent_turns("session_test", limit=2)

        assert [turn.turn_index for turn in turns] == [3, 4]
        assert [turn.user_query for turn in turns] == ["query 3", "query 4"]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_save_turn_handles_missing_query_understanding_trace() -> None:
    engine, db, db_path = _create_test_db("smartbuy_conversation_missing_trace_test.db")
    try:
        memory = ConversationMemoryService(db)

        turn = memory.save_turn(
            session_id="session_test",
            user_query="推荐一下",
            chat_response=_chat_response(include_query_trace=False),
        )

        saved_turn = db.scalar(select(ChatTurn).where(ChatTurn.id == turn.id))
        assert saved_turn is not None
        assert saved_turn.intent is None
        assert saved_turn.category_id is None
        assert json.loads(saved_turn.preferences_json) == []
        assert json.loads(saved_turn.product_ids_json) == ["phone_001"]
        assert json.loads(saved_turn.citation_chunk_ids_json) == ["chunk_001"]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_recent_turns_cache_reads_from_cache_after_first_query() -> None:
    engine, db, db_path = _create_test_db("smartbuy_conversation_cache_test.db")
    try:
        cache = InMemoryCacheService()
        memory = ConversationMemoryService(db, cache_service=cache)
        memory.save_turn(
            session_id="session_cache",
            user_query="query 1",
            chat_response=_chat_response(answer="answer 1"),
        )

        first = memory.get_recent_turns("session_cache", limit=5)
        assert [turn.user_query for turn in first] == ["query 1"]

        db.add(
            ChatTurn(
                session_id="session_cache",
                turn_index=2,
                user_query="query outside cache",
                assistant_answer="answer outside cache",
                preferences_json="[]",
                product_ids_json="[]",
                citation_chunk_ids_json="[]",
            )
        )
        db.commit()

        second = memory.get_recent_turns("session_cache", limit=5)
        assert [turn.user_query for turn in second] == ["query 1"]
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_save_turn_invalidates_recent_turns_cache_and_sets_last_candidates() -> None:
    engine, db, db_path = _create_test_db(
        "smartbuy_conversation_cache_refresh_test.db"
    )
    try:
        cache = InMemoryCacheService()
        memory = ConversationMemoryService(db, cache_service=cache)
        memory.save_turn(
            session_id="session_cache",
            user_query="query 1",
            chat_response=_chat_response(answer="answer 1"),
        )
        memory.get_recent_turns("session_cache", limit=5)

        memory.save_turn(
            session_id="session_cache",
            user_query="query 2",
            chat_response=_chat_response(answer="answer 2"),
        )

        turns = memory.get_recent_turns("session_cache", limit=5)
        assert [turn.user_query for turn in turns] == ["query 1", "query 2"]
        assert (
            cache.get_json("smartbuy:session:session_cache:last_candidates")
            == ["phone_001"]
        )
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
