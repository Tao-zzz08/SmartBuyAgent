from collections.abc import Generator
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.db import Base, get_db
from app.main import app
from app.models import ChatFeedback


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    return engine, TestingSessionLocal, db_path


def _override_db(TestingSessionLocal) -> None:
    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db


def _cleanup(engine, db_path: Path) -> None:
    app.dependency_overrides.clear()
    engine.dispose()
    db_path.unlink(missing_ok=True)


def test_submit_helpful_feedback_success() -> None:
    engine, TestingSessionLocal, db_path = _create_test_session(
        "smartbuy_feedback_helpful_test.db"
    )
    _override_db(TestingSessionLocal)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "session_feedback_001",
                "rating": "helpful",
                "reason": "recommendation_relevant",
                "query": "budget 3000 phone",
                "answer_preview": "Recommended phone candidates.",
            },
        )
        payload = response.json()

        assert response.status_code == 200
        assert payload["status"] == "saved"
        assert payload["id"] > 0

        db = TestingSessionLocal()
        try:
            saved = db.scalars(select(ChatFeedback)).one()
            assert saved.session_id == "session_feedback_001"
            assert saved.rating == "helpful"
            assert saved.reason == "recommendation_relevant"
            assert saved.query == "budget 3000 phone"
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path)


def test_submit_not_helpful_feedback_with_reason_and_comment() -> None:
    engine, TestingSessionLocal, db_path = _create_test_session(
        "smartbuy_feedback_not_helpful_test.db"
    )
    _override_db(TestingSessionLocal)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "session_feedback_002",
                "turn_id": None,
                "rating": "not_helpful",
                "reason": "unclear_answer",
                "comment": "  explanation was too broad  ",
            },
        )

        assert response.status_code == 200

        db = TestingSessionLocal()
        try:
            saved = db.scalars(select(ChatFeedback)).one()
            assert saved.rating == "not_helpful"
            assert saved.reason == "unclear_answer"
            assert saved.comment == "explanation was too broad"
            assert saved.turn_id is None
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path)


def test_submit_feedback_rejects_invalid_rating() -> None:
    engine, TestingSessionLocal, db_path = _create_test_session(
        "smartbuy_feedback_invalid_rating_test.db"
    )
    _override_db(TestingSessionLocal)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "session_feedback_003",
                "rating": "neutral",
            },
        )

        assert response.status_code == 422
    finally:
        _cleanup(engine, db_path)


def test_submit_feedback_rejects_long_comment() -> None:
    engine, TestingSessionLocal, db_path = _create_test_session(
        "smartbuy_feedback_long_comment_test.db"
    )
    _override_db(TestingSessionLocal)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "session_feedback_004",
                "rating": "not_helpful",
                "comment": "x" * 1001,
            },
        )

        assert response.status_code == 422
    finally:
        _cleanup(engine, db_path)


def test_submit_feedback_truncates_answer_preview() -> None:
    engine, TestingSessionLocal, db_path = _create_test_session(
        "smartbuy_feedback_preview_test.db"
    )
    _override_db(TestingSessionLocal)
    try:
        client = TestClient(app)
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "session_feedback_005",
                "rating": "helpful",
                "answer_preview": "a" * 650,
            },
        )

        assert response.status_code == 200

        db = TestingSessionLocal()
        try:
            saved = db.scalars(select(ChatFeedback)).one()
            assert saved.answer_preview == "a" * 500
        finally:
            db.close()
    finally:
        _cleanup(engine, db_path)
