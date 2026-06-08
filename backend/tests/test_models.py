from pathlib import Path

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from app.core.db import Base
from app.models import Category, Message, Product, Session as ChatSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]


EXPECTED_TABLES = {
    "categories",
    "category_attribute_defs",
    "category_profiles",
    "products",
    "product_attributes",
    "product_tags",
    "documents",
    "document_chunks",
    "sessions",
    "messages",
    "retrieval_logs",
    "recommendation_logs",
    "feedback",
}


def _create_test_engine(db_name: str):
    db_path = PROJECT_ROOT / "data" / db_name
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    return create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    ), db_path


def test_all_models_can_create_tables() -> None:
    engine, db_path = _create_test_engine("smartbuy_models_create_test.db")

    try:
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        assert EXPECTED_TABLES.issubset(set(inspector.get_table_names()))
    finally:
        engine.dispose()
        db_path.unlink(missing_ok=True)


def test_core_records_can_be_inserted_and_queried() -> None:
    engine, db_path = _create_test_engine("smartbuy_models_insert_test.db")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestingSessionLocal()
    try:
        category = Category(id="cat_phone", name="手机", level=2)
        product = Product(
            id="phone_001",
            category_id=category.id,
            title="测试手机",
            brand="SmartBuy",
            price=3999,
            stock=10,
        )
        session = ChatSession(id="session_001", title="测试导购会话")
        message = Message(
            id="message_001",
            session_id=session.id,
            role="user",
            content="我想买一台拍照好的手机",
        )

        db.add_all([category, product, session, message])
        db.commit()

        saved_category = db.execute(
            select(Category).where(Category.id == "cat_phone")
        ).scalar_one()
        saved_product = db.execute(
            select(Product).where(Product.id == "phone_001")
        ).scalar_one()
        saved_session = db.execute(
            select(ChatSession).where(ChatSession.id == "session_001")
        ).scalar_one()
        saved_message = db.execute(
            select(Message).where(Message.id == "message_001")
        ).scalar_one()

        assert saved_category.name == "手机"
        assert saved_product.price == 3999
        assert saved_product.status == "active"
        assert saved_session.user_id == "demo_user"
        assert saved_message.role == "user"
    finally:
        db.close()
        engine.dispose()
        db_path.unlink(missing_ok=True)
