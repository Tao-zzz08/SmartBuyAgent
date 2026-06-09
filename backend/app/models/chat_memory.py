from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.core.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    turn_index = Column(Integer, nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    assistant_answer = Column(Text, nullable=False)
    intent = Column(String(128), nullable=True, index=True)
    category_id = Column(String(64), nullable=True, index=True)
    category_path = Column(String(255), nullable=True)
    budget_min = Column(Integer, nullable=True)
    budget_max = Column(Integer, nullable=True)
    preferences_json = Column(Text, nullable=False, default="[]")
    product_ids_json = Column(Text, nullable=False, default="[]")
    citation_chunk_ids_json = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
