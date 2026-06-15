from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.core.db import Base


class ChatFeedback(Base):
    __tablename__ = "chat_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    turn_id = Column(Integer, nullable=True, index=True)
    rating = Column(String(32), nullable=False, index=True)
    reason = Column(String(128), nullable=True, index=True)
    comment = Column(Text, nullable=True)
    query = Column(Text, nullable=True)
    answer_preview = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
