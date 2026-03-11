"""
Chat session model.

Persists conversation containers in PostgreSQL so chat history survives restarts.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text

from backend.db.connection import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True)
    owner_id = Column(String, nullable=False, index=True)
    title = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)