"""
Embedding model.

Stores semantic vectors separately from chunk text so retrieval and storage
can scale more cleanly.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, String

from backend.db.connection import Base


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id = Column(String, ForeignKey("chunks.id"), primary_key=True)
    embedding = Column(Vector(1024), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)