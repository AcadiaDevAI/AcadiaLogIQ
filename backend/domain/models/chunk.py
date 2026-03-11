"""
Chunk model.

Stores only text + positioning info.
Embeddings are stored separately in the embeddings table.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from backend.db.connection import Base


class Chunk(Base):
    __tablename__ = "chunks"

    # String key is convenient because current code builds ids like:
    # "{file_id}:{job_id}:{chunk_index}"
    id = Column(String, primary_key=True)

    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    document_version_id = Column(UUID(as_uuid=True), ForeignKey("document_versions.id"), nullable=False, index=True)

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)