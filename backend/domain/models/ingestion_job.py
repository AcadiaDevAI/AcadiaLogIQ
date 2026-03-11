"""
Ingestion job model.

Persists upload processing state in PostgreSQL instead of in-memory only.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from backend.db.connection import Base


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    job_id = Column(String, primary_key=True)
    file_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    owner_id = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False, default="kb")
    file_hash = Column(String, nullable=True)

    status = Column(String, nullable=False, default="queued", index=True)
    processed_chunks = Column(String, nullable=False, default="0")
    total_chunks = Column(String, nullable=False, default="0")
    successful_chunks = Column(String, nullable=False, default="0")
    error = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)