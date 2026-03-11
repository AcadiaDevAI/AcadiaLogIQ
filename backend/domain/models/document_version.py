"""
Physical version of a document.

Each upload creates a document_version.
This lets us keep version history while retrieval only uses the active version.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from backend.db.connection import Base


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)

    version_number = Column(Integer, nullable=False)

    # Hash of file bytes for exact duplicate awareness.
    fingerprint = Column(String, nullable=True, index=True)

    # Where the raw file is stored. Local path now, S3 URI later.
    storage_uri = Column(Text, nullable=True)

    mime_type = Column(String, nullable=True)
    file_size_mb = Column(String, nullable=True)

    # Only one version per document should be active.
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    superseded_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)