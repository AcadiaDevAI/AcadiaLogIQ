"""
Logical document model.

A document represents the user-visible file entry.
Versions are stored in document_versions so the latest file can become active
while older versions remain traceable.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID

from backend.db.connection import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Which user owns this document.
    owner_id = Column(String, nullable=False, index=True)

    # Original file name shown in the UI.
    name = Column(String, nullable=False)

    # File type used by retrieval/UI filtering, e.g. "kb".
    file_type = Column(String, nullable=False, default="kb")

    # active / superseded / deleted
    status = Column(String, nullable=False, default="active", index=True)

    # Pointer to the latest active version of the document.
    current_version_id = Column(UUID(as_uuid=True), ForeignKey("document_versions.id"), nullable=True)

    # Optional note for future audit/debug.
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)