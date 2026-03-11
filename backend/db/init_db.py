"""
Initializes database tables for Phase 1.
Creates documents and chunks tables in PostgreSQL + pgvector.
"""

from backend.db.connection import engine
from backend.domain.models.document import Document
from backend.domain.models.chunk import Chunk
from sqlalchemy import text
from backend.db.connection import engine
from backend.db.migrate import run_migrations


def init_db():
    run_migrations()


if __name__ == "__main__":
    init_db()