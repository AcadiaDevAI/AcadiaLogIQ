"""
Database connection layer for PostgreSQL.
Keeps a single SQLAlchemy engine/session factory used by all Phase 1 modules.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

# SQLAlchemy engine used across the whole backend.
# pool_pre_ping=True helps recover from stale DB connections.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

# Session factory used by API routes and helper functions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

# Declarative base for ORM models.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()