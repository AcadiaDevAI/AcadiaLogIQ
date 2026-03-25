"""
Database connection layer for PostgreSQL.
Keeps a single SQLAlchemy engine/session factory used by all modules.

PERFORMANCE TUNED:
- pool_size=10 (was default 5) — supports concurrent metadata + embed workers
- max_overflow=20 — burst capacity for ingestion spikes
- pool_recycle=1800 — prevent stale connections on long-running servers
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,           # ← increased from default 5
    max_overflow=20,        # ← increased from default 10
    pool_recycle=1800,      # ← recycle connections every 30 min
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
