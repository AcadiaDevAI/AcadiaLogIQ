"""
Database connection layer for PostgreSQL.

Single SQLAlchemy engine + session factory shared by every module.

Pool sizing is env-driven via ``backend.config.Settings``:
    DB_POOL_SIZE       — baseline connections kept open per worker
    DB_MAX_OVERFLOW    — burst capacity above pool_size
    DB_POOL_RECYCLE    — seconds before recycling a connection
    DB_POOL_PRE_PING   — connection-validity ping on checkout

Why env-driven:
* Laptop dev wants a generous pool (10 + 20) so manual scripts don't
  starve the API.
* Production Fargate tasks must size the pool tight so the total
  connection demand fits inside the RDS ``max_connections`` budget.
  See ``backend/config.py`` for the connection-budget math.

Boot logs the resolved pool config so an operator can confirm the
production override actually landed.
"""

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings


logger = logging.getLogger("acadia-log-iq")


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    future=True,
)


# Boot-time confirmation of resolved pool config. Helps catch the
# "I thought I set DB_POOL_SIZE=3 in the task definition" misconfig
# class — operators see the actual value in CloudWatch.
logger.info(
    "[db.pool] pool_size=%d max_overflow=%d pool_recycle=%ds pre_ping=%s",
    settings.DB_POOL_SIZE, settings.DB_MAX_OVERFLOW,
    settings.DB_POOL_RECYCLE, settings.DB_POOL_PRE_PING,
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
