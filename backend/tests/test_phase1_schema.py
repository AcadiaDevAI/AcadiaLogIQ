"""
Basic schema tests for Phase 1.

These are lightweight examples that you can run after wiring pytest into the project.
"""

from sqlalchemy import text

from backend.db.connection import engine


def test_phase1_tables_exist():
    expected = {
        "documents",
        "document_versions",
        "document_metadata",
        "chunks",
        "embeddings",
        "ingestion_jobs",
        "chat_sessions",
        "chat_messages",
    }

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                """
            )
        ).fetchall()

    existing = {row[0] for row in rows}
    missing = expected - existing
    assert not missing, f"Missing Phase 1 tables: {missing}"