"""
Simple structured migration runner.

This is a Phase 1 alternative to Alembic when you want lightweight,
predictable SQL migrations.
"""

from pathlib import Path

from sqlalchemy import text

from backend.db.connection import engine


def run_migrations():
    migrations_dir = Path(__file__).parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

        applied = {
            row[0]
            for row in conn.execute(text("SELECT version FROM schema_migrations")).fetchall()
        }

        for migration_path in migration_files:
            version = migration_path.name
            if version in applied:
                continue

            sql = migration_path.read_text(encoding="utf-8")
            # Split on semicolons for a lightweight runner.
            # Good enough for this migration set.
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for statement in statements:
                conn.execute(text(statement))

            conn.execute(
                text("INSERT INTO schema_migrations(version) VALUES (:version)"),
                {"version": version},
            )
            print(f"Applied migration: {version}")

    print("All migrations completed successfully.")


if __name__ == "__main__":
    run_migrations()