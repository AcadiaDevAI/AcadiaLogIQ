"""
Job retention sweeper — drops ``report_jobs`` rows older than N days.

The ``report_jobs`` table is a workflow ledger, not the product. The
actual cached results live in ``report_cache`` (kept indefinitely —
it IS the product). Letting the jobs table grow forever would bloat
the DB without a corresponding business value.

Default retention: 30 days. Set via ``JOB_RETENTION_DAYS`` env var.

How it runs
-----------
* In production: an ECS scheduled task (or a tiny cron on the worker
  container) calls ``python -m backend.jobs.retention`` once per
  day. The runbook in ``docs/runbooks/job-retention.md`` documents
  the cron expression.
* In dev: run manually when you want to clean up after testing.

Why a separate module (not a cron inside the worker loop)
---------------------------------------------------------
* Single responsibility — easy to audit and dry-run.
* Schedulable independently — sweep frequency doesn't have to match
  worker scaling.
* No surprise queries inside the hot-path worker loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


def _engine():
    from backend.db.connection import engine  # type: ignore
    return engine


def sweep(retention_days: int = 30, dry_run: bool = False) -> int:
    """Delete ``report_jobs`` rows older than ``retention_days``.

    Returns the number of rows that were removed (or would be
    removed, in dry-run mode).

    We only sweep rows in a terminal status (done / failed /
    cancelled). Pending / running rows are left alone even if old
    so a stuck job is visible to ops rather than silently disappearing.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
    sql = text(
        """
        DELETE FROM report_jobs
         WHERE status IN ('done', 'failed', 'cancelled')
           AND COALESCE(finished_at, created_at) < :cutoff
        """
    )
    count_sql = text(
        """
        SELECT count(*)
          FROM report_jobs
         WHERE status IN ('done', 'failed', 'cancelled')
           AND COALESCE(finished_at, created_at) < :cutoff
        """
    )

    with _engine().connect() as conn:
        n = conn.execute(count_sql, {"cutoff": cutoff}).scalar() or 0

    if dry_run or n == 0:
        logger.info(
            "[jobs.retention] cutoff=%s would_delete=%d dry_run=%s",
            cutoff.isoformat(), n, dry_run,
        )
        return int(n)

    with _engine().begin() as conn:
        conn.execute(sql, {"cutoff": cutoff})

    logger.info(
        "[jobs.retention] cutoff=%s deleted=%d", cutoff.isoformat(), n,
    )
    return int(n)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days", type=int,
        default=int(os.environ.get("JOB_RETENTION_DAYS", "30")),
        help="Retention in days (default 30, env JOB_RETENTION_DAYS).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count rows that would be deleted; do not modify the table.",
    )
    args = parser.parse_args(argv)

    # MUST run before ``backend.config`` is imported. In production
    # the Fargate task carries only the Secrets Manager ARN; DATABASE_URL
    # lives in the secret blob and ``Settings()`` would crash without it.
    from backend.core.secrets import bootstrap_environment
    bootstrap_environment()

    # Defer logging setup to the main path so importing this module
    # in tests / shell sessions doesn't reconfigure root handlers.
    from backend.observability import configure_logging
    from backend.config import settings
    configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)

    n = sweep(retention_days=args.days, dry_run=args.dry_run)
    print(f"swept={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
