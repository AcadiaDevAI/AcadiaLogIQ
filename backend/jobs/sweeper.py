"""
Stuck-job sweeper — runtime watchdog for both queue tables.

Resets rows that have been ``status='running'`` for longer than the
per-kind runtime ceiling. A row is "stuck" when:

  * the worker died mid-job (OOM, container killed, deploy rollover), or
  * the handler itself is wedged (Bedrock infinite retry, network hang).

Either way the row sits in ``running`` forever unless someone resets
it. The mark_failed code path on both queue modules already does
the right thing — soft-fail with backoff if attempts remain,
permanent-fail otherwise. This module just identifies the stuck rows
and calls into those existing paths.

Why a separate sweeper (not in-line inside the worker loop)
-----------------------------------------------------------
The worker doing the sweep is the same one that might be stuck.
Putting the watchdog in the worker's main loop means a wedged
worker never gets reset. EventBridge Scheduler running the sweep
in a fresh Fargate task once per minute breaks that dependency
cycle — same primitive we use for queue-depth metrics.

Cost: ~$3-5/mo for 1440 daily Fargate task starts.

Per-kind ceilings
-----------------
Encoded here in code (not a DB column) because the values are
deployment-static — RCA / Gap Analysis prompts don't change at
runtime. Bumping a ceiling = code change + deploy, which is the
right cadence for "what counts as stuck".

  * rca_customer / rca_internal           : 10 min (Bedrock has a
                                             read_timeout=120, so 10
                                             min covers ~5 retries)
  * gap_analysis_gap                      : 15 min (longer output, can
                                             cross 4 min on a fresh run)
  * gap_analysis_pm                       : 10 min
  * ingest_document                       : 45 min (worst-case big PDF
                                             + Titan embedding batches)

Failure mode contract
---------------------
Sweeper never raises. DB / CloudWatch errors are logged at WARNING.
A missed sweep is a hundredfold less bad than a sweeper that blocks
the auto-scaler signal pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Tuple

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Per-kind runtime ceiling, in seconds. Sweeper resets any row in
# ``status='running'`` for longer than this.
MAX_RUNTIME_SECONDS: Dict[str, int] = {
    "rca_customer":     600,
    "rca_internal":     600,
    "gap_analysis_gap": 900,
    "gap_analysis_pm":  600,
    "ingest_document":  2700,
}

# Fallback ceiling for any unknown kind — must be longer than every
# documented ceiling so a genuinely-long but unseen kind isn't
# spuriously reset. Bump if you add a kind that legitimately takes
# longer than this.
_DEFAULT_RUNTIME_SECONDS = 1200


# Tables we sweep + their (id_column, mark_failed callable) tuple.
# Worker.py uses a different mark_failed per table (report_jobs vs
# ingestion_jobs) and we honour that here so a soft-fail reset on
# either table uses the correct retry-vs-fail branch.
def _table_specs() -> List[Tuple[str, str, Any]]:
    # Lazy imports so this module is import-safe in unit tests / static
    # analysis without a live DB.
    from .queue import mark_failed as mark_failed_report
    from .ingestion_queue import mark_failed as mark_failed_ingestion
    return [
        ("report_jobs",    "id",     mark_failed_report),
        ("ingestion_jobs", "job_id", mark_failed_ingestion),
    ]


def _engine():
    from backend.db.connection import engine  # type: ignore
    return engine


def _find_stuck(table: str, id_col: str) -> List[Dict[str, Any]]:
    """Return rows in ``status='running'`` past their per-kind ceiling.

    The SQL uses a ``CASE`` over ``kind`` to encode the per-kind
    ceiling so we do the filter server-side instead of pulling every
    running row and filtering in Python. Cheaper at scale and keeps
    the lock window short.

    Returns a list of dicts with: id (as string), kind, attempts,
    max_attempts, age_seconds.
    """
    case_sql = " ".join(
        f"WHEN '{k}' THEN {v}"
        for k, v in MAX_RUNTIME_SECONDS.items()
    )
    sql = text(
        f"""
        SELECT {id_col}::text AS id,
               kind,
               attempts,
               max_attempts,
               EXTRACT(EPOCH FROM (NOW() - started_at))::int AS age_seconds
          FROM {table}
         WHERE status = 'running'
           AND started_at IS NOT NULL
           AND NOW() - started_at > make_interval(
                 secs => CASE kind
                     {case_sql}
                     ELSE {int(_DEFAULT_RUNTIME_SECONDS)}
                 END
               )
        """
    )
    try:
        with _engine().connect() as conn:
            rows = conn.execute(sql).mappings().all()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(
            "[sweeper] find_stuck failed table=%s err=%s — skipping this tick",
            table, exc,
        )
        return []


def sweep_one_table(
    *,
    table: str,
    id_col: str,
    mark_failed_fn,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Sweep a single queue table. Returns a count dict.

    Returns ``{"stuck": N, "reset": M, "failed_permanent": K}`` where
    ``reset`` is soft-failures (re-queued) and ``failed_permanent`` is
    rows that burned through max_attempts.
    """
    stuck = _find_stuck(table, id_col)
    if not stuck:
        return {"stuck": 0, "reset": 0, "failed_permanent": 0}

    counts = {"stuck": len(stuck), "reset": 0, "failed_permanent": 0}
    for row in stuck:
        kind = row["kind"]
        ceiling = MAX_RUNTIME_SECONDS.get(kind, _DEFAULT_RUNTIME_SECONDS)
        attempts = int(row["attempts"])
        max_attempts = int(row["max_attempts"])
        age = int(row["age_seconds"])
        permanent = attempts >= max_attempts

        logger.warning(
            "[sweeper] stuck table=%s id=%s kind=%s age=%ds ceiling=%ds "
            "attempts=%d/%d → %s",
            table, row["id"], kind, age, ceiling,
            attempts, max_attempts,
            "permanent_fail" if permanent else "soft_reset",
        )
        if not dry_run:
            try:
                mark_failed_fn(
                    job_id=row["id"],
                    error_message=(
                        f"sweeper: runtime {age}s exceeded ceiling {ceiling}s"
                    ),
                    permanent=permanent,
                )
            except Exception as exc:
                # Defense in depth — mark_failed is meant to be
                # idempotent and fail-safe, but log if it does raise
                # so a regression doesn't silently wedge the sweeper.
                logger.warning(
                    "[sweeper] mark_failed failed table=%s id=%s err=%s",
                    table, row["id"], exc,
                )
                continue
        if permanent:
            counts["failed_permanent"] += 1
        else:
            counts["reset"] += 1

    return counts


def sweep(dry_run: bool = False) -> Dict[str, Dict[str, int]]:
    """Sweep both queue tables. Returns per-table count dicts."""
    out: Dict[str, Dict[str, int]] = {}
    for table, id_col, mark_failed_fn in _table_specs():
        out[table] = sweep_one_table(
            table=table,
            id_col=id_col,
            mark_failed_fn=mark_failed_fn,
            dry_run=dry_run,
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Find stuck rows + log them but DO NOT call mark_failed.",
    )
    args = parser.parse_args(argv)

    # MUST run before ``backend.config`` is imported — Fargate task
    # carries only the Secrets Manager ARN; DATABASE_URL etc. live
    # in the secret blob.
    from backend.core.secrets import bootstrap_environment
    bootstrap_environment()

    from backend.observability import configure_logging
    from backend.config import settings
    configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)

    results = sweep(dry_run=args.dry_run)
    for table, counts in results.items():
        logger.info(
            "[sweeper] table=%s stuck=%d reset=%d failed_permanent=%d dry_run=%s",
            table, counts["stuck"], counts["reset"],
            counts["failed_permanent"], args.dry_run,
        )
    # Also print a one-line summary to stdout for the EventBridge
    # invocation log, which captures container output.
    summary = ", ".join(
        f"{t}: {c['stuck']} stuck"
        for t, c in results.items()
    )
    print(f"sweep complete — {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
