"""
CloudWatch metric publisher — queue-depth signal for ECS auto-scaling.

Runs as an EventBridge-Scheduler-triggered ECS task once per minute
(see ``infra/terraform/scheduler_metrics.tf``). Two metrics emitted
into the ``Acadia/LogIQ`` namespace:

  * ``ReportJobsPending``  — runnable rows in ``report_jobs``
                              (status=pending AND not_before <= now).
  * ``IngestJobsPending``  — same shape but in ``ingestion_jobs``;
                              covers all supported upload types
                              (log, txt, md, json, pdf, docx — the
                              queue doesn't care which parser the
                              worker will dispatch to).

Why a separate sidecar (not in-process inside the API)
-------------------------------------------------------
* The API tier never needs to know its own queue depth — only the
  ECS auto-scaler does. Coupling the metric write to a request
  hot path would add latency for zero user benefit.
* A dedicated cron means the metric publishes even when the API is
  under load or briefly unreachable (e.g. mid-deploy). Auto-scaling
  signals must NOT depend on the very service they're scaling.
* Same Docker image as the workers — no second build, no new
  dependency. Container override at schedule time sets
  ``command=["python","-m","backend.jobs.metrics_publisher"]``.

Failure mode
------------
Every CloudWatch / DB error is logged at WARNING and swallowed —
this cron must NEVER prevent the worker pool from running. A missed
data point is dramatically less bad than a metric publisher that
blocks downstream auto-scaling.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Metric namespace — single source of truth shared with Terraform.
# Bump the version segment ("v1") if we ever rework the metric
# shape so the alarm + dashboard layer can opt in cleanly.
_NAMESPACE = "Acadia/LogIQ"
_METRIC_REPORT = "ReportJobsPending"
_METRIC_INGEST = "IngestJobsPending"

# Queue tables we measure. Tuple form so the iteration order is
# stable and obvious in logs.
_QUEUES: Tuple[Tuple[str, str, str], ...] = (
    # (table_name, metric_name, friendly_label)
    ("report_jobs",     _METRIC_REPORT, "report"),
    ("ingestion_jobs",  _METRIC_INGEST, "ingest"),
)


def _engine():
    """Lazy import so this module is import-safe without a live DB."""
    from backend.db.connection import engine  # type: ignore
    return engine


def _count_pending(table: str) -> int:
    """Return the runnable-now row count for one queue table.

    "Runnable" = ``status='pending' AND not_before <= NOW()``. Rows
    still in their exponential-backoff window are excluded so a
    burst of soft-failed jobs doesn't trigger spurious scale-out.

    Returns 0 on DB error (logged at WARNING). The auto-scaler
    treats "no data" as "no work", so reporting 0 on failure is
    the correct fail-safe.
    """
    sql = text(
        f"""
        SELECT count(*)
          FROM {table}
         WHERE status = 'pending'
           AND not_before <= NOW()
        """
    )
    try:
        with _engine().connect() as conn:
            return int(conn.execute(sql).scalar() or 0)
    except Exception as exc:
        logger.warning(
            "[metrics] count failed table=%s err=%s — emitting 0",
            table, exc,
        )
        return 0


def _cloudwatch_client():
    """Build a boto3 CloudWatch client. Region resolution mirrors the
    rest of the backend (``AWS_REGION`` env, falling back to the
    Bedrock region we already use)."""
    import boto3
    from botocore.config import Config as BotoConfig

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    # The cron is short-lived (seconds), so a tight read_timeout
    # surfaces issues quickly rather than hanging the ECS task.
    cfg = BotoConfig(
        retries={"max_attempts": 3, "mode": "standard"},
        read_timeout=10,
        connect_timeout=5,
    )
    return boto3.client("cloudwatch", region_name=region, config=cfg)


def _put_metric(
    client,
    *,
    metric: str,
    value: float,
    env: str,
    extra_dimensions: Optional[Dict[str, str]] = None,
) -> bool:
    """Emit a single data point. Logged on failure, never raises."""
    dimensions = [{"Name": "Environment", "Value": env}]
    if extra_dimensions:
        for k, v in extra_dimensions.items():
            dimensions.append({"Name": k, "Value": v})
    try:
        client.put_metric_data(
            Namespace=_NAMESPACE,
            MetricData=[{
                "MetricName": metric,
                "Dimensions": dimensions,
                "Value": float(value),
                "Unit": "Count",
                "Timestamp": datetime.now(timezone.utc),
            }],
        )
        return True
    except Exception as exc:
        logger.warning(
            "[metrics] put_metric_data failed metric=%s value=%s err=%s",
            metric, value, exc,
        )
        return False


def publish_once(env: Optional[str] = None) -> Dict[str, int]:
    """Read both queues + emit both metrics. Returns the values for
    optional callsite inspection (tests assert on the return dict).
    """
    resolved_env = (env or os.environ.get("APP_ENV") or "dev").strip().lower()
    client = _cloudwatch_client()

    results: Dict[str, int] = {}
    for table, metric, label in _QUEUES:
        depth = _count_pending(table)
        ok = _put_metric(client, metric=metric, value=depth, env=resolved_env)
        results[metric] = depth
        logger.info(
            "[metrics] %-7s pending=%-4d emit=%s metric=%s env=%s",
            label, depth, "ok" if ok else "FAIL", metric, resolved_env,
        )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read queue depths but do NOT call CloudWatch.",
    )
    args = parser.parse_args(argv)

    # MUST run before ``backend.config`` is imported — settings reads
    # DATABASE_URL etc. from os.environ at module load, and Secrets
    # Manager is the only source for those values in production.
    # Without this call the Fargate-launched cron crashes at import.
    from backend.core.secrets import bootstrap_environment
    bootstrap_environment()

    # Defer logging setup to the main path so importing this module
    # in tests / shell sessions doesn't reconfigure root handlers.
    from backend.observability import configure_logging
    from backend.config import settings
    configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)

    if args.dry_run:
        # Skip the CloudWatch hop entirely — just print depths.
        results: Dict[str, int] = {}
        for table, metric, label in _QUEUES:
            depth = _count_pending(table)
            results[metric] = depth
            print(f"[dry-run] {label:7s} pending={depth:<4d} (would emit {metric})")
        return 0

    publish_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
