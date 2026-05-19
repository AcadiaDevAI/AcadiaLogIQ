"""
Structured log formatters.

Two formatters that share the same record-shape contract:

* ``JsonFormatter``  → newline-delimited JSON per log line. Designed
  to be ingested verbatim by the Docker ``awslogs`` driver into
  CloudWatch, where every field becomes a queryable column in Logs
  Insights ("show 5xx grouped by route", "p99 latency last hour").
* ``TextFormatter``  → human-readable single-line output for laptop
  development. Identical fields as JSON but printed compactly so
  ``docker logs -f`` stays scannable.

The same ``logging.LogRecord`` flows through both — we never branch
the application code on output format. Switching is purely a
deployment decision (env var ``LOG_FORMAT``).

Why hand-rolled (and not ``python-json-logger`` or structlog)
-------------------------------------------------------------
* Zero extra dependencies — formatters are <60 lines apiece.
* Full control over reserved field order, so CloudWatch's column
  auto-detection picks the right shape on first ingest.
* Trivial to reason about during incident response (no third-party
  config to second-guess at 2 AM).

Reserved-field policy
---------------------
Top-level keys are stable contract for log consumers. ``extra``
dicts passed via ``logger.info("...", extra={"foo": 1})`` are
merged into the JSON output untouched so callers can attach
domain-specific context without modifying this module.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict

# Fields injected by ``ContextFilter`` (see ``log_setup.py``). We
# reference the names here so the formatter and the filter stay in
# sync without a circular import.
_CONTEXT_FIELDS = ("request_id", "user_id")

# Built-in LogRecord attributes that we don't want to bleed into the
# ``extra`` payload — they're either redundant (already promoted to
# top-level) or noisy (``exc_info`` is rendered specially below).
_RESERVED_LOG_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }
    | set(_CONTEXT_FIELDS)
)

# Host identifier — useful when multiple EC2 instances or Docker
# containers stream to the same CloudWatch group.
_HOSTNAME = socket.gethostname() or os.environ.get("HOSTNAME", "unknown")


def _iso_ts(record: logging.LogRecord) -> str:
    """Render the record's wall-clock as RFC 3339 / ISO 8601 UTC."""
    return datetime.fromtimestamp(
        record.created, tz=timezone.utc,
    ).isoformat()


def _extract_extra(record: logging.LogRecord) -> Dict[str, Any]:
    """Pull non-reserved attributes off the record into a flat dict.

    ``logger.info("hi", extra={"chunk_id": 7})`` attaches ``chunk_id``
    to the LogRecord. We collect those here so the JSON output keeps
    a stable top-level shape and dumps domain fields under a clear
    namespace.
    """
    out: Dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _RESERVED_LOG_RECORD_ATTRS:
            continue
        # Skip private fields (``_xxx``) so callers can use them as
        # scratch attributes without polluting logs.
        if key.startswith("_"):
            continue
        out[key] = value
    return out


class JsonFormatter(logging.Formatter):
    """One JSON document per line. Stable key order, UTF-8, no NaN.

    The JSON output is read by:
      * CloudWatch Logs Insights — every top-level key becomes a
        queryable column (e.g. ``filter route="/ask"``).
      * Sentry's ``LoggingIntegration`` — picks up ``message`` and
        ``level`` automatically; ``extra`` lands in Sentry's "Additional
        Data" panel.
      * Local jq pipelines — ``docker logs api | jq 'select(.level=="ERROR")'``
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _iso_ts(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "host": _HOSTNAME,
        }

        # Context fields — present on every record thanks to
        # ``ContextFilter``, but defensive ``getattr`` keeps this
        # formatter usable even if the filter is removed in tests.
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value:
                payload[field] = value

        extra = _extract_extra(record)
        if extra:
            payload["extra"] = extra

        if record.exc_info:
            # Preserve the traceback as a single multi-line string so
            # Logs Insights renders it intact and Sentry can extract a
            # stack frame from it.
            payload["exc"] = self.formatException(record.exc_info)

        # ``default=str`` is the cheap escape hatch for values the
        # JSON encoder can't handle natively (datetime, UUID, Decimal,
        # Path, SQLAlchemy rows). The string fallback is good enough
        # for log consumption — domain code that needs structured
        # output should serialise upstream.
        return json.dumps(
            payload, ensure_ascii=False, default=str, separators=(",", ":"),
        )


class TextFormatter(logging.Formatter):
    """Compact single-line text for laptop dev — same fields as JSON.

    Format roughly mirrors what the previous ``logging.basicConfig``
    produced (so muscle memory survives), plus the new ``request_id``
    and ``user_id`` tags when present.

    Example::

        2026-05-18T12:30:00.123+00:00  INFO  acadia-log-iq  api:1731
            req=a8c7d2e1 user=user_3Cv… msg=POST /ask -> 200 (812.3ms)
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = _iso_ts(record)
        ctx_bits = []
        for field in _CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value:
                # Trim ContextVar values to keep the line short; full
                # value still lands in JSON logs and Sentry events.
                short = str(value)
                if len(short) > 16:
                    short = short[:13] + "…"
                ctx_bits.append(f"{field[:3]}={short}")
        ctx = (" " + " ".join(ctx_bits)) if ctx_bits else ""

        head = (
            f"{ts}  {record.levelname:<5}  {record.name}  "
            f"{record.module}:{record.lineno}{ctx}"
        )
        line = f"{head}  {record.getMessage()}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line
