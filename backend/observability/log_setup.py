"""
Process-wide logging setup.

Single entry point — ``configure_logging(level, fmt)`` — that replaces
the old ``logging.basicConfig(...)`` call in ``backend/api.py``. Idempotent:
re-invocations during reload don't stack handlers.

Responsibilities
----------------
1. Pick a formatter (JSON or text) based on ``LOG_FORMAT``.
2. Install a single stdout ``StreamHandler`` so the Docker engine and
   CloudWatch ``awslogs`` driver pick up every line.
3. Attach a ``ContextFilter`` that copies the current request_id and
   user_id from the ContextVars onto every LogRecord — so the same
   ``logger.info("...")`` calls already scattered across the codebase
   start carrying correlation IDs with zero source changes.
4. Wire third-party noisy loggers (``uvicorn.access`` in particular)
   into the same formatter so the log stream is uniform.

Why not just use ``logging.config.dictConfig``
----------------------------------------------
dictConfig is fine for green-field projects but this codebase already
calls ``logging.basicConfig`` early in ``api.py`` and a tangle of
modules call ``logging.getLogger("acadia-log-iq")`` at import time.
A simple, imperative setup function is the lowest-risk migration path
— we don't have to rewrite anything else.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .log_formatters import JsonFormatter, TextFormatter
from .request_context import get_request_id, get_user_id


# Internal sentinel used to detect "already configured" so that
# uvicorn-reload doesn't pile up handlers.
_SETUP_MARKER = "_acadia_observability_configured"


class ContextFilter(logging.Filter):
    """Inject ``request_id`` / ``user_id`` from ContextVars onto every record.

    Filters run before formatters, so once this is attached every
    LogRecord that reaches the handler has the two attributes set
    (or left as ``None`` if no request is in flight). The formatters
    read them via ``getattr(record, ..., None)``.

    A filter (rather than a custom Logger subclass) is the canonical
    Python pattern for cross-cutting record decoration — it leaves
    every other piece of the logging machinery alone.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.user_id = get_user_id()
        return True


def _normalise_level(level: object) -> int:
    """Accept either an int (``logging.INFO``) or string (``"INFO"``).

    Robust against typos in the env var — an unknown level name falls
    back to ``logging.INFO`` rather than raising. This is a deliberate
    failure mode: a logging-config typo must never crash the entire
    API at startup; we'd rather emit a noisy boot-time warning and
    keep running.
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        looked_up = logging.getLevelName(level.upper())
        # ``getLevelName`` returns the int when the name exists, and
        # a ``"Level XYZ"`` string when it doesn't. Anything non-int
        # means the env var was junk — fall back to INFO and shout.
        if isinstance(looked_up, int):
            return looked_up
        logging.getLogger(__name__).warning(
            "[observability] unknown LOG_LEVEL=%r — defaulting to INFO. "
            "Fix the env var to silence this warning.",
            level,
        )
        return logging.INFO
    return logging.INFO


def configure_logging(
    *,
    level: object = logging.INFO,
    fmt: Optional[str] = None,
) -> None:
    """Install the process-wide logging configuration.

    Parameters
    ----------
    level : int | str
        Minimum level. Accepts both ``logging.INFO`` and ``"INFO"`` so
        the caller can pass ``settings.LOG_LEVEL`` directly.
    fmt : "json" | "text" | None
        Output format. ``"json"`` is the production default; ``"text"``
        is the friendlier laptop format. When ``None``, the function
        reads ``LOG_FORMAT`` from the environment, defaulting to JSON.

    Side effects
    ------------
    * Replaces handlers on the root logger and ``acadia-log-iq``.
    * Routes ``uvicorn.access`` through our formatter so JSON-shaped
      access logs land in the same CloudWatch stream as application
      logs.
    * Sets ``logging.captureWarnings(True)`` so ``warnings.warn(...)``
      flows into the same pipeline.
    """
    root = logging.getLogger()
    if getattr(root, _SETUP_MARKER, False):
        # Subsequent calls — e.g. ``--reload`` under uvicorn — must
        # not double-attach handlers. Update level only and return.
        root.setLevel(_normalise_level(level))
        return

    chosen_fmt = (fmt or "").strip().lower()
    if not chosen_fmt:
        import os
        chosen_fmt = os.environ.get("LOG_FORMAT", "json").strip().lower()

    formatter: logging.Formatter
    if chosen_fmt == "text":
        formatter = TextFormatter()
    else:
        formatter = JsonFormatter()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())

    # Wipe the root logger clean so ``basicConfig``'s default handler
    # (or a leftover from a previous import order) doesn't double-log.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(_normalise_level(level))

    # Application logger inherits via propagation; we don't add a
    # second handler. But we *do* clear any handlers attached
    # earlier (e.g. by ``logging.basicConfig``) so the inherited
    # handler is the only writer.
    app_logger = logging.getLogger("acadia-log-iq")
    for existing in list(app_logger.handlers):
        app_logger.removeHandler(existing)
    app_logger.propagate = True

    # uvicorn.access logs every request in its own format. Route it
    # through our formatter so CloudWatch ingests one schema, not
    # two. Setting propagate=True + clearing its handlers means it
    # bubbles up to root → our handler → our formatter.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        u_logger = logging.getLogger(name)
        for existing in list(u_logger.handlers):
            u_logger.removeHandler(existing)
        u_logger.propagate = True
        # uvicorn.access defaults to INFO; honour the configured level
        # so DEBUG-mode laptop runs aren't drowned out.
        u_logger.setLevel(_normalise_level(level))

    # Route ``warnings.warn(...)`` into logging so deprecation warnings
    # from dependencies (urllib3, etc.) land in the same stream.
    logging.captureWarnings(True)

    setattr(root, _SETUP_MARKER, True)
