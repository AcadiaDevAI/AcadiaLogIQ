"""Per-request progress channel for streaming staged UI updates.

The core ``/ask`` pipeline (``backend/api.py``) is a large, mostly synchronous
flow. To let the ``/ask/stream`` SSE endpoint show REAL phase updates
("Retrieving context", "Generating the answer") WITHOUT restructuring that
pipeline, we expose a tiny context-local sink.

The pipeline calls :func:`emit_progress` at real phase boundaries. When no sink
is registered — i.e. the normal ``/ask`` path — every call is a no-op, so
``/ask`` behavior stays byte-identical. The streaming endpoint registers a sink
(via :func:`set_progress_sink`) that forwards each milestone to its SSE queue.

A ``ContextVar`` is used so the sink is scoped to the request/thread that set
it and never leaks across concurrent requests.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Callable, Optional

logger = logging.getLogger("acadia-log-iq")

# stage-id + human label -> None. Absent (None) on the plain /ask path.
_progress_sink: contextvars.ContextVar[Optional[Callable[[str, str], None]]] = (
    contextvars.ContextVar("acadia_progress_sink", default=None)
)


def set_progress_sink(fn: Callable[[str, str], None]):
    """Register the current context's progress sink. Returns a reset token to
    pass to :func:`reset_progress_sink` when the request is done."""
    return _progress_sink.set(fn)


def reset_progress_sink(token) -> None:
    """Clear the sink registered by :func:`set_progress_sink`. Never raises."""
    try:
        _progress_sink.reset(token)
    except Exception:  # pragma: no cover - defensive
        pass


def emit_progress(stage: str, label: str) -> None:
    """Publish a progress milestone to the current request's sink.

    No-op when no sink is registered (the normal ``/ask`` path). NEVER raises —
    progress reporting must not be able to break the answer pipeline.
    """
    fn = _progress_sink.get()
    if fn is None:
        return
    try:
        fn(stage, label)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[progress] sink raised (ignored): %s", exc)
