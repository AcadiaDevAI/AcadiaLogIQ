"""
Answer Cache — Postgres-backed exact-match cache for ``/ask`` answers.

Keyed on (normalized_query, owner_id, sorted active_file_ids). Replaces
the previous in-process LRU; the public API (``ANSWER_CACHE.get(...)``,
``.put(...)``, ``CacheLookup``) is preserved verbatim so callers in
``backend/api.py`` need no changes.

Why we changed
--------------
The previous implementation kept state in a process-local OrderedDict.
Multi-replica deployments would have produced inconsistent answers
because two replicas don't share that dict. We moved to a Postgres
table (``answer_cache_exact``, migration 044) so the cache is shared,
durable across container restarts, and consistent under horizontal
scaling.

Behaviour preserved
-------------------
* Same key shape: sha1(normalized_query || owner_id || sorted files).
* Same payload contract: caller passes / receives a dict, opaque to us.
* TTL semantics unchanged (~10 min default).
* Same fail-safe behaviour — every public function swallows DB errors
  and reports a miss so a Postgres hiccup never breaks ``/ask``.
* Same instrumentation tags (``Answer cache HIT/PUT``) so existing log
  filters keep working.

What changed under the hood
---------------------------
* The OrderedDict is gone.
* ``size()`` now reports active (unexpired) row count via a single
  ``SELECT count(*)`` — adequate for the /health endpoint that calls it.
* ``clear()`` truncates the table; used only by tests and admin tools.
* Reads honour ``expires_at`` server-side so two replicas can't disagree
  about whether an entry is alive.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Tunables — kept as module constants so call sites can import them if they
# want to override (e.g. for a long-form report TTL). Same values as the
# previous in-process implementation so behaviour matches.
# ---------------------------------------------------------------------------
DEFAULT_TTL_SECONDS: int = 600     # 10 minutes
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CacheLookup:
    """Result envelope returned by ``AnswerCache.get``.

    ``hit=True`` and ``payload`` populated when the cache contained an
    unexpired entry; otherwise ``hit=False`` and ``reason`` tells you
    whether the miss was a true miss, an expiry, or an error.

    Identical shape to the previous in-process implementation so
    callers in ``backend/api.py`` continue working without changes.
    """
    hit: bool = False
    payload: Optional[Dict[str, Any]] = None
    key: str = ""
    reason: str = ""


def _normalize_query(q: str) -> str:
    """Lowercase, strip, collapse internal whitespace. Stable cache key.

    Matches the previous module's normalisation byte-for-byte so an
    upgrade with a warm DB cache (carried from a future ETL or
    re-seeded from logs) wouldn't re-key existing entries.
    """
    if not q:
        return ""
    return _WHITESPACE_RE.sub(" ", q.strip().lower())


def _build_key(
    query: str,
    owner_id: Optional[str],
    active_file_ids: Iterable[str],
) -> str:
    """sha1 of ``normalized_query || owner_id || sorted_file_ids``.

    Hash is stable across processes (no Python randomisation) so two
    replicas computing the same key produce identical 40-char hex.
    """
    nq = _normalize_query(query)
    oid = str(owner_id or "")
    files = ",".join(sorted(str(f) for f in (active_file_ids or [])))
    raw = f"{nq}||{oid}||{files}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _engine():
    """Lazy import of the SQLAlchemy engine.

    Matches the pattern used by other cache modules
    (``tier1_copilot/_shared/report_cache.py``,
    ``services/semantic_cache.py``) so this module stays importable
    in environments without a live DB (unit tests, static analysis).
    """
    from backend.db.connection import engine  # type: ignore
    return engine


# ---------------------------------------------------------------------------
# SQL statements — defined once so the SQLAlchemy text-cache can amortise
# parsing cost across requests.
# ---------------------------------------------------------------------------
_SQL_GET = text(
    """
    SELECT payload_json
      FROM answer_cache_exact
     WHERE cache_key  = :key
       AND expires_at > NOW()
     LIMIT 1
    """
)

_SQL_UPSERT = text(
    """
    INSERT INTO answer_cache_exact (cache_key, payload_json, expires_at)
    VALUES (:key, CAST(:payload AS JSONB), :expires)
    ON CONFLICT (cache_key) DO UPDATE
       SET payload_json = EXCLUDED.payload_json,
           expires_at   = EXCLUDED.expires_at
    """
)

_SQL_SIZE = text(
    "SELECT count(*) FROM answer_cache_exact WHERE expires_at > NOW()"
)

_SQL_CLEAR = text("TRUNCATE TABLE answer_cache_exact")


class AnswerCache:
    """
    Postgres-backed exact-match answer cache.

    The instance is constructed with a TTL only; the actual storage
    lives in the ``answer_cache_exact`` table. Multiple processes
    (workers, replicas) sharing the same DB share the same cache —
    that is the whole point.

    Failure mode contract
    ---------------------
    Every public method MUST swallow DB exceptions and treat the
    failure as a cache miss / put-skip. Caching is an optimisation;
    a Postgres hiccup must never break ``/ask``.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        # ttl_seconds is the only knob — no max_entries because
        # eviction is by expiry, not LRU.
        self._ttl = int(ttl_seconds)

    # ------------------------------------------------------------------
    def get(
        self,
        *,
        query: str,
        owner_id: Optional[str],
        active_file_ids: Iterable[str],
    ) -> CacheLookup:
        """Look up an active entry by exact-match key.

        Never raises. Returns ``CacheLookup(hit=False)`` on miss,
        expiry, malformed payload, or DB error — the caller's code
        path is identical for all of those cases.
        """
        try:
            key = _build_key(query, owner_id, active_file_ids)
        except Exception as exc:
            logger.warning("Answer cache GET key build failed (%s)", exc)
            return CacheLookup(hit=False, key="", reason=f"error: {exc}")

        try:
            with _engine().connect() as conn:
                row = conn.execute(_SQL_GET, {"key": key}).mappings().first()
        except Exception as exc:
            # DB unreachable / pool exhausted — fail safe to a miss
            # rather than propagating the exception into the request.
            logger.warning("Answer cache GET DB failed (%s) — treating as miss", exc)
            return CacheLookup(hit=False, key=key, reason=f"error: {exc}")

        if not row:
            return CacheLookup(hit=False, key=key, reason="miss")

        payload = row.get("payload_json")
        if not isinstance(payload, dict):
            # Corrupt row — log and ignore. The next .put() will
            # overwrite it via ON CONFLICT.
            logger.warning(
                "Answer cache GET key=%s — unexpected payload type %s",
                key[:12], type(payload).__name__,
            )
            return CacheLookup(hit=False, key=key, reason="corrupt")

        logger.info("Answer cache HIT key=%s", key[:12])
        return CacheLookup(hit=True, key=key, payload=payload, reason="hit")

    # ------------------------------------------------------------------
    def put(
        self,
        *,
        query: str,
        owner_id: Optional[str],
        active_file_ids: Iterable[str],
        payload: Dict[str, Any],
    ) -> bool:
        """UPSERT a payload at the computed key.

        Returns True on success, False on any failure. Failures are
        logged at WARNING — they're not silent — but never raised so
        the original request's response is not affected by a cache
        write hiccup.
        """
        if not isinstance(payload, dict):
            logger.warning(
                "Answer cache PUT skipped — payload not a dict (%s)",
                type(payload).__name__,
            )
            return False

        try:
            key = _build_key(query, owner_id, active_file_ids)
            expires = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
            # We pass the JSON as a string so SQLAlchemy uses ``CAST(:payload
            # AS JSONB)`` server-side. Using JSONB's native pyjsonb encoder
            # is also valid but tying the binding to plain text keeps this
            # module dependency-free of psycopg internals.
            import json as _json
            payload_str = _json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("Answer cache PUT serialise failed (%s)", exc)
            return False

        try:
            with _engine().begin() as conn:
                conn.execute(
                    _SQL_UPSERT,
                    {"key": key, "payload": payload_str, "expires": expires},
                )
        except Exception as exc:
            logger.warning("Answer cache PUT DB failed (%s) — skipping", exc)
            return False

        logger.info("Answer cache PUT key=%s", key[:12])
        return True

    # ------------------------------------------------------------------
    def size(self) -> int:
        """Active (unexpired) row count.

        Returns -1 on DB error so the /health probe surfaces the
        problem rather than reporting a misleading zero.
        """
        try:
            with _engine().connect() as conn:
                n = conn.execute(_SQL_SIZE).scalar()
                return int(n or 0)
        except Exception:
            return -1

    def clear(self) -> None:
        """Truncate the cache table. Intended for tests / admin use."""
        try:
            with _engine().begin() as conn:
                conn.execute(_SQL_CLEAR)
        except Exception as exc:
            logger.warning("Answer cache CLEAR failed (%s)", exc)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by api.py.
#
# Singleton instantiation is safe at import time because no DB I/O happens
# in ``__init__`` — the engine is resolved lazily on the first get/put.
# ---------------------------------------------------------------------------
ANSWER_CACHE = AnswerCache()
