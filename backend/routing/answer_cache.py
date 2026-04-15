"""
Answer Cache — thread-safe in-memory TTL + LRU cache for /ask answers.
Keyed on (normalized_query, owner_id, sorted active_file_ids). All ops
are fail-safe: any exception is swallowed and treated as a cache miss.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_TTL_SECONDS: int = 600     # 10 minutes
DEFAULT_MAX_ENTRIES: int = 500     # LRU-evict above this
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CacheLookup:
    """hit=True with `payload` on a cache hit; otherwise hit=False."""
    hit: bool = False
    payload: Optional[Dict[str, Any]] = None
    key: str = ""
    reason: str = ""


def _normalize_query(q: str) -> str:
    """Lowercase, strip, collapse internal whitespace. Stable cache key."""
    if not q:
        return ""
    return _WHITESPACE_RE.sub(" ", q.strip().lower())


def _build_key(query: str, owner_id: Optional[str], active_file_ids: Iterable[str]) -> str:
    """Build a short sha1 key from the normalized components."""
    nq = _normalize_query(query)
    oid = str(owner_id or "")
    files = ",".join(sorted(str(f) for f in (active_file_ids or [])))
    raw = f"{nq}||{oid}||{files}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    """
    Thread-safe LRU + TTL cache. Never raises on get/put — failures are
    logged and treated as a miss so the request proceeds normally.
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def get(
        self,
        *,
        query: str,
        owner_id: Optional[str],
        active_file_ids: Iterable[str],
    ) -> CacheLookup:
        """Return CacheLookup. Never raises."""
        try:
            key = _build_key(query, owner_id, active_file_ids)
            with self._lock:
                entry = self._store.get(key)
                if entry is None:
                    return CacheLookup(hit=False, key=key, reason="miss")
                ts, payload = entry
                if (time.time() - ts) > self._ttl:
                    # Expired — evict
                    self._store.pop(key, None)
                    return CacheLookup(hit=False, key=key, reason="expired")
                # Move to MRU end
                self._store.move_to_end(key)
                logger.info("Answer cache HIT key=%s", key[:12])
                return CacheLookup(hit=True, key=key, payload=payload, reason="hit")
        except Exception as exc:
            logger.warning("Answer cache GET failed (%s) — treating as miss", exc)
            return CacheLookup(hit=False, key="", reason=f"error: {exc}")

    # ------------------------------------------------------------------
    def put(
        self,
        *,
        query: str,
        owner_id: Optional[str],
        active_file_ids: Iterable[str],
        payload: Dict[str, Any],
    ) -> bool:
        """Store a payload. Returns True on success, False on failure."""
        try:
            key = _build_key(query, owner_id, active_file_ids)
            with self._lock:
                self._store[key] = (time.time(), payload)
                self._store.move_to_end(key)
                while len(self._store) > self._max:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug("Answer cache evicted LRU key=%s", evicted_key[:12])
            logger.info("Answer cache PUT key=%s size=%d", key[:12], len(self._store))
            return True
        except Exception as exc:
            logger.warning("Answer cache PUT failed (%s) — skipping", exc)
            return False

    # ------------------------------------------------------------------
    def size(self) -> int:
        try:
            with self._lock:
                return len(self._store)
        except Exception:
            return -1

    def clear(self) -> None:
        try:
            with self._lock:
                self._store.clear()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton — imported by api.py
# ---------------------------------------------------------------------------
ANSWER_CACHE = AnswerCache()
