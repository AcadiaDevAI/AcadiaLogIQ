"""
Retrieval shadow-diff logger.

During the BM25 → FTS migration (Phase 0.1) every retrieval call
records the top-K chunk IDs from BOTH channels into ``retrieval_eval``
so we can answer "do the two channels agree on the same incidents?"
with one SQL query.

Decision pipeline
-----------------
1. After ~2 weeks of writes, run::

     SELECT
         date_trunc('day', created_at) AS day,
         count(*)                       AS n,
         avg(overlap::float / NULLIF(bm25_size,0))  AS recall,
         avg(overlap::float / NULLIF(fts_size,0))   AS precision
     FROM retrieval_eval
     WHERE created_at > NOW() - INTERVAL '14 days'
     GROUP BY 1 ORDER BY 1;

2. If ``recall >= 0.85`` and ``precision >= 0.85`` for every day,
   the migration is safe.
3. Flip ``RETRIEVAL_BM25_ENABLED=false`` in Secrets Manager. BM25
   stops being queried; the channel still exists but always returns
   ``[]``.
4. Optionally flip ``RETRIEVAL_SHADOW_LOG_ENABLED=false`` to stop
   the writes once you've stopped reading them.
5. After 30 days of clean prod operation, delete the BM25 module
   from the codebase (one-way door).

Design contract
---------------
* Logging is FAIL-OPEN. A DB error here MUST NOT propagate to the
  request that triggered it — caching is an optimisation; retrieval
  measurement is one level removed and must be even less intrusive.
* All work runs in a single ``INSERT`` so latency added to the
  request is ~1 ms.
* Caller is responsible for passing already-sorted top-K lists; we
  don't re-rank here.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


_SQL_INSERT = text(
    """
    INSERT INTO retrieval_eval
        (query_text, file_set_size, bm25_top_k, fts_top_k,
         overlap, bm25_size, fts_size)
    VALUES
        (:q, :n_files, CAST(:bm25 AS JSONB), CAST(:fts AS JSONB),
         :overlap, :bm25_n, :fts_n)
    """
)


def _engine():
    """Lazy import — keeps this module testable without a live DB."""
    from backend.db.connection import engine  # type: ignore
    return engine


def _normalise_id_list(items: Iterable) -> List[str]:
    """Coerce retrieved-result entries to a flat list of chunk ID strings.

    The orchestrator's per-channel result lists are tuples or dicts
    depending on the channel. The shadow log only cares about chunk
    IDs in rank order, so we extract them defensively.
    """
    out: List[str] = []
    for entry in items or []:
        cid: Optional[str] = None
        if isinstance(entry, dict):
            cid = entry.get("id") or entry.get("chunk_id")
        elif isinstance(entry, tuple) and len(entry) >= 1:
            cid = entry[0]
        if cid:
            out.append(str(cid))
    return out


def log_diff(
    *,
    query: str,
    bm25_results: Iterable,
    fts_results: Iterable,
    file_set_size: int = 0,
    top_k: int = 10,
) -> None:
    """Insert a single row comparing BM25 vs FTS for one query.

    Fails open — any DB error is logged at WARNING and swallowed.
    """
    try:
        bm25_ids = _normalise_id_list(bm25_results)[:top_k]
        fts_ids = _normalise_id_list(fts_results)[:top_k]
        overlap = len(set(bm25_ids) & set(fts_ids))

        # Bail without writing if BOTH channels returned nothing —
        # the row would be vacuous and wasteful at scale.
        if not bm25_ids and not fts_ids:
            return

        with _engine().begin() as conn:
            conn.execute(
                _SQL_INSERT,
                {
                    "q": (query or "")[:2000],  # safety cap
                    "n_files": int(file_set_size or 0),
                    "bm25": json.dumps(bm25_ids),
                    "fts": json.dumps(fts_ids),
                    "overlap": int(overlap),
                    "bm25_n": len(bm25_ids),
                    "fts_n": len(fts_ids),
                },
            )
    except Exception as exc:
        # Shadow logging must NEVER break a real request. A DB hiccup
        # here just means a missing row in the migration verdict —
        # not a user-visible failure.
        logger.warning("[retrieval_eval] log_diff failed (%s)", exc)
