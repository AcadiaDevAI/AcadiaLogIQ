"""
Report cache + feedback store.

Used by both RCA (``backend/tier1_copilot/rca/``) and Gap Analysis
(``backend/tier1_copilot/gap_analysis/``) to:

  1. Read the most recent cached Markdown for a given
     (report_kind, incident_number) pair before invoking an LLM.
  2. Persist a freshly-generated Markdown so the next request for
     the same pair returns instantly.
  3. Record 👍 / 👎 feedback events to an append-only audit table.
  4. Invalidate (delete) a cached row when a 👎 fires, so the next
     "Generate" call runs the LLM and replaces the bad output.

Design intent
-------------
* **Per-incident, global scope** — two engineers viewing the same
  incident see the same cached report. The 👎 signal is a public
  vote: one dislike marks the cache stale for everyone.
* **One 👎 = delete-row** (not flag-as-stale). Next generate hits
  cache-miss → LLM → UPSERT writes the fresh row. Simpler than a
  ``stale BOOLEAN`` column and impossible to leak (no chance of a
  stale-but-not-marked row sneaking through).
* **Feedback is append-only.** Likes and dislikes both write a row
  in ``report_feedback``. Cache deletion never touches that log so
  analytics queries see the full history (which engineer disliked
  which incident's report and when).
* **Failure-open.** Every public function in this module returns
  on DB error with a warning log; the caller behaves as though the
  cache simply missed (and re-runs the LLM). The cache is an
  optimisation, never a correctness gate.

DB schema (migration 043_report_cache.sql)
------------------------------------------
``report_cache``::

    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_kind     VARCHAR(64) NOT NULL,
    incident_number VARCHAR(128) NOT NULL,
    markdown        TEXT NOT NULL,
    model_id        VARCHAR(128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (report_kind, incident_number)

``report_feedback``::

    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_kind     VARCHAR(64) NOT NULL,
    incident_number VARCHAR(128) NOT NULL,
    feedback_type   VARCHAR(16) NOT NULL   -- 'like' | 'dislike'
    feedback_by     VARCHAR(256),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()

Public API
----------
* ``get_cached_report(report_kind, incident_number) -> str | None``
* ``save_cached_report(report_kind, incident_number, markdown, model_id) -> None``
* ``invalidate_cached_report(report_kind, incident_number) -> bool``
* ``record_feedback(report_kind, incident_number, feedback_type, user_id) -> bool``

The constants ``REPORT_KIND_*`` enumerate the four panels that
exist today; callers should prefer the constants to string literals
so a rename can be done in one place.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# ── Whitelist of valid report_kind strings ─────────────────────────
# Callers MUST use these constants. The DB column is free-form text
# (so adding a new kind doesn't need a migration), but the
# application layer keeps a strict whitelist to prevent typos from
# silently sharding the cache by accident.
REPORT_KIND_RCA_CUSTOMER = "rca_customer"
REPORT_KIND_RCA_INTERNAL = "rca_internal"
REPORT_KIND_GAP_ANALYSIS = "gap_analysis_gap"
REPORT_KIND_POST_MORTEM = "gap_analysis_pm"

_VALID_REPORT_KINDS = frozenset({
    REPORT_KIND_RCA_CUSTOMER,
    REPORT_KIND_RCA_INTERNAL,
    REPORT_KIND_GAP_ANALYSIS,
    REPORT_KIND_POST_MORTEM,
})

# Valid feedback strings — must match the CHECK constraint in
# migration 043. Same whitelist guard rationale.
FEEDBACK_LIKE = "like"
FEEDBACK_DISLIKE = "dislike"
_VALID_FEEDBACK = frozenset({FEEDBACK_LIKE, FEEDBACK_DISLIKE})


def _normalise_incident(inc: str) -> str:
    """Trim and validate. Empty input is caller error — raise."""
    s = (inc or "").strip()
    if not s:
        raise ValueError("incident_number required")
    return s


def _engine():
    """Lazy-import the SQLAlchemy engine. Mirrors the pattern used
    by ``rca/ticket_lookup.py`` so this module stays importable
    even when there's no live DB (unit tests, --check-only runs).
    """
    from backend.db.connection import engine  # type: ignore
    return engine


# ── READ ─────────────────────────────────────────────────────────


def get_cached_report(
    report_kind: str,
    incident_number: str,
) -> Optional[str]:
    """Return the cached Markdown for ``(report_kind, incident_number)``
    or ``None`` on cache miss, DB error, or unknown report_kind.

    Never raises. A miss and an error are indistinguishable to the
    caller on purpose — the caller's recovery path is the same:
    invoke the LLM and call ``save_cached_report`` afterwards.
    """
    if report_kind not in _VALID_REPORT_KINDS:
        logger.warning(
            "[report_cache] unknown report_kind=%r — treating as miss",
            report_kind,
        )
        return None
    try:
        inc = _normalise_incident(incident_number)
    except ValueError:
        return None

    sql = text(
        """
        SELECT markdown
          FROM report_cache
         WHERE report_kind = :kind
           AND incident_number = :inc
         LIMIT 1
        """
    )
    try:
        with _engine().connect() as conn:
            row = conn.execute(
                sql, {"kind": report_kind, "inc": inc},
            ).mappings().first()
    except Exception as exc:
        logger.warning(
            "[report_cache] read failed kind=%s inc=%s err=%s",
            report_kind, inc, exc,
        )
        return None

    if not row:
        return None
    md = row.get("markdown")
    if not isinstance(md, str) or not md.strip():
        return None
    return md


# ── WRITE ────────────────────────────────────────────────────────


def save_cached_report(
    report_kind: str,
    incident_number: str,
    markdown: str,
    model_id: Optional[str] = None,
) -> None:
    """UPSERT a freshly-generated Markdown blob.

    Conflict resolution: on the ``(report_kind, incident_number)``
    unique constraint, the existing row is overwritten and
    ``updated_at`` advances to ``NOW()``. The combination of "👎
    deletes the row" + "save UPSERTs" means a single dislike
    cleanly produces one fresh row on the next generate.

    Silent on validation failure / DB error — caching is an
    optimisation and must never poison the request that succeeded.
    """
    if report_kind not in _VALID_REPORT_KINDS:
        logger.warning(
            "[report_cache] write skipped — unknown report_kind=%r",
            report_kind,
        )
        return
    if not isinstance(markdown, str) or not markdown.strip():
        return
    try:
        inc = _normalise_incident(incident_number)
    except ValueError:
        return

    sql = text(
        """
        INSERT INTO report_cache (report_kind, incident_number, markdown, model_id)
        VALUES (:kind, :inc, :md, :model_id)
        ON CONFLICT (report_kind, incident_number)
        DO UPDATE SET
            markdown   = EXCLUDED.markdown,
            model_id   = EXCLUDED.model_id,
            updated_at = NOW()
        """
    )
    try:
        with _engine().begin() as conn:
            conn.execute(
                sql,
                {
                    "kind": report_kind,
                    "inc": inc,
                    "md": markdown,
                    "model_id": (model_id or "").strip() or None,
                },
            )
    except Exception as exc:
        logger.warning(
            "[report_cache] write failed kind=%s inc=%s err=%s",
            report_kind, inc, exc,
        )


# ── INVALIDATE ───────────────────────────────────────────────────


def invalidate_cached_report(
    report_kind: str,
    incident_number: str,
) -> bool:
    """Delete the cached row, if any. Returns ``True`` when a row
    was actually removed.

    Two callers:
      * The 👎 feedback handler — invalidate after recording the
        dislike so the next "Generate" produces a fresh report.
      * The frontend's "Regenerate" button — invalidate without
        recording a dislike so the same engineer can refresh a
        report they otherwise rated fine.
    """
    if report_kind not in _VALID_REPORT_KINDS:
        return False
    try:
        inc = _normalise_incident(incident_number)
    except ValueError:
        return False

    sql = text(
        """
        DELETE FROM report_cache
         WHERE report_kind = :kind
           AND incident_number = :inc
        """
    )
    try:
        with _engine().begin() as conn:
            res = conn.execute(sql, {"kind": report_kind, "inc": inc})
            removed = (res.rowcount or 0) > 0
    except Exception as exc:
        logger.warning(
            "[report_cache] invalidate failed kind=%s inc=%s err=%s",
            report_kind, inc, exc,
        )
        return False

    if removed:
        logger.info(
            "[report_cache] invalidated kind=%s inc=%s",
            report_kind, inc,
        )
    return removed


# ── FEEDBACK ─────────────────────────────────────────────────────


def record_feedback(
    report_kind: str,
    incident_number: str,
    feedback_type: str,
    user_id: Optional[str] = None,
) -> bool:
    """Append a 👍 or 👎 event to ``report_feedback``.

    Returns ``True`` on success (event was logged). Returns
    ``False`` on validation failure or DB error — the route layer
    decides whether to surface that to the user.

    NOTE: This function does NOT invalidate the cache; the caller
    is responsible for chaining ``invalidate_cached_report()`` when
    ``feedback_type == 'dislike'``. Keeping the two operations
    independent here means analytics queries always see the
    complete feedback history regardless of cache TTL strategy.
    """
    if report_kind not in _VALID_REPORT_KINDS:
        logger.warning(
            "[report_cache] feedback rejected — unknown report_kind=%r",
            report_kind,
        )
        return False
    if feedback_type not in _VALID_FEEDBACK:
        logger.warning(
            "[report_cache] feedback rejected — bad feedback_type=%r",
            feedback_type,
        )
        return False
    try:
        inc = _normalise_incident(incident_number)
    except ValueError:
        return False

    sql = text(
        """
        INSERT INTO report_feedback
            (report_kind, incident_number, feedback_type, feedback_by)
        VALUES
            (:kind, :inc, :ft, :user_id)
        """
    )
    try:
        with _engine().begin() as conn:
            conn.execute(
                sql,
                {
                    "kind": report_kind,
                    "inc": inc,
                    "ft": feedback_type,
                    "user_id": (user_id or "").strip() or None,
                },
            )
    except Exception as exc:
        logger.warning(
            "[report_cache] feedback insert failed kind=%s inc=%s err=%s",
            report_kind, inc, exc,
        )
        return False

    logger.info(
        "[report_cache] feedback kind=%s inc=%s type=%s by=%s",
        report_kind, inc, feedback_type, user_id or "(anon)",
    )
    return True
