"""Sprint 10 Stage 5 — Escalation Package wrapper.

Per spec §3.7: reuses Sprint 7's `escalation_package.build_package`
wholesale. The Sprint 10 contribution is a richer `client_what_tried`
list that includes the engineer's stage-traversal log assembled from
`tier1_journey_events`.

Output type is the existing Sprint 7 `Tier1EscalationPackage`. No
new schema. The route handler combines the cohort-derived ticket
metadata + the journey-traversal log here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text


logger = logging.getLogger("acadia-log-iq")


_STAGE_LABELS = {
    # Sprint 12 — labels aligned with the frontend STAGE_LABELS
    # constant so the escalation traversal log uses the same names
    # the engineer just saw in the UI. The "Stage N — " prefix is
    # also dropped at the format site below; we render only the
    # human-readable label.
    "stage_0": "Best Historical Match & Recommended Resolution",
    "pivot_insights": "Pivot Insights",
    # Legacy keys kept for back-compat with rows written before
    # the Sprint 10.2 1A/1B → pivot_insights merge.
    "stage_1a": "Smoking Gun",
    "stage_1b": "Do Not Chase",
    "stage_2": "Related Incidents & Probable Causes",
    "stage_3": "Guided Troubleshooting Workflow",
    "stage_4": "Knowledge Base & SOP Reference",
    "stage_5": "Operational Handoff",
}


def _format_duration(seconds: float) -> str:
    """Render an elapsed-seconds float as a compact human string:
       <60s  → "Xs"
       <1h   → "Xm Ys" (or "Xm" when seconds == 0)
       ≥1h   → "Xh Ym" (or "Xh" when minutes == 0)

    Used for per-stage time-spent in the escalation traversal log so
    Tier-2 / analytics can see how long the engineer dwelled on each
    stage in addition to the absolute timestamps.
    """
    if seconds is None or seconds < 0:
        return "0s"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec}s" if sec > 0 else f"{m}m"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m" if m2 > 0 else f"{h}h"


def compute_journey_time_metrics(session_id: str) -> Dict[str, Any]:
    """Sprint 13.26 — aggregate time-on-task signals for the Tier-2
    handoff note's opener bullets.

    Returns a dict with:
      * `stage_durations` — {stage_id: int_seconds, ...} for any
        stage with two or more events (rendered + advance).
      * `discuss_chat_count`, `discuss_chat_seconds` — Stage 0
        per-bullet "Discuss with LogIQ" chats (chat_sessions where
        `scope_incident_id` is set + `metadata.journey_session_id`
        matches this journey).
      * `kb_chat_count`, `kb_chat_seconds` — Stage 4 Search-KB
        chats (same parentage, scope_incident_id is NULL).
      * `total_journey_seconds` — last-event-minus-first-event
        across the whole tier1_journey_events row set.

    Failure-open: any DB error returns the dict with whatever
    metrics were computable; never raises.
    """
    metrics: Dict[str, Any] = {
        "stage_durations": {},
        "discuss_chat_count": 0,
        "discuss_chat_seconds": 0,
        # Sprint 13.29 — per-chat breakdown for the Discuss-with-LogIQ
        # bullet so the handoff note can name the engaged ticket(s)
        # alongside total time. Each entry: {"incident_id": str,
        # "seconds": int}. Order matches chat_sessions.created_at ASC.
        "discuss_chats": [],
        "kb_chat_count": 0,
        "kb_chat_seconds": 0,
        "total_journey_seconds": 0,
    }
    if not session_id:
        return metrics

    try:
        from backend.db.connection import engine as _engine
    except Exception as exc:
        logger.warning("[journey.metrics] engine import failed: %s", exc)
        return metrics

    # ── Per-stage durations from tier1_journey_events ──
    try:
        with _engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT stage, created_at
                    FROM tier1_journey_events
                    WHERE session_id = :sid
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"sid": session_id},
            ).mappings().all()
    except Exception as exc:
        logger.warning("[journey.metrics] events query failed: %s", exc)
        rows = []

    if rows:
        per_stage_first: Dict[str, Any] = {}
        per_stage_last: Dict[str, Any] = {}
        first_ts = rows[0].get("created_at")
        last_ts = rows[-1].get("created_at")
        for r in rows:
            st = r.get("stage")
            ts = r.get("created_at")
            if not st or ts is None:
                continue
            if st not in per_stage_first:
                per_stage_first[st] = ts
            per_stage_last[st] = ts
        for st, fts in per_stage_first.items():
            lts = per_stage_last.get(st, fts)
            try:
                d = (lts - fts).total_seconds()
                if d > 0:
                    metrics["stage_durations"][st] = int(round(d))
            except Exception:
                pass
        try:
            metrics["total_journey_seconds"] = int(
                round((last_ts - first_ts).total_seconds())
            )
        except Exception:
            pass

    # ── Chat-session aggregates: Discuss-with-LogIQ vs Search-KB ──
    # Sprint 13.28 — corrected JOIN. The journey_session_id is NOT on
    # chat_sessions (that table has no metadata_json column); it's
    # folded into the FIRST chat_messages row's sources_json under the
    # namespaced `_session_metadata` key by save_message_to_session()
    # (see backend/vector_store.py:1225-1231 and
    # backend/tier1_copilot/journey/stage4_search_kb_handoff.py:92-93).
    # The pre-13.28 query targeted a column that doesn't exist, so
    # discuss_chat_count and kb_chat_count were always 0 in the
    # Tier-2 handoff opener — even when the engineer had clearly
    # opened a Discuss-with-LogIQ chat. Fix: EXISTS sub-select against
    # chat_messages.sources_json.
    # `scope_incident_id` (a real column on chat_sessions, set by the
    # per-bullet handoff's UPDATE) still distinguishes Discuss chats
    # (set) from generic Search-KB chats (NULL).
    try:
        with _engine.connect() as conn:
            chat_rows = conn.execute(
                text(
                    """
                    SELECT cs.id::text                              AS chat_id,
                           cs.created_at                            AS started_at,
                           cs.scope_incident_id                     AS scope_incident_id,
                           (SELECT MAX(created_at)
                              FROM chat_messages
                             WHERE session_id = cs.id::text)        AS last_msg_at
                    FROM chat_sessions cs
                    WHERE EXISTS (
                        SELECT 1 FROM chat_messages cm
                        WHERE cm.session_id = cs.id::text
                          AND cm.sources_json->'_session_metadata'->>'journey_session_id'
                              = :sid
                    )
                    """
                ),
                {"sid": session_id},
            ).mappings().all()
    except Exception as exc:
        logger.warning("[journey.metrics] chat_sessions query failed: %s", exc)
        chat_rows = []

    # Order chat aggregation by start time so the handoff note's
    # per-ticket Discuss list reads in the order the engineer opened
    # each chat (oldest → newest).
    sorted_rows = sorted(
        chat_rows,
        key=lambda r: r.get("started_at") or 0,
    )
    for r in sorted_rows:
        started = r.get("started_at")
        last = r.get("last_msg_at") or started
        scope = r.get("scope_incident_id")
        try:
            secs = max(0, int(round((last - started).total_seconds())))
        except Exception:
            secs = 0
        if scope:
            metrics["discuss_chat_count"] += 1
            metrics["discuss_chat_seconds"] += secs
            metrics["discuss_chats"].append(
                {"incident_id": str(scope), "seconds": int(secs)}
            )
        else:
            metrics["kb_chat_count"] += 1
            metrics["kb_chat_seconds"] += secs

    return metrics


def fetch_traversal_log(session_id: str) -> List[Dict[str, Any]]:
    """Read tier1_journey_events for this session, return as a flat list
    of `{step, result, note}` dicts compatible with build_package's
    `client_what_tried` arg shape.

    Failure → []. Never raises.
    """
    if not session_id:
        return []
    try:
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[journey.stage5] DB engine import failed: %s", exc)
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT stage, event_type, created_at
                    FROM tier1_journey_events
                    WHERE session_id = :sid
                    ORDER BY created_at ASC, id ASC
                    """
                ),
                {"sid": session_id},
            ).mappings().all()
    except Exception as exc:
        logger.warning("[journey.stage5] traversal log fetch failed: %s", exc)
        return []

    # Per-stage roll-up: for each stage we visited, append a what_tried
    # entry. Sprint 11 expansion vs. Sprint 10:
    #   - Time format is now "%Y-%m-%d %H:%M UTC" (was "%H:%M") so the
    #     Tier-2 reader knows the timezone unambiguously. Source rows
    #     are stored as naive Postgres timestamps that already represent
    #     UTC, so this is a label change, not a value change.
    #   - For stage_4 specifically, count `kb_chat_engaged` events and
    #     surface as "opened KB chat (N exchanges)" so Tier-2 sees
    #     whether the Tier-1 actually engaged the chat (vs. clicked Open
    #     Chat and immediately escalated without trying anything).
    per_stage: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        stage = r.get("stage")
        et = r.get("event_type")
        ts = r.get("created_at")
        if not stage:
            continue
        bucket = per_stage.setdefault(stage, {"events": [], "first_seen": ts})
        bucket["events"].append((et, ts))

    out: List[Dict[str, Any]] = []
    for stage, bucket in sorted(per_stage.items(), key=lambda kv: kv[1]["first_seen"]):
        label = _STAGE_LABELS.get(stage, stage)
        events = bucket["events"]
        actions: List[str] = []
        if any(et == "stage_rendered" for et, _ in events):
            actions.append("viewed")
        if any(et == "helpful_clicked" for et, _ in events):
            actions.append("marked Helpful")
        # Sprint 11 — engagement signal for Stage 4 KB chat handoff.
        # Empty list when this stage isn't stage_4 or no engagement
        # events were posted. Renders as "opened KB chat (1 exchange)"
        # / "(N exchanges)" so Tier-2 sees the depth.
        kb_engagements = sum(1 for et, _ in events if et == "kb_chat_engaged")
        if kb_engagements > 0:
            noun = "exchange" if kb_engagements == 1 else "exchanges"
            actions.append(f"opened KB chat ({kb_engagements} {noun})")
        adv = next((ts for et, ts in events if et == "next_stage_clicked"), None)
        if adv is not None:
            try:
                # Sprint 11 — second-level precision. The earlier
                # %H:%M format rounded to the minute, which masked
                # rapid stage-to-stage transitions (an engineer who
                # clicked through 4 stages in 14s would see four
                # identical "18:24 UTC" entries). Seconds are the
                # right granularity for journey traversal.
                actions.append(f"advanced at {adv.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            except Exception:
                actions.append("advanced")
        if not actions:
            actions.append("seen")

        # Sprint 12 — per-stage time-spent. Useful for Tier-2 review and
        # for engagement analytics ("how long did the engineer actually
        # sit with each stage before advancing?"). Duration is computed
        # as (advance_ts - first_seen) when the engineer clicked the
        # next-stage CTA; otherwise as (latest_event_ts - first_seen),
        # which captures dwell time even when they didn't advance
        # (e.g., the final viewed stage before escalation).
        first_ts = bucket["first_seen"]
        last_ts = events[-1][1] if events else first_ts
        end_ts = adv if adv is not None else last_ts
        try:
            duration_sec = (end_ts - first_ts).total_seconds()
        except Exception:
            duration_sec = 0
        duration_str = _format_duration(duration_sec)

        out.append({
            # Sprint 12 — drop the "Stage N — " prefix; the label alone
            # matches what the engineer saw in the journey UI cards
            # ("Best Historical Match & Recommended Resolution",
            # "Related Incidents & Probable Causes", etc.).
            "step": label,
            # Append "/ <duration>" so the rendered line reads
            # "viewed, advanced at 2026-05-05 23:50:13 UTC / 1m 3s"
            "result": f"{', '.join(actions)} / {duration_str}",
            "note": None,
        })
    return out


def build_journey_escalation_package(
    *,
    session_id: str,
    ticket_metadata: Dict[str, Any],
    alert_payload: Dict[str, Any],
    session_what_tried: Optional[List[Dict[str, Any]]] = None,
    extra_what_tried: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Build the Sprint 10 escalation package by reusing Sprint 7's
    `build_package` with a merged what_tried list.

    Returns a `Tier1EscalationPackage` (Sprint 7 schema, unchanged).
    """
    try:
        from backend.tier1_copilot.diagnostics.escalation_package import build_package
    except Exception as exc:
        logger.error("[journey.stage5] cannot import Sprint 7 build_package: %s", exc)
        raise

    traversal = fetch_traversal_log(session_id)
    client_what_tried = list(extra_what_tried or []) + traversal

    try:
        from backend.db.connection import engine as db_engine
    except Exception:
        db_engine = None  # build_package degrades gracefully without it

    return build_package(
        ticket_metadata=ticket_metadata or {},
        alert_payload=alert_payload or {},
        session_what_tried=session_what_tried,
        client_what_tried=client_what_tried,
        engine=db_engine,
        related_incidents=None,
    )
