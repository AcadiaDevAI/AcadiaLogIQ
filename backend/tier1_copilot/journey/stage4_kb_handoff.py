"""Sprint 10 Stage 4 — Search KB / SOP handoff.

Per spec §3.6: Stage 4 hands off to the existing chat (`/ask`) with a
pre-filled first message. The frontend dispatches that message into
`ChatContext` via `ADD_USER_MESSAGE` and route-transitions to ChatArea.
The `allowed_doc_kinds` field tells the chat backend to prefer SOP / KB
chunks over historical tickets for this turn (Sprint 3-PREP-B
mechanism, exposed via the optional `allowed_doc_kinds` request param
that step 15 of Phase 2 will add to /ask).

Pure assembly — no DB, no LLM. The dominant_root_cause string is sourced
from Stage 0's profile_match (the modal cohort signature); the alert
payload comes from the engineer's intake form.
"""
from __future__ import annotations

from typing import Optional

from .schemas import Stage4SearchKB


def _truncate(s: str, max_chars: int = 240) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def build_stage4(
    *,
    severity: Optional[str],
    asset_name: Optional[str],
    alert_type: Optional[str],
    notes: Optional[str] = None,
    dominant_root_cause: Optional[str] = None,
) -> Stage4SearchKB:
    """Assemble the prefilled chat message per spec §3.6.

    Format:
        Severity {sev} — {alert_type} on {asset_name}.
        Past tickets suggest {dominant_root_cause}.
        {notes (if provided)}
    """
    sev = (severity or "P3").strip()
    alert = (alert_type or "(unspecified alert)").strip()
    asset = (asset_name or "(unspecified asset)").strip()

    line1 = f"Severity {sev} — {alert} on {asset}."
    parts = [line1]

    if dominant_root_cause and dominant_root_cause.strip():
        parts.append(f"Past tickets suggest {_truncate(dominant_root_cause.strip())}.")

    if notes and notes.strip():
        parts.append(notes.strip())

    return Stage4SearchKB(
        prefilled_message="\n".join(parts).strip(),
        allowed_doc_kinds=["sop", "kb"],
    )
