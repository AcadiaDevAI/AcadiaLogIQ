"""Sprint 13.19 — Tier-2 Escalation Handoff Note generator (deterministic).

Architectural reset: the prior LLM-driven generator was hallucinating
diagnostic checks pulled from cohort context even when the engineer
had never opened the Guided Troubleshooting Workflow or ticked any
boxes. Every section of this note is structured data with a 1:1
mapping to text — there is no synthesis task left for an LLM to add
value to. So this module now does pure template-fill and returns the
same shape (`(note: str, used_fallback: bool)`) the route expects.

Inputs the route gathers and passes in:
  * cohort               — for incident-number list
  * attempted_steps      — Stage 3 consolidated steps the engineer
                           ACTUALLY ticked (filtered upstream by the
                           POST body's `attempted_step_numbers`).
                           When the list is empty, no diagnostic
                           bullets render.
  * stage_3_visited      — bool from traversal_log.
  * stage_4_visited      — bool from traversal_log.
  * kb_chat_engaged      — bool (whether engineer chatted with the
                           KB after opening Stage 4).
  * routing              — Stage 5 routing aggregation.
  * contacts_payload     — affected_customer + customer_contacts +
                           vendor_contacts + directory_contacts.

Output rules — strict, no LLM:
  * Diagnostic Summary section adapts to the engineer's actual
    journey:
      - Stage 3 visited AND ≥ 1 step ticked → bulleted list of
        ticked steps' action text.
      - Stage 3 visited AND zero ticks       → "Tier 1 reviewed the
        Guided Troubleshooting Workflow but did not mark any
        individual steps as attempted."
      - Stage 3 NOT visited                   → "Tier 1 did not open
        the Guided Troubleshooting Workflow during this triage."
  * Stage 4 / KB chat status appended as a short coverage note when
    the engineer didn't open Search KB / SOP or didn't engage the
    chat — so Tier-2 sees gaps explicitly.
  * Reason for Escalation, Routing & Vendor block, Contact Details
    block — bullets, deterministic.

`used_fallback` always returns False under this pure-deterministic
build (the prior flag tracked LLM failure; with no LLM there's no
failure mode to flag). Kept on the response for back-compat.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .schemas import ConsolidatedStep, EscalationRouting
from .stage2_historical import _flatten, _safe_get


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Source extraction helpers
# ─────────────────────────────────────────────────────────────
def _coerce_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _extract_incident_numbers(cohort: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for ticket in cohort or []:
        if not isinstance(ticket, dict):
            continue
        inc = _coerce_str(_safe_get(ticket, "Metadata", "Incident_Number"))
        if inc and inc not in seen:
            seen.add(inc)
            out.append(inc)
    return out


# ─────────────────────────────────────────────────────────────
# Block builders — each one is independent and never invents data
# ─────────────────────────────────────────────────────────────
def _format_duration_secs(seconds: int) -> str:
    """Local copy of stage5_escalation._format_duration so the handoff
    note module stays import-light (no circular dep risk)."""
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


def _opening_paragraph(
    incidents: List[str],
    stage_2_visited: bool,
    stage_3_visited: bool,
    stage_4_visited: bool,
    kb_chat_engaged: bool,
    time_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Sprint 13.20 / 13.26 — dynamic bullet-formatted opener.

    Each clause that used to be in a single prose sentence is now its
    own bullet, with time-on-task data appended when available
    (stage durations from tier1_journey_events; chat-session totals
    from chat_sessions joined by journey_session_id metadata).

    Time data sources:
      * Stage 3 (Guided Troubleshooting Workflow) duration
      * Stage 4 (Search KB / SOP Reference) duration
      * Discuss-with-Logic chat sessions (count + total seconds)
      * Search-KB chat sessions (count + total seconds)
    """
    metrics = time_metrics or {}
    stage_durations: Dict[str, int] = metrics.get("stage_durations", {}) or {}

    def _dur_for(stage_id: str) -> Optional[str]:
        secs = stage_durations.get(stage_id)
        if secs and secs > 0:
            return _format_duration_secs(int(secs))
        return None

    discuss_count = int(metrics.get("discuss_chat_count", 0) or 0)
    discuss_secs = int(metrics.get("discuss_chat_seconds", 0) or 0)
    # Sprint 13.29 — per-chat breakdown so the Discuss bullet can name
    # the engaged ticket(s). Falls back to [] when missing for back-compat.
    discuss_chats = metrics.get("discuss_chats") or []
    kb_count = int(metrics.get("kb_chat_count", 0) or 0)
    kb_secs = int(metrics.get("kb_chat_seconds", 0) or 0)
    total_secs = int(metrics.get("total_journey_seconds", 0) or 0)

    # Set of incident IDs the engineer engaged via Discuss-with-Logic
    # chat. Powers the historical bullet's engagement-aware phrasing —
    # opening a per-ticket chat IS opening that ticket "in detail" even
    # when Stage 2's panel was never visited.
    engaged_via_chat = {
        str(c.get("incident_id") or "").strip()
        for c in discuss_chats
        if c.get("incident_id")
    }
    engaged_via_chat.discard("")

    bullets: List[str] = []
    bullets.append("- Tier 1 has completed initial triage.")

    # Sprint 13.28 — bullet order rearranged at user's request to mirror
    # the engineer's natural triage flow:
    #   1. triage complete (always)
    #   2. historical tickets (Stage 0 / 2 / pivot — surfaced first)
    #   3. discuss with logic (per-bullet chats off the historical list)
    #   4. guided troubleshooting workflow (Stage 3)
    #   5. search KB / SOP reference (Stage 4 — last-resort lookup)
    #   6. total time
    # The clauses themselves are unchanged; only the order is different.

    # ── 2. Historical tickets ────────────────────────────────────
    # Sprint 13.28 — append `(Xs across context panels)` time when
    # available. The "context panels" cover any stage that surfaces
    # the cohort to the engineer: stage_0 (Best Historical Match),
    # pivot_insights (Pivot Insights), and stage_2 (Related
    # Incidents). We sum durations for whichever of these the
    # engineer actually visited so the bullet reflects how long they
    # studied the historical evidence — even when stage_2 itself was
    # never opened in detail.
    history_secs = 0
    for st in ("stage_0", "pivot_insights", "stage_2"):
        v = stage_durations.get(st)
        if isinstance(v, int) and v > 0:
            history_secs += v
    history_dur = (
        _format_duration_secs(history_secs) if history_secs > 0 else None
    )
    inc_str = ", ".join(incidents) if incidents else ""

    # Sprint 13.29 — engagement-aware phrasing. If the engineer
    # opened a Discuss-with-Logic chat on at least one of the cohort
    # tickets, that ticket WAS opened in detail — even if Stage 2's
    # panel was never visited. Cohort-restricted set so a stray chat
    # scoped to a non-cohort ticket doesn't change the framing.
    chat_engaged_in_cohort = sorted(
        engaged_via_chat.intersection({str(i) for i in incidents})
    )

    # Sprint 13.29 — multi-line layout to fix wrap-alignment in the
    # <pre>-rendered note. With pre-wrap and word-break, a long
    # single-line bullet wraps to column 0 (no hanging indent), so
    # the time suffix appears unaligned with the bullet text. Splitting
    # the bullet across a header line + 2-space-indented continuation
    # lines makes alignment explicit regardless of viewport width.
    if stage_2_visited:
        header = "- Similar historical tickets reviewed in detail:"
    elif chat_engaged_in_cohort:
        engaged_str = ", ".join(chat_engaged_in_cohort)
        header = (
            "- Historical tickets surfaced for context "
            f"(engaged via per-ticket chat for {engaged_str}):"
        )
    else:
        header = (
            "- Historical tickets surfaced for context "
            "(not opened in detail):"
        )
    if inc_str:
        block = [header, f"  {inc_str}"]
        if history_dur:
            block.append(f"  ({history_dur} across context panels)")
        bullets.append("\n".join(block))
    else:
        if history_dur:
            bullets.append(
                f"{header[:-1]} ({history_dur} across context panels)."
            )
        else:
            bullets.append(header[:-1] + ".")

    # ── 3. Discuss with Logic — per-bullet ticket chats ──────────
    # Sprint 13.29 — name the engaged ticket(s) alongside their
    # individual durations. Single chat → "INC-X (1m 21s)";
    # multi-chat → "INC-A (1m 21s), INC-B (45s); 2m 6s total".
    if discuss_count > 0 and discuss_chats:
        per_chat = ", ".join(
            f"{str(c.get('incident_id') or '?')} "
            f"({_format_duration_secs(int(c.get('seconds') or 0))})"
            for c in discuss_chats
        )
        if discuss_count == 1:
            bullets.append(
                f"- Discuss with Logic: per-ticket chat for {per_chat}."
            )
        else:
            total_str = (
                f"; {_format_duration_secs(discuss_secs)} total"
                if discuss_secs > 0 else ""
            )
            bullets.append(
                f"- Discuss with Logic: {discuss_count} per-ticket chat "
                f"sessions — {per_chat}{total_str}."
            )
    elif discuss_count > 0:
        # Defensive fallback: per-chat list missing but counts present.
        # Shouldn't happen with the 13.29 metrics, but keeps older
        # callers / pre-migration data renderable.
        dur_str = (
            f", {_format_duration_secs(discuss_secs)} total"
            if discuss_secs > 0 else ""
        )
        bullets.append(
            f"- Discuss with Logic: {discuss_count} per-ticket chat session"
            + ("s" if discuss_count != 1 else "")
            + dur_str + "."
        )
    else:
        bullets.append(
            "- Discuss with Logic: no per-ticket chat sessions opened."
        )

    # ── 4. Guided Troubleshooting Workflow ───────────────────────
    stage3_dur = _dur_for("stage_3")
    if stage_3_visited:
        bullets.append(
            "- Guided Troubleshooting Workflow: opened"
            + (f" ({stage3_dur} on the panel)" if stage3_dur else "")
            + "."
        )
    else:
        bullets.append(
            "- Guided Troubleshooting Workflow: NOT opened during this triage."
        )

    # ── 5. Search KB / SOP reference ─────────────────────────────
    stage4_dur = _dur_for("stage_4")
    if stage_4_visited and kb_chat_engaged:
        bits = ["consulted"]
        if stage4_dur:
            bits.append(f"{stage4_dur} on the panel")
        if kb_count > 0 and kb_secs > 0:
            bits.append(
                f"{kb_count} chat session"
                + ("s" if kb_count != 1 else "")
                + f", {_format_duration_secs(kb_secs)} total"
            )
        elif kb_count > 0:
            bits.append(f"{kb_count} chat session" + ("s" if kb_count != 1 else ""))
        bullets.append(
            f"- Search KB / SOP reference: {' (' .join([bits[0]] + [', '.join(bits[1:])]) + ')' if len(bits) > 1 else bits[0]}."
        )
    elif stage_4_visited:
        bits = ["opened, no chat engaged"]
        if stage4_dur:
            bits.append(f"{stage4_dur} on the panel")
        bullets.append(
            f"- Search KB / SOP reference: {bits[0]}"
            + (f" ({bits[1]})" if len(bits) > 1 else "")
            + "."
        )
    else:
        bullets.append(
            "- Search KB / SOP reference: NOT consulted during this triage."
        )

    # ── 6. Total journey time (when meaningfully > 0) ────────────
    if total_secs > 0:
        bullets.append(
            f"- Total time on this triage: {_format_duration_secs(total_secs)}."
        )

    return (
        "Please review this ticket for advanced intervention. The "
        "triage activity summary is below:\n"
        + "\n".join(bullets)
    )


def _diagnostic_block(
    attempted_steps: List[ConsolidatedStep],
    stage_3_visited: bool,
    stage_4_visited: bool,
    kb_chat_engaged: bool,
) -> str:
    """Diagnostic Summary — strictly mirrors the engineer's journey.

    Bullets ONLY render for steps the engineer explicitly ticked;
    coverage gaps for Stage 3 / Stage 4 / KB chat are appended as
    short status sentences so Tier-2 sees what was and wasn't done.
    """
    lines: List[str] = []

    # Primary line — Stage 3 ticks
    if not stage_3_visited:
        lines.append(
            "Tier 1 did not open the Guided Troubleshooting Workflow "
            "during this triage."
        )
    elif not attempted_steps:
        lines.append(
            "Tier 1 reviewed the Guided Troubleshooting Workflow but "
            "did not mark any individual steps as attempted."
        )
    else:
        lines.append(
            "Tier 1 has already reviewed/attempted the following "
            "diagnostic checks:"
        )
        for s in attempted_steps:
            lines.append(f"- {s.action}")

    # Coverage gap — Stage 4 (Search KB / SOP)
    if not stage_4_visited:
        lines.append("")
        lines.append(
            "- Search KB / SOP reference was not consulted during "
            "this triage."
        )
    elif not kb_chat_engaged:
        lines.append("")
        lines.append(
            "- Search KB / SOP reference was opened, but no chat "
            "was engaged with it."
        )

    return "\n".join(lines)


def _routing_block(routing: Optional[EscalationRouting]) -> str:
    """Escalation Routing & Vendor/OEM Engagement section."""
    lines: List[str] = ["*Escalation Routing & Vendor/OEM Engagement:*"]
    if routing is None or routing.empty:
        lines.append("- Resolution Groups: None recorded in this cohort.")
        lines.append("- Recommended Tier-2 entry: Not determinable from cohort.")
        lines.append("- Historical team paths: None recorded.")
        lines.append(
            "- Forensic data required before vendor/OEM engagement: "
            "No vendor/OEM engagement records found in this cohort."
        )
        return "\n".join(lines)

    if routing.resolution_groups:
        lines.append(
            f"- Resolution Groups: {', '.join(routing.resolution_groups)}"
        )
    else:
        lines.append("- Resolution Groups: None recorded in this cohort.")

    if routing.recommended_tier2_teams:
        teams = ", ".join(
            f"{t.team} (×{t.occurrence_count})" if t.occurrence_count > 1 else t.team
            for t in routing.recommended_tier2_teams
        )
        lines.append(f"- Recommended Tier-2 entry: {teams}")
    else:
        lines.append("- Recommended Tier-2 entry: Not determinable from cohort.")

    if routing.team_paths:
        lines.append("- Historical team paths:")
        for p in routing.team_paths:
            lines.append(f"  - {p}")
    else:
        lines.append("- Historical team paths: None recorded.")

    if routing.forensic_data_required:
        lines.append(
            f"- Forensic data required before vendor/OEM engagement: "
            f"{', '.join(routing.forensic_data_required)}"
        )
    else:
        lines.append(
            "- Forensic data required before vendor/OEM engagement: "
            "No vendor/OEM engagement records found in this cohort."
        )
    return "\n".join(lines)


def _contact_row(contact: Any) -> Optional[str]:
    """Render a Tier1Contact-like object as 'Name (Role) — phone — email'."""
    if contact is None:
        return None
    if hasattr(contact, "model_dump"):
        c = contact.model_dump()
    elif isinstance(contact, dict):
        c = contact
    else:
        return None
    name = (c.get("name") or "").strip()
    role = (c.get("role") or "").strip()
    phone = (c.get("phone") or "").strip()
    email = (c.get("email") or "").strip()
    if not (name or role or phone or email):
        return None
    pieces: List[str] = []
    if name:
        pieces.append(name + (f" ({role})" if role else ""))
    elif role:
        pieces.append(role)
    if phone:
        pieces.append(phone)
    if email:
        pieces.append(email)
    return " — ".join(pieces) if pieces else None


def _directory_row(contact: Any) -> Optional[str]:
    """Render a Tier1DirectoryContact as '(Label) Name — detail'."""
    if contact is None:
        return None
    if hasattr(contact, "model_dump"):
        c = contact.model_dump()
    elif isinstance(contact, dict):
        c = contact
    else:
        return None
    label = (c.get("label") or "").strip()
    name = (c.get("name") or "").strip()
    detail = (c.get("detail") or "").strip()
    if not (label or name or detail):
        return None
    pieces: List[str] = []
    if label and name:
        pieces.append(f"({label}) {name}")
    elif name:
        pieces.append(name)
    elif label:
        pieces.append(f"({label})")
    if detail:
        pieces.append(detail)
    return " — ".join(pieces) if pieces else None


def _contacts_block(contacts_payload: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = ["*Contact Details:*"]
    payload = contacts_payload or {}

    affected = (payload.get("affected_customer") or "").strip()
    lines.append(
        f"- Affected Customer: {affected if affected else 'Not specified'}"
    )

    customer = payload.get("customer_contacts") or []
    customer_rows = [r for r in (_contact_row(c) for c in customer) if r]
    if customer_rows:
        lines.append("- Customer Contacts:")
        for r in customer_rows:
            lines.append(f"  - {r}")
    else:
        lines.append("- Customer Contacts: None recorded.")

    vendor = payload.get("vendor_contacts") or []
    vendor_rows = [r for r in (_contact_row(c) for c in vendor) if r]
    if vendor_rows:
        lines.append("- Vendor Contacts:")
        for r in vendor_rows:
            lines.append(f"  - {r}")
    else:
        lines.append("- Vendor Contacts: None recorded.")

    directory = payload.get("directory_contacts") or []
    directory_rows = [r for r in (_directory_row(c) for c in directory) if r]
    if directory_rows:
        lines.append("- Internal Directory:")
        for r in directory_rows:
            lines.append(f"  - {r}")
    else:
        lines.append("- Internal Directory: None recorded.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def generate_handoff_note(
    cohort: List[Dict[str, Any]],
    *,
    attempted_steps: Optional[List[ConsolidatedStep]] = None,
    stage_2_visited: bool = False,
    stage_3_visited: bool = False,
    stage_4_visited: bool = False,
    kb_chat_engaged: bool = False,
    routing: Optional[EscalationRouting] = None,
    contacts_payload: Optional[Dict[str, Any]] = None,
    # Sprint 13.26 — time metrics from compute_journey_time_metrics().
    # Optional; when missing, the opener bullets render without
    # duration suffixes (back-compat with older callers).
    time_metrics: Optional[Dict[str, Any]] = None,
    # Sprint 13.19 — `consolidated_steps`, `traversal_log`, and
    # `generate_fn` accepted for back-compat with any caller that
    # still passes them; ignored under the deterministic build.
    consolidated_steps: Optional[List[ConsolidatedStep]] = None,
    traversal_log: Optional[List[Dict[str, Any]]] = None,
    generate_fn: Any = None,
) -> Tuple[str, bool]:
    """Build the Tier-2 escalation handoff note (pure deterministic).

    Returns ``(note_text, used_fallback)``. ``used_fallback`` is
    always False under this build — there is no LLM to fail.
    """
    incidents = _extract_incident_numbers(cohort)

    steps = list(attempted_steps or [])
    opener = _opening_paragraph(
        incidents=incidents,
        stage_2_visited=stage_2_visited,
        stage_3_visited=stage_3_visited,
        stage_4_visited=stage_4_visited,
        kb_chat_engaged=kb_chat_engaged,
        time_metrics=time_metrics,
    )
    diagnostic = _diagnostic_block(
        attempted_steps=steps,
        stage_3_visited=stage_3_visited,
        stage_4_visited=stage_4_visited,
        kb_chat_engaged=kb_chat_engaged,
    )
    routing_block = _routing_block(routing)
    contacts_block = _contacts_block(contacts_payload)

    # Sprint 13.21 — render structure trimmed at user's request:
    #   * 2nd & 3rd Reason-for-Escalation bullets commented out
    #   * Entire Escalation Routing block commented out
    #   * Entire Contact Details block commented out
    #   * Single bolded residual line at the end about vendor/OEM
    # Reinstate any of the suppressed sections by uncommenting the
    # f-string lines below — the helpers (_routing_block,
    # _contacts_block) are still wired up so re-enabling is one edit.
    note = (
        "*Escalation to Tier 2: Triage Complete*\n\n"
        f"{opener}\n\n"
        "*Diagnostic Summary:*\n"
        f"{diagnostic}\n\n"
        "*Reason for Escalation:*\n"
        "- Tier 1 has successfully identified the issue pattern and "
        "completed the standard diagnostics. The ticket is being "
        "handed off for further investigation.\n"
        # "- Historical documentation lacks the granular, issue-specific "
        # "details required for Tier 1 to safely execute a final fix in "
        # "the current environment.\n"
        # "- Strict knowledge and access boundaries for this specific "
        # "scenario have been reached; referring to Tier 2 for advanced "
        # "investigation and resolution.\n"
        "\n"
        # f"{routing_block}\n\n"
        # f"{contacts_block}"
        # Sprint 13.23 — bolded "No vendor/OEM engagement records ..."
        # residual removed from the copy-paste note. The Tier-2 reader
        # already sees that fact prominently in the
        # EscalationRoutingSection card above the note (frontend
        # styles the line as bold black there), so duplicating it
        # inside the copyable note added clutter without info gain.
    )

    logger.info(
        "[stage5_handoff_note] deterministic note built incidents=%d "
        "stage2=%s stage3=%s stage4=%s kb_chat=%s ticked_steps=%d",
        len(incidents), stage_2_visited, stage_3_visited,
        stage_4_visited, kb_chat_engaged, len(steps),
    )
    return (note, False)
