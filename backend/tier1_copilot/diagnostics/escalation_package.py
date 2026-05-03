"""Sprint 7 — Escalation Package builder.

Pure template assembly from:
  - matched ticket metadata_json (Engagement_Analysis, Metadata, SSM)
  - session state (what_tried log)
  - optional Sprint 3C contact enrichment (doc_kind='contact_customer')

No LLM call. Everything is deterministic so the output is paste-ready
into ServiceNow / Jira the moment the engineer clicks Escalate.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from backend.tier1_copilot.diagnostics.escalation_directory import (
    lookup_directory_contacts,
)
from backend.tier1_copilot.schemas import (
    Tier1Contact,
    Tier1DirectoryContact,
    Tier1EscalationPackage,
)

logger = logging.getLogger("acadia-log-iq")


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _what_tried_lines(
    session_what_tried: Optional[List[Dict[str, Any]]],
    client_what_tried: Optional[List[Dict[str, Any]]],
) -> List[str]:
    merged: List[Dict[str, Any]] = []
    for source in (session_what_tried or [], client_what_tried or []):
        for item in source:
            if isinstance(item, dict):
                merged.append(item)

    # Dedupe by (step, result).
    seen: set = set()
    out: List[str] = []
    for item in merged:
        step = str(item.get("step", "")).strip()
        result = str(item.get("result", "")).strip()
        if not step:
            continue
        key = (step.lower(), result.lower())
        if key in seen:
            continue
        seen.add(key)
        if result:
            out.append(f"{step} → {result}")
        else:
            out.append(step)
    return out


def fetch_customer_contacts(
    *,
    customer_name: Optional[str],
    engine: Any,
    limit: int = 3,
) -> List[Tier1Contact]:
    """Pull top-N contact_customer rows for this customer.

    Graceful: returns [] when Sprint 3C data isn't ingested, the
    customer is unknown, or the query fails."""
    if not customer_name or engine is None:
        return []
    customer_norm = customer_name.strip()
    if not customer_norm:
        return []

    try:
        from sqlalchemy import text
        sql = text(
            """
            SELECT metadata_json
            FROM chunks
            WHERE metadata_json->>'doc_kind' = 'contact_customer'
              AND (
                    metadata_json->'organization'->>'name' ILIKE :c
                 OR metadata_json->>'customer_name' ILIKE :c
                 OR metadata_json->>'organization_name' ILIKE :c
              )
            ORDER BY created_at DESC
            LIMIT :lim
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(
                sql, {"c": customer_norm, "lim": int(limit * 3)},
            ).mappings().all()
    except Exception as exc:
        logger.warning(
            "[tier1_copilot:sprint7] contact lookup failed customer=%s: %s",
            customer_name, exc,
        )
        return []

    out: List[Tier1Contact] = []
    for r in rows:
        md = r.get("metadata_json") or {}
        if not isinstance(md, dict):
            continue
        org = md.get("organization") if isinstance(md.get("organization"), dict) else {}
        team = md.get("team") if isinstance(md.get("team"), dict) else {}
        out.append(Tier1Contact(
            name=md.get("name") or md.get("contact_name") or org.get("primary_contact"),
            role=md.get("role") or md.get("title") or team.get("role"),
            phone=md.get("phone") or md.get("contact_phone") or md.get("phone_number"),
            email=md.get("email") or md.get("contact_email"),
            escalation_level=(
                str(team.get("escalation_level"))
                if isinstance(team, dict) and team.get("escalation_level") is not None
                else md.get("escalation_level")
            ),
        ))
        if len(out) >= limit:
            break
    return out


def build_package(
    *,
    ticket_metadata: Dict[str, Any],
    alert_payload: Dict[str, Any],
    session_what_tried: Optional[List[Dict[str, Any]]] = None,
    client_what_tried: Optional[List[Dict[str, Any]]] = None,
    engine: Any = None,
    related_incidents: Optional[Iterable[str]] = None,
) -> Tier1EscalationPackage:
    meta = ticket_metadata.get("Metadata") \
        if isinstance(ticket_metadata.get("Metadata"), dict) else {}
    ssm = ticket_metadata.get("Symptom_Solution_Mapping") \
        if isinstance(ticket_metadata.get("Symptom_Solution_Mapping"), dict) else {}
    ea = ticket_metadata.get("Engagement_Analysis") \
        if isinstance(ticket_metadata.get("Engagement_Analysis"), dict) else {}

    priority = alert_payload.get("severity") or meta.get("priority") \
        or meta.get("Priority") or ""
    customer = alert_payload.get("customer") or meta.get("customer_name") \
        or meta.get("Customer_Name")
    assets = _coerce_list(
        alert_payload.get("asset_name") or meta.get("Affected_Assets")
        or meta.get("Target_Service")
    )
    suggested_owner = None
    team_path_raw = ea.get("Team_Path") if isinstance(ea, dict) else None
    team_path = _coerce_list(team_path_raw)
    if team_path:
        suggested_owner = team_path[-1]

    primary_fix = ssm.get("Primary_Fix") or (
        ticket_metadata.get("remediation_payload") or {}
    ).get("Summary") if isinstance(ticket_metadata.get("remediation_payload"), dict) \
        else ssm.get("Primary_Fix")

    what_tried_lines = _what_tried_lines(session_what_tried, client_what_tried)
    contacts = fetch_customer_contacts(customer_name=customer, engine=engine)

    directory_contacts = lookup_directory_contacts(
        customer=customer,
        alert_type=alert_payload.get("alert_type") or meta.get("Alert_Type"),
        asset_name=(assets[0] if assets else None)
        or alert_payload.get("asset_name"),
        technology=alert_payload.get("technology") or meta.get("Technology"),
        notes=alert_payload.get("notes"),
    )

    rel = list(related_incidents or [])
    inc = meta.get("Incident_Number")
    if inc and inc not in rel:
        rel.insert(0, inc)

    summary = (
        ssm.get("Detected_Symptom")
        or alert_payload.get("alert_type")
        or "Tier-1 alert under investigation."
    )

    formatted_text = _format_paste_block(
        summary=summary,
        priority=priority,
        customer=customer,
        assets=assets,
        suggested_owner=suggested_owner,
        team_path=team_path,
        contacts=contacts,
        directory_contacts=directory_contacts,
        what_tried_lines=what_tried_lines,
        recommended_next_action=primary_fix,
        relevant_tickets=rel,
    )

    return Tier1EscalationPackage(
        summary=str(summary)[:500],
        priority=str(priority or ""),
        affected_customer=customer,
        affected_assets=assets,
        suggested_owner_team=suggested_owner,
        escalation_path=team_path,
        customer_contacts=contacts,
        vendor_contacts=[],
        directory_contacts=directory_contacts,
        what_was_tried=what_tried_lines,
        recommended_next_action=str(primary_fix) if primary_fix else None,
        relevant_tickets=rel,
        formatted_text=formatted_text,
    )


# ─────────────────────────────────────────────────────────────
# Paste-ready text block
# ─────────────────────────────────────────────────────────────
def _format_paste_block(
    *,
    summary: str,
    priority: str,
    customer: Optional[str],
    assets: List[str],
    suggested_owner: Optional[str],
    team_path: List[str],
    contacts: List[Tier1Contact],
    directory_contacts: List[Tier1DirectoryContact],
    what_tried_lines: List[str],
    recommended_next_action: Optional[str],
    relevant_tickets: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"Summary: {summary}")
    if priority:
        lines.append(f"Priority: {priority}")
    # if customer:
    #     lines.append(f"Customer: {customer}")
    if assets:
        lines.append(f"Affected assets: {', '.join(assets)}")
    if suggested_owner:
        lines.append(f"Suggested owner: {suggested_owner}")
    if team_path:
        lines.append(f"Escalation path: {' → '.join(team_path)}")
    if contacts:
        lines.append("")
        lines.append("Customer contacts:")
        for c in contacts:
            bits = [c.name or "", c.role or "", c.phone or "", c.email or ""]
            lines.append("  - " + " | ".join(b for b in bits if b))
    if what_tried_lines:
        lines.append("")
        lines.append("What has been tried:")
        for w in what_tried_lines:
            lines.append(f"  - {w}")
    if recommended_next_action:
        lines.append("")
        lines.append(f"Recommended next action: {recommended_next_action}")
    if relevant_tickets:
        lines.append("")
        lines.append(f"Relevant tickets: {', '.join(relevant_tickets)}")
    if directory_contacts:
        lines.append("")
        lines.append("Recommended contacts (Acadia Escalation Directory):")
        for d in directory_contacts:
            bits = [b for b in (d.name, d.detail) if b]
            prefix = f"[{d.label}] " if d.label else ""
            lines.append("  - " + prefix + " · ".join(bits))
    return "\n".join(lines).strip()
