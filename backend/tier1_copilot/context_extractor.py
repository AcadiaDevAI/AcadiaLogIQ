"""Extract the 11-field compact context the Tier-1 prompt needs.

The LLM NEVER sees full gold-schema JSON — just these fields. This
keeps the prompt payload small (~1KB typical) and prevents the model
from hallucinating off sections like Executive_Sharable_RCA or
QA_Auditor_Feedback that have no place in a Tier-1 answer.

Missing fields are set to `None`. Callers that serialize the context
into the prompt must drop None entries outright (never render "N/A"
— that's a rule the system prompt enforces).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def extract_compact_context(ticket_metadata_json: Dict[str, Any]) -> Dict[str, Any]:
    md = ticket_metadata_json if isinstance(ticket_metadata_json, dict) else {}
    meta = md.get("Metadata") if isinstance(md.get("Metadata"), dict) else {}
    ssm = md.get("Symptom_Solution_Mapping") \
        if isinstance(md.get("Symptom_Solution_Mapping"), dict) else {}
    sop = md.get("Operational_SOP") \
        if isinstance(md.get("Operational_SOP"), dict) else {}
    exec_rca = md.get("Executive_Sharable_RCA") \
        if isinstance(md.get("Executive_Sharable_RCA"), dict) else {}
    remed = md.get("remediation_payload") \
        if isinstance(md.get("remediation_payload"), dict) else \
        (sop.get("remediation_payload") \
         if isinstance(sop.get("remediation_payload"), dict) else {})

    recommended_checks: List[str] = []
    diag_chunks = sop.get("diagnostic_logic_chunks") or []
    if isinstance(diag_chunks, list):
        for c in diag_chunks:
            if not isinstance(c, dict):
                continue
            command = c.get("command") or c.get("action")
            if command:
                recommended_checks.append(str(command).strip())
    recommended_checks = recommended_checks[:6] or None

    return {
        "incident_number": meta.get("Incident_Number"),
        "customer": meta.get("customer_name") or meta.get("Customer_Name"),
        "priority": meta.get("priority") or meta.get("Priority"),
        "technology": meta.get("component_category") or meta.get("Target_Service"),
        "issue": md.get("Header") or ssm.get("Origin_Event"),
        "detected_symptom": ssm.get("Detected_Symptom"),
        "root_cause": exec_rca.get("Root_Cause") or ssm.get("Origin_Event"),
        "primary_fix": ssm.get("Primary_Fix") or remed.get("Summary"),
        "recommended_checks": recommended_checks,
        "validation": ssm.get("Validation_Metric"),
        "escalation_path": exec_rca.get("Escalation_Path")
            or (meta.get("Resolution_Groups") or [None])[0]
            if isinstance(meta.get("Resolution_Groups"), list)
            else exec_rca.get("Escalation_Path"),
    }


def prune_none(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose value is None / empty so the prompt stays clean."""
    return {k: v for k, v in ctx.items() if v not in (None, "", [], {})}
