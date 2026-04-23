"""Sprint 5 — Template-First Expert Copilot renderer (pure Python).

Deterministic rendering of the gold-schema JSON ticket sections that do
NOT need LLM synthesis:

  - Header + Fingerprints block
  - Phase 2: Branching Diagnostics (iterates diagnostic_logic_chunks)
  - Phase 3: Validated Fix (Symptom_Solution_Mapping + remediation_payload
             + Remediation_As_Code snippet)
  - KB citations (Knowledge_Base.semantic_unit_educational.knowledge_id)

The detector `is_gold_schema_ticket` is the gate for the entire Sprint 5
fast path: if it returns False, the caller MUST fall back to the Sprint 4
full LLM pipeline. This module never imports config / DB / Bedrock — it
is safe to call from any thread and has zero external dependencies
beyond the standard library.
"""
from __future__ import annotations

from typing import Any, Dict

_REQUIRED_GOLD_SECTIONS = (
    "Symptom_Solution_Mapping",
    "Operational_SOP",
    "remediation_payload",
)


def is_gold_schema_ticket(metadata_json: Any) -> bool:
    """Strict detector. True iff the record has all three required
    top-level sections AND a non-empty Metadata.Fingerprints array.

    Defensive: accepts anything (PDFs, KB rows, partial tickets, bad
    inputs) and returns False for everything that isn't the gold
    schema. This is the only gate protecting Sprint 4 LLM behavior for
    non-gold retrievals.

    Note: remediation_payload may live either at the top level OR nested
    under Operational_SOP (ingestion has copied it top-level since
    Sprint 4 §872-886 of _ingest_gold_ticket_json), so we accept either.
    """
    if not isinstance(metadata_json, dict):
        return False
    meta = metadata_json.get("Metadata")
    if not isinstance(meta, dict):
        return False
    fps = meta.get("Fingerprints")
    if not fps or not isinstance(fps, (list, tuple)):
        return False

    for key in ("Symptom_Solution_Mapping", "Operational_SOP"):
        val = metadata_json.get(key)
        if not isinstance(val, (dict, list)):
            return False

    # remediation_payload may be top-level or nested under Operational_SOP.
    top = metadata_json.get("remediation_payload")
    nested = (metadata_json.get("Operational_SOP") or {}).get("remediation_payload") \
        if isinstance(metadata_json.get("Operational_SOP"), dict) else None
    if not isinstance(top, (dict, list)) and not isinstance(nested, (dict, list)):
        return False
    return True


def render_header_and_fingerprints(json_ticket: Dict[str, Any]) -> str:
    """Render the opening header block (title + incident number + priority
    + fingerprint list). No LLM call."""
    header = json_ticket.get("Header") or ""
    meta = json_ticket.get("Metadata") or {}
    inc = meta.get("Incident_Number", "unknown")
    priority = meta.get("priority", meta.get("Priority", "n/a"))
    fps = meta.get("Fingerprints") or []

    out = [
        f"# Troubleshooting Guide: {header}".rstrip(": ").rstrip(),
        f"**Incident ID:** {inc} | **Priority:** {priority}",
        "",
        "---",
        "",
        "**Detected Fingerprints:**",
    ]
    for fp in fps:
        out.append(f"- `{fp}`")
    return "\n".join(out)


def render_phase_2_branching(json_ticket: Dict[str, Any]) -> str:
    """Render Phase 2 Branching Diagnostics from Operational_SOP's
    diagnostic_logic_chunks array. No LLM call."""
    sop = json_ticket.get("Operational_SOP") or {}
    if not isinstance(sop, dict):
        return ""
    chunks = sop.get("diagnostic_logic_chunks") or []
    if not chunks:
        return ""

    out_lines = ["## Phase 2: Branching Diagnostics", ""]
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        step_id = chunk.get("step_id", "")
        action = chunk.get("action", "")
        command = chunk.get("command")
        branching = chunk.get("branching_logic", "")

        heading = f"### {step_id}: {action}".strip()
        out_lines.append(heading)
        if command:
            out_lines.append(f"- **Command:** `{command}`")
        if branching:
            out_lines.append(f"- **Branching Logic:** {branching}")
        out_lines.append("")

    return "\n".join(out_lines).rstrip()


def render_phase_3_remediation(json_ticket: Dict[str, Any]) -> str:
    """Render Phase 3 Validated Fix from Symptom_Solution_Mapping +
    remediation_payload. remediation_payload is checked under
    Operational_SOP first, then top-level (Sprint 4 ingestion copies it
    top-level; legacy records may only have the nested form). No LLM
    call."""
    ssm = json_ticket.get("Symptom_Solution_Mapping") or {}
    if not isinstance(ssm, dict):
        ssm = {}

    sop = json_ticket.get("Operational_SOP") or {}
    nested = sop.get("remediation_payload") if isinstance(sop, dict) else None
    top = json_ticket.get("remediation_payload")
    remed = nested if isinstance(nested, dict) else (top if isinstance(top, dict) else {})

    out = ["## Phase 3: Validated Fix", ""]
    if ssm.get("Primary_Fix"):
        out.append(f"**Primary Fix:** {ssm['Primary_Fix']}")
    if ssm.get("Primary_Fix_Confidence_Interval"):
        out.append(f"**Confidence:** {ssm['Primary_Fix_Confidence_Interval']}")
    if ssm.get("Validation_Metric"):
        out.append(f"**Validation Metric:** {ssm['Validation_Metric']}")

    steps = remed.get("execution_steps") or []
    if steps:
        out.append("")
        out.append("**Remediation Steps:**")
        for i, step in enumerate(steps, 1):
            if not isinstance(step, dict):
                continue
            task = step.get("task", "")
            action = step.get("action", "")
            out.append(f"{i}. **{task}:** `{action}`")

    rac = remed.get("Remediation_As_Code") or {}
    if isinstance(rac, dict) and rac.get("Executable_Snippet"):
        lang = (rac.get("IaC_Language") or "").lower()
        out.append("")
        out.append(f"**Remediation as Code ({lang or 'snippet'}):**")
        out.append(f"```{lang}")
        out.append(rac["Executable_Snippet"])
        out.append("```")

    return "\n".join(out)


def render_kb_citations(json_ticket: Dict[str, Any]) -> str:
    """Render KB citations list. No LLM call. Returns an empty string if
    no knowledge_ids are present so the stitcher can skip the section
    cleanly."""
    kb_list = json_ticket.get("Knowledge_Base") or []
    if not isinstance(kb_list, list):
        return ""
    ids = []
    for kb in kb_list:
        if not isinstance(kb, dict):
            continue
        unit = kb.get("semantic_unit_educational") or {}
        if not isinstance(unit, dict):
            continue
        kid = unit.get("knowledge_id")
        if kid:
            ids.append(kid)
    if not ids:
        return ""
    return "**Referenced KB:** " + ", ".join(f"`{k}`" for k in ids)
