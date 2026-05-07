"""Sprint 13 Stage 3 — Per-ticket Guided Workflow harvester.

Replaces the merged-ledger architecture (Sprint 10/12.6's
``stage3_troubleshooting.build_stage3``) with one *grouped* workflow
per cohort ticket. Pure structural pass — no LLM here; the synthesis
pass (`stage3_synthesis.synthesize_workflows`) layers Intent/Pivot
prose on top.

Why per-ticket?
  The merged ledger inherited dangling cross-references like
  *"If scan clean, proceed to Step 2 for CPE log check."* — the
  source author's "Step 2" pointed at *their* local Resolution_Steps
  ordering, not ours. After dedupe + global re-numbering, the
  reference dangles. Per-ticket grouping eliminates the class of
  bug entirely: each section keeps the source ticket's internal
  numbering and reads as a self-contained playbook.

Source field families (per cohort ticket), walked in this order so
the rendered ledger reads diagnostic-then-fix:

  1. Operational_SOP.diagnostic_logic_chunks      (the only source
                                                   carrying Intent/
                                                   Pivot/Command in
                                                   the gold schema)
  2. Troubleshooting_Ledger.Diagnostic_Tests_Executed   (forward-
                                                   compat; 0/213
                                                   in current corpus)
  3. Forensic_Performance_Audit.Key_Movements_Timeline.Action
  4. Executive_Sharable_RCA.Resolution_Steps
  5. Forensic_Performance_Audit.Critical_Intervention
  6. Key_Contributors.Key_Impact_Players[0].Hero_Action

`Executive_Sharable_RCA.Technical_Snapshot` rides on the workflow
header as ``technical_snapshot`` (not a step — it's whole-ticket
narrative).

Numbering is **local** to the workflow (1..N per ticket; restarts
at 1 for the next match). This is the contract that lets the
synthesis layer write Pivot prose without ever pointing at
"Step N" — the scope is unambiguous within a single panel.

Pure function — no DB, no LLM, no logging beyond debug.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import GuidedWorkflow, GuidedWorkflowStep
# Re-use the array-aware path walker + flattener from stage 2 so all
# stages traverse the corpus the same way.
from .stage2_historical import _flatten, _safe_get


logger = logging.getLogger("acadia-log-iq")


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _incident_number(ticket: Dict[str, Any]) -> Optional[str]:
    return _safe_str(_safe_get(ticket, "Metadata", "Incident_Number"))


def _harvest_steps(ticket: Dict[str, Any]) -> List[GuidedWorkflowStep]:
    """Extract every actionable step from one ticket, in
    diagnostic-then-fix order. Each step keeps verbatim source
    text for action/intent/pivot/command — the synthesis layer
    refines them per-ticket-context afterwards. Steps with no
    populated source intent/pivot still emit (those fields stay
    None until the synthesis pass fills them in).
    """
    if not isinstance(ticket, dict):
        return []

    raw: List[Dict[str, Any]] = []

    # ── Source 1 — Operational_SOP.diagnostic_logic_chunks[] ──
    # The ONLY source that populates Intent + Pivot + Command in
    # the gold schema. Field-name cascade preserves Sprint 10
    # fixtures that used "rationale" / "intent" / "pivot" keys.
    chunks = _safe_get(ticket, "Operational_SOP", "diagnostic_logic_chunks")
    if isinstance(chunks, list):
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            action = _safe_str(ch.get("action") or ch.get("step_id") or ch.get("step"))
            if not action:
                continue
            raw.append({
                "action": action,
                "intent": _safe_str(
                    ch.get("context") or ch.get("rationale") or ch.get("intent")
                ),
                "pivot": _safe_str(ch.get("branching_logic") or ch.get("pivot")),
                "command": _safe_str(ch.get("command")),
                "source_field": "diagnostic_logic_chunks",
            })

    # ── Source 2 — Troubleshooting_Ledger.Diagnostic_Tests_Executed[] ──
    # 0/213 populated in the current corpus; the branch is wired
    # forward-compat so a future ingestion auto-lights it up.
    tests = _safe_get(ticket, "Troubleshooting_Ledger", "Diagnostic_Tests_Executed")
    if isinstance(tests, list):
        for entry in tests:
            if isinstance(entry, dict):
                action = _safe_str(
                    entry.get("name") or entry.get("description")
                    or entry.get("test") or entry.get("action")
                )
            else:
                action = _safe_str(entry)
            if not action:
                continue
            raw.append({
                "action": action,
                "intent": None,
                "pivot": None,
                "command": None,
                "source_field": "Diagnostic_Tests_Executed",
            })

    # ── Source 3 — Forensic_Performance_Audit[0].Key_Movements_Timeline[].Action ──
    timeline = _safe_get(ticket, "Forensic_Performance_Audit", "Key_Movements_Timeline")
    if isinstance(timeline, list):
        for mv in timeline:
            if not isinstance(mv, dict):
                continue
            action = _safe_str(mv.get("Action"))
            if not action:
                continue
            raw.append({
                "action": action,
                "intent": None,
                "pivot": None,
                "command": None,
                "source_field": "Key_Movements_Timeline",
            })

    # ── Source 4 — Executive_Sharable_RCA.Resolution_Steps[] ──
    rsteps = _safe_get(ticket, "Executive_Sharable_RCA", "Resolution_Steps")
    if isinstance(rsteps, list):
        for s in rsteps:
            if isinstance(s, dict):
                action = _safe_str(
                    s.get("description") or s.get("action") or s.get("step")
                )
            else:
                action = _safe_str(s)
            if not action:
                continue
            raw.append({
                "action": action,
                "intent": None,
                "pivot": None,
                "command": None,
                "source_field": "Resolution_Steps",
            })
    elif isinstance(rsteps, str) and rsteps.strip():
        raw.append({
            "action": rsteps.strip(),
            "intent": None,
            "pivot": None,
            "command": None,
            "source_field": "Resolution_Steps",
        })

    # ── Source 5 — Forensic_Performance_Audit[0].Critical_Intervention ──
    ci_raw = _safe_get(ticket, "Forensic_Performance_Audit", "Critical_Intervention")
    ci = _safe_str(ci_raw) if isinstance(ci_raw, str) else _flatten(ci_raw)
    if ci:
        raw.append({
            "action": ci,
            "intent": None,
            "pivot": None,
            "command": None,
            "source_field": "Critical_Intervention",
        })

    # ── Source 6 — Key_Contributors.Key_Impact_Players[0].Hero_Action ──
    ha_raw = _safe_get(
        ticket, "Key_Contributors", "Key_Impact_Players", "Hero_Action",
    )
    ha = _safe_str(ha_raw) if isinstance(ha_raw, str) else _flatten(ha_raw)
    if ha:
        raw.append({
            "action": ha,
            "intent": None,
            "pivot": None,
            "command": None,
            "source_field": "Hero_Action",
        })

    # Build typed records with local 1..N numbering.
    return [
        GuidedWorkflowStep(
            step_number=i + 1,
            action=r["action"],
            intent=r["intent"],
            pivot=r["pivot"],
            command=r["command"],
            source_field=r["source_field"],
        )
        for i, r in enumerate(raw)
    ]


def build_guided_workflows(
    cohort: List[Dict[str, Any]],
) -> List[GuidedWorkflow]:
    """One GuidedWorkflow per cohort ticket, in rank order.

    Tickets without ANY harvestable step are skipped (zero-step
    workflows would render as empty panels). Tickets without an
    Incident_Number are skipped — without an ID the engineer can't
    cite the source, which defeats the purpose of grouping.

    Match ranks are re-numbered 1..N over the survivors so the
    visible labels stay sequential after sibling drops. The first
    survivor is marked ``expanded_by_default=True``; the rest
    collapsed.

    `header` and per-step Intent/Pivot are left at their structural
    defaults (raw `Incident_Summary.INCIDENT` for header; verbatim
    source text for Intent/Pivot if present, None otherwise). The
    LLM synthesis pass refines them in-place afterwards.
    """
    if not cohort:
        return []

    out: List[GuidedWorkflow] = []
    for ticket in cohort:
        if not isinstance(ticket, dict):
            continue
        inc = _incident_number(ticket)
        if not inc:
            continue
        steps = _harvest_steps(ticket)
        if not steps:
            continue

        # Default header — raw INCIDENT line. The LLM synthesis pass
        # rewrites this into a 3-7 word topical title; this default
        # only ever ships when synthesis is skipped (LLM down).
        default_header = (
            _flatten(_safe_get(ticket, "Incident_Summary", "INCIDENT"))
            or inc
        )
        snapshot = _flatten(
            _safe_get(ticket, "Executive_Sharable_RCA", "Technical_Snapshot")
        )

        out.append(GuidedWorkflow(
            match_rank=0,           # filled in below after re-rank
            incident_number=inc,
            header=default_header,
            technical_snapshot=snapshot,
            expanded_by_default=False,
            steps=steps,
            synthesis_skipped=True,    # synthesis pass flips this False
        ))

    # Re-rank survivors and mark only the first as default-open.
    for new_rank, wf in enumerate(out, start=1):
        out[new_rank - 1] = wf.model_copy(update={
            "match_rank": new_rank,
            "expanded_by_default": (new_rank == 1),
        })

    logger.info(
        "[stage3_grouped] cohort=%d workflows=%d (steps_per_ticket=%s)",
        len(cohort),
        len(out),
        [len(wf.steps) for wf in out],
    )
    return out
