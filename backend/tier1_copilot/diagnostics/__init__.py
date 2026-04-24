"""Sprint 7 — Tier-1 diagnostics subpackage.

Four pure-logic modules (no FastAPI / no direct DB coupling beyond the
session store):

  deeper_diagnostics        Builds the severity-aware step card from the
                            matched ticket's Operational_SOP JSON.
  escalation_package        Assembles the pastable escalation bundle,
                            optionally enriched with Sprint 3C contacts.
  explain_recommendation    Score-breakdown + field-match evidence.
  stuck_detector            Pure nudge threshold logic.
"""
