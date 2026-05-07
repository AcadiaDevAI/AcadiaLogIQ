"""Sprint 12.7 — Tier-2 Escalation Handoff Note generator.

Implements the user-supplied prompt verbatim. The note is produced
when an engineer clicks "Generate Tier 2 Escalation Handoff" on the
Operational Handoff card; the resulting text is meant to be pasted
straight into the Tier-2 ticket / ServiceNow / runbook.

Inputs (per cohort ticket):
  - Metadata.Incident_Number                              → list of IDs
  - Executive_Sharable_RCA.Resolution_Steps               → diagnostic context
  - Operational_SOP.diagnostic_logic_chunks               → diagnostic context
  - Troubleshooting_Ledger.Diagnostic_Tests_Executed      → diagnostic context

Output:
  Plain-text note matching the spec template (4 sections: header,
  ticket-list line, Diagnostic Summary, Reason for Escalation).

Failure stance:
  Failure-open. LLM errors fall back to a deterministic template-fill
  using a heuristic 1-sentence summary so the button never produces
  a blank. ``used_fallback=True`` flagged on the response so callers
  can surface a small "regenerate" hint.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .stage2_historical import _flatten, _safe_get


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Source extraction
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


def _extract_diagnostic_corpus(cohort: List[Dict[str, Any]]) -> List[str]:
    """Pull every diagnostic-relevant string from the three source
    fields. The LLM reads this flat list and synthesises a 1-2
    sentence summary; the heuristic fallback also uses it.
    """
    bag: List[str] = []
    seen: set = set()

    def _add(s: Any) -> None:
        v = _coerce_str(s)
        if v and v.casefold() not in seen:
            seen.add(v.casefold())
            bag.append(v)

    for ticket in cohort or []:
        if not isinstance(ticket, dict):
            continue

        # Resolution_Steps
        rsteps = _safe_get(ticket, "Executive_Sharable_RCA", "Resolution_Steps")
        if isinstance(rsteps, list):
            for s in rsteps:
                if isinstance(s, dict):
                    _add(s.get("description") or s.get("action") or s.get("step"))
                else:
                    _add(s)
        elif isinstance(rsteps, str):
            _add(rsteps)

        # diagnostic_logic_chunks → action + intent (context)
        chunks = _safe_get(ticket, "Operational_SOP", "diagnostic_logic_chunks")
        if isinstance(chunks, list):
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                _add(ch.get("action") or ch.get("step_id") or ch.get("step"))
                _add(ch.get("context") or ch.get("rationale") or ch.get("intent"))

        # Troubleshooting_Ledger.Diagnostic_Tests_Executed
        tests = _safe_get(ticket, "Troubleshooting_Ledger", "Diagnostic_Tests_Executed")
        if isinstance(tests, list):
            for entry in tests:
                if isinstance(entry, dict):
                    _add(
                        entry.get("name")
                        or entry.get("description")
                        or entry.get("test")
                        or entry.get("action")
                    )
                else:
                    _add(entry)

    return bag


# ─────────────────────────────────────────────────────────────
# LLM prompt — spec verbatim
# ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """*Role & Objective:*
You are an Expert IT Service Management Assistant. When the user triggers the "Generate Tier 2 Escalation Handoff" action, your task is to analyze the provided JSON payload of historical tickets and generate a concise, professional escalation note.

The note must instill confidence in Tier 2 that Tier 1 did their job, while clearly stating that Tier 1 reached a hard knowledge/tooling boundary.

*Data Extraction Rules:*
1. *Ticket Numbers:* Extract all ticket numbers from the Metadata.Incident_Number field in the provided JSON and combine them into a comma-separated list.
2. *Diagnostic Summary:* Extract the execution steps from Executive_Sharable_RCA.Resolution_Steps, Operational_SOP.diagnostic_logic_chunks and Troubleshooting_Ledger.Diagnostic_Tests_Executed. Synthesize these steps into a single, brief summary sentence that describes the types of diagnostics performed (e.g., "We validated X, monitored Y for resource exhaustion, and evaluated Z."). Do not list individual steps, commands, intents, or pivots.

*Output Constraint:*
Do not output any conversational filler (e.g., "Here is your note"). Output ONLY the following formatted template, replacing the bracketed [...] sections with the dynamically generated data.

---
*Escalation to Tier 2: Triage Complete*

Please review this ticket for advanced intervention. Tier 1 has completed all initial triage and exhausted standard operating procedures (SOPs). We have verified the core symptoms and performed a comprehensive review of similar historical tickets (*[Insert Comma-Separated List of Ticket Numbers here]*).

*Diagnostic Summary:*
[Insert the dynamically generated 1-2 sentence summary of the diagnostic actions performed here.]

*Reason for Escalation:*
While we successfully identified the issue pattern and completed the diagnostics mentioned above, the historical documentation lacks the granular, issue-specific details required for Tier 1 to safely execute a final fix in the current environment. Due to strict knowledge and access boundaries for this specific scenario, we have reached the limit of our capabilities. We are referring this to your queue for advanced investigation and resolution.
"""


def _render_payload(incidents: List[str], diagnostic_bag: List[str]) -> str:
    """Compact JSON-ish payload the prompt analyses. We don't ship the
    raw cohort dicts (too noisy for a 1-sentence synthesis); just the
    fields the prompt actually reads, pre-extracted."""
    lines = ["{"]
    lines.append('  "Metadata": [')
    for inc in incidents:
        lines.append(f'    {{"Incident_Number": "{inc}"}},')
    if incidents:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("  ],")
    lines.append('  "diagnostic_actions": [')
    for s in diagnostic_bag:
        # JSON-escape just the dangerous chars; we're inside a prompt,
        # not strict JSON, so a lightweight sanitiser is enough.
        clean = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
        lines.append(f'    "{clean}",')
    if diagnostic_bag:
        lines[-1] = lines[-1].rstrip(",")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Deterministic fallback — fires on LLM error / empty response
# ─────────────────────────────────────────────────────────────
def _heuristic_summary(diagnostic_bag: List[str]) -> str:
    """Build a fallback 1-sentence diagnostic summary without an LLM.
    Picks the 3 longest strings (most descriptive) and stitches them
    into a single sentence. Not as polished as the LLM output, but
    grounded in real source content rather than boilerplate."""
    if not diagnostic_bag:
        return (
            "We completed the cohort's standard diagnostics; specific "
            "actions were not surfaced in the historical record."
        )
    picks = sorted(diagnostic_bag, key=len, reverse=True)[:3]
    # Lowercase the first letter of each clause for natural prose.
    parts = [p[0].lower() + p[1:] if p else p for p in picks]
    if len(parts) == 1:
        return f"We {parts[0]}."
    if len(parts) == 2:
        return f"We {parts[0]}, and {parts[1]}."
    return f"We {parts[0]}, {parts[1]}, and {parts[2]}."


def _render_template(incidents: List[str], summary_sentence: str) -> str:
    """Template-fill the spec output verbatim. Used by both the
    fallback path AND as a sanity wrapper if the LLM returns plain
    summary text (we re-frame it into the template)."""
    inc_str = ", ".join(incidents) if incidents else "(no incident numbers found)"
    return (
        "*Escalation to Tier 2: Triage Complete*\n\n"
        "Please review this ticket for advanced intervention. Tier 1 has "
        "completed all initial triage and exhausted standard operating "
        "procedures (SOPs). We have verified the core symptoms and "
        "performed a comprehensive review of similar historical tickets "
        f"(*{inc_str}*).\n\n"
        "*Diagnostic Summary:*\n"
        f"{summary_sentence}\n\n"
        "*Reason for Escalation:*\n"
        "While we successfully identified the issue pattern and completed "
        "the diagnostics mentioned above, the historical documentation "
        "lacks the granular, issue-specific details required for Tier 1 "
        "to safely execute a final fix in the current environment. Due "
        "to strict knowledge and access boundaries for this specific "
        "scenario, we have reached the limit of our capabilities. We are "
        "referring this to your queue for advanced investigation and "
        "resolution."
    )


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def generate_handoff_note(
    cohort: List[Dict[str, Any]],
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> Tuple[str, bool]:
    """Generate the Tier-2 escalation handoff note.

    Args:
        cohort: list of ticket metadata_json dicts (the same shape
            ticket_loader.load_cohort_metadata returns).
        generate_fn: LLM call. Signature ``(prompt, max_tokens) -> str``.
            Defaults to :func:`backend.api.safe_generate`. Injected for
            tests.

    Returns:
        ``(note_text, used_fallback)``. Never raises.
    """
    incidents = _extract_incident_numbers(cohort)
    bag = _extract_diagnostic_corpus(cohort)

    if generate_fn is None:
        try:
            from backend.api import safe_generate as generate_fn  # type: ignore
        except Exception as exc:
            logger.warning(
                "[stage5_handoff_note] safe_generate import failed (%s) — "
                "using deterministic template fallback", exc,
            )
            return (
                _render_template(incidents, _heuristic_summary(bag)),
                True,
            )

    payload = _render_payload(incidents, bag)
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"INPUT JSON PAYLOAD:\n{payload}\n\n"
        f"BEGIN OUTPUT (template only, no preamble):"
    )

    try:
        raw = generate_fn(full_prompt, 600)
        out = (raw or "").strip()
        # Strip an accidental "---" lead-in if the model echoed the
        # prompt's separator.
        if out.startswith("---"):
            out = out[3:].lstrip("\n").lstrip()
        # Strip a leading code-fence if the model wrapped the template.
        if out.startswith("```"):
            out = out.strip("`").lstrip("markdown").lstrip("text").strip()
        if not out:
            raise RuntimeError("empty LLM response")
        # Sanity check — the spec template ALWAYS contains the literal
        # header line "*Escalation to Tier 2: Triage Complete*". If the
        # model went off-script we drop to fallback.
        if "Escalation to Tier 2" not in out:
            raise RuntimeError("LLM response missing template header")
        logger.info(
            "[stage5_handoff_note] LLM note generated incidents=%d "
            "diag_signals=%d chars=%d",
            len(incidents), len(bag), len(out),
        )
        return (out, False)
    except Exception as exc:
        logger.warning(
            "[stage5_handoff_note] LLM call failed (%s) — falling back "
            "to deterministic template", exc,
        )
        return (
            _render_template(incidents, _heuristic_summary(bag)),
            True,
        )
