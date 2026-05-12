"""Sprint 13.12 Stage 3 — Consolidated 5-step Tier-1 ledger.

Replaces the per-ticket Guided Workflows surface as the primary
Stage 3 view. The user's feedback was concrete: 8 steps × 5 tickets
overwhelms a Tier-1 engineer; what they actually need is the merged
best-of-five read-only diagnostic sequence.

Pipeline:
  1. Harvest every candidate action from every cohort ticket via
     `stage3_guided_workflows._harvest_steps`. The Sprint 13 harvest
     already covers the 7 spec source paths (Technical_Snapshot
     demoted to subtitle; six step-shaped sources).
  2. Bundle the candidates + cohort RCA context into a single
     prompt and call Claude Haiku with the user-supplied
     spec verbatim.
  3. LLM dedupes semantically equivalent checks, filters out
     anything that modifies config / restarts / could cause an
     outage, picks the 5 best read-only steps, synthesises Intent
     + Pivot in plain prose. Output is strict JSON that we parse
     deterministically.
  4. Failure-open: any LLM error or schema deviation returns an
     empty list with ``consolidated_synthesis_skipped=True`` so the
     panel's per-ticket detail accordion still renders below.

What the LLM is NOT allowed to do:
  * Invent new actions, IPs, ticket numbers, or commands not
    grounded in the input candidates / cohort context.
  * Suggest config edits, restarts, service changes, deletions, or
    any action that can degrade production.
  * Exceed 5 steps.
  * Reference "Step N" inside Intent / Pivot prose.

Per-ticket Guided Workflows remain on the schema (back-compat) but
are no longer the primary render. The per-ticket detail accordion
(Sprint 11) is preserved in the panel below this consolidated view.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schemas import ConsolidatedStep
from .stage2_historical import _safe_get
from .stage3_guided_workflows import _harvest_steps


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Prompt — user-supplied spec verbatim.
# The OUTPUT REQUIREMENTS block is the mechanical JSON-shape contract
# that `_parse_response` relies on; without it the parser falls back
# to an empty ledger on every call and the panel silently breaks.
# ─────────────────────────────────────────────────────────────
# ── OLD prompt (commented per user request — kept on disk for reference) ──
# _SYSTEM_PROMPT = """Merge the diagnostic steps from all tickets into one optimal, sequential execution ledger. Deduplicate identical checks. For each consolidated test, state the Intent (hypothesis) and the Pivot (how the result dictates the next step). If Ticket A and Ticket B used different successful interventions, present them as sequential options (e.g., 'Attempt Fix A; if it fails, proceed to Fix B').
# Generate troubleshooting steps specifically suitable for a Tier 1 engineer with read-only access.
#
# Guidelines:
# - Only include safe, non-intrusive actions such as:
#   • executing show/read-only commands
#   • capturing logs
#   • performing observational checks
# - Do NOT suggest any actions that modify configurations, restart services, or could cause outages or performance degradation.
# - Keep all steps low-risk and safe for production environments.
# - Limit the total number of steps to a maximum of 5.
#
# Ensure the output is concise, practical, and immediately executable by a Tier 1 engineer.
#
# OUTPUT REQUIREMENTS — STRICT:
# 1. Output ONLY a single JSON object. No prose before or after. No code fences.
# 2. JSON shape:
#    {
#      "consolidated_steps": [
#        {
#          "step_number": <int 1..5>,
#          "action": "<one short, read-only action>",
#          "intent": "<one sentence hypothesis>",
#          "pivot": "<one or two sentences: what the result means and what to do next>",
#          "command": "<show / read-only command if directly applicable, else omit>"
#        },
#        ...
#      ]
#    }
# 3. step_number values MUST be 1, 2, 3, 4, 5 in order (omit later ones if fewer than 5 steps qualify).
# 4. Maximum 5 entries. Fewer is fine if only fewer safe candidates exist.
#
# WRITING RULES:
# - action: ONE sentence, ≤ 18 words. Begin with a read-only verb (Verify, Check, Capture, Inspect, Run, Review, Confirm, Compare). NEVER use Restart, Apply, Configure, Set, Modify, Delete, Remove, Adjust, Update, Push, Reload, Clear-counters, Reset.
# - intent: ONE sentence, ≤ 20 words. Diagnostic hypothesis the step is testing.
# - pivot: ONE OR TWO sentences, ≤ 40 words total. Describe the result interpretation + next action in plain prose. NEVER reference step numbers ("go to Step 3").
# - command: ONLY when the input candidates include a real command string. Drawn verbatim from input — do NOT invent commands or flags.
# - For divergent successful interventions across tickets: prefer framing the SAFER read-only diagnostic that distinguishes which fix path applies, not the fix itself. Only include "Attempt Fix A; if it fails, proceed to Fix B" framing when both fixes are themselves read-only / safe.
# - Do NOT invent IPs, hostnames, ASNs, ticket numbers, or specific values not present in the input."""

# ── INTERIM cross-ticket prompt (commented — superseded by per-ticket scope) ──
# _SYSTEM_PROMPT = """5. The Troubleshooting Approach: Merge the diagnostic steps from all tickets into one optimal,
# sequential execution ledger. Deduplicate identical checks. For each consolidated test,
# state the Intent (hypothesis) and the Pivot (how the result dictates the next step).
# If Ticket A and Ticket B used different successful interventions,
# present them as sequential options (e.g., 'Attempt Fix A; if it fails, proceed to Fix B').
#
# Generate troubleshooting steps
#
# Ensure the output is concise, practical, and immediately executable.
#
# OUTPUT REQUIREMENTS — STRICT:
# 1. Output ONLY a single JSON object. No prose before or after. No code fences.
# 2. JSON shape:
#    {
#      "consolidated_steps": [
#        {
#          "step_number": <int>,
#          "action": "<one short action sentence>",
#          "intent": "<one sentence hypothesis>",
#          "pivot": "<one or two sentences: what the result means and what to do next>",
#          "command": "<command if directly applicable, else omit>"
#        },
#        ...
#      ]
#    }
# 3. step_number values MUST be sequential integers starting at 1."""

# ── ACTIVE per-ticket prompt ──
# One LLM call per cohort ticket. The model sees ONE ticket's seven
# spec source fields and produces ONE consolidated Activity grounded
# only in that ticket's data. Cohort of 5 → 5 calls → 5 Activities.
_SYSTEM_PROMPT = """The Troubleshooting Approach: You will receive ONE historical ticket's full troubleshooting record (technical snapshot, resolution steps, diagnostic tests, diagnostic logic chunks, key movements timeline, critical intervention, and hero action).

Consolidate THIS ticket's diagnostic activity into a single Activity for a Tier-1 engineer. State the Intent (the hypothesis the activity is testing) and the Pivot (how the result dictates the next step).

Ground every clause in THIS ticket's own fields. Do not invent commands, IPs, hostnames, ticket numbers, ASNs, or values not present in the input.

Ensure the output is concise, practical, and immediately executable.

OUTPUT REQUIREMENTS — STRICT:
1. Output ONLY a single JSON object. No prose before or after. No code fences.
2. JSON shape:
   {
     "action": "<one short action sentence — the consolidated diagnostic activity for this ticket>",
     "intent": "<one sentence: the hypothesis this activity is testing>",
     "pivot": "<one or two sentences: what the result means and what to do next>",
     "command": "<command verbatim from input if directly applicable, else omit>"
   }
3. action ≤ 25 words. intent ≤ 20 words. pivot ≤ 40 words.
4. NEVER reference "Step N" in intent or pivot — describe the next check in plain prose."""


# ─────────────────────────────────────────────────────────────
# Token budget — per-ticket call. One ticket's input is bounded by
# its 7 source fields (≤ ~3000 tokens after truncation); the output
# is a single Activity JSON object (action + intent + pivot + command
# ≈ 200 tokens). 800 gives Haiku comfortable headroom without burning
# tokens that no per-ticket call will use.
# ─────────────────────────────────────────────────────────────
_MAX_TOKENS = 800


_REQUIRED_KEYS = ("step_number", "action", "intent", "pivot")


# Defensive read-only-action filter applied AFTER LLM synthesis as a
# belt-and-suspenders. If the model ignores the prompt and emits a
# config-changing verb, we drop the step. The list is INTENTIONALLY
# narrow — false-positives here would also drop legitimate read-only
# steps. Verbs are checked at word-boundaries and case-insensitively.
_DISRUPTIVE_VERBS = (
    r"\brestart\b", r"\breload\b", r"\breboot\b", r"\bshutdown\b",
    r"\bpush\b", r"\bapply\b", r"\bconfigure\b", r"\bconfig\b",
    r"\bmodify\b", r"\bset\b", r"\bdelete\b", r"\bremove\b",
    r"\bclear\b", r"\breset\b", r"\bupdate config\b",
    r"\bwrite memory\b", r"\bcopy run\b", r"\bcommit\b",
)
_DISRUPTIVE_RE = re.compile("|".join(_DISRUPTIVE_VERBS), re.IGNORECASE)


# Belt-and-suspenders against an LLM that occasionally ignores the
# "NEVER reference step numbers" rule and writes pivots like
# "proceed to Step 2 for CPE log check". The replacement keeps the
# conditional flavour intact while removing the dangling pointer.
_STEP_REF_RE = re.compile(
    r"\b(?:go to|proceed to|return to|jump to|see)\s+step\s*\d+\b",
    re.IGNORECASE,
)


def _strip_step_refs(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    cleaned = _STEP_REF_RE.sub("continue with the next check", s)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _is_read_only_action(action: str) -> bool:
    """True when `action` looks read-only — i.e., no disruptive verb
    detected at word-boundary. Belt-and-suspenders against an LLM
    that slips a config-changing step past the prompt."""
    if not isinstance(action, str) or not action.strip():
        return False
    return _DISRUPTIVE_RE.search(action) is None


# ─────────────────────────────────────────────────────────────
# LEGACY cross-ticket payload (commented — kept on disk for reference).
# The per-ticket build_consolidated_ledger no longer calls this; the
# new flow uses `_build_single_ticket_payload` once per cohort ticket.
# ─────────────────────────────────────────────────────────────
# def _build_payload(
#     cohort: List[Dict[str, Any]],
# ) -> Tuple[str, int]:
#     """Render the cohort's candidate actions + RCA context into the
#     LLM input JSON. Returns ``(json_string, candidate_count)``."""
#     cohort_context: List[Dict[str, Any]] = []
#     candidates: List[Dict[str, Any]] = []
#     seen_signatures: set = set()    # casefold dedupe at the source
#
#     for ticket in cohort or []:
#         if not isinstance(ticket, dict):
#             continue
#         meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
#         rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
#         inc = meta.get("Incident_Number") if isinstance(meta, dict) else None
#         summary = rca.get("Executive_Summary") if isinstance(rca, dict) else None
#         if inc and summary:
#             cohort_context.append({
#                 "incident_number": str(inc),
#                 "executive_summary": str(summary)[:500],
#             })
#
#         for s in _harvest_steps(ticket):
#             sig = s.action.casefold().strip()
#             if not sig or sig in seen_signatures:
#                 continue
#             seen_signatures.add(sig)
#             candidates.append({
#                 "incident_number": str(inc) if inc else None,
#                 "action": s.action,
#                 "source_intent": s.intent,
#                 "source_pivot": s.pivot,
#                 "command": s.command,
#                 "source_field": s.source_field,
#             })
#
#     payload = {
#         "cohort_context": cohort_context[:5],
#         "candidate_actions": candidates,
#     }
#     try:
#         return json.dumps(payload, ensure_ascii=False, default=str), len(candidates)
#     except Exception:
#         return str(payload), len(candidates)


# ─────────────────────────────────────────────────────────────
# ACTIVE per-ticket payload.
# Reads all seven spec source fields for ONE ticket and renders them
# as the JSON payload sent to the LLM. Each field is truncated
# defensively so a single verbose ticket can't blow the prompt window.
# Caller is responsible for skipping tickets where every field is empty.
# ─────────────────────────────────────────────────────────────
_FIELD_TRUNCATE = 1200   # per-field char cap before sending to LLM


def _trunc(value: Any, limit: int = _FIELD_TRUNCATE) -> Optional[str]:
    """Stringify + truncate. Returns None for empty/missing input so
    the JSON payload omits the field cleanly (Haiku reads "absent"
    correctly; "" tends to be paraphrased as "no data" in the output)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _build_single_ticket_payload(
    ticket: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Render ONE ticket's seven source fields into LLM input JSON.

    Returns ``(json_string, incident_number)``. Both are None when the
    ticket lacks an Incident_Number or has no usable content across
    any of the seven fields (caller skips → no LLM call for that ticket).
    """
    if not isinstance(ticket, dict):
        return (None, None)

    meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
    inc = _trunc(meta.get("Incident_Number"), 64) if isinstance(meta, dict) else None
    if not inc:
        return (None, None)

    rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    sop = ticket.get("Operational_SOP") if isinstance(ticket.get("Operational_SOP"), dict) else {}

    # ── Field 1: Technical_Snapshot ──
    technical_snapshot = _trunc(rca.get("Technical_Snapshot") if rca else None)

    # ── Field 2: Resolution_Steps ──
    resolution_steps: List[str] = []
    rsteps = rca.get("Resolution_Steps") if rca else None
    if isinstance(rsteps, list):
        for s in rsteps:
            if isinstance(s, dict):
                txt = _trunc(s.get("description") or s.get("action") or s.get("step"))
            else:
                txt = _trunc(s)
            if txt:
                resolution_steps.append(txt)
    elif isinstance(rsteps, str):
        t = _trunc(rsteps)
        if t:
            resolution_steps.append(t)

    # ── Field 3: Diagnostic_Tests_Executed ──
    diagnostic_tests: List[str] = []
    tle = _safe_get(ticket, "Troubleshooting_Ledger", "Diagnostic_Tests_Executed")
    if isinstance(tle, list):
        for entry in tle:
            if isinstance(entry, dict):
                txt = _trunc(
                    entry.get("name")
                    or entry.get("description")
                    or entry.get("test")
                    or entry.get("action")
                )
            else:
                txt = _trunc(entry)
            if txt:
                diagnostic_tests.append(txt)

    # ── Field 4: diagnostic_logic_chunks (carries Intent + Pivot + Command) ──
    diagnostic_logic_chunks: List[Dict[str, Optional[str]]] = []
    chunks = sop.get("diagnostic_logic_chunks") if isinstance(sop, dict) else None
    if isinstance(chunks, list):
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            action = _trunc(ch.get("action") or ch.get("step_id") or ch.get("step"))
            if not action:
                continue
            diagnostic_logic_chunks.append({
                "action": action,
                "intent": _trunc(ch.get("context") or ch.get("rationale") or ch.get("intent")),
                "pivot": _trunc(ch.get("branching_logic") or ch.get("pivot")),
                "command": _trunc(ch.get("command")),
            })

    # ── Field 5: Key_Movements_Timeline.Action[] ──
    timeline_actions: List[str] = []
    timeline = _safe_get(ticket, "Forensic_Performance_Audit", "Key_Movements_Timeline")
    if isinstance(timeline, list):
        for mv in timeline:
            if isinstance(mv, dict):
                txt = _trunc(mv.get("Action"))
                if txt:
                    timeline_actions.append(txt)

    # ── Field 6: Critical_Intervention ──
    ci_raw = _safe_get(ticket, "Forensic_Performance_Audit", "Critical_Intervention")
    critical_intervention = _trunc(ci_raw)

    # ── Field 7: Hero_Action ──
    ha_raw = _safe_get(ticket, "Key_Contributors", "Key_Impact_Players", "Hero_Action")
    hero_action = _trunc(ha_raw)

    # Drop ticket entirely if zero of the seven fields carried content.
    has_any = any([
        technical_snapshot,
        resolution_steps,
        diagnostic_tests,
        diagnostic_logic_chunks,
        timeline_actions,
        critical_intervention,
        hero_action,
    ])
    if not has_any:
        return (None, inc)

    executive_summary = _trunc(rca.get("Executive_Summary") if rca else None, 600)

    payload: Dict[str, Any] = {
        "incident_number": inc,
        "executive_summary": executive_summary,
        "technical_snapshot": technical_snapshot,
        "resolution_steps": resolution_steps,
        "diagnostic_tests_executed": diagnostic_tests,
        "diagnostic_logic_chunks": diagnostic_logic_chunks,
        "key_movements_timeline_actions": timeline_actions,
        "critical_intervention": critical_intervention,
        "hero_action": hero_action,
    }
    try:
        return (json.dumps(payload, ensure_ascii=False, default=str), inc)
    except Exception:
        return (str(payload), inc)


# ─────────────────────────────────────────────────────────────
# LEGACY cross-ticket parser (commented — kept on disk for reference).
# Parsed the old `consolidated_steps[]` shape. Replaced by
# `_parse_single_activity` which parses the per-ticket Activity object.
# ─────────────────────────────────────────────────────────────
# def _parse_response(raw: str) -> List[ConsolidatedStep]:
#     """Parse the LLM JSON into a list of ConsolidatedStep. Raises on
#     schema deviation so the caller can fall back."""
#     s = (raw or "").strip()
#     if not s:
#         raise ValueError("empty response")
#     if s.startswith("```"):
#         s = s.strip("`")
#         s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
#     open_idx = s.find("{")
#     close_idx = s.rfind("}")
#     if open_idx < 0 or close_idx < 0 or close_idx <= open_idx:
#         raise ValueError("no JSON object found")
#     parsed = json.loads(s[open_idx:close_idx + 1])
#     raw_steps = parsed.get("consolidated_steps")
#     if not isinstance(raw_steps, list) or not raw_steps:
#         raise ValueError("missing/empty consolidated_steps")
#
#     out: List[ConsolidatedStep] = []
#     for entry in raw_steps:
#         if not isinstance(entry, dict):
#             continue
#         if not all(k in entry for k in _REQUIRED_KEYS):
#             continue
#         try:
#             n = int(entry["step_number"])
#         except (TypeError, ValueError):
#             continue
#         action = entry.get("action")
#         intent = entry.get("intent")
#         pivot = entry.get("pivot")
#         command = entry.get("command")
#         if not (isinstance(action, str) and action.strip()):
#             continue
#         if not (isinstance(intent, str) and intent.strip()):
#             continue
#         if not (isinstance(pivot, str) and pivot.strip()):
#             continue
#         if not _is_read_only_action(action):
#             logger.warning(
#                 "[stage3_consolidated] dropping non-read-only step: %r",
#                 action[:80],
#             )
#             continue
#         out.append(ConsolidatedStep(
#             step_number=n,
#             action=_strip_step_refs(action.strip()),
#             intent=_strip_step_refs(intent.strip()),
#             pivot=_strip_step_refs(pivot.strip()),
#             command=command.strip() if isinstance(command, str) and command.strip() else None,
#         ))
#         if len(out) >= 5:
#             break
#
#     if not out:
#         raise ValueError("no valid steps after read-only filter")
#
#     return [
#         s.model_copy(update={"step_number": i + 1})
#         for i, s in enumerate(out)
#     ]


# ─────────────────────────────────────────────────────────────
# ACTIVE per-ticket parser.
# Reads a single Activity JSON object (action / intent / pivot /
# command) from one LLM call's output. Returns the parsed dict, or
# raises on schema deviation so the per-ticket loop can skip that
# ticket without crashing the whole ledger.
# ─────────────────────────────────────────────────────────────
def _parse_single_activity(raw: str) -> Dict[str, Optional[str]]:
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty response")
    if s.startswith("```"):
        s = s.strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
    open_idx = s.find("{")
    close_idx = s.rfind("}")
    if open_idx < 0 or close_idx < 0 or close_idx <= open_idx:
        raise ValueError("no JSON object found")
    parsed = json.loads(s[open_idx:close_idx + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")

    action = parsed.get("action")
    intent = parsed.get("intent")
    pivot = parsed.get("pivot")
    command = parsed.get("command")
    if not (isinstance(action, str) and action.strip()):
        raise ValueError("missing/empty action")
    if not (isinstance(intent, str) and intent.strip()):
        raise ValueError("missing/empty intent")
    if not (isinstance(pivot, str) and pivot.strip()):
        raise ValueError("missing/empty pivot")

    return {
        "action": _strip_step_refs(action.strip()),
        "intent": _strip_step_refs(intent.strip()),
        "pivot": _strip_step_refs(pivot.strip()),
        "command": command.strip() if isinstance(command, str) and command.strip() else None,
    }


# ─────────────────────────────────────────────────────────────
# Public entry point — per-ticket consolidation.
#
# One LLM call PER cohort ticket. Each call produces ONE Activity
# grounded in that ticket's seven source fields. Cohort of N tickets
# → up to N Activities (skips tickets with no source content). The
# function signature is preserved so build_stage3 / the consolidated
# cache wrapper don't need to change — only the body and the meaning
# of `consolidated_synthesis_skipped` (now: True iff EVERY ticket's
# call failed, not just one).
# ─────────────────────────────────────────────────────────────
def build_consolidated_ledger(
    cohort: List[Dict[str, Any]],
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> Tuple[List[ConsolidatedStep], bool]:
    """Returns ``(activities, synthesis_skipped)``.

    Each Activity in the returned list corresponds to one cohort ticket
    and carries that ticket's ``incident_number``. Failure-open: a per-
    ticket parse error is logged and that ticket is silently skipped;
    the function only sets ``synthesis_skipped=True`` when every ticket
    failed (or the SDK import failed up front), so the panel can show
    the unavailable banner.
    """
    if not cohort:
        return ([], False)

    if generate_fn is None:
        try:
            from backend.api import safe_generate as generate_fn  # type: ignore
        except Exception as exc:
            logger.warning(
                "[stage3_consolidated] safe_generate import failed (%s) — "
                "shipping empty ledger", exc,
            )
            return ([], True)

    activities: List[ConsolidatedStep] = []
    attempted = 0
    skipped: List[str] = []

    for ticket in cohort:
        payload, inc = _build_single_ticket_payload(ticket)
        if payload is None:
            # No usable content across the seven fields for this ticket.
            # Not an LLM failure — just skip silently.
            if inc:
                logger.info(
                    "[stage3_consolidated] inc=%s no source content — skipping",
                    inc,
                )
            continue

        attempted += 1
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"INPUT TICKET JSON:\n{payload}\n\n"
            f"OUTPUT JSON:"
        )

        try:
            raw = generate_fn(prompt, _MAX_TOKENS)
            parsed = _parse_single_activity(raw)
        except Exception as exc:
            logger.warning(
                "[stage3_consolidated] inc=%s synthesis failed (%s)",
                inc, exc,
            )
            skipped.append(inc or "<unknown>")
            continue

        # The read-only / disruptive-verb filter is intentionally left
        # in place — it runs after the LLM and drops Activities whose
        # action verb suggests a config-changing operation. Belt-and-
        # suspenders against an off-script generation. Remove later
        # only if logs confirm legitimate Activities are being dropped.
        if not _is_read_only_action(parsed["action"]):
            logger.warning(
                "[stage3_consolidated] inc=%s dropping non-read-only "
                "Activity: %r", inc, parsed["action"][:80],
            )
            skipped.append(inc or "<unknown>")
            continue

        activities.append(ConsolidatedStep(
            step_number=len(activities) + 1,
            action=parsed["action"],
            intent=parsed["intent"],
            pivot=parsed["pivot"],
            command=parsed["command"],
            incident_number=inc,
        ))

    logger.info(
        "[stage3_consolidated] cohort=%d attempted=%d activities=%d skipped=%d",
        len(cohort), attempted, len(activities), len(skipped),
    )

    # `synthesis_skipped` is the banner gate. Show the banner only
    # when we attempted at least one call AND every attempt failed.
    # If the cohort produced zero candidates (no LLM calls made),
    # leave it False so the UI renders an empty section rather than
    # a misleading error banner.
    if attempted > 0 and not activities:
        return ([], True)
    return (activities, False)
