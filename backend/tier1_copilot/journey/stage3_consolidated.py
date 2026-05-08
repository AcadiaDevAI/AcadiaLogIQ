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
# Prompt — user spec verbatim, with output discipline appended
# ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """Merge the diagnostic steps from all tickets into one optimal, sequential execution ledger. Deduplicate identical checks. For each consolidated test, state the Intent (hypothesis) and the Pivot (how the result dictates the next step). If Ticket A and Ticket B used different successful interventions, present them as sequential options (e.g., 'Attempt Fix A; if it fails, proceed to Fix B').
Generate troubleshooting steps specifically suitable for a Tier 1 engineer with read-only access.

Guidelines:
- Only include safe, non-intrusive actions such as:
  • executing show/read-only commands
  • capturing logs
  • performing observational checks
- Do NOT suggest any actions that modify configurations, restart services, or could cause outages or performance degradation.
- Keep all steps low-risk and safe for production environments.
- Limit the total number of steps to a maximum of 5.

Ensure the output is concise, practical, and immediately executable by a Tier 1 engineer.

OUTPUT REQUIREMENTS — STRICT:
1. Output ONLY a single JSON object. No prose before or after. No code fences.
2. JSON shape:
   {
     "consolidated_steps": [
       {
         "step_number": <int 1..5>,
         "action": "<one short, read-only action>",
         "intent": "<one sentence hypothesis>",
         "pivot": "<one or two sentences: what the result means and what to do next>",
         "command": "<show / read-only command if directly applicable, else omit>"
       },
       ...
     ]
   }
3. step_number values MUST be 1, 2, 3, 4, 5 in order (omit later ones if fewer than 5 steps qualify).
4. Maximum 5 entries. Fewer is fine if only fewer safe candidates exist.

WRITING RULES:
- action: ONE sentence, ≤ 18 words. Begin with a read-only verb (Verify, Check, Capture, Inspect, Run, Review, Confirm, Compare). NEVER use Restart, Apply, Configure, Set, Modify, Delete, Remove, Adjust, Update, Push, Reload, Clear-counters, Reset.
- intent: ONE sentence, ≤ 20 words. Diagnostic hypothesis the step is testing.
- pivot: ONE OR TWO sentences, ≤ 40 words total. Describe the result interpretation + next action in plain prose. NEVER reference step numbers ("go to Step 3").
- command: ONLY when the input candidates include a real command string. Drawn verbatim from input — do NOT invent commands or flags.
- For divergent successful interventions across tickets: prefer framing the SAFER read-only diagnostic that distinguishes which fix path applies, not the fix itself. Only include "Attempt Fix A; if it fails, proceed to Fix B" framing when both fixes are themselves read-only / safe.
- Do NOT invent IPs, hostnames, ASNs, ticket numbers, or specific values not present in the input."""


# ─────────────────────────────────────────────────────────────
# Token budget — cap at 5 steps so the budget is fixed.
#   step: action ~30 + intent ~30 + pivot ~60 + command ~25 + JSON ~20 = ~165 tokens
#   header / outer JSON = ~80 tokens
#   safety multiplier = 1.5×
# = ~1300 tokens. Use 1500 for headroom.
# ─────────────────────────────────────────────────────────────
_MAX_TOKENS = 1500


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


def _build_payload(
    cohort: List[Dict[str, Any]],
) -> Tuple[str, int]:
    """Render the cohort's candidate actions + RCA context into the
    LLM input JSON. Returns ``(json_string, candidate_count)``."""
    cohort_context: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    seen_signatures: set = set()    # casefold dedupe at the source

    for ticket in cohort or []:
        if not isinstance(ticket, dict):
            continue
        meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
        rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
        inc = meta.get("Incident_Number") if isinstance(meta, dict) else None
        summary = rca.get("Executive_Summary") if isinstance(rca, dict) else None
        if inc and summary:
            cohort_context.append({
                "incident_number": str(inc),
                "executive_summary": str(summary)[:500],
            })

        for s in _harvest_steps(ticket):
            sig = s.action.casefold().strip()
            if not sig or sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            candidates.append({
                "incident_number": str(inc) if inc else None,
                "action": s.action,
                "source_intent": s.intent,
                "source_pivot": s.pivot,
                "command": s.command,
                "source_field": s.source_field,
            })

    payload = {
        "cohort_context": cohort_context[:5],
        "candidate_actions": candidates,
    }
    try:
        return json.dumps(payload, ensure_ascii=False, default=str), len(candidates)
    except Exception:
        return str(payload), len(candidates)


def _parse_response(raw: str) -> List[ConsolidatedStep]:
    """Parse the LLM JSON into a list of ConsolidatedStep. Raises on
    schema deviation so the caller can fall back."""
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
    raw_steps = parsed.get("consolidated_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("missing/empty consolidated_steps")

    out: List[ConsolidatedStep] = []
    for entry in raw_steps:
        if not isinstance(entry, dict):
            continue
        if not all(k in entry for k in _REQUIRED_KEYS):
            continue
        try:
            n = int(entry["step_number"])
        except (TypeError, ValueError):
            continue
        action = entry.get("action")
        intent = entry.get("intent")
        pivot = entry.get("pivot")
        command = entry.get("command")
        if not (isinstance(action, str) and action.strip()):
            continue
        if not (isinstance(intent, str) and intent.strip()):
            continue
        if not (isinstance(pivot, str) and pivot.strip()):
            continue
        # Belt-and-suspenders: drop any step whose action contains a
        # disruptive verb. Better to ship 4 safe steps than 5 with a
        # restart slipped in.
        if not _is_read_only_action(action):
            logger.warning(
                "[stage3_consolidated] dropping non-read-only step: %r",
                action[:80],
            )
            continue
        out.append(ConsolidatedStep(
            step_number=n,
            action=_strip_step_refs(action.strip()),
            intent=_strip_step_refs(intent.strip()),
            pivot=_strip_step_refs(pivot.strip()),
            command=command.strip() if isinstance(command, str) and command.strip() else None,
        ))
        if len(out) >= 5:
            break

    if not out:
        raise ValueError("no valid steps after read-only filter")

    # Re-number 1..N so any gaps from the read-only filter are closed.
    return [
        s.model_copy(update={"step_number": i + 1})
        for i, s in enumerate(out)
    ]


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def build_consolidated_ledger(
    cohort: List[Dict[str, Any]],
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> Tuple[List[ConsolidatedStep], bool]:
    """Returns ``(consolidated_steps, synthesis_skipped)``. Never
    raises. ``synthesis_skipped=True`` means the LLM call failed and
    the caller should fall back to whatever non-LLM surfaces it has
    (per-ticket detail accordion remains rendered downstream)."""
    if not cohort:
        return ([], False)

    payload, n_candidates = _build_payload(cohort)
    if n_candidates == 0:
        logger.info(
            "[stage3_consolidated] cohort=%d candidates=0 — skipping LLM",
            len(cohort),
        )
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

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"INPUT JSON:\n{payload}\n\n"
        f"OUTPUT JSON:"
    )

    try:
        raw = generate_fn(prompt, _MAX_TOKENS)
        steps = _parse_response(raw)
    except Exception as exc:
        logger.warning(
            "[stage3_consolidated] synthesis failed (%s) — shipping "
            "empty ledger", exc,
        )
        return ([], True)

    logger.info(
        "[stage3_consolidated] cohort=%d candidates=%d shown=%d",
        len(cohort), n_candidates, len(steps),
    )
    return (steps, False)
