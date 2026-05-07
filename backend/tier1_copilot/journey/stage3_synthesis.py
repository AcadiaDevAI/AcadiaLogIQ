"""Sprint 13 Stage 3 — LLM synthesis pass for Guided Workflows.

Layered on top of `stage3_guided_workflows.build_guided_workflows`.
Takes the structurally harvested per-ticket workflows and rewrites:

  * ``header`` — short 3-7 word topical title (e.g. "BGP Session
    Flap + BFD Timer Expiration + Control Plane CPU Saturation")
  * ``intent`` — one-line hypothesis the step is testing, in the
    voice of *this* ticket's playbook
  * ``pivot``  — two-clause prose decision: "If X, then Y. If not,
    then Z." Strict rule: NEVER references "Step N" — within a
    single ticket scope we describe outcomes in plain action prose.

What we do NOT touch:
  * ``action`` — verbatim source text. The engineer might paste it
    into a console; never let the LLM rephrase it.
  * ``command`` — verbatim.
  * ``source_field`` — telemetry only; never re-derived.

Failure stance:
  Failure-open. LLM error / bad JSON / missing template fields →
  return the original verbatim workflow with ``synthesis_skipped=True``.
  The frontend renders verbatim Intent/Pivot (or omits them when
  missing) — same UX as pre-synthesis.

Cost shape:
  One Haiku call per ticket. Cohort of 5 → 5 calls. Cohort cache
  upstream means re-renders of /stage-3 don't re-call the LLM
  within a session.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .schemas import GuidedWorkflow, GuidedWorkflowStep


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
# Why such a strict prompt:
#   - Action text MUST stay verbatim — engineer pastes commands.
#   - Pivot MUST NOT reference step numbers. Reason: even though
#     step numbers are local to this workflow now, the prose reads
#     better when it describes the *signal interpretation* and
#     next *action* directly. "If clean, the carrier path is
#     healthy — focus on CPE logs next" beats "If clean, go to
#     Step 3" every time.
#   - Output is strict JSON so we can parse deterministically and
#     fall back cleanly when the model goes off-script.
_SYSTEM_PROMPT = """You are an Expert Tier-1 IT Service Management author. You will be given ONE historical ticket's diagnostic playbook. Your job is to rewrite the human-facing prose for a Tier-1 engineer who is about to follow these steps live.

OUTPUT REQUIREMENTS — STRICT:
1. Output ONLY a single JSON object. No prose before or after. No code fences.
2. JSON shape:
   {
     "header": "<3-7 word topical title for this ticket's playbook>",
     "steps": [
       {"step_number": <int>, "intent": "<one sentence>", "pivot": "<one or two sentences>"},
       ...
     ]
   }
3. The "step_number" values MUST match the input verbatim — same count, same numbers, same order.

WRITING RULES:
- Intent: ONE sentence, MAXIMUM 20 words, stating the diagnostic hypothesis the step is testing. Prefer "Confirm whether..." / "Determine whether..." / "Identify whether...". Plain prose; no bullets.
- Pivot: ONE OR TWO sentences, MAXIMUM 40 words TOTAL. State what the result MEANS and what to DO next, in actions. Use "If X, then Y. If not, then Z." structure when a binary outcome is meaningful. Otherwise a single-clause guidance sentence.
- NEVER reference "Step N" in either Intent or Pivot. Describe the next action in plain words, not by index. (The user reads them sequentially and doesn't think in step numbers.)
- NEVER invent commands, IP addresses, ticket numbers, or specific values not present in the input.
- Keep technical vocabulary that's already in the action text (BGP, BFD, MED, GTSM, MTU, etc.).
- Header: name the technical pattern this ticket represents in 3-7 words. Use "+" to join multiple themes. Example shape: "BGP Session Flap + BFD Timer Expiration".

If the input action lacks any source-side intent or pivot, synthesize them anyway from the action text and the surrounding context. Do not return null fields."""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
_STEP_REF_RE = re.compile(r"\b(?:go to|proceed to|return to|jump to|see)\s+step\s*\d+\b", re.IGNORECASE)


# Sprint 13.1 — token budget scales with workflow size so a future
# cohort with a long-tail ticket (12+ steps) can't truncate the JSON
# response and lose the trailing steps to validation failure.
#
# Per-step rough budget (Haiku output, post-prompt-tightening):
#   intent  ≤ 20 words ≈ 30 tokens
#   pivot   ≤ 40 words ≈ 60 tokens
#   JSON noise (commas, braces, keys) ≈ 15 tokens
#   safety multiplier (variance / occasional verbose Haiku) ≈ 1.6×
# → ~170 tokens/step rounded up to 250 for headroom.
#
# Plus 120 tokens fixed for the header line + outer JSON wrapper.
_TOK_PER_STEP = 250
_TOK_FIXED_OVERHEAD = 120
_TOK_MIN = 800   # floor for short workflows so Haiku never under-budgets


def _max_tokens_for(workflow) -> int:
    """Workflow-aware output budget for the synthesis call."""
    n = len(workflow.steps) if workflow and workflow.steps else 0
    return max(_TOK_MIN, _TOK_FIXED_OVERHEAD + n * _TOK_PER_STEP)


def _strip_step_refs(s: Optional[str]) -> Optional[str]:
    """Defensive scrub — strip any "go to Step N" / "proceed to
    Step N" pattern from prose that slips through. Only fires as
    a belt-and-suspenders against a model that occasionally drops
    the rule."""
    if not s:
        return s
    cleaned = _STEP_REF_RE.sub("continue with the next check", s)
    # Collapse double spaces caused by the substitution.
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _build_payload(workflow: GuidedWorkflow, ticket: Dict[str, Any]) -> str:
    """Render the workflow + just-enough RCA context as the LLM input.
    We send:
      - ticket id + raw incident headline
      - executive summary + technical snapshot for grounding
      - per-step records (action verbatim + any verbatim intent/pivot/command)
    Ticket dict is passed in so we can pull RCA fields without leaking
    them into the typed schema.
    """
    rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    inc_summary = ticket.get("Incident_Summary") if isinstance(ticket.get("Incident_Summary"), dict) else {}

    payload: Dict[str, Any] = {
        "incident_number": workflow.incident_number,
        "incident_headline": (inc_summary.get("INCIDENT") if inc_summary else None) or workflow.incident_number,
        "executive_summary": rca.get("Executive_Summary") if rca else None,
        "technical_snapshot": workflow.technical_snapshot,
        "steps": [
            {
                "step_number": s.step_number,
                "action": s.action,
                "source_intent": s.intent,
                "source_pivot": s.pivot,
                "command": s.command,
                "source_field": s.source_field,
            }
            for s in workflow.steps
        ],
    }
    # JSON serialise; truncate any over-long string defensively so the
    # prompt stays under Haiku's window even on pathological tickets.
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        # Last-ditch — stringify whatever survived.
        return str(payload)


def _parse_response(raw: str, expected_step_numbers: List[int]) -> Dict[str, Any]:
    """Parse the LLM response into ``{header: str, steps: {step_number: {intent, pivot}}}``.
    Raises on any deviation so the caller can fall back."""
    s = (raw or "").strip()
    if not s:
        raise ValueError("empty response")
    # Strip code fences if the model adds them despite the instruction.
    if s.startswith("```"):
        s = s.strip("`")
        # Drop a leading "json" / "JSON" token after the fence.
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
    # The model occasionally prepends a one-word ack; locate the JSON.
    open_idx = s.find("{")
    close_idx = s.rfind("}")
    if open_idx < 0 or close_idx < 0 or close_idx <= open_idx:
        raise ValueError("no JSON object found")
    parsed = json.loads(s[open_idx:close_idx + 1])
    header = parsed.get("header")
    steps = parsed.get("steps")
    if not isinstance(header, str) or not header.strip():
        raise ValueError("missing/empty header")
    if not isinstance(steps, list) or not steps:
        raise ValueError("missing/empty steps")
    by_num: Dict[int, Dict[str, Any]] = {}
    for entry in steps:
        if not isinstance(entry, dict):
            continue
        try:
            n = int(entry.get("step_number"))
        except (TypeError, ValueError):
            continue
        intent = entry.get("intent")
        pivot = entry.get("pivot")
        if isinstance(intent, str) and intent.strip() and isinstance(pivot, str) and pivot.strip():
            by_num[n] = {
                "intent": intent.strip(),
                "pivot": pivot.strip(),
            }
    # Require coverage of every input step. Anything less = drop to fallback.
    missing = [n for n in expected_step_numbers if n not in by_num]
    if missing:
        raise ValueError(f"missing synthesis for step_numbers={missing}")
    return {"header": header.strip(), "steps": by_num}


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def synthesize_workflows(
    workflows: List[GuidedWorkflow],
    ticket_lookup: Dict[str, Dict[str, Any]],
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> List[GuidedWorkflow]:
    """Run one Haiku call per workflow and merge the synthesised
    header + per-step Intent/Pivot back into the typed records.

    Args:
        workflows: structurally-built workflows from
            :func:`stage3_guided_workflows.build_guided_workflows`.
        ticket_lookup: ``{incident_number: ticket_metadata_json}`` —
            so the synthesizer can ground prose in the RCA / executive
            summary without round-tripping through the typed schema.
        generate_fn: LLM call. Signature ``(prompt, max_tokens) -> str``.
            Defaults to :func:`backend.api.safe_generate`.

    Returns:
        New list of GuidedWorkflow (originals are not mutated). On any
        per-workflow error the original verbatim workflow is preserved
        with ``synthesis_skipped=True``. Never raises.
    """
    if not workflows:
        return []

    if generate_fn is None:
        try:
            from backend.api import safe_generate as generate_fn  # type: ignore
        except Exception as exc:
            logger.warning(
                "[stage3_synthesis] safe_generate import failed (%s) — "
                "all workflows ship verbatim", exc,
            )
            return [w.model_copy(update={"synthesis_skipped": True}) for w in workflows]

    out: List[GuidedWorkflow] = []
    skipped = 0
    for wf in workflows:
        ticket = ticket_lookup.get(wf.incident_number) or {}
        payload = _build_payload(wf, ticket)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"INPUT TICKET (JSON):\n{payload}\n\n"
            f"OUTPUT JSON:"
        )
        try:
            raw = generate_fn(prompt, _max_tokens_for(wf))
            parsed = _parse_response(
                raw, [s.step_number for s in wf.steps],
            )
            new_steps: List[GuidedWorkflowStep] = []
            for s in wf.steps:
                p = parsed["steps"].get(s.step_number)
                if not p:
                    # Shouldn't reach here — coverage validated in
                    # _parse_response — but defensive.
                    new_steps.append(s)
                    continue
                new_steps.append(s.model_copy(update={
                    "intent": _strip_step_refs(p["intent"]),
                    "pivot": _strip_step_refs(p["pivot"]),
                }))
            out.append(wf.model_copy(update={
                "header": parsed["header"],
                "steps": new_steps,
                "synthesis_skipped": False,
            }))
        except Exception as exc:
            logger.warning(
                "[stage3_synthesis] synthesis failed for %s (%s) — "
                "shipping verbatim", wf.incident_number, exc,
            )
            out.append(wf.model_copy(update={"synthesis_skipped": True}))
            skipped += 1

    logger.info(
        "[stage3_synthesis] workflows=%d synthesised=%d skipped=%d",
        len(workflows), len(workflows) - skipped, skipped,
    )
    return out
