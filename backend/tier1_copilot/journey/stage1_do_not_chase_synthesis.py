"""Sprint 13.3 Stage 1B — LLM synthesis pass for "What NOT to chase".

Layered on top of `stage1_do_not_chase.build_stage1b`. Takes the
structurally harvested + casefold-deduped `Stage1bDoNotChase`
result and rewrites each entry's prose into a strict 'Anti-Waste'
checklist sentence per the user-supplied spec, grounding the
rewrite in the cohort's RCA / Executive_Summary context so the
warning reads like a real Tier-2 author wrote it for the current
engineer.

What this rewrites:
  - misleading_signal — keep the technical claim, sharpen wording
  - rule_out_logic    — replace the generic placeholder
                        ("Checked and found healthy in past
                        incidents — verify but do not deep-dive.")
                        with grounded prose that warns the engineer
                        why this signal looks broken but isn't.

What this does NOT touch:
  - occurrence_count       — structural metric, preserved 1:1
  - seen_in_incidents      — preserved 1:1 (input order matches output)
  - reason / empty flags   — preserved on the result wrapper

1:1 mapping — input entry N maps to output entry N. We don't
consolidate semantically equivalent entries via the LLM (that
would shuffle counts/incidents, raising failure modes); the
caller keeps casefold-dedupe as the dedupe layer. Future work
can extend this to semantic consolidation with explicit source
indices.

Failure stance:
  Failure-open. LLM error / bad JSON / missing entries → return
  the original `Stage1bDoNotChase` unchanged with
  ``synthesis_skipped=True``. The frontend renders verbatim
  source text (today's behaviour).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .schemas import DoNotChaseEntry, Stage1bDoNotChase


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Prompt — user spec verbatim, plus output discipline.
# ─────────────────────────────────────────────────────────────
# The first paragraph is the user's spec wording; the second adds
# the JSON-output contract we need to parse deterministically. We
# intentionally do NOT ask the model to invent new entries or
# reorder the list — the harvested set is the universe; the model
# only polishes prose per row.
_SYSTEM_PROMPT = """You are an Expert IT Service Management author writing a strict 'Anti-Waste' checklist for a Tier-1 engineer who is currently triaging an active ticket.

PROCEDURE (verbatim spec):
Aggregate all false paths, red herrings, and misleading signals from the ticket batch. Deduplicate them into a single, strict 'Anti-Waste' checklist. Provide the rule-out logic for each, warning the current engineer exactly which systems might look broken but were proven to be secondary symptoms in all historical cases.

YOUR JOB:
You will receive an INPUT JSON payload with:
  - "cohort_context": short executive summaries of the historical tickets the entries came from
  - "entries": an ordered list, each with {index, misleading_signal, rule_out_logic, occurrence_count, seen_in_incidents}

For EACH entry, output a polished version of misleading_signal and rule_out_logic.

OUTPUT REQUIREMENTS — STRICT:
1. Output ONLY a single JSON object. No prose before or after. No code fences.
2. JSON shape:
   {
     "entries": [
       {"index": <int>, "misleading_signal": "<rewritten>", "rule_out_logic": "<rewritten>"},
       ...
     ]
   }
3. The "index" values MUST exactly match the input — same count, same indices, same order. Do not add, drop, or reorder entries.

WRITING RULES:
- misleading_signal: ONE sentence naming the specific system/symptom that LOOKED like the cause. ≤ 18 words. Use plain technical English. Lead with the noun (the system/component), e.g., "Individual fax device hardware fault." Do not start with "If".
- rule_out_logic: ONE OR TWO sentences explaining why this looks broken but is a secondary symptom in all historical cases, plus what to focus on instead. ≤ 40 words total. Mention the actual diagnostic evidence that ruled it out when the input includes it.
- NEVER invent specifics (IPs, ticket numbers, command flags) not present in the input.
- NEVER reference step numbers — this checklist is a flat list of what NOT to chase.
- If the input rule_out_logic is the generic placeholder ("Checked and found healthy in past incidents — verify but do not deep-dive."), REPLACE it with grounded prose drawn from the misleading_signal + cohort_context. Do not echo the placeholder back.
- Keep technical vocabulary that's already in the input (BGP, BFD, MED, GTSM, MTU, CPU, firewall, etc.).
- Preserve the spirit of "warning the current engineer" — second-person voice ("you'll see…", "this looks like…") is fine; first-person is not."""


# ─────────────────────────────────────────────────────────────
# Token budget — scales with entry count so a 30-entry corpus
# doesn't truncate. Per the prompt's word caps:
#   misleading_signal ≤ 18 words ≈ 28 tokens
#   rule_out_logic    ≤ 40 words ≈ 60 tokens
#   JSON noise        ≈ 15 tokens
#   safety multiplier ≈ 1.5×
# = ~150 tokens / entry, plus ~120 fixed overhead.
# ─────────────────────────────────────────────────────────────
def _max_tokens_for(n_entries: int) -> int:
    return max(800, 120 + n_entries * 150)


_INDEX_REQUIRED_KEYS = ("index", "misleading_signal", "rule_out_logic")


def _build_payload(
    do_not_chase: Stage1bDoNotChase,
    cohort: List[Dict[str, Any]],
) -> str:
    """Render the harvested entries + a compact cohort RCA context
    into the LLM input JSON. Cohort context is capped to the top 3
    tickets' executive summaries — enough grounding without bloating
    the prompt."""
    cohort_context: List[Dict[str, Any]] = []
    for ticket in (cohort or [])[:3]:
        if not isinstance(ticket, dict):
            continue
        meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
        rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
        inc = meta.get("Incident_Number") if isinstance(meta, dict) else None
        summary = rca.get("Executive_Summary") if isinstance(rca, dict) else None
        if inc and summary:
            cohort_context.append({
                "incident_number": str(inc),
                "executive_summary": str(summary)[:600],
            })

    payload = {
        "cohort_context": cohort_context,
        "entries": [
            {
                "index": i,
                "misleading_signal": e.misleading_signal,
                "rule_out_logic": e.rule_out_logic,
                "occurrence_count": e.occurrence_count,
                "seen_in_incidents": list(e.seen_in_incidents or [])[:3],
            }
            for i, e in enumerate(do_not_chase.entries)
        ],
    }
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)


def _parse_response(raw: str, n_entries: int) -> Dict[int, Dict[str, str]]:
    """Parse the LLM JSON into ``{index: {misleading_signal, rule_out_logic}}``.
    Raises on any deviation so the caller can fall back."""
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
    entries = parsed.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing/empty entries")

    by_index: Dict[int, Dict[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not all(k in entry for k in _INDEX_REQUIRED_KEYS):
            continue
        try:
            idx = int(entry["index"])
        except (TypeError, ValueError):
            continue
        sig = entry.get("misleading_signal")
        rule = entry.get("rule_out_logic")
        if not isinstance(sig, str) or not sig.strip():
            continue
        if not isinstance(rule, str) or not rule.strip():
            continue
        by_index[idx] = {
            "misleading_signal": sig.strip(),
            "rule_out_logic": rule.strip(),
        }

    missing = [i for i in range(n_entries) if i not in by_index]
    if missing:
        raise ValueError(f"missing rewrites for index={missing}")
    return by_index


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def synthesize_do_not_chase(
    do_not_chase: Stage1bDoNotChase,
    cohort: List[Dict[str, Any]],
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> Stage1bDoNotChase:
    """Run one Haiku call; merge the rewritten signal/rule_out text
    back into the structural records. Returns a new
    `Stage1bDoNotChase` (originals not mutated). Never raises."""
    # Skip cleanly when there's nothing to rewrite — saves an LLM
    # call on empty cohorts and on the ``no_data`` reason path.
    if not do_not_chase or not do_not_chase.entries:
        return do_not_chase.model_copy(update={"synthesis_skipped": True}) if do_not_chase else do_not_chase

    if generate_fn is None:
        try:
            from backend.api import safe_generate as generate_fn  # type: ignore
        except Exception as exc:
            logger.warning(
                "[stage1b_synthesis] safe_generate import failed (%s) — "
                "shipping verbatim", exc,
            )
            return do_not_chase.model_copy(update={"synthesis_skipped": True})

    payload = _build_payload(do_not_chase, cohort)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"INPUT JSON:\n{payload}\n\n"
        f"OUTPUT JSON:"
    )

    try:
        raw = generate_fn(prompt, _max_tokens_for(len(do_not_chase.entries)))
        parsed = _parse_response(raw, len(do_not_chase.entries))
    except Exception as exc:
        logger.warning(
            "[stage1b_synthesis] synthesis failed (%s) — shipping verbatim",
            exc,
        )
        return do_not_chase.model_copy(update={"synthesis_skipped": True})

    new_entries: List[DoNotChaseEntry] = []
    for i, e in enumerate(do_not_chase.entries):
        rewrite = parsed.get(i)
        if not rewrite:
            new_entries.append(e)
            continue
        new_entries.append(e.model_copy(update={
            "misleading_signal": rewrite["misleading_signal"],
            "rule_out_logic": rewrite["rule_out_logic"],
        }))

    logger.info(
        "[stage1b_synthesis] entries=%d synthesised=1 reason=%s",
        len(new_entries), do_not_chase.reason,
    )
    return do_not_chase.model_copy(update={
        "entries": new_entries,
        "synthesis_skipped": False,
    })
