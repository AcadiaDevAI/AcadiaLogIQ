"""Sprint 10 Stage 1A — Smoking Gun aggregator (no LLM in MVP).

Per spec §3.2:
  1. Walk every cohort ticket's Knowledge_Base[*]; extract
     (pivot_data_point, shift_in_logic, the_pivot_signal) triples.
  2. Normalize pivot_data_point (lower + whitespace-collapse) and group.
  3. Keep groups where count >= ceil(N * threshold).
  4. From the highest-frequency kept group:
       - pivot_signal       = quality-weighted canonical pick
       - bypass_instruction = quality-weighted canonical (longest at tie)
       - recommended_action = quality-weighted canonical pick
       - seen_in_incidents  = first 5 incident numbers
       - frequency_percent  = round(count / N * 100)
  5. When threshold isn't met but at least one ticket has documented
     pivot data, surface that observation (derived_from=
     "mental_pivot_single"). Sparse-data cohorts shouldn't have
     real findings hidden behind a pure-aggregation gate.
  6. Primary_Fix fallback only when zero tickets have any pivot data.
  7. Empty only when even the Primary_Fix fields are missing.

Sprint 10.1 corrections:
  - Quality-weighted canonical pick: when picking text from a group,
    sort members by Resolution_Quality_Score DESC, then length DESC. A
    score-5 ticket's text trumps a score-3 ticket's even if shorter.
  - Primary_Fix fallback: when no cohort ticket has a populated
    Knowledge_Base.the_mental_pivot, distil from the highest-quality
    ticket's Symptom_Solution_Mapping.Primary_Fix +
    Executive_Sharable_RCA.Root_Cause_Technical_High_Level. Mark
    derived_from="primary_fix_fallback" so the frontend re-labels.

Logging:
  Emits one INFO line per call summarising cohort coverage:
  ``[stage1a] cohort=N with_pivot=M (P%) threshold=T derived=...``
  This makes ingestion-quality regressions (pivot fields missing
  from a new batch of tickets) visible in prod without requiring
  every caller to inspect the result object.
"""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


logger = logging.getLogger("acadia-log-iq")

from .schemas import Stage1aSmokingGun


_WHITESPACE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WHITESPACE.sub(" ", str(s).lower().strip())


def _incident_number(ticket: Dict[str, Any]) -> Optional[str]:
    meta = ticket.get("Metadata") if isinstance(ticket, dict) else None
    if not isinstance(meta, dict):
        return None
    inc = meta.get("Incident_Number")
    return str(inc).strip() if inc else None


def _quality_score(ticket: Dict[str, Any]) -> int:
    """Read Resolution_Quality_Score from the ticket; 0 on any miss."""
    if not isinstance(ticket, dict):
        return 0
    meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
    raw = meta.get("Resolution_Quality_Score")
    try:
        return int(raw or 0)
    except (ValueError, TypeError):
        return 0


def _kb_entries(ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Knowledge_Base may be a single dict or a list-of-dicts depending on
    ingestion shape. Normalise to a list."""
    if not isinstance(ticket, dict):
        return []
    kb = ticket.get("Knowledge_Base")
    if isinstance(kb, dict):
        return [kb]
    if isinstance(kb, list):
        return [k for k in kb if isinstance(k, dict)]
    return []


def _pivot_payload(kb_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{the_mental_pivot, diagnostic_logic}`` for a KB entry,
    tolerating two known shapes:

    * Nested (current ingest):
        ``semantic_unit_educational.diagnostic_pathway.the_mental_pivot``
        ``semantic_unit_educational.diagnostic_logic``
    * Flat (legacy / hand-written tickets):
        ``the_mental_pivot`` and ``diagnostic_logic`` at the KB root.

    Always returns a dict with both keys; missing pieces become ``{}`` so
    the caller can ``.get(...)`` without None-guards. Field paths come
    from the gold-ticket schema actually present in
    ``chunks.metadata_json`` for the cohort tickets — see commit message.
    """
    if not isinstance(kb_entry, dict):
        return {"the_mental_pivot": {}, "diagnostic_logic": {}}

    sue = kb_entry.get("semantic_unit_educational")
    if not isinstance(sue, dict):
        sue = {}

    pathway = sue.get("diagnostic_pathway")
    if not isinstance(pathway, dict):
        pathway = {}

    nested_pivot = pathway.get("the_mental_pivot")
    nested_logic = sue.get("diagnostic_logic")

    flat_pivot = kb_entry.get("the_mental_pivot")
    flat_logic = kb_entry.get("diagnostic_logic")

    pivot = nested_pivot if isinstance(nested_pivot, dict) else (
        flat_pivot if isinstance(flat_pivot, dict) else {}
    )
    logic = nested_logic if isinstance(nested_logic, dict) else (
        flat_logic if isinstance(flat_logic, dict) else {}
    )

    return {"the_mental_pivot": pivot, "diagnostic_logic": logic}


def _pick_canonical(group_members: List[Dict[str, Any]]) -> str:
    """Sprint 10.1 — sort by quality DESC, then length DESC, return the
    winning text. Members are dicts of {"text": str, "metadata_json": dict}.

    Rationale: a score-5 ticket's pivot description trumps a score-3
    ticket's even if shorter. When scores tie, prefer the longer (more
    descriptive) text.
    """
    if not group_members:
        return ""

    def sort_key(m):
        try:
            score = int(m.get("metadata_json", {}).get("Metadata", {}).get(
                "Resolution_Quality_Score", "0") or 0)
        except (ValueError, TypeError):
            score = 0
        return (-score, -len(m.get("text", "")))

    return sorted(group_members, key=sort_key)[0]["text"]


def _highest_quality_ticket(cohort: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the cohort ticket with the highest Resolution_Quality_Score.
    Ties resolve to the first one (stable sort)."""
    if not cohort:
        return None
    scored = [(t, _quality_score(t)) for t in cohort if isinstance(t, dict)]
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[0][0]


def _safe_str_path(d: Any, *path: str) -> Optional[str]:
    """Walk a dotted path through nested dicts; return string or None."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if cur is None:
        return None
    s = str(cur).strip()
    return s or None


def _primary_fix_fallback(
    cohort: List[Dict[str, Any]],
) -> Stage1aSmokingGun:
    """Sprint 10.1 — when no cohort ticket has Knowledge_Base.the_mental_pivot,
    distil from the highest-quality ticket's Primary_Fix + Root_Cause.

    Returns an empty Stage1aSmokingGun with derived_from="empty" if even
    the fallback fields are unavailable on the highest-quality ticket.
    """
    best = _highest_quality_ticket(cohort)
    if best is None:
        return Stage1aSmokingGun(empty=True, derived_from="empty")

    primary_fix = _safe_str_path(best, "Symptom_Solution_Mapping", "Primary_Fix")
    root_cause = _safe_str_path(best, "Executive_Sharable_RCA", "Root_Cause_Technical_High_Level")

    if not primary_fix and not root_cause:
        return Stage1aSmokingGun(empty=True, derived_from="empty")

    inc = _incident_number(best)
    return Stage1aSmokingGun(
        pivot_signal=root_cause,
        bypass_instruction=primary_fix,
        recommended_action=None,
        frequency_in_cohort_percent=0,  # not a frequency-based result
        seen_in_incidents=[inc] if inc else [],
        empty=False,
        derived_from="primary_fix_fallback",
    )


def build_stage1a(
    cohort: List[Dict[str, Any]],
    min_frequency_ratio: float = 0.4,
) -> Stage1aSmokingGun:
    """See module docstring."""
    n = len(cohort or [])
    if n == 0:
        return Stage1aSmokingGun(empty=True, derived_from="empty")

    threshold = max(1, math.ceil(n * float(min_frequency_ratio)))

    # Group by normalised pivot_data_point. Each candidate text is
    # tracked alongside its source ticket's metadata_json so the
    # quality-weighted pick has the score available. `_pivot_payload`
    # resolves both the current nested layout
    # (``Knowledge_Base[*].semantic_unit_educational.diagnostic_pathway
    # .the_mental_pivot``) and the legacy flat layout, so callers get
    # the same answer regardless of ingestion vintage.
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "raw_signals": [],     # [{"text": str, "metadata_json": dict}, ...]
        "shift_strings": [],   # [{"text": str, "metadata_json": dict}, ...]
        "actions": [],         # [{"text": str, "metadata_json": dict}, ...]
        "incidents": [],       # incident numbers
    })
    tickets_with_pivot: int = 0

    for ticket in cohort:
        inc = _incident_number(ticket) or ""
        ticket_contributed = False
        for kb in _kb_entries(ticket):
            payload = _pivot_payload(kb)
            mental_pivot = payload["the_mental_pivot"]
            diagnostic_logic = payload["diagnostic_logic"]
            pivot_raw = mental_pivot.get("pivot_data_point")
            shift_raw = mental_pivot.get("shift_in_logic")
            action_raw = diagnostic_logic.get("the_pivot_signal")

            if not pivot_raw or not str(pivot_raw).strip():
                continue
            key = _normalize(str(pivot_raw))
            g = groups[key]
            g["raw_signals"].append({"text": str(pivot_raw), "metadata_json": ticket})
            if shift_raw and str(shift_raw).strip():
                g["shift_strings"].append({"text": str(shift_raw), "metadata_json": ticket})
            if action_raw and str(action_raw).strip():
                g["actions"].append({"text": str(action_raw), "metadata_json": ticket})
            if inc and inc not in g["incidents"]:
                g["incidents"].append(inc)
            ticket_contributed = True
        if ticket_contributed:
            tickets_with_pivot += 1

    coverage_pct = round(100.0 * tickets_with_pivot / n) if n else 0

    def _emit(result: Stage1aSmokingGun) -> Stage1aSmokingGun:
        # One log line per call so prod monitoring can spot ingestion
        # regressions (e.g. a batch of tickets without populated
        # mental_pivot fields shows up as a coverage drop).
        logger.info(
            "[stage1a] cohort=%d with_pivot=%d (%d%%) threshold=%d "
            "derived=%s freq_pct=%d",
            n, tickets_with_pivot, coverage_pct, threshold,
            result.derived_from, result.frequency_in_cohort_percent,
        )
        return result

    # No usable pivot anywhere in the cohort → degrade to Primary_Fix
    # distillation from the highest-quality ticket. Same behaviour as
    # the legacy "no any_pivot_present" early-out, just deferred so
    # the coverage log line still fires.
    if not groups:
        return _emit(_primary_fix_fallback(cohort))

    # Aggregate path — at least `threshold` cohort tickets share the
    # same normalized pivot_data_point. This is the strongest signal.
    candidates = [
        (key, g, len(g["incidents"]))
        for key, g in groups.items()
        if len(g["incidents"]) >= threshold
    ]
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        _, winner, count = candidates[0]
        pivot_signal = _pick_canonical(winner["raw_signals"])
        bypass_instruction = (
            _pick_canonical(winner["shift_strings"]) if winner["shift_strings"] else None
        )
        recommended_action = (
            _pick_canonical(winner["actions"]) if winner["actions"] else None
        )
        return _emit(Stage1aSmokingGun(
            pivot_signal=pivot_signal or None,
            bypass_instruction=bypass_instruction or None,
            recommended_action=recommended_action or None,
            frequency_in_cohort_percent=round(100.0 * count / n),
            seen_in_incidents=winner["incidents"][:5],
            empty=False,
            derived_from="mental_pivot_aggregate",
        ))

    # Single-pivot path — at least one ticket has documented pivot
    # data, but nothing repeats often enough to clear the threshold.
    # Production prefers surfacing the lone observation over hiding
    # it behind a Primary_Fix fallback that duplicates Stage 0. Pick
    # the group with the highest-quality canonical text — same
    # quality-weighted comparator used inside `_pick_canonical` —
    # so we always lead with the best-rated ticket's pivot.
    def _group_best_score(g: Dict[str, Any]) -> int:
        best = 0
        for m in g["raw_signals"]:
            try:
                s = int(((m.get("metadata_json") or {}).get("Metadata") or {})
                        .get("Resolution_Quality_Score") or 0)
            except (ValueError, TypeError):
                s = 0
            if s > best:
                best = s
        return best

    ranked_groups = sorted(
        groups.values(),
        key=lambda g: (_group_best_score(g), len(g["incidents"])),
        reverse=True,
    )
    winner = ranked_groups[0]
    count = len(winner["incidents"])
    pivot_signal = _pick_canonical(winner["raw_signals"])
    bypass_instruction = (
        _pick_canonical(winner["shift_strings"]) if winner["shift_strings"] else None
    )
    recommended_action = (
        _pick_canonical(winner["actions"]) if winner["actions"] else None
    )
    return _emit(Stage1aSmokingGun(
        pivot_signal=pivot_signal or None,
        bypass_instruction=bypass_instruction or None,
        recommended_action=recommended_action or None,
        frequency_in_cohort_percent=round(100.0 * count / n),
        seen_in_incidents=winner["incidents"][:5],
        empty=False,
        derived_from="mental_pivot_single",
    ))
