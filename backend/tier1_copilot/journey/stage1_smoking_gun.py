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
  5. Empty when no group meets threshold.

Sprint 10.1 corrections:
  - Quality-weighted canonical pick: when picking text from a group,
    sort members by Resolution_Quality_Score DESC, then length DESC. A
    score-5 ticket's text trumps a score-3 ticket's even if shorter.
  - Primary_Fix fallback: when no cohort ticket has a populated
    Knowledge_Base.the_mental_pivot, distil from the highest-quality
    ticket's Symptom_Solution_Mapping.Primary_Fix +
    Executive_Sharable_RCA.Root_Cause_Technical_High_Level. Mark
    derived_from="primary_fix_fallback" so the frontend re-labels.

Pure function — no DB, no LLM, no logging beyond debug.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

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

    # ── First, scan whether ANY ticket has a usable mental_pivot ──
    # If zero tickets have populated KB.the_mental_pivot.pivot_data_point,
    # we go straight to the Primary_Fix fallback rather than running the
    # aggregator on empty input.
    any_pivot_present = False
    for ticket in cohort:
        for kb in _kb_entries(ticket):
            mp = kb.get("the_mental_pivot") if isinstance(kb.get("the_mental_pivot"), dict) else {}
            if mp and str(mp.get("pivot_data_point", "")).strip():
                any_pivot_present = True
                break
        if any_pivot_present:
            break

    if not any_pivot_present:
        return _primary_fix_fallback(cohort)

    # Group by normalised pivot_data_point. Each candidate text is
    # tracked alongside its source ticket's metadata_json so the
    # quality-weighted pick has the score available.
    groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "raw_signals": [],     # [{"text": str, "metadata_json": dict}, ...]
        "shift_strings": [],   # [{"text": str, "metadata_json": dict}, ...]
        "actions": [],         # [{"text": str, "metadata_json": dict}, ...]
        "incidents": [],       # incident numbers
    })

    for ticket in cohort:
        inc = _incident_number(ticket) or ""
        for kb in _kb_entries(ticket):
            mental_pivot = kb.get("the_mental_pivot") if isinstance(kb.get("the_mental_pivot"), dict) else {}
            diagnostic_logic = kb.get("diagnostic_logic") if isinstance(kb.get("diagnostic_logic"), dict) else {}
            pivot_raw = mental_pivot.get("pivot_data_point") if isinstance(mental_pivot, dict) else None
            shift_raw = mental_pivot.get("shift_in_logic") if isinstance(mental_pivot, dict) else None
            action_raw = diagnostic_logic.get("the_pivot_signal") if isinstance(diagnostic_logic, dict) else None

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

    if not groups:
        # Edge case: pivot_present=True but normalising stripped them all
        return _primary_fix_fallback(cohort)

    # Pick highest-frequency group above threshold (count = unique-incident count)
    candidates = [
        (key, g, len(g["incidents"]))
        for key, g in groups.items()
        if len(g["incidents"]) >= threshold
    ]
    if not candidates:
        # Cohort has KB pivot data but nothing repeats enough — fall
        # back to Primary_Fix from the highest-quality ticket.
        return _primary_fix_fallback(cohort)

    candidates.sort(key=lambda x: x[2], reverse=True)
    _, winner, count = candidates[0]

    # Quality-weighted canonical pick for each text field.
    pivot_signal = _pick_canonical(winner["raw_signals"])
    bypass_instruction = (
        _pick_canonical(winner["shift_strings"]) if winner["shift_strings"] else None
    )
    recommended_action = (
        _pick_canonical(winner["actions"]) if winner["actions"] else None
    )

    return Stage1aSmokingGun(
        pivot_signal=pivot_signal or None,
        bypass_instruction=bypass_instruction or None,
        recommended_action=recommended_action or None,
        frequency_in_cohort_percent=round(100.0 * count / n),
        seen_in_incidents=winner["incidents"][:5],
        empty=False,
        derived_from="mental_pivot_aggregate",
    )
