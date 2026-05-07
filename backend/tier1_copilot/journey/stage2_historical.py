"""Sprint 10 Stage 2 — Historical Matches & Possible Causes.

Per spec §3.4: one HistoricalMatchCard per cohort ticket, in rank
order (rank 1 first). Pure structural transform — no LLM, no DB.

Field-mapping rules (verbatim from spec):
  Headline      = Incident_Summary.INCIDENT
  Summary       = Executive_Sharable_RCA.Executive_Summary
  Snapshot      = Executive_Sharable_RCA.Technical_Snapshot
  Symptoms      = Operational_SOP.signal_identification.human_symptom
  Root Cause    = Metadata.outage_cause + Executive_Sharable_RCA.Root_Cause_Technical_High_Level
                  joined with " — " when both present
  Resolution    = Executive_Sharable_RCA.Resolution_Steps
                  + Forensic_Performance_Audit.Critical_Intervention
                  + Key_Contributors.Hero_Action
                  rendered as a flat list of bullet strings
  Footnote      = customer + ttr + closed_without_recurrence (each independent)

Missing fields are OMITTED, not rendered as "N/A".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schemas import HistoricalMatchCard, Stage2HistoricalMatches


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _join_em_dash(*parts: Any) -> Optional[str]:
    """Join non-empty stringified parts with ' — '. Returns None when
    every part is empty."""
    cleaned = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    return " — ".join(cleaned) if cleaned else None


# ─────────────────────────────────────────────────────────────
# Sprint 10.8.1 §3 — array-aware nested access + value flattener.
#
# Replaces Sprint 10.8's _read_path / _flatten_field with helpers
# realigned to the verified 27-ticket corpus shape:
#   - Forensic_Performance_Audit is a single-element list, not a dict
#   - Key_Contributors.Key_Impact_Players is a list before Hero_Action
# _safe_get auto-steps into the [0] element when it encounters a list
# mid-path so callers can write the logical key sequence regardless
# of which links are arrays vs dicts.
# ─────────────────────────────────────────────────────────────
def _safe_get(obj: Any, *path: str) -> Any:
    """Walk a JSON path, auto-stepping into single-element lists.

    Examples:
        _safe_get({'a': {'b': 'x'}}, 'a', 'b')         -> 'x'
        _safe_get({'a': [{'b': 'x'}]}, 'a', 'b')       -> 'x'   (auto-step)
        _safe_get({'a': [{'b': [{'c': 'x'}]}]}, 'a', 'b', 'c') -> 'x'
        _safe_get({}, 'a', 'b')                        -> None
        _safe_get(None, 'a')                           -> None
    """
    cur = obj
    for key in path:
        if cur is None:
            return None
        if isinstance(cur, list):
            if not cur:
                return None
            cur = cur[0]
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _flatten(value: Any) -> Optional[str]:
    """Convert a JSON value into a display-ready string, or None if
    effectively empty.

      - str:                stripped; None when empty
      - list of str:        joined with "; " (Resolution_Steps shape)
      - list of dict:       each flattened, joined with "; "
      - dict:               prefer "description"/"text"/"summary"/
                            "Action"/"value" keys; fall back to the
                            first non-empty string value
      - None / other:       None
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(v, str) for v in value):
            joined = "; ".join(s.strip() for s in value if s.strip())
            return joined or None
        parts = []
        for v in value:
            f = _flatten(v)
            if f:
                parts.append(f)
        return "; ".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("description", "text", "summary", "Action", "value"):
            sub = value.get(key)
            if sub:
                f = _flatten(sub)
                if f:
                    return f
        for v in value.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    s = str(value).strip()
    return s or None


# Sprint 10.8 backward-compat aliases — older internal callers still
# reference the prior names. Drop these aliases when no in-tree caller
# uses them anymore.
_flatten_field = _flatten
_read_path = _safe_get


def _harvest_error_codes(ticket: Dict[str, Any]) -> List[str]:
    """Sprint 12.5 — collect per-ticket error-code fingerprints from
    three independent fields, deduplicated case-insensitively with
    first-seen casing preserved.

    Sources (all optional — silently skipped when missing):
      - ``Metadata.Fingerprints``                       (List[str])
      - ``Operational_SOP.primary_error_fingerprint``   (str)
      - ``semantic_faq_block[*].related_signals``       (List[str])

    The three streams are intentionally merged into one list rather
    than kept separate: from the engineer's perspective they are all
    "tokens you can grep against the current incident" — a single
    deduped chip row reads cleaner than three near-empty rows.
    """
    if not isinstance(ticket, dict):
        return []

    seen: Dict[str, str] = {}      # casefold key → first-seen casing
    ordered: List[str] = []

    def _add(value: Any) -> None:
        if value is None:
            return
        s = str(value).strip()
        if not s:
            return
        key = s.casefold()
        if key in seen:
            return
        seen[key] = s
        ordered.append(s)

    # Metadata.Fingerprints — list of strings
    fps = _safe_get(ticket, "Metadata", "Fingerprints")
    if isinstance(fps, list):
        for fp in fps:
            _add(fp)
    elif isinstance(fps, str):
        _add(fps)

    # Operational_SOP.primary_error_fingerprint — single string
    pef = _safe_get(ticket, "Operational_SOP", "primary_error_fingerprint")
    _add(pef)

    # semantic_faq_block[*].related_signals — list of dicts, each
    # carrying a related_signals list. Walked across ALL faq entries
    # (not just [0]) so every populated block contributes.
    faq = ticket.get("semantic_faq_block")
    if isinstance(faq, list):
        for entry in faq:
            if not isinstance(entry, dict):
                continue
            sigs = entry.get("related_signals")
            if isinstance(sigs, list):
                for sig in sigs:
                    _add(sig)
            elif isinstance(sigs, str):
                _add(sigs)

    return ordered


def _resolution_bullets(ticket: Dict[str, Any]) -> List[str]:
    """Concat Resolution_Steps + Critical_Intervention + Hero_Action.
    Resolution_Steps may be a list-of-strings, list-of-dicts, or a single
    string — all three normalised to a flat list of bullet strings.
    Critical_Intervention and Hero_Action are single strings each."""
    rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    fpa = ticket.get("Forensic_Performance_Audit") if isinstance(ticket.get("Forensic_Performance_Audit"), dict) else {}
    contributors = ticket.get("Key_Contributors") if isinstance(ticket.get("Key_Contributors"), dict) else {}

    bullets: List[str] = []

    steps = rca.get("Resolution_Steps") if isinstance(rca, dict) else None
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict):
                # Common keys: "step", "action", "description"
                txt = s.get("description") or s.get("action") or s.get("step") or ""
                txt = str(txt).strip()
                if txt:
                    bullets.append(txt)
            elif s is not None:
                txt = str(s).strip()
                if txt:
                    bullets.append(txt)
    elif isinstance(steps, str) and steps.strip():
        bullets.append(steps.strip())

    ci = fpa.get("Critical_Intervention") if isinstance(fpa, dict) else None
    if isinstance(ci, str) and ci.strip():
        bullets.append(ci.strip())
    elif isinstance(ci, list):
        for s in ci:
            if isinstance(s, str) and s.strip():
                bullets.append(s.strip())

    ha = contributors.get("Hero_Action") if isinstance(contributors, dict) else None
    if isinstance(ha, str) and ha.strip():
        bullets.append(ha.strip())
    elif isinstance(ha, list):
        for s in ha:
            if isinstance(s, str) and s.strip():
                bullets.append(s.strip())

    return bullets


def _build_card(rank: int, ticket: Dict[str, Any]) -> HistoricalMatchCard:
    """Sprint 10.8.1 §4 + Sprint 11 — card builder realigned to the verified
    corpus shape.

    Critical path corrections vs. Sprint 10.8:
      - Forensic_Performance_Audit.Critical_Intervention is read via
        _safe_get's auto-list-step (the parent is a single-element
        list `[{...}]` in this corpus, not a dict).
      - Key_Contributors.Key_Impact_Players[0].Hero_Action — the real
        path with two-level array nesting; was mistakenly treated as
        Key_Contributors.Hero_Action in 10.8.

    Sprint 11 reinstates Executive_Sharable_RCA.Technical_Snapshot —
    populated in 72/180 tickets across the four reachable source files.
    """
    if not isinstance(ticket, dict):
        return HistoricalMatchCard(rank=rank)

    incident_number = (
        _flatten(_safe_get(ticket, "Metadata", "Incident_Number"))
        or f"UNKNOWN-{rank}"
    )

    # ── Sprint 10.0 backward-compat fields ──
    # Existing internal tests still read these. The new merged surfaces
    # (incident_summary / resolution_approach) are what the post-10.8
    # frontend renders.
    headline = _flatten(_safe_get(ticket, "Incident_Summary", "INCIDENT"))
    summary = _flatten(_safe_get(ticket, "Executive_Sharable_RCA", "Executive_Summary"))
    symptoms = _flatten(
        _safe_get(ticket, "Operational_SOP", "signal_identification", "human_symptom"),
    )
    root_cause = _join_em_dash(
        _flatten(_safe_get(ticket, "Metadata", "outage_cause")),
        _flatten(_safe_get(ticket, "Executive_Sharable_RCA", "Root_Cause_Technical_High_Level")),
    )
    resolution = _resolution_bullets(ticket)
    # Sprint 11 — Technical Snapshot row (string in corpus; numbered narrative).
    technical_snapshot = _flatten(
        _safe_get(ticket, "Executive_Sharable_RCA", "Technical_Snapshot"),
    )

    meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
    customer = _safe_str(meta.get("customer_name") or meta.get("Customer_Name"))
    ttr = _safe_int(meta.get("time_to_resolve_minutes"))

    qa = ticket.get("QA_Auditor_Feedback") if isinstance(ticket.get("QA_Auditor_Feedback"), dict) else {}
    rework = qa.get("Rework_Detected") if isinstance(qa, dict) else None
    closed_status = _safe_str(meta.get("ticket_status"))
    closed_without_recurrence = (
        bool(closed_status and closed_status.lower() == "closed")
        and not bool(rework)
    )

    # ── Sprint 10.8.1 §4 — em-dash-merged surfaces with corrected paths. ──
    incident_summary = _join_em_dash(
        _flatten(_safe_get(ticket, "Incident_Summary", "INCIDENT")),
        _flatten(_safe_get(ticket, "Executive_Sharable_RCA", "Executive_Summary")),
    )
    resolution_approach = _join_em_dash(
        _flatten(_safe_get(ticket, "Executive_Sharable_RCA", "Resolution_Steps")),
        # Forensic_Performance_Audit IS A LIST in the real corpus —
        # _safe_get auto-steps into [0] before reading the next key.
        _flatten(_safe_get(ticket, "Forensic_Performance_Audit", "Critical_Intervention")),
        # Key_Contributors.Key_Impact_Players IS A LIST before Hero_Action.
        _flatten(_safe_get(ticket, "Key_Contributors", "Key_Impact_Players", "Hero_Action")),
    )

    # Sprint 12.5 — harvest the three fingerprint streams.
    error_codes = _harvest_error_codes(ticket)

    return HistoricalMatchCard(
        rank=rank,
        incident_number=incident_number,
        headline=headline,
        summary=summary,
        symptoms=symptoms,
        root_cause=root_cause,
        resolution=resolution,
        customer=customer,
        time_to_resolve_minutes=ttr,
        closed_without_recurrence=closed_without_recurrence,
        technical_snapshot=technical_snapshot,
        incident_summary=incident_summary,
        resolution_approach=resolution_approach,
        error_codes=error_codes,
    )


# Sprint 10.8.1 §3 — _join_em_dash restated with the spec's signature
# (a list, not *args) so callers can match the spec verbatim. The old
# *args version still works via the wrapper above.
def _join_em_dash_list(parts):
    surviving = [p for p in parts if p]
    return " — ".join(surviving) if surviving else None


# Sprint 11 — placeholder strings the source data uses for "we know
# nothing about this field". Treat as missing for the purpose of
# card-usefulness scoring; otherwise we end up rendering a card that
# says "Root Cause: Data Not Present in Log" with nothing else, which
# is worse than not showing the card at all.
_TRIVIAL_PLACEHOLDERS = frozenset({
    "data not present in log",
    "data not present",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "unknown",
    "tbd",
    "to be determined",
    "none",
    "null",
    "-",
    "—",
})


def _is_trivial(value) -> bool:
    """True when `value` is missing, empty, or a known placeholder."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_trivial(v) for v in value)
    s = str(value).strip().lower()
    if not s:
        return True
    return s in _TRIVIAL_PLACEHOLDERS


def _is_useful_card(card: HistoricalMatchCard) -> bool:
    """A card is useful if it has a real incident_number AND at least
    one body field carries non-trivial signal.

    Reasons a card gets dropped:
      - incident_number missing or starts with "UNKNOWN-" (chunk had
        no Metadata.Incident_Number — typically a non-ticket chunk
        that scored into top_5 by accident).
      - Every body field (incident_summary, technical_snapshot,
        symptoms, root_cause, resolution_approach, resolution[]) is
        missing OR equal to a placeholder like "Data Not Present in
        Log". The card would render as a near-empty stub that wastes
        the engineer's attention.
    """
    inc = (card.incident_number or "").strip()
    if not inc or inc.upper().startswith("UNKNOWN-"):
        return False
    body_fields = (
        card.incident_summary,
        card.technical_snapshot,
        card.symptoms,
        card.root_cause,
        card.resolution_approach,
        card.resolution,
    )
    return any(not _is_trivial(f) for f in body_fields)


def build_stage2(
    cohort: List[Dict[str, Any]],
    max_matches_shown: int = 5,
    *,
    filter_empty: bool = True,
) -> Stage2HistoricalMatches:
    """Build the Stage 2 surface from the cohort.

    Sprint 11 — three changes vs. the original 1:1 build:
      1. Filter empty / placeholder-only cards via _is_useful_card
         (when filter_empty=True, the production default). Stops
         "MATCH 3 OF 5 — INC-10000 / Root Cause: Data Not Present in
         Log" and "MATCH 4 OF 5 — UNKNOWN-4" from polluting the
         surface. Pass filter_empty=False to inspect the raw 1:1
         build (used by the resilience tests that intentionally feed
         sparse fixtures).
      2. Re-rank surviving cards 1..N so visible labels stay
         sequential after sibling drops (engineer sees "MATCH 1 OF 3"
         cleanly).
      3. Return ALL useful cards — do NOT cap on the backend.
         `max_matches_shown` is a frontend rendering hint: render
         this many by default, surface a "View more matches (N)"
         button when more are available client-side. The "show more"
         reveal becomes a one-line frontend addition once retrieval
         ever expands beyond 5 candidates per cohort, with zero
         backend pagination plumbing required.
    """
    raw_cards = [_build_card(i + 1, t) for i, t in enumerate(cohort or [])]
    surviving = [c for c in raw_cards if _is_useful_card(c)] if filter_empty else raw_cards
    # Re-number rank so visible labels stay 1..N after sibling drops.
    matches: List[HistoricalMatchCard] = []
    for new_rank, card in enumerate(surviving, start=1):
        # Pydantic models are immutable by default — use model_copy.
        matches.append(card.model_copy(update={"rank": new_rank}))
    return Stage2HistoricalMatches(
        matches=matches,
        total_available_matches=len(matches),
        max_matches_shown=max_matches_shown,
    )
