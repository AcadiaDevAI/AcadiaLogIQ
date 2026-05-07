"""Sprint 10 Stage 3 — The Troubleshooting Approach.

Per spec §3.5: build a sequenced ledger from the cohort's diagnostic
steps, deduplicating identical actions across tickets, and detecting
divergent successful interventions as alternative branches.

Sprint 12.6 — capping is per-category so interventions always reach
the engineer. The pre-12.6 single ``max_steps`` cap let the priority
order (diagnostic → timeline → intervention) silently truncate every
intervention on a 5-ticket BGP cohort (5 diag + 5 timeline filled
the cap of 8; all 30 candidate interventions were dropped). The
spec's "Attempt Fix A; if it fails, proceed to Fix B" framing is
unreachable when zero fixes survive the cap. New defaults guarantee
slots for each category; if a category has fewer candidates than
its quota, the leftover slots refill from any category in
sequencing order (so heavy diagnostic-only cohorts still cap cleanly
at ``max_steps``).

Algorithm:
  1. Extract candidate Step records from every cohort ticket. Sources:
       - Operational_SOP.diagnostic_logic_chunks[]
       - Troubleshooting_Ledger.Diagnostic_Tests_Executed[]
       - Executive_Sharable_RCA.Resolution_Steps[]
       - Forensic_Performance_Audit.Key_Movements_Timeline[].Action
       - Forensic_Performance_Audit.Critical_Intervention
       - Key_Contributors.Hero_Action
  2. Normalise action text (lower, strip punct, collapse whitespace).
     Group by the normalised key.
  3. Per group: longest verbatim action; first non-empty intent;
     first non-empty pivot; average ordinal.
  4. Sequence groups by ascending average ordinal.
  5. Branching: if two cohort tickets have distinct Critical_Intervention
     strings that did NOT collapse into the same group, tag them as
     "Alt A" / "Alt B" / "Primary".
  6. Cap at 8 consolidated steps.

Pure function — no LLM, no DB.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger("acadia-log-iq")

from .schemas import (
    DiagnosticLogicEntry,
    Stage3TroubleshootingApproach,
    TicketTroubleshootingDetail,
    TimelineEntry,
    TroubleshootingStep,
)
# Sprint 10.8.1 §3 — share the array-aware path walker + flattener
# from stage2_historical so both stages traverse the corpus the same
# way. (Defining once, used twice — single source of truth.)
from .stage2_historical import _flatten, _safe_get


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-]")


def _normalize(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", str(s).lower())).strip()


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _incident(ticket: Dict[str, Any]) -> str:
    meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
    return _safe_str(meta.get("Incident_Number")) or ""


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


def _pick_canonical(group_members: List[Dict[str, Any]]) -> str:
    """Sprint 10.1 — sort by quality DESC, then length DESC, return the
    winning text. Mirrors stage1_smoking_gun._pick_canonical (same
    semantics; kept module-local to avoid a cross-stage import). Members
    are dicts of {"text": str, "metadata_json": dict}."""
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


@dataclass
class _Step:
    action: str
    intent: Optional[str] = None
    pivot: Optional[str] = None
    command: Optional[str] = None
    incident: str = ""
    source_field: str = ""
    ordinal: int = 0
    is_critical_intervention: bool = False  # branching detection feeds off this
    # Sprint 10.1 — carry the originating ticket's metadata_json so the
    # quality-weighted canonical pick can read Resolution_Quality_Score.
    metadata_json: Optional[Dict[str, Any]] = None
    # Sprint 10.8 §3.2 — sequencing-priority bucket. diagnostic first,
    # timeline second, intervention last. "context" comes from
    # Technical_Snapshot only — harvested for the raw count but not
    # surfaced as a user-facing step.
    category: str = "diagnostic"
    # Sprint 10.8 — index of the cohort ticket this step came from.
    # Used as the "rank-of-first-appearance" tiebreaker when sequencing
    # within a category (lower = earlier-rank ticket = sequenced sooner).
    cohort_rank: int = 0


def _harvest_steps(ticket: Dict[str, Any], cohort_rank: int = 0) -> List[_Step]:
    """Extract every candidate step from one cohort ticket.

    Sprint 10.8.1 §5 + Sprint 11 — realigned to the verified corpus:
      - Sprint 11: REINSTATED Technical_Snapshot harvest as a
        category="context" source (populated in 72/180 tickets across
        the four reachable source files). "context" steps are bucketed
        at _CATEGORY_PRIORITY=99 so they're counted in
        total_unique_steps_before_cap but excluded from the user-facing
        playbook (line 400-403 of build_stage3).
      - Dropped Troubleshooting_Ledger.Diagnostic_Tests_Executed source
        (parent key 0/180 populated across all reachable source files;
        re-add when a future upload populates it).
      - Forensic_Performance_Audit traversed via _safe_get's auto-list-
        step (parent is `[{...}]` in this corpus).
      - Key_Contributors.Key_Impact_Players[0].Hero_Action — the real
        path with two-level array nesting.
      - diagnostic_logic_chunks sub-fields remapped per §2:
          context         → intent
          branching_logic → pivot
          command         → command
        Existing fixtures using "rationale" / "intent" / "pivot" keys
        still resolve via the cascade, preserving Sprint 10.0/10.8
        backward compat.

    Sprint 10.8 §3.7 commitment unchanged: Intent / Pivot / Command
    populated ONLY on steps sourced from diagnostic_logic_chunks.
    """
    if not isinstance(ticket, dict):
        return []
    inc = _incident(ticket)
    steps: List[_Step] = []

    # ── Source 1 — Executive_Sharable_RCA.Technical_Snapshot (Sprint 11) ──
    # String in this corpus, typically a numbered narrative ("1. … 2. …").
    # Bucketed as `category="context"` so it raises the unique-step count
    # without polluting the actionable playbook.
    ts_raw = _safe_get(ticket, "Executive_Sharable_RCA", "Technical_Snapshot")
    ts = _flatten(ts_raw)
    if ts:
        steps.append(_Step(
            action=ts,
            incident=inc,
            source_field="Technical_Snapshot",
            ordinal=-1,
            metadata_json=ticket,
            category="context",
            cohort_rank=cohort_rank,
        ))

    # ── Source 3 — Troubleshooting_Ledger.Diagnostic_Tests_Executed[] ──
    # Sprint 12.6 — re-added (Sprint 10.8.1 dropped it as 0/180 in the
    # then-corpus; still 0/213 today, but the field is in the user's
    # spec so we wire it forward-compatibly: when a future ticket
    # populates it, no code change required). Treated as `diagnostic`
    # — these are explicit tests/checks the prior engineer ran.
    # Accepts list-of-str and list-of-dict (name | description | test
    # | action keys), mirroring _build_diagnostic_tests_executed.
    tests = _safe_get(ticket, "Troubleshooting_Ledger", "Diagnostic_Tests_Executed")
    if isinstance(tests, list):
        for i, entry in enumerate(tests):
            if isinstance(entry, dict):
                action = _safe_str(
                    entry.get("name")
                    or entry.get("description")
                    or entry.get("test")
                    or entry.get("action")
                )
            else:
                action = _safe_str(entry)
            if not action:
                continue
            steps.append(_Step(
                action=action,
                incident=inc,
                source_field="Diagnostic_Tests_Executed",
                ordinal=i,
                metadata_json=ticket,
                category="diagnostic",
                cohort_rank=cohort_rank,
            ))

    # ── Source 4 — Operational_SOP.diagnostic_logic_chunks[] ──
    # The ONLY source that populates Intent + Pivot + Command.
    sop = ticket.get("Operational_SOP") if isinstance(ticket.get("Operational_SOP"), dict) else {}
    chunks = sop.get("diagnostic_logic_chunks") if isinstance(sop, dict) else None
    if isinstance(chunks, list):
        for i, ch in enumerate(chunks):
            if not isinstance(ch, dict):
                continue
            action = _safe_str(ch.get("action") or ch.get("step_id") or ch.get("step"))
            if not action:
                continue
            steps.append(_Step(
                action=action,
                # Sprint 10.8.1 §2 — `context` is the canonical source
                # for Intent in the corpus; cascade preserves Sprint 10.0
                # fixtures that use "rationale" / "intent".
                intent=_safe_str(
                    ch.get("context") or ch.get("rationale") or ch.get("intent")
                ),
                pivot=_safe_str(ch.get("branching_logic") or ch.get("pivot")),
                command=_safe_str(ch.get("command")),
                incident=inc,
                source_field="diagnostic_logic_chunks",
                ordinal=i,
                metadata_json=ticket,
                category="diagnostic",
                cohort_rank=cohort_rank,
            ))

    # ── Source 2 — Executive_Sharable_RCA.Resolution_Steps[] ──
    rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    rsteps = rca.get("Resolution_Steps") if isinstance(rca, dict) else None
    if isinstance(rsteps, list):
        for i, s in enumerate(rsteps):
            if isinstance(s, dict):
                action = _safe_str(s.get("description") or s.get("action") or s.get("step"))
            else:
                action = _safe_str(s)
            if not action:
                continue
            steps.append(_Step(
                action=action,
                incident=inc,
                source_field="Resolution_Steps",
                ordinal=i,
                metadata_json=ticket,
                category="intervention",
                cohort_rank=cohort_rank,
            ))
    elif isinstance(rsteps, str) and rsteps.strip():
        steps.append(_Step(
            action=rsteps.strip(),
            incident=inc,
            source_field="Resolution_Steps",
            ordinal=0,
            metadata_json=ticket,
            category="intervention",
            cohort_rank=cohort_rank,
        ))

    # ── Source 5 — Forensic_Performance_Audit[0].Key_Movements_Timeline[].Action ──
    # Use _safe_get so the auto-list-step at the FPA level Just Works
    # whether the parent is a dict or a single-element list.
    timeline = _safe_get(ticket, "Forensic_Performance_Audit", "Key_Movements_Timeline")
    if isinstance(timeline, list):
        for i, mv in enumerate(timeline):
            if not isinstance(mv, dict):
                continue
            action = _safe_str(mv.get("Action"))
            if not action:
                continue
            steps.append(_Step(
                action=action,
                incident=inc,
                source_field="Key_Movements_Timeline",
                ordinal=i,
                metadata_json=ticket,
                category="timeline",
                cohort_rank=cohort_rank,
            ))

    # ── Source 6 — Forensic_Performance_Audit[0].Critical_Intervention ──
    ci_raw = _safe_get(ticket, "Forensic_Performance_Audit", "Critical_Intervention")
    ci = _safe_str(ci_raw) if isinstance(ci_raw, str) else _flatten(ci_raw)
    if ci:
        steps.append(_Step(
            action=ci,
            incident=inc,
            source_field="Critical_Intervention",
            ordinal=99,
            is_critical_intervention=True,
            metadata_json=ticket,
            category="intervention",
            cohort_rank=cohort_rank,
        ))

    # ── Source 7 — Key_Contributors.Key_Impact_Players[0].Hero_Action ──
    # Two-level array path: Key_Contributors.Key_Impact_Players is the
    # array; the hero record sits at [0]. _safe_get auto-steps in.
    ha_raw = _safe_get(ticket, "Key_Contributors", "Key_Impact_Players", "Hero_Action")
    ha = _safe_str(ha_raw) if isinstance(ha_raw, str) else _flatten(ha_raw)
    if ha:
        steps.append(_Step(
            action=ha,
            incident=inc,
            source_field="Hero_Action",
            ordinal=100,
            metadata_json=ticket,
            category="intervention",
            cohort_rank=cohort_rank,
        ))

    return steps


@dataclass
class _Group:
    canonical_action: str
    intents: List[str] = field(default_factory=list)
    pivots: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    incidents: List[str] = field(default_factory=list)
    ordinals: List[int] = field(default_factory=list)
    has_critical_intervention: bool = False
    # Sprint 10.1 — quality-weighted canonical-pick candidates. Each
    # entry is {"text": str, "metadata_json": dict} so _pick_canonical
    # can read Resolution_Quality_Score from the originating ticket.
    action_candidates: List[Dict[str, Any]] = field(default_factory=list)
    intent_candidates: List[Dict[str, Any]] = field(default_factory=list)
    pivot_candidates: List[Dict[str, Any]] = field(default_factory=list)
    command_candidates: List[Dict[str, Any]] = field(default_factory=list)
    # Sprint 10.8 §3.5 — sequencing-priority bucket of the FIRST step
    # contributed to this group. When members from different categories
    # collapse, the FIRST contributor's category wins (mirrors how
    # diagnostic-then-intervention sequences read most naturally to
    # the engineer). cohort_rank_first = lowest cohort rank that
    # contributed to this group (§3.5 within-category tiebreak).
    category: str = "diagnostic"
    cohort_rank_first: int = 999_999
    # Sprint 10.8 — preserve the order incidents appeared so the
    # §3.10 test can assert rank-ordered seen_in_incidents.
    incident_order: List[int] = field(default_factory=list)


# Sprint 10.8 §3.5 — explicit category-priority order. Lower number =
# sequenced earlier in the user-facing ledger. "context" steps are
# harvested for the raw count but excluded from output (they're
# whole-ticket descriptions, not actionable steps).
_CATEGORY_PRIORITY = {
    "diagnostic": 0,
    "timeline": 1,
    "intervention": 2,
    "context": 99,
}


# ─────────────────────────────────────────────────────────────
# Sprint 11 — Per-ticket detail (raw breakdown).
#
# Companion to the consolidated `steps` list. Mirrors the source
# structure with minimal flattening. Reads exactly the same JSON
# paths as `_harvest_steps`, but does NOT dedupe, normalize, group,
# or cap. Each cohort ticket → one TicketTroubleshootingDetail.
#
# Single source of truth: every Stage 3 path used by `_harvest_steps`
# is also walked here. If a future schema field is added to one place,
# the other should follow — `_build_per_ticket_details` is the
# verbatim view; `_harvest_steps` is the synthesized view.
# ─────────────────────────────────────────────────────────────
def _build_diagnostic_logic_entries(
    chunks: Any,
) -> List[DiagnosticLogicEntry]:
    """Operational_SOP.diagnostic_logic_chunks → typed entries.
    Same key cascade as the harvester (context → rationale → intent;
    branching_logic → pivot) so the per-ticket view matches the
    consolidated view."""
    if not isinstance(chunks, list):
        return []
    out: List[DiagnosticLogicEntry] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        action = _safe_str(ch.get("action") or ch.get("step_id") or ch.get("step"))
        intent = _safe_str(ch.get("context") or ch.get("rationale") or ch.get("intent"))
        pivot = _safe_str(ch.get("branching_logic") or ch.get("pivot"))
        command = _safe_str(ch.get("command"))
        if any((action, intent, pivot, command)):
            out.append(DiagnosticLogicEntry(
                action=action, intent=intent, pivot=pivot, command=command,
            ))
    return out


def _build_resolution_step_strings(rsteps: Any) -> List[str]:
    """Executive_Sharable_RCA.Resolution_Steps → list[str]. Accepts
    list-of-str, list-of-dict (description / action / step), or
    single-string forms."""
    out: List[str] = []
    if isinstance(rsteps, list):
        for s in rsteps:
            if isinstance(s, dict):
                txt = _safe_str(
                    s.get("description") or s.get("action") or s.get("step")
                )
            else:
                txt = _safe_str(s)
            if txt:
                out.append(txt)
    elif isinstance(rsteps, str):
        s = rsteps.strip()
        if s:
            out.append(s)
    return out


def _build_timeline_entries(timeline: Any) -> List[TimelineEntry]:
    """Forensic_Performance_Audit[0].Key_Movements_Timeline →
    typed entries. Both Time and Action carried verbatim; entries
    with neither are skipped."""
    if not isinstance(timeline, list):
        return []
    out: List[TimelineEntry] = []
    for mv in timeline:
        if not isinstance(mv, dict):
            continue
        t = _safe_str(mv.get("Time") or mv.get("time"))
        a = _safe_str(mv.get("Action") or mv.get("action"))
        if t or a:
            out.append(TimelineEntry(time=t, action=a))
    return out


def _build_diagnostic_tests_executed(value: Any) -> List[str]:
    """Forward-compat for Troubleshooting_Ledger.Diagnostic_Tests_Executed.
    Parent key is 0/180 populated in the current corpus; the function
    handles list-of-str and list-of-dict (with 'name' / 'description'
    / 'test') so future schema variations don't require a code change."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for entry in value:
        if isinstance(entry, dict):
            txt = _safe_str(
                entry.get("name")
                or entry.get("description")
                or entry.get("test")
                or entry.get("action")
            )
        else:
            txt = _safe_str(entry)
        if txt:
            out.append(txt)
    return out


def _build_ticket_detail(
    rank: int, ticket: Dict[str, Any]
) -> TicketTroubleshootingDetail:
    """Walk every Stage 3 source path and return a verbatim per-ticket
    detail record. Missing sources stay at their schema default
    (None / [])."""
    if not isinstance(ticket, dict):
        return TicketTroubleshootingDetail(rank=rank)

    incident_number = _safe_str(_safe_get(ticket, "Metadata", "Incident_Number"))

    technical_snapshot = _flatten(
        _safe_get(ticket, "Executive_Sharable_RCA", "Technical_Snapshot"),
    )

    resolution_steps = _build_resolution_step_strings(
        _safe_get(ticket, "Executive_Sharable_RCA", "Resolution_Steps"),
    )

    sop = ticket.get("Operational_SOP") if isinstance(ticket.get("Operational_SOP"), dict) else {}
    diagnostic_logic = _build_diagnostic_logic_entries(
        sop.get("diagnostic_logic_chunks") if isinstance(sop, dict) else None,
    )

    timeline = _build_timeline_entries(
        _safe_get(ticket, "Forensic_Performance_Audit", "Key_Movements_Timeline"),
    )

    ci_raw = _safe_get(ticket, "Forensic_Performance_Audit", "Critical_Intervention")
    critical_intervention = (
        _safe_str(ci_raw) if isinstance(ci_raw, str) else _flatten(ci_raw)
    )

    ha_raw = _safe_get(
        ticket, "Key_Contributors", "Key_Impact_Players", "Hero_Action",
    )
    hero_action = (
        _safe_str(ha_raw) if isinstance(ha_raw, str) else _flatten(ha_raw)
    )

    diagnostic_tests_executed = _build_diagnostic_tests_executed(
        _safe_get(ticket, "Troubleshooting_Ledger", "Diagnostic_Tests_Executed"),
    )

    return TicketTroubleshootingDetail(
        rank=rank,
        incident_number=incident_number,
        technical_snapshot=technical_snapshot,
        resolution_steps=resolution_steps,
        diagnostic_logic=diagnostic_logic,
        timeline=timeline,
        critical_intervention=critical_intervention,
        hero_action=hero_action,
        diagnostic_tests_executed=diagnostic_tests_executed,
    )


# Sprint 11 — placeholder strings the source data uses for "we know
# nothing about this field". Mirrors stage2_historical._TRIVIAL_PLACEHOLDERS;
# kept module-local so a future tweak in either stage stays scoped.
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


def _is_useful_detail(detail: TicketTroubleshootingDetail) -> bool:
    """A per-ticket detail is useful if it has a real incident_number
    AND at least one body section carries non-trivial content.

    Same usefulness contract as Stage 2's _is_useful_card — engineers
    care about the same kind of signal whether they're reading the
    Historical Matches summary or the Troubleshooting per-ticket
    breakdown.
    """
    inc = (detail.incident_number or "").strip()
    if not inc or inc.upper().startswith("UNKNOWN-"):
        return False
    body = (
        detail.technical_snapshot,
        detail.resolution_steps,
        detail.diagnostic_logic,
        detail.timeline,
        detail.critical_intervention,
        detail.hero_action,
        detail.diagnostic_tests_executed,
    )
    return any(not _is_trivial(f) for f in body)


def _build_per_ticket_details(
    cohort: List[Dict[str, Any]],
    *,
    filter_empty: bool = True,
) -> List[TicketTroubleshootingDetail]:
    """One detail entry per cohort ticket, in rank order.

    Sprint 11 — empty / placeholder-only details are filtered when
    filter_empty=True (production default). Pass False to inspect
    raw 1:1 build (the resilience tests use this). Survivors are
    re-ranked 1..N so the visible labels stay sequential after
    sibling drops."""
    raw = [_build_ticket_detail(i + 1, t) for i, t in enumerate(cohort or [])]
    if not filter_empty:
        return raw
    surviving = [d for d in raw if _is_useful_detail(d)]
    re_ranked: List[TicketTroubleshootingDetail] = []
    for new_rank, detail in enumerate(surviving, start=1):
        re_ranked.append(detail.model_copy(update={"rank": new_rank}))
    return re_ranked


def build_stage3(
    cohort: List[Dict[str, Any]],
    max_steps: int = 18,
    *,
    max_diagnostic_steps: int = 6,
    max_timeline_steps: int = 4,
    max_intervention_steps: int = 8,
    max_details_shown: int = 5,
    filter_empty_details: bool = True,
) -> Stage3TroubleshootingApproach:
    """See module docstring.

    Sprint 11 — `max_details_shown` is a frontend rendering hint for
    the per-ticket detail accordion; backend returns ALL useful
    details, frontend caps display at `max_details_shown` and
    surfaces a "View more ticket details" reveal for the rest.
    Mirrors the Stage 2 contract added in the same sprint.

    `filter_empty_details=True` (default) hides per-ticket details
    that carry no useful signal — same usefulness contract as
    Stage 2's _is_useful_card. Pass False to keep the raw 1:1
    build (resilience tests use this).

    Sprint 12.6 — per-category caps. Each category gets a guaranteed
    quota (default diag=6 / timeline=4 / intervention=8); ``max_steps``
    is the absolute total. After per-category quotas are spent, any
    remaining slots up to ``max_steps`` are filled from candidates
    skipped solely due to quota (preserving the ordered sequencing).
    A category with fewer candidates than its quota silently donates
    the leftover slots to other categories. Default ``max_steps``
    raised 8→18 so a 5-ticket cohort with the typical category mix
    (~5 diag + ~5 timeline + ~10 intervention) surfaces all of them.
    Callers passing an explicit lower ``max_steps`` still get exact
    capping — the per-category quotas are advisory upper-bounds.
    """
    if not cohort:
        return Stage3TroubleshootingApproach(
            steps=[], total_unique_steps_before_cap=0, cohort_size=0,
            per_ticket_details=[],
            total_available_details=0,
            max_details_shown=max_details_shown,
        )

    # Sprint 11 — per-ticket raw breakdown is built once, in rank order,
    # independent of the consolidation pipeline. Returned alongside the
    # consolidated `steps` list so the frontend can render both views.
    per_ticket_details = _build_per_ticket_details(
        cohort, filter_empty=filter_empty_details,
    )

    # Harvest + group
    groups: Dict[str, _Group] = {}
    ci_keys_per_incident: Dict[str, set] = defaultdict(set)

    for cohort_idx, ticket in enumerate(cohort):
        for step in _harvest_steps(ticket, cohort_rank=cohort_idx):
            key = _normalize(step.action)
            if not key:
                continue
            g = groups.get(key)
            if g is None:
                # First contributor's category + rank wins for the
                # group. Sprint 10.8 §3.5 — within-category tiebreak
                # uses cohort_rank_first.
                g = _Group(
                    canonical_action=step.action,
                    category=step.category,
                    cohort_rank_first=step.cohort_rank,
                )
                groups[key] = g
            # Sprint 10.1 — accumulate text candidates with their
            # source metadata so _pick_canonical can quality-weight.
            g.action_candidates.append({
                "text": step.action, "metadata_json": step.metadata_json or {},
            })
            if step.intent:
                g.intent_candidates.append({
                    "text": step.intent, "metadata_json": step.metadata_json or {},
                })
                if step.intent not in g.intents:
                    g.intents.append(step.intent)
            if step.pivot:
                g.pivot_candidates.append({
                    "text": step.pivot, "metadata_json": step.metadata_json or {},
                })
                if step.pivot not in g.pivots:
                    g.pivots.append(step.pivot)
            if step.command:
                g.command_candidates.append({
                    "text": step.command, "metadata_json": step.metadata_json or {},
                })
                if step.command not in g.commands:
                    g.commands.append(step.command)
            if step.incident and step.incident not in g.incidents:
                g.incidents.append(step.incident)
                g.incident_order.append(step.cohort_rank)
            g.ordinals.append(step.ordinal)
            if step.is_critical_intervention:
                g.has_critical_intervention = True
                ci_keys_per_incident[step.incident].add(key)
            # Track the lowest cohort_rank that contributed (for
            # within-category sequencing tiebreak per §3.5).
            if step.cohort_rank < g.cohort_rank_first:
                g.cohort_rank_first = step.cohort_rank

    # Sprint 10.1 — quality-weighted canonical pick of action text.
    # Sprint 10.8 §3.4 — when the quality-weighted pick ties, longer
    # display text wins (handled inside _pick_canonical via length
    # tiebreak).
    for g in groups.values():
        if g.action_candidates:
            g.canonical_action = _pick_canonical(g.action_candidates)

    if not groups:
        return Stage3TroubleshootingApproach(
            steps=[], total_unique_steps_before_cap=0, cohort_size=len(cohort),
            per_ticket_details=per_ticket_details,
            total_available_details=len(per_ticket_details),
            max_details_shown=max_details_shown,
        )

    # ── §3.6 fallback detection ──
    # Two intervention-category groups that came from non-overlapping
    # tickets and did NOT collapse → second one is marked is_fallback.
    # Sprint 10.0's branch_label ("Primary"/"Alt A") stays populated in
    # parallel so existing tests + analytics dashboards keep working.
    ci_groups: Dict[str, int] = {}
    for key, g in groups.items():
        if g.has_critical_intervention:
            ci_groups[key] = len(g.incidents)

    branch_label_for: Dict[str, str] = {}
    if len(ci_groups) >= 2:
        ci_sorted = sorted(ci_groups.items(), key=lambda x: x[1], reverse=True)
        branch_label_for[ci_sorted[0][0]] = "Primary"
        alt_letters = ["Alt A", "Alt B", "Alt C", "Alt D"]
        for i, (key, _count) in enumerate(ci_sorted[1:5]):
            branch_label_for[key] = alt_letters[i]

    # ── §3.5 sequencing — exclude context steps, then sort by
    # (category priority, cohort_rank_first, average ordinal). ──
    def _avg(xs: List[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    user_facing = [
        (key, g) for key, g in groups.items()
        if g.category != "context"
    ]
    sequenced = sorted(
        user_facing,
        key=lambda kv: (
            _CATEGORY_PRIORITY.get(kv[1].category, 99),
            kv[1].cohort_rank_first,
            _avg(kv[1].ordinals),
            len(kv[1].incidents) * -1,
        ),
    )

    total_unique = len(sequenced)

    # ── Sprint 12.6 — per-category two-pass admission ──
    # Pass 1: walk in sequenced order; admit each item if its category
    # quota isn't exhausted AND total < max_steps. Pass 2: walk the
    # same list again admitting anything skipped in pass 1, up to
    # max_steps. The two-pass shape means heavy diagnostic-only
    # cohorts still fill to max_steps with diagnostics (existing
    # behaviour, used by tests) while mixed cohorts guarantee
    # interventions reach the engineer.
    quota_remaining: Dict[str, int] = {
        "diagnostic":   max_diagnostic_steps,
        "timeline":     max_timeline_steps,
        "intervention": max_intervention_steps,
    }
    admitted_keys: set = set()
    admitted: List = []
    for key, g in sequenced:
        if len(admitted) >= max_steps:
            break
        cat = g.category
        if quota_remaining.get(cat, 0) <= 0:
            continue
        admitted.append((key, g))
        admitted_keys.add(key)
        quota_remaining[cat] = quota_remaining.get(cat, 0) - 1

    # Pass 2 — fill remaining max_steps with quota-skipped items.
    if len(admitted) < max_steps:
        for key, g in sequenced:
            if len(admitted) >= max_steps:
                break
            if key in admitted_keys:
                continue
            admitted.append((key, g))
            admitted_keys.add(key)

    # Per-category breakdown for the observability log line below.
    shown_by_cat: Dict[str, int] = {"diagnostic": 0, "timeline": 0, "intervention": 0}
    for _key, g in admitted:
        shown_by_cat[g.category] = shown_by_cat.get(g.category, 0) + 1

    logger.info(
        "[stage3] cohort=%d raw_unique=%d shown=%d "
        "(diag=%d timeline=%d intervention=%d)",
        len(cohort), total_unique, len(admitted),
        shown_by_cat.get("diagnostic", 0),
        shown_by_cat.get("timeline", 0),
        shown_by_cat.get("intervention", 0),
    )

    # ── Build output ──
    # Sprint 10.8 §3.6 — among intervention steps, the FIRST one keeps
    # is_fallback=False; every subsequent intervention that ALSO has a
    # non-overlapping incident set is marked is_fallback=True.
    out_steps: List[TroubleshootingStep] = []
    seen_intervention_incidents: set = set()
    interventions_emitted = 0

    for n, (key, g) in enumerate(admitted, start=1):
        intent = (
            _pick_canonical(g.intent_candidates) if g.intent_candidates
            else (" / ".join(g.intents[:2]) if g.intents else None)
        )
        pivot = (
            _pick_canonical(g.pivot_candidates) if g.pivot_candidates
            else (g.pivots[0] if g.pivots else None)
        )
        command = (
            _pick_canonical(g.command_candidates) if g.command_candidates
            else (g.commands[0] if g.commands else None)
        )

        # Sprint 10.8 §3.6 — fallback marker for second-and-later
        # intervention groups whose incidents don't overlap the
        # already-emitted intervention's incident set.
        is_fallback = False
        if g.category == "intervention":
            current_incidents = set(g.incidents)
            if (
                interventions_emitted >= 1
                and current_incidents.isdisjoint(seen_intervention_incidents)
            ):
                is_fallback = True
            seen_intervention_incidents.update(current_incidents)
            interventions_emitted += 1

        out_steps.append(TroubleshootingStep(
            step_number=n,
            action=g.canonical_action,
            intent=intent or None,
            pivot=pivot or None,
            command=command or None,
            branch_label=branch_label_for.get(key),
            seen_in_incidents=g.incidents[:5],
            is_fallback=is_fallback,
            category=g.category if g.category in (
                "diagnostic", "timeline", "intervention", "context",
            ) else "diagnostic",
        ))

    return Stage3TroubleshootingApproach(
        steps=out_steps,
        total_unique_steps_before_cap=total_unique,
        cohort_size=len(cohort),
        per_ticket_details=per_ticket_details,
        total_available_details=len(per_ticket_details),
        max_details_shown=max_details_shown,
    )
