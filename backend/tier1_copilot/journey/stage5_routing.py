"""Sprint 12.7 — Escalation Routing & Vendor/OEM Engagement aggregator.

Per spec:
  Synthesize the final escalation flow. Based on the aggregated
  Team_Path and Vendor_OEM_Engagement data, define a unified
  escalation threshold. Instruct the engineer exactly which internal
  Tier 2/3 group handles this cluster of issues, and compile a master
  list of all forensic data (PCAPs, logs, serial numbers) required
  before engaging the vendor or OEM.

Source field families (per cohort ticket):
  - Metadata.Resolution_Groups          (List[str])
  - Engagement_Analysis.Team_Path       (str, '->' separated path)
  - Vendor_OEM_Engagement               (dict — 0/213 in current corpus,
                                         wired forward-compat)

Aggregation rules:
  - Resolution_Groups: deduped (case-insensitive, first-seen casing
    preserved), in cohort first-seen order.
  - Team_Path: full path strings deduped verbatim.
  - Recommended Tier-2 entry: parsed second-hop of each Team_Path
    (the natural Tier-2 entry point after the L1 origin like NOC),
    deduped, with frequency counts. When every cohort ticket carries
    a different Tier-2 entry — common for diverse clusters — we
    surface them all rather than picking a synthesised winner the
    data can't actually support.
  - Vendor records / forensic data: collected when present; empty
    list/dict when not. Frontend renders an honest "no vendor records
    in this cohort" empty state rather than fabricating boilerplate.

Pure function — no LLM, no DB.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import EscalationRouting, Tier2EntryCandidate


logger = logging.getLogger("acadia-log-iq")


def _safe_get(obj: Any, *path: str) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _coerce_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _add_unique(seen: Dict[str, str], ordered: List[str], value: str) -> None:
    """Insert preserving first-seen casing, case-insensitive dedupe."""
    if not value:
        return
    key = value.casefold()
    if key in seen:
        return
    seen[key] = value
    ordered.append(value)


def _parse_team_path_hops(path: str) -> List[str]:
    """Split a Team_Path string on '->' (with or without surrounding
    whitespace) and return the cleaned hop list. Tolerant of '→' as a
    fallback separator."""
    if not isinstance(path, str) or not path.strip():
        return []
    # Normalise both ASCII '->' and unicode '→'.
    normalised = path.replace("→", "->")
    return [h.strip() for h in normalised.split("->") if h.strip()]


def build_escalation_routing(cohort: List[Dict[str, Any]]) -> EscalationRouting:
    """See module docstring."""
    n = len(cohort or [])
    if n == 0:
        return EscalationRouting(
            cohort_size=0,
            tickets_with_data=0,
            empty=True,
        )

    resolution_groups: List[str] = []
    seen_groups: Dict[str, str] = {}

    team_paths: List[str] = []
    seen_paths: Dict[str, str] = {}

    # Recommended Tier-2 candidates: case-insensitive group → frequency
    # count + an example Team_Path string for context. We surface the
    # SECOND hop of the path (the L1→L2 transition) — the first hop is
    # almost always the L1 origin (NOC / MNOps) which doesn't help the
    # current engineer figure out where to escalate.
    tier2_counts: Dict[str, int] = {}
    tier2_canonical: Dict[str, str] = {}
    tier2_example_path: Dict[str, str] = {}

    vendor_records: List[Dict[str, Any]] = []
    forensic_data_required: List[str] = []
    seen_forensic: Dict[str, str] = {}

    tickets_with_data = 0

    for ticket in cohort:
        if not isinstance(ticket, dict):
            continue
        contributed = False

        # Metadata.Resolution_Groups → deduped list
        groups_raw = _safe_get(ticket, "Metadata", "Resolution_Groups")
        if isinstance(groups_raw, list):
            for g in groups_raw:
                s = _coerce_str(g)
                if s:
                    _add_unique(seen_groups, resolution_groups, s)
                    contributed = True
        elif isinstance(groups_raw, str):
            s = _coerce_str(groups_raw)
            if s:
                _add_unique(seen_groups, resolution_groups, s)
                contributed = True

        # Engagement_Analysis.Team_Path → deduped path string + parsed
        # second-hop tier-2 candidate.
        path_str = _coerce_str(_safe_get(ticket, "Engagement_Analysis", "Team_Path"))
        if path_str:
            _add_unique(seen_paths, team_paths, path_str)
            hops = _parse_team_path_hops(path_str)
            if len(hops) >= 2:
                tier2 = hops[1]
                key = tier2.casefold()
                tier2_counts[key] = tier2_counts.get(key, 0) + 1
                if key not in tier2_canonical:
                    tier2_canonical[key] = tier2
                    tier2_example_path[key] = path_str
            contributed = True

        # Vendor_OEM_Engagement → forward-compat. Field is 0/213 in
        # the corpus today; we collect any dict/list shape verbatim
        # so future ingestion lights up automatically. Forensic
        # requirements (PCAPs / logs / serial numbers) are pulled from
        # well-known sub-keys when present; when not, the list stays
        # empty and the frontend renders an honest "no vendor records"
        # state.
        vendor_raw = ticket.get("Vendor_OEM_Engagement")
        if isinstance(vendor_raw, dict) and vendor_raw:
            vendor_records.append(vendor_raw)
            for forensic_key in (
                "forensic_data_required",
                "required_artifacts",
                "evidence_required",
                "required_data",
            ):
                items = vendor_raw.get(forensic_key)
                if isinstance(items, list):
                    for it in items:
                        s = _coerce_str(it)
                        if s:
                            _add_unique(seen_forensic, forensic_data_required, s)
                elif isinstance(items, str):
                    s = _coerce_str(items)
                    if s:
                        _add_unique(seen_forensic, forensic_data_required, s)
            contributed = True
        elif isinstance(vendor_raw, list) and vendor_raw:
            for entry in vendor_raw:
                if isinstance(entry, dict):
                    vendor_records.append(entry)
            contributed = True

        if contributed:
            tickets_with_data += 1

    # Build the Tier-2 candidate list, sorted desc by frequency then
    # alphabetical for stable rendering.
    recommended_tier2 = [
        Tier2EntryCandidate(
            team=tier2_canonical[key],
            occurrence_count=tier2_counts[key],
            example_path=tier2_example_path[key],
        )
        for key in sorted(
            tier2_counts.keys(),
            key=lambda k: (-tier2_counts[k], tier2_canonical[k].lower()),
        )
    ]

    empty = not (
        resolution_groups
        or team_paths
        or recommended_tier2
        or vendor_records
        or forensic_data_required
    )

    coverage_pct = round(100.0 * tickets_with_data / n) if n else 0
    logger.info(
        "[stage5_routing] cohort=%d tickets_with_data=%d (%d%%) "
        "groups=%d paths=%d tier2_candidates=%d vendor_records=%d "
        "forensic_items=%d empty=%s",
        n, tickets_with_data, coverage_pct,
        len(resolution_groups), len(team_paths), len(recommended_tier2),
        len(vendor_records), len(forensic_data_required), empty,
    )

    return EscalationRouting(
        resolution_groups=resolution_groups,
        team_paths=team_paths,
        recommended_tier2_teams=recommended_tier2,
        vendor_records=vendor_records,
        forensic_data_required=forensic_data_required,
        cohort_size=n,
        tickets_with_data=tickets_with_data,
        empty=empty,
    )
