"""Sprint 10.2 — Stage 0 Best-Ticket Distillation.

Replaces the Sprint 10 / 10.1 corpus-aggregate Confidence Lead. The new
architecture:
  1. Read cohort metadata via existing ticket_loader.
  2. Sort cohort by Resolution_Quality_Score DESC (recency tiebreak).
  3. Distil best ticket's Primary_Fix + Resolution_Steps as the headline.
  4. Run a SECOND, narrower SQL aggregate ONLY for corpus_size +
     platform_median_minutes (the only fields that need a corpus scan).
     This narrower query has no nullable filters — no parameter-binding
     bug surface.

Sparse fallback: cohort empty → render "Limited historical data" empty
state (sparse=True). All other fields default to None / 0.

No LLM. No retrieval, no ranking. Failure modes degrade gracefully —
DB failure on the corpus aggregate just leaves corpus_size +
platform_median_minutes None; the cohort-distilled fields still
populate.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Integer, String, bindparam, text

from .schemas import Stage0BestTicketDistillation


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Profile-signature builder (kept from Sprint 10 — feeds the
# `profile_match` field which is preserved for backward-compat
# tooling). Logic is unchanged from prior sprints.
# ─────────────────────────────────────────────────────────────
def _modal(values: List[str]) -> Optional[str]:
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _safe_str(d: Any, *path: str) -> Optional[str]:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if cur is None:
        return None
    s = str(cur).strip()
    return s or None


def _safe_int(v: Any) -> Optional[int]:
    if v in (None, "", "N/A"):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _derive_profile(
    cohort: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Return (component_category, asset_family, detected_symptom,
    human_readable_signature). Lazy-imports derive_asset_family from
    the intake catalog package."""
    try:
        from backend.tier1_copilot.intake.catalogs import derive_asset_family
    except Exception:
        derive_asset_family = None  # type: ignore[assignment]

    components: List[str] = []
    asset_families: List[str] = []
    symptoms: List[str] = []
    for t in cohort or []:
        if not isinstance(t, dict):
            continue
        cc = _safe_str(t, "Metadata", "component_category")
        if cc:
            components.append(cc)
        target = _safe_str(t, "Metadata", "Target_Service")
        if not target:
            aa = (t.get("Metadata") or {}).get("Affected_Assets") if isinstance(t.get("Metadata"), dict) else None
            if isinstance(aa, list) and aa:
                target = str(aa[0]).strip() if aa[0] else None
            elif isinstance(aa, str):
                target = aa.strip() or None
        if target and derive_asset_family is not None:
            try:
                fam = derive_asset_family(target)
            except Exception:
                fam = None
            if fam:
                asset_families.append(fam)
        symp = _safe_str(t, "Symptom_Solution_Mapping", "Detected_Symptom")
        if symp:
            symptoms.append(symp)

    component = _modal(components)
    family = _modal(asset_families)
    symptom = _modal(symptoms)
    parts = [p for p in (symptom, component, family) if p]
    signature = " · ".join(parts) if parts else "unrecognised cohort profile"
    return component, family, symptom, signature


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — minimal corpus aggregate (no nullable filters)
# Bindparam-typed via SQLAlchemy text-binding idiom so psycopg
# can never trip on type inference. Drops the Sprint 10.1 SQL-
# level CAST workaround.
# ─────────────────────────────────────────────────────────────
_CORPUS_SQL = """
SELECT
  COUNT(*) AS corpus_size,
  PERCENTILE_DISC(0.5) WITHIN GROUP (
    ORDER BY NULLIF(metadata_json->'Metadata'->>'time_to_resolve_minutes','')::numeric
  ) FILTER (
    WHERE metadata_json->'Metadata'->>'time_to_resolve_minutes' ~ '^[0-9]+$'
  ) AS platform_median_minutes
FROM chunks
WHERE metadata_json->>'doc_kind' = 'ticket'
"""


def _fetch_corpus_stats() -> Tuple[int, Optional[int]]:
    """Run the narrow corpus aggregate. Returns (corpus_size,
    platform_median_minutes). On any failure returns (0, None) — the
    rest of Stage 0 still populates from cohort distillation."""
    try:
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[journey.stage0] DB engine import failed: %s", exc)
        return 0, None

    try:
        # Sprint 10.2 — single dict params (NOT list-of-dicts) so
        # SQLAlchemy doesn't interpret as executemany() and chokes.
        # No bindparams needed since there are no nullable filter args.
        with engine.connect() as conn:
            row = conn.execute(text(_CORPUS_SQL), {}).mappings().first()
    except Exception as exc:
        logger.warning("[journey.stage0] corpus aggregate failed: %s", exc)
        return 0, None

    if not row:
        return 0, None
    corpus_size = int(row.get("corpus_size") or 0)
    median_raw = row.get("platform_median_minutes")
    median_min = int(round(float(median_raw))) if median_raw is not None else None
    return corpus_size, median_min


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — typed bindparams for nullable-filter SQL.
# Kept here as a public helper because the Sprint 10.1 spec called
# for the bindparam idiom. Currently unused inside this module
# (the new architecture removed the nullable-filter aggregate),
# but exposed so other call-sites that need typed nullable-filter
# binding can use the same pattern.
# ─────────────────────────────────────────────────────────────
def make_typed_filter_stmt(sql: str):
    """Return a SQLAlchemy text() statement with explicit bindparam
    typing for the standard journey filter params. Eliminates the
    `could not determine data type of parameter` failure that occurs
    when nullable params appear only in `IS NULL` checks."""
    return text(sql).bindparams(
        bindparam("lookback_months", type_=Integer),
        bindparam("component", type_=String),
        bindparam("asset_family", type_=String),
        bindparam("asset_family_pattern", type_=String),
        bindparam("symptom", type_=String),
        bindparam("symptom_pattern", type_=String),
    )


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — best-ticket distillation
# ─────────────────────────────────────────────────────────────
def _quality_score(ticket: Dict[str, Any]) -> int:
    if not isinstance(ticket, dict):
        return 0
    raw = (ticket.get("Metadata") or {}).get("Resolution_Quality_Score") if isinstance(ticket.get("Metadata"), dict) else None
    try:
        return int(raw or 0)
    except (ValueError, TypeError):
        return 0


def _ticket_recency_key(ticket: Dict[str, Any]) -> str:
    """Best-effort recency tiebreaker. Prefer Created_At / Updated_At
    timestamp strings; lexicographic compare works for ISO-8601 format.
    Falls back to incident-number lexicographic order so the sort is
    deterministic even when timestamps are absent."""
    if not isinstance(ticket, dict):
        return ""
    meta = ticket.get("Metadata") if isinstance(ticket.get("Metadata"), dict) else {}
    for k in ("Updated_At", "Closed_At", "Created_At"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v
    inc = meta.get("Incident_Number")
    return str(inc) if inc else ""


def _best_ticket(cohort: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Sprint 10.3 §4.2 — pick the best-rated cohort ticket with a
    STABLE rank-tiebreak. When multiple tickets share the same
    Resolution_Quality_Score, prefer the one with the lowest retrieval
    rank (i.e., the cohort's index 0 wins ties).

    The cohort is already ordered by retrieval rank (rank 1 first) by
    `ticket_loader.load_cohort_metadata`. Sorting by (-score, idx)
    ascending gives us score-descending with stable rank-ascending
    tiebreak. The previous implementation tied on a recency-key string
    that fell back to incident-number lexicographic order, which is
    why an all-score-3 cohort picked INC-ALPHA-028 over the rank-0
    INC-ALPHA-027.
    """
    if not cohort:
        return None
    candidates = [(idx, t) for idx, t in enumerate(cohort) if isinstance(t, dict)]
    if not candidates:
        return None
    # (-score, idx) ascending → highest score wins; ties broken by rank
    candidates.sort(key=lambda pair: (-_quality_score(pair[1]), pair[0]))
    return candidates[0][1]


def _derive_evidence_strength(score: Optional[int]) -> str:
    """Sprint 10.3 §4.3 — bucket the picked ticket's score into a
    strength label. Frontend renders different headline copy per
    bucket so a score-3 ticket isn't billed as "best-rated past
    resolution" when the cohort genuinely doesn't have strong
    evidence."""
    if score is None or score <= 0:
        return "none"
    if score >= 4:
        return "strong"
    if score == 3:
        return "adequate"
    # score in (1, 2)
    return "weak"


def _resolution_steps(
    ticket: Dict[str, Any],
    incident_number: Optional[str] = None,
) -> List[str]:
    """Extract Resolution_Steps as a flat string list, capped at 5.
    Handles list-of-strings, list-of-dicts, and single-string shapes.

    Sprint 12 — when `incident_number` is provided (e.g. the best
    ticket's Incident_Number), each step is suffixed with " - <id>"
    so the engineer (and any downstream Tier-2 reader) can trace the
    step back to the source ticket. The suffix is only appended when
    a non-empty incident_number is supplied; legacy callers that
    don't pass it get the original behavior."""
    rca = ticket.get("Executive_Sharable_RCA") if isinstance(ticket.get("Executive_Sharable_RCA"), dict) else {}
    raw = rca.get("Resolution_Steps") if isinstance(rca, dict) else None
    suffix = f" - {incident_number}" if incident_number else ""
    out: List[str] = []
    if isinstance(raw, list):
        for s in raw:
            if isinstance(s, dict):
                txt = s.get("description") or s.get("action") or s.get("step") or ""
            else:
                txt = s
            if txt is None:
                continue
            t = str(txt).strip()
            if t:
                out.append(f"{t}{suffix}")
            if len(out) >= 5:
                break
    elif isinstance(raw, str) and raw.strip():
        out.append(f"{raw.strip()}{suffix}")
    return out


def _top5_first_steps(cohort: List[Dict[str, Any]]) -> List[str]:
    """Sprint 12.2 — span the top-5 cohort tickets, ordered by
    retrieval similarity (most similar first).

    Design notes
    ------------
    Two orderings exist on the cohort and they are NOT the same:

      * Retrieval similarity rank — produced by upstream retrieval
        and reflected in the cohort's own list order (rank 1 first)
        per ``ticket_loader.load_cohort_metadata``'s contract.
      * Resolution_Quality_Score — used by ``_best_ticket()`` to
        pick the *headline* ticket (the "Best fix came from INC-XXX"
        line). High-quality writeups deserve the headline even if
        they aren't the closest match.

    The bullet list ("How they did it") tracks the FIRST ordering:
    bullet 1 is the most-similar past ticket, bullet 2 the second
    most-similar, and so on. The headline keeps tracking the SECOND
    ordering. This lets the engineer read the bullet list as a
    similarity-ranked tour of past evidence, while the headline
    still surfaces the highest-quality precedent.

    For each top-5 ticket in similarity order, we extract its first
    resolution step via the existing ``_resolution_steps`` helper —
    same field, same coercion logic, same trailing
    ``" - <Incident_Number>"`` suffix that powers the per-bullet
    Ask-in-Chat scope on the frontend. Tickets without an
    extractable step are skipped (never padded). Return type is
    ``List[str]``; the Stage 0 schema is unchanged.
    """
    if not cohort:
        return []
    # Preserve cohort order — index 0 = retrieval rank 1 = most
    # similar to the engineer's incident. We do NOT re-sort by
    # quality score here; that's `_best_ticket()`'s job for the
    # headline.
    seen_incidents: set = set()
    out: List[str] = []
    for ticket in cohort:
        if not isinstance(ticket, dict):
            continue
        if len(out) >= 5:
            break
        inc = _safe_str(ticket, "Metadata", "Incident_Number")
        # Defensive dedupe in case the cohort accidentally contains
        # the same Incident_Number twice — never show duplicate
        # sources side-by-side.
        if inc and inc in seen_incidents:
            continue
        steps = _resolution_steps(ticket, inc)
        if not steps:
            continue
        out.append(steps[0])
        if inc:
            seen_incidents.add(inc)
    return out


def _avg_minutes_cohort(cohort: List[Dict[str, Any]]) -> Optional[int]:
    """Average time_to_resolve_minutes across cohort tickets that have
    parseable numeric values. Returns None when no ticket has one."""
    values: List[int] = []
    for t in cohort or []:
        if not isinstance(t, dict):
            continue
        v = (t.get("Metadata") or {}).get("time_to_resolve_minutes") if isinstance(t.get("Metadata"), dict) else None
        i = _safe_int(v)
        if i is not None:
            values.append(i)
    if not values:
        return None
    return int(round(sum(values) / len(values)))


def _empty(profile_signature: str) -> Stage0BestTicketDistillation:
    return Stage0BestTicketDistillation(
        cohort_size=0,
        profile_match=profile_signature or None,
        sparse=True,
        evidence_strength="none",
    )


def compute_stage0(
    cohort: List[Dict[str, Any]],
    lookback_months: int = 18,  # kept for backward-compat call signature
) -> Stage0BestTicketDistillation:
    """Sprint 10.2 — distil the cohort's best ticket + corpus tail.

    Always returns a Stage0BestTicketDistillation. Empty cohort →
    sparse=True with profile_match populated. Corpus aggregate failure
    → corpus_size=0 and platform_median_minutes=None, but the
    cohort-distilled fields still populate."""
    _component, _family, _symptom, signature = _derive_profile(cohort)

    if not cohort:
        return _empty(signature)

    best = _best_ticket(cohort)
    if best is None:
        return _empty(signature)

    # ── Cohort-derived fields ──
    meta = best.get("Metadata") if isinstance(best.get("Metadata"), dict) else {}
    best_incident = _safe_str(best, "Metadata", "Incident_Number")
    best_quality = _safe_int(meta.get("Resolution_Quality_Score"))
    best_time = _safe_int(meta.get("time_to_resolve_minutes"))
    what_worked = _safe_str(best, "Symptom_Solution_Mapping", "Primary_Fix")
    # Sprint 12.1 — span the top-5 cohort tickets instead of pulling
    # all bullets from the single best ticket. Each bullet now carries
    # its own " - <Incident_Number>" suffix, so the per-bullet
    # Ask-in-Chat handoff can extract the source incident from the
    # suffix and scope the resulting chat session to that one ticket
    # (see `_top5_first_steps` docstring above for the full rationale).
    # Schema is unchanged — `how_they_did_it` is still List[str].
    how_they_did_it = _top5_first_steps(cohort)
    critical_intervention = _safe_str(best, "Forensic_Performance_Audit", "Critical_Intervention")

    # ── Cohort stats (collapsed tail) ──
    cohort_size = len([t for t in cohort if isinstance(t, dict)])
    clean_count = sum(1 for t in cohort if _quality_score(t) >= 4)
    clean_pct = round(100.0 * clean_count / cohort_size) if cohort_size else 0
    avg_min_cohort = _avg_minutes_cohort(cohort)

    # ── Corpus stats (separate, narrow SQL aggregate) ──
    corpus_size, platform_median = _fetch_corpus_stats()

    return Stage0BestTicketDistillation(
        cohort_size=cohort_size,
        best_incident=best_incident,
        best_quality_score=best_quality,
        best_time_minutes=best_time,
        what_worked=what_worked,
        how_they_did_it=how_they_did_it,
        critical_intervention=critical_intervention,
        clean_resolution_count=clean_count,
        clean_resolution_percent=clean_pct,
        avg_minutes_to_resolve_cohort=avg_min_cohort,
        corpus_size=corpus_size,
        platform_median_minutes=platform_median,
        profile_match=signature or None,
        sparse=False,
        # Sprint 10.3 — bucket the picked ticket's score so the
        # frontend headline can vary by strength.
        evidence_strength=_derive_evidence_strength(best_quality),
    )
