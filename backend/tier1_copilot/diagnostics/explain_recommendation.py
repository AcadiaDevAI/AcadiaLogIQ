"""Sprint 7 — Explain Recommendation card.

Produces the score-breakdown + matched-fields checklist + historical
success rate for the currently matched ticket. Pure computation on the
ranked-candidate object that `retrieval._weighted_rank` already
populated under `_score_components`.

No LLM call. The score formula here must stay in lockstep with
`retrieval._weighted_rank` — both pull from settings.TIER1_RANKING_WEIGHTS
via `_active_weights()` so a weight change ripples into the explain
payload automatically.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from backend.tier1_copilot.schemas import (
    Tier1ExplainResponse,
    Tier1FieldMatch,
    Tier1HistoricalSuccess,
    Tier1ScoreBreakdown,
)

logger = logging.getLogger("acadia-log-iq")


def _match_signal(a: Optional[str], b: Optional[str]) -> Any:
    if not a or not b:
        return False
    al = str(a).strip().lower()
    bl = str(b).strip().lower()
    if not al or not bl:
        return False
    if al == bl:
        return True
    if al in bl or bl in al:
        return "partial"
    return False


def build_matched_fields(
    *,
    alert_payload: Dict[str, Any],
    ticket_metadata: Dict[str, Any],
) -> List[Tier1FieldMatch]:
    meta = ticket_metadata.get("Metadata") or {}
    meta = meta if isinstance(meta, dict) else {}
    ssm = ticket_metadata.get("Symptom_Solution_Mapping") or {}
    ssm = ssm if isinstance(ssm, dict) else {}

    ticket_asset = None
    assets = meta.get("Affected_Assets")
    if isinstance(assets, list) and assets:
        ticket_asset = str(assets[0])
    elif isinstance(assets, str):
        ticket_asset = assets
    else:
        ticket_asset = meta.get("Target_Service")

    rows: List[Tier1FieldMatch] = [
        Tier1FieldMatch(
            field="Asset",
            your_value=alert_payload.get("asset_name"),
            ticket_value=ticket_asset,
            match=_match_signal(alert_payload.get("asset_name"), ticket_asset),
        ),
        Tier1FieldMatch(
            field="Alert type",
            your_value=alert_payload.get("alert_type"),
            ticket_value=ssm.get("Detected_Symptom"),
            match=_match_signal(
                alert_payload.get("alert_type"), ssm.get("Detected_Symptom"),
            ),
        ),
        Tier1FieldMatch(
            field="Severity",
            your_value=alert_payload.get("severity"),
            ticket_value=meta.get("priority") or meta.get("Priority"),
            match=_match_signal(
                alert_payload.get("severity"),
                meta.get("priority") or meta.get("Priority"),
            ),
        ),
        Tier1FieldMatch(
            field="Customer",
            your_value=alert_payload.get("customer"),
            ticket_value=meta.get("customer_name") or meta.get("Customer_Name"),
            match=_match_signal(
                alert_payload.get("customer"),
                meta.get("customer_name") or meta.get("Customer_Name"),
            ),
        ),
        Tier1FieldMatch(
            field="Technology",
            your_value=alert_payload.get("technology"),
            ticket_value=meta.get("component_category"),
            match=_match_signal(
                alert_payload.get("technology"), meta.get("component_category"),
            ),
        ),
    ]
    return rows


def compute_historical_success(
    *,
    top_matches: Iterable[Dict[str, Any]],
) -> Tier1HistoricalSuccess:
    """Approximate success rate from the ranked candidate list.

    A "success" here is any sibling ticket whose
    QA_Auditor_Feedback.Rework_Detected is falsy AND whose
    resolution_quality_score >= 3. This is a heuristic (we don't have a
    dedicated success-telemetry column) but matches the Sprint 6 ranking
    fallback in ingestion and keeps the UI honest about uncertainty.
    """
    total = 0
    succeeded: List[str] = []
    failed = 0
    primary_fix: Optional[str] = None

    for c in top_matches:
        md = c.get("metadata_json") if isinstance(c, dict) else {}
        md = md if isinstance(md, dict) else {}
        meta = md.get("Metadata") or {}
        meta = meta if isinstance(meta, dict) else {}
        ssm = md.get("Symptom_Solution_Mapping") or {}
        ssm = ssm if isinstance(ssm, dict) else {}
        qa = md.get("QA_Auditor_Feedback") or {}
        qa = qa if isinstance(qa, dict) else {}

        if primary_fix is None and ssm.get("Primary_Fix"):
            primary_fix = str(ssm.get("Primary_Fix"))

        total += 1
        rework = bool(qa.get("Rework_Detected", False))
        qscore = 0
        try:
            qscore = int(meta.get("Resolution_Quality_Score") or 0)
        except Exception:
            qscore = 0
        inc = meta.get("Incident_Number")
        if (not rework) and qscore >= 3:
            if inc:
                succeeded.append(str(inc))
        else:
            failed += 1

    succeeded_count = len(succeeded)
    rate = int(round((succeeded_count / total) * 100)) if total else 0
    return Tier1HistoricalSuccess(
        total_similar=total,
        primary_fix=primary_fix,
        succeeded_count=succeeded_count,
        succeeded_in=succeeded[:10],
        failed_count=failed,
        success_rate_percent=rate,
    )


def build_explain(
    *,
    alert_payload: Dict[str, Any],
    top_matches: List[Dict[str, Any]],
    target_chunk_id: Optional[str] = None,
) -> Tier1ExplainResponse:
    if not top_matches:
        return Tier1ExplainResponse(
            matched_incident=None,
            score_breakdown=Tier1ScoreBreakdown(),
        )

    picked = None
    if target_chunk_id:
        picked = next(
            (c for c in top_matches if c.get("chunk_id") == target_chunk_id),
            None,
        )
    if picked is None:
        picked = top_matches[0]

    comps = picked.get("_score_components") or {}

    breakdown = Tier1ScoreBreakdown(
        alert_type_match=float(comps.get("alert_type_match", 0.0)),
        asset_match=float(comps.get("asset_match", 0.0)),
        fingerprint_match=float(comps.get("fingerprint_match", 0.0)),
        technology_match=float(comps.get("technology_match", 0.0)),
        vector_similarity=float(comps.get("vector_similarity", 0.0)),
        resolution_quality=float(comps.get("resolution_quality", 0.0)),
        recency=float(comps.get("recency", 0.0)),
        success_frequency=float(comps.get("success_frequency", 0.0)),
        same_customer_boost=float(comps.get("same_customer_boost", 0.0)),
        same_asset_family_boost=float(comps.get("same_asset_family", 0.0)),
        final_score=float(picked.get("final_score", 0.0)),
    )

    matched_md = picked.get("metadata_json") or {}
    meta = matched_md.get("Metadata") or {}
    matched_incident = (
        meta.get("Incident_Number") if isinstance(meta, dict) else None
    )

    return Tier1ExplainResponse(
        matched_incident=matched_incident,
        score_breakdown=breakdown,
        matched_fields=build_matched_fields(
            alert_payload=alert_payload, ticket_metadata=matched_md,
        ),
        historical_success=compute_historical_success(top_matches=top_matches),
    )
