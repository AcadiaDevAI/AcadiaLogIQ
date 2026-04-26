"""Sprint 9.2 — Substring-grounded validator.

Three-stage validation pipeline:

  Stage 1 — Substring grounding (anti-hallucination).
            Each non-null field's `evidence` must appear as a substring
            (case-insensitive, whitespace-collapsed) of `raw_text`.
            Failed grounding → field nulled before any catalog lookup
            even runs. This is the primary defence against the Sprint 9
            hint-contamination bug where the LLM picked from the
            catalog hint list instead of the email body.

  Stage 2 — Catalog matching (deterministic, post-extraction).
            Verified extracted values get an exact-lowercase lookup
            against the full catalog first; a fuzzy fallback at the
            field-specific threshold (asset 0.85, alert 0.80, customer
            0.85) follows. The full catalog is searched — Sprint 9's
            top-30-hint sampling is gone.

  Stage 3 — Confidence derivation.
            "High" requires all three mandatory fields catalog-matched
            AND substring-grounded AND customer matched-and-grounded.
            Any grounding miss demotes to Low even if the catalog
            lookup happened to "work" — because we can't prove the LLM
            extracted from the email vs. invented the value.

Stdlib only (`difflib.SequenceMatcher`). The catalogs module continues
to lowercase-normalise terms at insert time, so all comparisons here
operate on lower-cased strings.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Set

from backend.config import settings
from backend.tier1_copilot.intake.catalogs import (
    IntakeCatalogs,
    derive_asset_family,
    normalise_alert_type,
    normalise_customer,
)
from backend.tier1_copilot.intake.schemas import (
    CanonicalForm,
    ValidatedCandidate,
    ValidationStatus,
)

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Stage 1 — substring grounding helpers
# ─────────────────────────────────────────────────────────────
def _verify_evidence_substring(evidence: Optional[str], raw_lower: str) -> bool:
    """True iff `evidence` appears as a substring of `raw_lower`
    (case-insensitive, whitespace-collapsed). Empty/None → False."""
    if not evidence or not str(evidence).strip():
        return False
    if not raw_lower:
        return False
    ev = " ".join(str(evidence).lower().split())
    raw = " ".join(raw_lower.split())
    return ev in raw


# ─────────────────────────────────────────────────────────────
# Stage 2 — catalog matching helpers
# ─────────────────────────────────────────────────────────────
def _exact_catalog_match(query: Optional[str], catalog: Set[str]) -> Optional[str]:
    """Lowercase exact / containment match against the full catalog.

    Returns the canonical catalog string when a match is found; None
    otherwise. Containment hits require the shorter side to be at
    least 70% of the longer (or 6 chars) so noise tokens like "the"
    don't latch onto every entry.
    """
    if not query or not catalog:
        return None
    q = str(query).lower().strip()
    if not q:
        return None
    if q in catalog:
        return q
    for term in catalog:
        if q == term:
            return term
        if q in term and len(q) >= max(int(0.7 * len(term)), 6):
            return term
        if term in q and len(term) >= max(int(0.7 * len(q)), 6):
            return term
    return None


def _fuzzy_catalog_match(
    query: Optional[str], catalog: Set[str], *, threshold: float,
) -> Optional[str]:
    """SequenceMatcher fuzzy match across the full catalog. Strict
    threshold per field — see settings.INTAKE_FUZZY_THRESHOLD_*."""
    if not query or not catalog:
        return None
    q = str(query).lower().strip()
    if not q:
        return None
    best_term = None
    best_score = float(threshold)
    for term in catalog:
        score = SequenceMatcher(None, q, term).ratio()
        if score > best_score:
            best_score = score
            best_term = term
    return best_term


# ─────────────────────────────────────────────────────────────
# Sprint 9 fuzzy_match_* — kept as thin wrappers so existing tests
# that import them continue to pass. They delegate to the new
# _exact + _fuzzy stack with field-specific thresholds.
# ─────────────────────────────────────────────────────────────
def fuzzy_match_asset(
    query: Optional[str], catalogs: IntakeCatalogs,
    *, threshold: Optional[float] = None,
) -> Optional[str]:
    if not query:
        return None
    thr = float(
        threshold if threshold is not None
        else getattr(settings, "INTAKE_FUZZY_THRESHOLD_ASSET", 0.85)
    )
    family_query = derive_asset_family(query) or str(query).lower().strip()
    if not family_query:
        return None
    direct = catalogs.asset_family_index.get(family_query)
    if direct:
        return direct
    raw_alias = catalogs.asset_family_index.get(str(query).lower().strip())
    if raw_alias:
        return raw_alias
    exact = _exact_catalog_match(family_query, catalogs.asset_families)
    if exact:
        return exact
    return _fuzzy_catalog_match(
        family_query, catalogs.asset_families, threshold=thr,
    )


def fuzzy_match_alert_type(
    query: Optional[str], catalogs: IntakeCatalogs,
    *, threshold: Optional[float] = None,
) -> Optional[str]:
    if not query:
        return None
    thr = float(
        threshold if threshold is not None
        else getattr(settings, "INTAKE_FUZZY_THRESHOLD_ALERT", 0.80)
    )
    norm = normalise_alert_type(query)
    if not norm:
        return None
    direct = catalogs.alert_type_index.get(norm)
    if direct:
        return direct
    exact = _exact_catalog_match(norm, catalogs.alert_types)
    if exact:
        return exact
    return _fuzzy_catalog_match(norm, catalogs.alert_types, threshold=thr)


def fuzzy_match_customer(
    query: Optional[str], catalogs: IntakeCatalogs,
    *, threshold: Optional[float] = None,
) -> Optional[str]:
    if not query:
        return None
    thr = float(
        threshold if threshold is not None
        else getattr(settings, "INTAKE_FUZZY_THRESHOLD_CUSTOMER", 0.85)
    )
    norm = normalise_customer(query)
    if not norm:
        return None
    direct = catalogs.customer_index.get(norm)
    if direct:
        return direct
    exact = _exact_catalog_match(norm, catalogs.customers)
    if exact:
        return exact
    return _fuzzy_catalog_match(norm, catalogs.customers, threshold=thr)


# ─────────────────────────────────────────────────────────────
# Severity canonicalisation — enum check + regex fallback
# ─────────────────────────────────────────────────────────────
_SEV_RE = re.compile(r"\bP\s*[1-4]\b|\bSev(?:erity)?\s*[1-4]\b", re.IGNORECASE)


def _canonicalize_severity(
    severity: Optional[str], evidence: Optional[str],
) -> Optional[str]:
    """Return one of P1/P2/P3/P4 or None.

    Looks at the LLM's severity value first (enum check), then falls
    back to a regex over the severity evidence text (catches "this is
    a P2", "Severity 1", "Sev2" phrasings the LLM sometimes returns
    verbatim instead of the enum)."""
    if severity in {"P1", "P2", "P3", "P4"}:
        return severity
    for source in (severity, evidence):
        if not source:
            continue
        m = _SEV_RE.search(str(source))
        if m:
            num = re.search(r"[1-4]", m.group(0))
            if num:
                return f"P{num.group(0)}"
    return None


# ─────────────────────────────────────────────────────────────
# Confidence derivation (Sprint 9.2 — evidence grounding mandatory)
# ─────────────────────────────────────────────────────────────
def _derive_confidence(
    *,
    severity_status: str,
    asset_status: str,
    alert_status: str,
    customer_status: str,
    evidence_grounded: Dict[str, bool],
) -> str:
    mandatory_matched = (
        severity_status == "valid"
        and asset_status == "matched"
        and alert_status == "matched"
    )
    mandatory_grounded = (
        evidence_grounded.get("severity", False)
        and evidence_grounded.get("asset_name", False)
        and evidence_grounded.get("alert_type", False)
    )
    if not (mandatory_matched and mandatory_grounded):
        return "Low"
    if (
        customer_status == "matched"
        and evidence_grounded.get("customer", False)
    ):
        return "High"
    return "Medium"


# ─────────────────────────────────────────────────────────────
# Diversity signature — lowercase + whitespace-collapse the triple
# (mirrors diversifier._normalize_signature_part)
# ─────────────────────────────────────────────────────────────
def _normalize_signature_part(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(str(value).lower().split())


def _signature(
    severity: Optional[str],
    asset_name: Optional[str],
    alert_type: Optional[str],
) -> str:
    sev = _normalize_signature_part(severity) or "?"
    family = _normalize_signature_part(
        derive_asset_family(asset_name) if asset_name else None,
    ) or "?"
    alert = _normalize_signature_part(
        normalise_alert_type(alert_type) if alert_type else None,
    ) or "?"
    return f"{sev}|{family}|{alert}"


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def validate_candidate(
    cand: ValidatedCandidate,
    catalogs: IntakeCatalogs,
    raw_text: str = "",
) -> ValidatedCandidate:
    """Sprint 9.2 — three-stage validation.

    `raw_text` is the original engineer-pasted content. When supplied,
    each evidence string is verified to appear as a substring of
    raw_text before catalog matching runs; when omitted (empty string),
    grounding is permissive (treated as True for every field) so older
    callers keep working — but anti-hallucination protection is only
    active when raw_text is threaded through. The Sprint 9.2 routes.py
    call site passes it explicitly per spec §6.5.

    Mutates and returns `cand` for backward compatibility with Sprint 9
    callers that expect the same instance back.
    """
    # ── Stage 1 — substring grounding ─────────────────────────
    raw_lower = (raw_text or "").lower()
    grounding_active = bool(raw_lower.strip())
    ev = cand.evidence
    if grounding_active:
        sev_grounded = _verify_evidence_substring(ev.severity, raw_lower)
        asset_grounded = _verify_evidence_substring(ev.asset_name, raw_lower)
        alert_grounded = _verify_evidence_substring(ev.alert_type, raw_lower)
        cust_grounded = _verify_evidence_substring(ev.customer, raw_lower)
    else:
        # No raw_text → preserve Sprint 9 behaviour exactly. Older
        # callers (test fixtures, any pre-9.2 site that hasn't been
        # updated) populate ValidatedCandidate without evidence and
        # expect High/Medium confidence to remain reachable. We
        # treat every field as grounded so the confidence derivation
        # collapses to "catalog match alone" — the Sprint 9 contract.
        sev_grounded = True
        asset_grounded = True
        alert_grounded = True
        cust_grounded = True

    # If grounding is active and an evidence string isn't a substring of
    # raw_text, the field is hallucinated → null it before catalog
    # matching even runs. (When grounding is inactive the raw values
    # pass through unchanged.)
    severity_raw = cand.severity if (not grounding_active or sev_grounded) else None
    asset_raw = cand.asset_name if (not grounding_active or asset_grounded) else None
    alert_raw = cand.alert_type if (not grounding_active or alert_grounded) else None
    customer_raw = cand.customer if (not grounding_active or cust_grounded) else None

    # ── Stage 2 — catalog matching ────────────────────────────
    severity_canonical = _canonicalize_severity(severity_raw, ev.severity)
    severity_status = "valid" if severity_canonical else "invalid"

    asset_canonical = (
        _exact_catalog_match(asset_raw, catalogs.asset_families)
        or _fuzzy_catalog_match(
            asset_raw,
            catalogs.asset_families,
            threshold=getattr(settings, "INTAKE_FUZZY_THRESHOLD_ASSET", 0.85),
        )
    ) if asset_raw else None
    asset_status = (
        "matched" if asset_canonical
        else ("unknown" if asset_raw else "absent")
    )

    alert_canonical = (
        _exact_catalog_match(
            normalise_alert_type(alert_raw) if alert_raw else None,
            catalogs.alert_types,
        )
        or _fuzzy_catalog_match(
            normalise_alert_type(alert_raw) if alert_raw else None,
            catalogs.alert_types,
            threshold=getattr(settings, "INTAKE_FUZZY_THRESHOLD_ALERT", 0.80),
        )
    ) if alert_raw else None
    alert_status = (
        "matched" if alert_canonical
        else ("unknown" if alert_raw else "absent")
    )

    customer_canonical = (
        _exact_catalog_match(
            normalise_customer(customer_raw) if customer_raw else None,
            catalogs.customers,
        )
        or _fuzzy_catalog_match(
            normalise_customer(customer_raw) if customer_raw else None,
            catalogs.customers,
            threshold=getattr(settings, "INTAKE_FUZZY_THRESHOLD_CUSTOMER", 0.85),
        )
    ) if customer_raw else None
    customer_status = (
        "matched" if customer_canonical
        else ("unknown" if customer_raw else "absent")
    )

    # ── Stage 3 — confidence band ─────────────────────────────
    evidence_grounded = {
        "severity": sev_grounded,
        "asset_name": asset_grounded,
        "alert_type": alert_grounded,
        "customer": cust_grounded,
    }
    confidence = _derive_confidence(
        severity_status=severity_status,
        asset_status=asset_status,
        alert_status=alert_status,
        customer_status=customer_status,
        evidence_grounded=evidence_grounded,
    )

    # ── Mutate the candidate for the caller ───────────────────
    cand.severity = severity_canonical
    cand.asset_name = asset_canonical or asset_raw
    cand.alert_type = alert_canonical or alert_raw
    cand.customer = customer_canonical or customer_raw
    cand.validation = ValidationStatus(
        severity_status=severity_status,
        asset_status=asset_status,
        alert_type_status=alert_status,
        customer_status=customer_status,
    )
    cand.confidence = confidence
    cand.diversity_signature = _signature(
        severity_canonical, asset_canonical, alert_canonical,
    )
    cand.canonical_form = CanonicalForm(
        severity=severity_canonical,
        asset_name=cand.asset_name,
        alert_type=cand.alert_type,
        customer=cand.customer,
        location=cand.location,
    )
    return cand
