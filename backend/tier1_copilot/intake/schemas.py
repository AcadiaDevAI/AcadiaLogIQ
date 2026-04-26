"""Pydantic models for the Universal Intake module.

Single source of truth for the API contract. Used by the route
handler (`routes.py`), the extractor (`extractor.py`), and the audit
logger (`audit.py`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Source enum — the five "non-Alert" intake modes Sprint 9 adds.
# ─────────────────────────────────────────────────────────────
SourceType = Literal["email", "phone", "portal", "chat", "note"]
VALID_SOURCE_TYPES = ("email", "phone", "portal", "chat", "note")


# ─────────────────────────────────────────────────────────────
# Request / response shapes
# ─────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    source: SourceType
    raw_text: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = Field(None, max_length=64)
    context_hints: Optional[Dict[str, Any]] = None


class FieldEvidence(BaseModel):
    """Exact substrings copied from raw_text by the LLM. The validator
    does NOT reject candidates whose evidence isn't a literal substring;
    it just shows them as-is so the engineer can spot drift."""
    severity: Optional[str] = None
    asset_name: Optional[str] = None
    alert_type: Optional[str] = None
    customer: Optional[str] = None


class ValidationStatus(BaseModel):
    severity_status: Literal["valid", "invalid"] = "valid"
    asset_status: Literal["matched", "unknown", "absent"] = "absent"
    alert_type_status: Literal["matched", "unknown", "absent"] = "absent"
    customer_status: Literal["matched", "unknown", "absent"] = "absent"


class CanonicalForm(BaseModel):
    """Sprint 9.1 — catalog-mapped values ready to populate the Tier-1
    form when the engineer clicks "Use this interpretation".

    These are the post-fuzzy-match canonical strings (asset family,
    alert type, customer name) — the same byte form Alert mode types
    against. Populating the form with these is what restores
    BM25/exact-match contributions during retrieval and prevents the
    rank-3-instead-of-rank-1 drift seen with raw LLM extractions.

    For unknown values (validator marked "unknown"), these fields fall
    back to the raw LLM string so the engineer can still edit and
    submit; the "Unknown — verify" chip in the UI signals which fields
    are unverified."""
    severity: Optional[Literal["P1", "P2", "P3", "P4"]] = None
    asset_name: Optional[str] = None
    alert_type: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None


class ValidatedCandidate(BaseModel):
    severity: Optional[Literal["P1", "P2", "P3", "P4"]] = None
    asset_name: Optional[str] = None
    alert_type: Optional[str] = None
    customer: Optional[str] = None
    location: Optional[str] = None
    users_impacted_count: Optional[int] = None
    evidence: FieldEvidence = Field(default_factory=FieldEvidence)
    validation: ValidationStatus = Field(default_factory=ValidationStatus)
    confidence: Literal["High", "Medium", "Low"] = "Low"
    diversity_signature: str = ""
    # Sprint 9.1 — canonical mapping for downstream form prefill.
    canonical_form: CanonicalForm = Field(default_factory=CanonicalForm)


class ExtractResponse(BaseModel):
    extraction_id: Optional[str] = None
    candidates: List[ValidatedCandidate] = Field(default_factory=list)
    error: Optional[str] = None


class ExtractionFeedbackRequest(BaseModel):
    picked_index: Optional[int] = Field(None, ge=0, le=20)
    edits: Optional[Dict[str, Any]] = None
    was_rejected: bool = False


class ExtractionFeedbackResponse(BaseModel):
    ok: bool
    extraction_id: str


class IntakeHealthResponse(BaseModel):
    ok: bool
    flag_on: bool
    catalogs_built: bool
    severities: int
    asset_families: int
    alert_types: int
    customers: int
