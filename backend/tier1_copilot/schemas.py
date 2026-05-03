"""Pydantic request/response models for Tier-1 Copilot endpoints.

Single source of truth for the API contract. All model classes live
here so `routes.py`, `prompt_builder.py`, and `cache.py` can type
against a consistent shape without scattering field definitions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# /tier1/analyze
# ─────────────────────────────────────────────────────────────
class Tier1AnalyzeRequest(BaseModel):
    # Mandatory (3)
    severity: Literal["P1", "P2", "P3", "P4"]
    asset_name: str = Field(..., min_length=1, max_length=200)
    alert_type: str = Field(..., min_length=1, max_length=200)
    # Optional (6)
    customer: Optional[str] = Field(None, max_length=200)
    location: Optional[str] = Field(None, max_length=200)
    technology: Optional[str] = Field(None, max_length=100)
    ip_or_device_id: Optional[str] = Field(None, max_length=200)
    error_code: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)
    # Session (used for feedback correlation; not itself a filter)
    session_id: str = Field(..., min_length=1, max_length=64)


class Tier1AnswerSection(BaseModel):
    """The 8-section fixed-format answer body. Every field is a string
    except `recommended_first_checks` which is a list. Missing sections
    that the LLM could not populate are rendered as empty strings — the
    frontend decides whether to show "no evidence" copy."""
    issue_understanding: str = ""
    historical_match: str = ""
    most_likely_cause: str = ""
    recommended_first_checks: List[str] = Field(default_factory=list)
    most_likely_fix: str = ""
    validation: str = ""
    escalate_if: str = ""
    follow_up_question: str = ""


class Tier1AnalyzeResponse(BaseModel):
    matched_incident: Optional[str] = None
    confidence: Literal["High", "Medium", "Low", "None"]
    similar_count: int = 0
    answer: Tier1AnswerSection
    cache_hit: bool = False
    response_id: str
    # Sprint 7 additions — all optional so Sprint 6 clients still pass
    # validation when the progressive flag is off (fields default to
    # None / empty). Non-breaking change for existing callers.
    top_5_match_ids: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    started_at: Optional[str] = None
    # Sprint 8 additions — populated only by the rank-N match endpoint.
    # Sprint 6/7 consumers that receive this response from /tier1/analyze
    # will see None / 0; that is intentional. Leaving them on the
    # baseline schema keeps the rank-N endpoint's return shape aligned
    # with /tier1/analyze instead of diverging.
    match_index: Optional[int] = None
    total_matches: int = 0


# ─────────────────────────────────────────────────────────────
# /tier1/feedback
# ─────────────────────────────────────────────────────────────
class Tier1FeedbackRequest(BaseModel):
    response_id: str = Field(..., min_length=1, max_length=40)
    helpful: bool
    follow_up_action: Optional[Literal[
        "next_best_solution",
        "deeper_diagnostics",
        "escalation_note",
        "search_kb_sop",
        "explain_recommendation",
    ]] = None
    session_id: str = Field(..., min_length=1, max_length=64)


class Tier1FeedbackResponse(BaseModel):
    ok: bool
    response_id: str
    action_taken: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# /tier1/health
# ─────────────────────────────────────────────────────────────
class Tier1HealthResponse(BaseModel):
    ok: bool
    alias_term_count: int
    cache_size: int
    flag_on: bool


# ─────────────────────────────────────────────────────────────
# Sprint 7 — Progressive workflow models
# ─────────────────────────────────────────────────────────────
class Tier1SessionCreateRequest(BaseModel):
    alert_signature: str = Field(..., min_length=1, max_length=400)
    alert_payload: Dict[str, Any] = Field(default_factory=dict)
    top_5_match_ids: List[str] = Field(default_factory=list)


class Tier1SessionStatus(BaseModel):
    session_id: str
    elapsed_seconds: int
    stuck_nudge: bool
    thumbs_down_count: int = 0
    current_match_index: int = 0
    escalated: bool = False
    resolved: bool = False


class Tier1MatchIndexRequest(BaseModel):
    match_index: int = Field(..., ge=0, le=20)


class Tier1ActionLogRequest(BaseModel):
    step: str = Field(..., min_length=1, max_length=200)
    result: str = Field(..., min_length=1, max_length=40)
    # One of: "normal", "abnormal", "skipped", "tried", "escalated"
    note: Optional[str] = Field(None, max_length=500)


class Tier1ActionLogResponse(BaseModel):
    ok: bool
    what_tried: List[Dict[str, Any]] = Field(default_factory=list)


# ── Deeper Diagnostics ──────────────────────────────────────
class Tier1DeeperDiagnosticsRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    matched_incident: Optional[str] = None


class Tier1DiagnosticStep(BaseModel):
    step_number: int
    title: str = ""
    what_to_check: str = ""
    why: str = ""
    command: Optional[str] = None
    expected_result: str = ""
    next_action_if_abnormal: str = ""
    next_action_if_normal: str = ""


class Tier1NextQuestion(BaseModel):
    prompt: str = ""
    options: List[str] = Field(default_factory=list)


class Tier1DeeperDiagnosticsResponse(BaseModel):
    goal: str = ""
    severity: str = ""
    steps: List[Tier1DiagnosticStep] = Field(default_factory=list)
    validation: str = ""
    escalation_path: List[str] = Field(default_factory=list)
    next_question: Tier1NextQuestion = Field(default_factory=Tier1NextQuestion)
    llm_used: bool = False


# ── Escalation Package ──────────────────────────────────────
class Tier1EscalationPackageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    matched_incident: Optional[str] = None
    # Client can ALSO push what_tried from local state; we merge with the
    # server-side log before building the package.
    what_tried: List[Dict[str, Any]] = Field(default_factory=list)


class Tier1Contact(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    escalation_level: Optional[str] = None


class Tier1DirectoryContact(BaseModel):
    """One-line contact recommendation sourced from the Acadia Escalation
    Contact Directory. `label` is the bucket (Customer / Vendor / Internal /
    Directory), `name` is the contact or team, `detail` is the one-line
    summary (phone, email, portal, entitlement)."""
    label: str
    name: Optional[str] = None
    detail: Optional[str] = None
    source: str = "Acadia Escalation Contact Directory"


class Tier1EscalationPackage(BaseModel):
    summary: str = ""
    priority: str = ""
    affected_customer: Optional[str] = None
    affected_assets: List[str] = Field(default_factory=list)
    suggested_owner_team: Optional[str] = None
    escalation_path: List[str] = Field(default_factory=list)
    customer_contacts: List[Tier1Contact] = Field(default_factory=list)
    vendor_contacts: List[Tier1Contact] = Field(default_factory=list)
    directory_contacts: List[Tier1DirectoryContact] = Field(default_factory=list)
    what_was_tried: List[str] = Field(default_factory=list)
    recommended_next_action: Optional[str] = None
    relevant_tickets: List[str] = Field(default_factory=list)
    formatted_text: str = ""


class Tier1EscalationPackageResponse(BaseModel):
    package: Tier1EscalationPackage


# ── Explain Recommendation ──────────────────────────────────
class Tier1ExplainRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    matched_incident: Optional[str] = None


class Tier1ScoreBreakdown(BaseModel):
    alert_type_match: float = 0.0
    asset_match: float = 0.0
    fingerprint_match: float = 0.0
    technology_match: float = 0.0
    vector_similarity: float = 0.0
    resolution_quality: float = 0.0
    recency: float = 0.0
    success_frequency: float = 0.0
    same_customer_boost: float = 0.0
    same_asset_family_boost: float = 0.0
    final_score: float = 0.0


class Tier1FieldMatch(BaseModel):
    field: str
    your_value: Optional[str] = None
    ticket_value: Optional[str] = None
    # True / False / "partial"
    match: Any = False


class Tier1HistoricalSuccess(BaseModel):
    total_similar: int = 0
    primary_fix: Optional[str] = None
    succeeded_count: int = 0
    succeeded_in: List[str] = Field(default_factory=list)
    failed_count: int = 0
    success_rate_percent: int = 0


class Tier1ExplainResponse(BaseModel):
    matched_incident: Optional[str] = None
    score_breakdown: Tier1ScoreBreakdown
    matched_fields: List[Tier1FieldMatch] = Field(default_factory=list)
    historical_success: Tier1HistoricalSuccess = Field(
        default_factory=Tier1HistoricalSuccess
    )
