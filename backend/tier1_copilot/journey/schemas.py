"""Sprint 10 — Pydantic schemas for the Resolution Journey response objects.

Mirrors spec §3 + §7 verbatim. `JourneyInitial` is what /initial returns
(Stage 0 + 1A + 1B, all eager). Per-stage GETs return their specific
model. Telemetry POST takes `JourneyEventRequest` and returns `{ok}`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Stage 0 — Best-Ticket Distillation (Sprint 10.2 redesign)
#
# Replaces Stage0ConfidenceLead. Instead of generic corpus stats,
# Stage 0 now distils the highest-quality cohort ticket's actual
# resolution: Primary_Fix as headline + Resolution_Steps as the
# step list. Corpus stats (cohort_size, clean_resolution_percent,
# platform_median_minutes) are collapsed-by-default frontend tail.
# ─────────────────────────────────────────────────────────────
class Stage0BestTicketDistillation(BaseModel):
    """Sprint 10.2 — replaces ConfidenceLead with best-ticket framing."""
    cohort_size: int = 0
    best_incident: Optional[str] = None
    best_quality_score: Optional[int] = None
    best_time_minutes: Optional[int] = None
    what_worked: Optional[str] = None
    how_they_did_it: List[str] = Field(default_factory=list)
    # Sprint 13.7 — top-5 cohort tickets' Incident_Summary.INCIDENT
    # lines, suffixed with " - <Incident_Number>". Drives the Stage 0
    # "Possible details are" bullet list. Same retrieval-rank order
    # and same trailing-id contract as `how_they_did_it` so the
    # per-bullet Discuss-with-Logic scoping regex still works.
    top5_incident_summaries: List[str] = Field(default_factory=list)
    critical_intervention: Optional[str] = None
    # Collapsed stats tail (frontend renders behind a <Collapse>)
    clean_resolution_count: int = 0
    clean_resolution_percent: int = 0
    avg_minutes_to_resolve_cohort: Optional[int] = None
    corpus_size: int = 0
    platform_median_minutes: Optional[int] = None
    # Profile signature kept for backward-compat tooling
    profile_match: Optional[str] = None
    sparse: bool = False
    # Sprint 10.3 — honest labelling per spec §4.3 so a score-3 ticket
    # isn't billed as "best-rated past resolution" (which implies high
    # confidence). Frontend headline copy varies by this field.
    #   strong   — best_quality_score >= 4
    #   adequate — best_quality_score == 3
    #   weak     — best_quality_score in (1, 2)
    #   none     — empty cohort or no quality scores at all
    evidence_strength: Literal["strong", "adequate", "weak", "none"] = "none"


# ─────────────────────────────────────────────────────────────
# Stage 1A — Smoking Gun (40% threshold across cohort)
# ─────────────────────────────────────────────────────────────
class Stage1aSmokingGun(BaseModel):
    pivot_signal: Optional[str] = None
    bypass_instruction: Optional[str] = None
    recommended_action: Optional[str] = None
    frequency_in_cohort_percent: int = 0
    seen_in_incidents: List[str] = Field(default_factory=list)
    empty: bool = True
    # Sprint 10.1 — tells the frontend how to label the panel:
    #   "mental_pivot_aggregate" → multiple cohort tickets share the
    #     same pivot_data_point (count >= ceil(N * 0.4)); the standard
    #     "Smoking Gun" panel renders with the cohort-frequency badge.
    #   "mental_pivot_single"    → at least one ticket has documented
    #     pivot data, but the cohort threshold isn't met (sparse data,
    #     not enough tickets carry it). Surface what we have rather
    #     than dropping it. Frontend renders the same panel but with
    #     a caption that contextualises the lone observation.
    #   "primary_fix_fallback"  → no cohort ticket has populated
    #     Knowledge_Base.the_mental_pivot at all; we distilled from
    #     the highest-quality ticket's Primary_Fix +
    #     Root_Cause_Technical_High_Level. Frontend re-titles to
    #     "Best Historical Fix" with a small italic caption.
    #   "empty"                 → no usable data; render existing
    #     empty-state copy.
    derived_from: Literal[
        "mental_pivot_aggregate",
        "mental_pivot_single",
        "primary_fix_fallback",
        "empty",
    ] = "empty"


# ─────────────────────────────────────────────────────────────
# Stage 1B — Do Not Chase (anti-waste checklist, count >= 2, cap 8)
# ─────────────────────────────────────────────────────────────
class DoNotChaseEntry(BaseModel):
    misleading_signal: str
    rule_out_logic: str
    occurrence_count: int
    seen_in_incidents: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Sprint 12.7 — Escalation Routing & Vendor/OEM Engagement.
# Lives next to (not inside) the Sprint 7 Tier1EscalationPackage so
# the existing chat-Escalate flow is byte-identical. Exposed via a
# dedicated GET endpoint that the journey frontend fetches alongside
# /stage-5 when the engineer reveals Operational Handoff.
# ─────────────────────────────────────────────────────────────
class Tier2EntryCandidate(BaseModel):
    team: str
    occurrence_count: int = 1
    example_path: Optional[str] = None


class EscalationRouting(BaseModel):
    resolution_groups: List[str] = Field(default_factory=list)
    team_paths: List[str] = Field(default_factory=list)
    recommended_tier2_teams: List[Tier2EntryCandidate] = Field(default_factory=list)
    # Vendor_OEM_Engagement is 0/213 populated in the current corpus
    # — wired forward-compat. `vendor_records` carries the raw dicts
    # (rendered lazily by the frontend when present); the dedicated
    # `forensic_data_required` list is pulled from well-known
    # sub-keys (forensic_data_required / required_artifacts /
    # evidence_required / required_data) so the master forensic-data
    # list lights up automatically once ingestion populates them.
    vendor_records: List[Dict[str, Any]] = Field(default_factory=list)
    forensic_data_required: List[str] = Field(default_factory=list)
    cohort_size: int = 0
    tickets_with_data: int = 0
    empty: bool = True


# ─────────────────────────────────────────────────────────────
# Sprint 12.7 — Escalation Handoff Note (LLM-generated).
# POST /tier1/journey/{sid}/escalation-handoff-note returns this.
# ─────────────────────────────────────────────────────────────
# Sprint 13.19 — request body for the handoff-note POST. Carries the
# Tier-1 engineer's tick-box state (which Stage 3 consolidated steps
# they marked attempted). Optional / defaults to empty so older
# callers without the body still work — they'll see the "no steps
# attempted" branch of the deterministic note.
class EscalationHandoffNoteRequest(BaseModel):
    attempted_step_numbers: List[int] = Field(default_factory=list)
    # Sprint 13.24 PERF — when true, bypass the session-keyed
    # consolidated-ledger cache and force a fresh LLM call. Powers
    # the Regenerate button in the Stage 5 panel; default False so
    # auto-fetch on mount uses the cache.
    force: bool = False


class EscalationHandoffNoteResponse(BaseModel):
    note: str
    # Diagnostic flag — true when the deterministic template-fill
    # fallback fired (LLM call failed). Frontend can show a small
    # indicator so the engineer knows to expect a less polished
    # diagnostic-summary sentence and re-run if desired.
    used_fallback: bool = False


class Stage1bDoNotChase(BaseModel):
    entries: List[DoNotChaseEntry] = Field(default_factory=list)
    empty: bool = True
    # Sprint 10.4 §4.3 — reason taxonomy collapsed to three values.
    # "populated"    = entries returned.
    # "no_data"      = cohort has zero false_path_red_herrings /
    #                  elimination_checklist entries.
    # "no_recurring" = entries exist but every one is unique
    #                  (kept as a safety branch — should never fire
    #                  with min_count=1).
    # "below_threshold" was dropped in 10.4: structurally unreachable
    # once MIN_OCCURRENCE_COUNT lowered to 1.
    reason: Literal[
        "populated",
        "no_data",
        "no_recurring",
    ] = "no_data"
    # Sprint 13.3 — true when the LLM synthesis pass failed and the
    # frontend is rendering the verbatim source text (today's pre-13.3
    # behaviour). Lets the panel render a small "(verbatim source —
    # synthesis unavailable)" footnote so engineers know to expect
    # less polished prose.
    synthesis_skipped: bool = False


# ─────────────────────────────────────────────────────────────
# Stage 2 — Historical Matches (per-ticket cards, ordered by rank)
# ─────────────────────────────────────────────────────────────
class HistoricalMatchCard(BaseModel):
    rank: int
    incident_number: Optional[str] = None
    # Sprint 10.0 backward-compat fields. The post-10.8 frontend
    # prefers the merged surfaces below; existing internal tests still
    # read these.
    headline: Optional[str] = None
    summary: Optional[str] = None
    symptoms: Optional[str] = None
    root_cause: Optional[str] = None
    resolution: List[str] = Field(default_factory=list)
    customer: Optional[str] = None
    time_to_resolve_minutes: Optional[int] = None
    closed_without_recurrence: bool = False
    # Sprint 11 — `technical_snapshot` REINSTATED. Direct file inspection
    # of the four reachable source uploads (180 tickets total) shows
    # `Executive_Sharable_RCA.Technical_Snapshot` populated in 72 of 180
    # — file3 (48/48) + Hypothetical_goldschema (24/24). The 10.8.1 audit
    # only inspected file1's 27 tickets (which are 0/27); the broader
    # corpus has the field. The reader omits the row when missing, so
    # tickets without it are unaffected.
    technical_snapshot: Optional[str] = None
    # Sprint 10.8 §2.5 — em-dash-merged surfaces, paths corrected by
    # 10.8.1 §4 to traverse the real array shapes:
    #   incident_summary    = Incident_Summary.INCIDENT
    #                         + Executive_Sharable_RCA.Executive_Summary
    #   resolution_approach = Executive_Sharable_RCA.Resolution_Steps
    #                         + Forensic_Performance_Audit[0].Critical_Intervention
    #                         + Key_Contributors.Key_Impact_Players[0].Hero_Action
    incident_summary: Optional[str] = None
    resolution_approach: Optional[str] = None
    # Sprint 12.5 — Error-code fingerprints aggregated per ticket from:
    #   - Metadata.Fingerprints                       (List[str])
    #   - Operational_SOP.primary_error_fingerprint   (str)
    #   - semantic_faq_block[*].related_signals       (List[str])
    # Deduped case-insensitively, first-seen casing wins. Surfaces in
    # the historical-match card so the engineer can grep these tokens
    # against current-incident logs. Empty list when none populated.
    error_codes: List[str] = Field(default_factory=list)


class Stage2HistoricalMatches(BaseModel):
    matches: List[HistoricalMatchCard] = Field(default_factory=list)
    # Sprint 11 — total useful matches available BEFORE the
    # max_matches_shown cap. Equals len(matches) today (retrieval
    # still fetches max 5 candidates, so the cap rarely bites). Once
    # the retrieval layer expands beyond 5 candidates per cohort, the
    # frontend uses the `total_available_matches > max_matches_shown`
    # condition to render a "View more matches" reveal — no second
    # rank pass needed because the surplus is already filtered for
    # usefulness on the backend.
    total_available_matches: int = 0
    max_matches_shown: int = 5


# ─────────────────────────────────────────────────────────────
# Stage 3 — Troubleshooting Approach (sequenced ledger, cap 8)
# ─────────────────────────────────────────────────────────────
class TroubleshootingStep(BaseModel):
    step_number: int
    action: str
    intent: Optional[str] = None
    pivot: Optional[str] = None
    command: Optional[str] = None
    branch_label: Optional[str] = None    # "Primary" | "Alt A" | "Alt B" | None
    seen_in_incidents: List[str] = Field(default_factory=list)
    # Sprint 10.8 §3.6 — second distinct successful intervention is
    # marked is_fallback=True so the frontend prefixes the action with
    # "(Fallback)". Stays False on every other step.
    is_fallback: bool = False
    # Sprint 10.8 §3.5 — sequencing-priority bucket. "diagnostic" steps
    # render first (rule out before fixing), "timeline" second
    # (observed behaviour), "intervention" last (the actual fix).
    # "context" is harvested for the pre-cap raw count but does not
    # add to the user-facing ledger.
    category: Literal[
        "diagnostic", "timeline", "intervention", "context",
    ] = "diagnostic"


# ─────────────────────────────────────────────────────────────
# Sprint 11 — Per-ticket detail expansion (UX parity with Stage 2).
#
# The consolidated `steps` list above is the cross-cohort
# deduplicated playbook. The per-ticket detail list below is the
# raw breakdown — one entry per cohort ticket, every source rendered
# verbatim. Frontend renders both: consolidated playbook on top, then
# a per-ticket accordion (first card expanded, rest collapsed,
# Show more / Show less on long narrative fields like
# Technical_Snapshot).
#
# Design rules:
#   - Every field is Optional / has an empty default — sparse tickets
#     produce sparse details, not errors. Frontend omits missing rows.
#   - Order of `per_ticket_details` matches the cohort rank order
#     (rank 1 first), same as Stage 2 cards.
#   - `diagnostic_tests_executed` is the forward-compat field for
#     Troubleshooting_Ledger.Diagnostic_Tests_Executed. The parent
#     key is 0/180 populated in the current corpus; the field is
#     wired now so any future upload that carries it lights up
#     automatically with no code change.
# ─────────────────────────────────────────────────────────────
class DiagnosticLogicEntry(BaseModel):
    """One Operational_SOP.diagnostic_logic_chunks[] entry.

    The cascade in stage3_troubleshooting handles legacy fixture key
    names — `context` is the canonical Sprint 10.8.1 source for
    `intent`, with `rationale` / `intent` as fallbacks.
    """
    action: Optional[str] = None
    intent: Optional[str] = None
    pivot: Optional[str] = None
    command: Optional[str] = None


class TimelineEntry(BaseModel):
    """One Forensic_Performance_Audit[0].Key_Movements_Timeline[]
    entry. `time` is whatever string the source carries (HH:MM,
    relative offset, or ISO timestamp — preserved verbatim)."""
    time: Optional[str] = None
    action: Optional[str] = None


class TicketTroubleshootingDetail(BaseModel):
    """Per-ticket raw breakdown. Mirrors the source structure with
    just enough flattening to keep the frontend render straightforward
    (e.g., Key_Contributors.Key_Impact_Players[0].Hero_Action becomes
    a single `hero_action` string)."""
    rank: int
    incident_number: Optional[str] = None
    technical_snapshot: Optional[str] = None
    resolution_steps: List[str] = Field(default_factory=list)
    diagnostic_logic: List[DiagnosticLogicEntry] = Field(default_factory=list)
    timeline: List[TimelineEntry] = Field(default_factory=list)
    critical_intervention: Optional[str] = None
    hero_action: Optional[str] = None
    # Sprint 11 — forward-compat. The parent key Troubleshooting_Ledger
    # is 0/180 populated in the current corpus. Wired now so any future
    # upload auto-lights without a code change.
    diagnostic_tests_executed: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Sprint 13 — Per-ticket Guided Workflows (replaces the merged
# ledger surface). Each cohort ticket renders as one collapsible
# panel; numbering is local to the ticket; Intent + Pivot rewritten
# by an LLM synthesis pass to read as human decision text rather
# than verbatim source strings.
# ─────────────────────────────────────────────────────────────
class GuidedWorkflowStep(BaseModel):
    step_number: int            # 1..N within this workflow only
    action: str                  # verbatim from source — never LLM-rewritten
    intent: Optional[str] = None     # LLM-synthesized (or verbatim source if synthesis skipped)
    pivot: Optional[str] = None      # LLM-synthesized prose; never references step numbers
    command: Optional[str] = None    # verbatim from source when present
    source_field: str = ""           # diagnostic_logic_chunks | Resolution_Steps | etc.


class GuidedWorkflow(BaseModel):
    match_rank: int                   # 1, 2, 3, ... — re-numbered after sibling drops
    incident_number: str
    header: str = ""                 # 3-7 word topical title (LLM-synthesized; falls back to Incident_Summary.INCIDENT)
    technical_snapshot: Optional[str] = None
    expanded_by_default: bool = False
    steps: List[GuidedWorkflowStep] = Field(default_factory=list)
    synthesis_skipped: bool = False  # true when LLM call failed → action+intent+pivot are verbatim


# ─────────────────────────────────────────────────────────────
# Sprint 13.12 — single consolidated 5-step ledger. Replaces the
# per-ticket Guided Workflows surface as the primary Stage 3 view.
# A Tier-1 engineer with read-only access doesn't need 8 steps × 5
# tickets = 40 actions; they need the merged best-of-five sequence
# of safe diagnostic checks. LLM filters out config-changing /
# disruptive actions and synthesises Intent + Pivot for each step.
# ─────────────────────────────────────────────────────────────
class ConsolidatedStep(BaseModel):
    step_number: int            # 1..5 (hard cap)
    action: str                  # one safe, executable read-only action
    intent: str                  # one-sentence hypothesis being tested
    pivot: str                   # what the result means + what to do next
    command: Optional[str] = None    # show / read-only command when applicable


class Stage3TroubleshootingApproach(BaseModel):
    # Sprint 13.12 — primary content of the Stage 3 panel. One merged
    # ledger of up to 5 read-only-safe diagnostic steps for a Tier-1
    # engineer. LLM-synthesised; failure-open to empty list.
    consolidated_steps: List[ConsolidatedStep] = Field(default_factory=list)
    consolidated_synthesis_skipped: bool = False
    # Sprint 13 — per-ticket Guided Workflows. Kept on the schema for
    # back-compat / future toggle; the frontend no longer renders this
    # field as of Sprint 13.12. Backend still computes it so any
    # downstream consumer (analytics, escalation package) keeps working.
    guided_workflows: List[GuidedWorkflow] = Field(default_factory=list)
    cohort_size: int = 0
    # Sprint 11 — per-ticket raw breakdown (companion to `steps`).
    # See DiagnosticLogicEntry / TimelineEntry / TicketTroubleshootingDetail
    # docs above. Empty list on empty cohort; never None. Already
    # filtered for usefulness on the backend so empty / placeholder-
    # only details never reach the frontend.
    per_ticket_details: List[TicketTroubleshootingDetail] = Field(default_factory=list)
    # Sprint 11 — total useful per-ticket details available BEFORE the
    # max_details_shown cap. Equals len(per_ticket_details) today
    # (cohort still capped at 5). Once cohort retrieval expands, the
    # frontend uses `total_available_details > max_details_shown` to
    # render a "View more ticket details (N)" reveal.
    total_available_details: int = 0
    max_details_shown: int = 5


# ─────────────────────────────────────────────────────────────
# Stage 4 — Search KB / SOP handoff (no DB, no LLM)
# ─────────────────────────────────────────────────────────────
class Stage4SearchKB(BaseModel):
    prefilled_message: str
    allowed_doc_kinds: List[str] = Field(default_factory=lambda: ["sop", "kb"])


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — Pivot Insights (merged Smoking Gun + Do Not Chase)
# Frontend renders a SINGLE card with both sections, ONE Helpful
# button, ONE next-stage button. The two child schemas are unchanged
# — only the wire shape merges.
# ─────────────────────────────────────────────────────────────
class PivotInsights(BaseModel):
    smoking_gun: Stage1aSmokingGun
    do_not_chase: Stage1bDoNotChase


# ─────────────────────────────────────────────────────────────
# Sprint 12.4 — Environment Context & Tech Component Profile.
# Lead-in panel rendered ABOVE Stage 0. Aggregates the cohort's
# technology landscape (domains, components, clusters, products,
# technical entities) into a single deduplicated profile so the
# engineer sees the full blast radius before drilling into any
# single ticket. Source fields:
#   - Metadata.Dynamic_Domain_Payload.Domain_Type       → domain_types
#   - Metadata.component_category                       → component_categories
#   - RAG_Potency_Metadata.Synaptic_Cluster_ID          → synaptic_cluster_ids
#   - Engagement_Analysis.Products_Involved             → products_involved
#   - Metadata.technical_entities                       → technical_entities
# All five are deduped (case-insensitive) and emitted in first-seen
# order across the cohort (rank-1 ticket's values lead).
# ─────────────────────────────────────────────────────────────
class EnvironmentProfile(BaseModel):
    domain_types: List[str] = Field(default_factory=list)
    component_categories: List[str] = Field(default_factory=list)
    synaptic_cluster_ids: List[str] = Field(default_factory=list)
    products_involved: List[str] = Field(default_factory=list)
    technical_entities: List[str] = Field(default_factory=list)
    cohort_size: int = 0
    tickets_with_data: int = 0
    empty: bool = True


# ─────────────────────────────────────────────────────────────
# /initial bundle — what paints on first journey load
# Sprint 10.2 — stage_1a + stage_1b replaced with pivot_insights.
# Sprint 12.4 — environment_profile prepended.
# ─────────────────────────────────────────────────────────────
class JourneyInitial(BaseModel):
    session_id: str
    environment_profile: EnvironmentProfile
    stage_0: Stage0BestTicketDistillation
    pivot_insights: PivotInsights


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 — Stage 4 Search KB handoff response
# Sprint 10.6 §3.4 — `has_corpus` removed: Search KB now invokes
# /ask the same way regular chat does (no doc-kind gate, no filter),
# so the boolean was meaningless and the frontend toast it drove was
# misleading (regular /ask finds plain-text uploads with no filter).
# ─────────────────────────────────────────────────────────────
# Sprint 11 — optional request body. Sprint 10.2's contract had no
# body — the prefilled message was always derived from the journey's
# Stage 4 build. Sprint 11 adds an optional override so the new
# "Ask in chat" link beside individual Stage 0 / Stage 3 steps can
# carry an arbitrary step text into the chat session, while every
# existing caller (no body) keeps its prior behaviour byte-for-byte.
class SearchKBHandoffRequest(BaseModel):
    prefilled_message_override: Optional[str] = None
    # Sprint 12.1 — When the Stage 0 "Ask in Chat" button is clicked
    # on a per-bullet step, the frontend extracts the bullet's source
    # Incident_Number (parsed from the existing " - INC-XXX" suffix)
    # and sends it here. The handoff persists it on the new
    # chat_sessions row (scope_incident_id column, migration 042) so
    # every subsequent /ask call in that chat session is scoped to
    # this one ticket's chunks. NULL/omitted = original global
    # Search-in-KB behavior is preserved.
    scope_incident_id: Optional[str] = None


class SearchKBHandoffResponse(BaseModel):
    chat_session_id: str
    redirect_url: str


# ─────────────────────────────────────────────────────────────
# Telemetry — POST /tier1/journey/{session_id}/event
# ─────────────────────────────────────────────────────────────
class JourneyEventRequest(BaseModel):
    # Sprint 10.2 — added "pivot_insights" for the merged Stage 1A+1B
    # panel. Old "stage_1a" and "stage_1b" values stay so historical
    # event rows remain queryable; new events should use pivot_insights.
    stage: Literal[
        "environment_context",   # Sprint 12.4 — lead-in profile panel
        "stage_0",
        "stage_1a", "stage_1b",  # legacy, kept for historical rows
        "pivot_insights",        # Sprint 10.2 — merged 1A+1B
        "stage_2", "stage_3", "stage_4", "stage_5",
    ]
    event_type: Literal[
        "stage_rendered", "helpful_clicked",
        # Sprint 13.2 — engineer disagreement signal (Dislike button).
        "disliked_clicked",
        "next_stage_clicked", "abandoned",
        # Sprint 10.5 §3.2 — fired when an engineer clicks
        # "Escalate Ticket" inline on a chat message footer. Recorded
        # before the navigate to /tier1/journey/{id}?stage=5 so the
        # event survives the route unmount.
        "escalation_initiated_from_chat",
        # Sprint 10.7 §3 — fired when the engineer reaches a new
        # stage (next-stage advance, or chat-Escalate jumping to
        # stage_5). The /resume-state endpoint reads the most recent
        # row with this event_type to decide where the journey should
        # re-mount on a remount/revisit. Distinct from the FROM-stage
        # `next_stage_clicked` and the cohort-wide `stage_rendered`.
        "stage_advanced",
        # Sprint 11 — fired AFTER the first /ask round-trip in a
        # Stage 4 spawned chat session. Distinguishes "user opened
        # the KB chat and got an answer" from "user clicked Open
        # Chat and immediately escalated without engaging". Always
        # carries `stage="stage_4"`. The escalation traversal log
        # surfaces this as "opened KB chat (N exchanges)" so the
        # Tier-2 reader sees how engaged the Tier-1 was before the
        # handoff.
        "kb_chat_engaged",
    ]
    payload: Optional[dict] = None


class JourneyEventResponse(BaseModel):
    ok: bool


# ─────────────────────────────────────────────────────────────
# Sprint 10.7 §3.1 — resume-state response.
#
# Read-only derivation of the engineer's last-viewed stage from the
# tier1_journey_events table. Powers ResolutionJourney's resume-on-
# mount behaviour so chat-Escalate and chat-Return open the journey at
# the correct stage instead of always restarting at Stage 0.
# ─────────────────────────────────────────────────────────────
class ResumeStateResponse(BaseModel):
    session_id: str
    current_stage: Literal[
        "stage_0",
        "pivot_insights",
        "stage_2",
        "stage_3",
        "stage_4",
        "stage_5",
    ] = "stage_0"
    # ISO-8601 string when serialized; None for fresh sessions with
    # no advance events. Kept as Optional[str] (not datetime) so the
    # response shape is JSON-native and Pydantic doesn't need timezone
    # serialization config.
    last_event_at: Optional[str] = None
