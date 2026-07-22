"""US Pharma org profile.

Phase 1: inherits ALL shared/Acadia behavior — only the public identity
(display name + accent) differs, which is the visible proof that the org
switch works end-to-end. Phase 2 adds the real overrides here: the Store ID +
symptom intake, the Historic → KB → Escalate journey variant, and any
per-org prompt/matcher hooks. Each override is added to this one file; Acadia
stays untouched.
"""
from __future__ import annotations

from backend.orgs.base import OrgProfile


class USPharmaProfile(OrgProfile):
    slug = "uspharma"
    display_name = "US Pharma"
    # Distinct accent so the frontend can visibly confirm the module switched.
    theme = {"accent": "#0b7285"}

    # US Pharma's KB is a "search-everything" surface: store lookups live in
    # Excel/PDF/JSON, so on an identifier miss we always run the full hybrid
    # search rather than returning "not found". (Acadia keeps the default
    # natural-language-gated behavior.)
    kb_search_all_on_identifier_miss = True

    # US Pharma KB search ALSO includes JSON ticket documents. Stage 4 scopes
    # KB search to ("sop", "kb"); JSON uploads auto-classify as
    # doc_kind="ticket", so we fold "ticket" into the KB search scope to make
    # that JSON corpus searchable from the KB step. (Acadia keeps SOP/KB-only.)
    kb_search_extra_doc_kinds = ("ticket",)

    # Run each hybrid retrieval channel in its OWN request-context copy so
    # BM25 + keyword stop crashing with "cannot enter context: ... is already
    # entered" (they share one Context on the default path). Restores true
    # hybrid KB search for US Pharma. (Acadia keeps the legacy shared path.)
    retrieval_per_channel_context = True

    # US Pharma tier-1 intake is Store ID + symptom; historic matches are
    # hard-scoped to that store. Store ID is mandatory.
    tier1_requires_store_id = True

    # US Pharma KB answers are structured data pulls (full escalation matrices,
    # vendor/contact tables, per-store config) where completeness matters more
    # than brevity — the shared response-class caps (600/1200 tokens) truncate
    # them mid-record. Lift the answer output cap so these complete. Still
    # bounded by the agent token budget + the model's output limit, so it can't
    # over-spend. (Acadia keeps the shared response-class caps.)
    answer_max_output_tokens = 8000

    # Ingest ANY uploaded JSON (any structure) through the recursive, lossless
    # chunker so raw escalation / config JSON is fully searchable in chat.
    # (Acadia keeps the generic one-chunk-per-record / text path.)
    json_recursive_chunking = True

    # Escalation Procedure = upload-then-chat (JSON + PDF), not the fixed-PDF
    # vendor-section template. Uploading drops the engineer into the chatbot.
    escalation_mode = "upload_chat"

    # US Pharma answers are vendor / escalation contact directories — the email
    # addresses ARE the requested content, not PII to hide. Keep them visible
    # (all other PII types stay scrubbed). Set back to True to restore email
    # redaction. (Acadia keeps the shared email scrubbing.)
    redact_contact_emails = False
