"""Sprint 13.15 — JSON-first scoped chat for "Discuss with Logic".

When a chat session is scoped to a specific Incident_Number (set via
``chat_sessions.scope_incident_id`` by the per-bullet
Discuss-with-Logic handoff), every ``/ask`` should answer from THAT
ticket's complete structured JSON — not via cosine-similarity search
over a subset of its chunks.

Why a separate module from `scoped_retrieval`?
-------------------------------------------------
``scoped_retrieval.retrieve_within_incident`` performs cosine-ranked
chunk retrieval inside one ticket. That works for free-form prose
questions where the answer lives narratively inside a chunk. It does
NOT work for STRUCTURED-FIELD questions like *"what are the affected
assets?"* — the answer lives in ``Metadata.Affected_Assets`` (a list
of strings), and a prose chunk may not surface those tokens with
high cosine similarity to the user's plain-English question.

The fix: give the LLM the **complete ticket JSON** as input and let
it navigate the structure semantically. Engineers never type
``Affected_Assets`` with the underscore; they say *"affected
assets"*, *"what got hit"*, *"impacted devices"*. An LLM trained on
JSON can map those phrasings to the right field instantly. The
existing retrieval layer cannot.

Architecture decisions (Sprint 13.15)
-------------------------------------
* **Full JSON in prompt — no re-chunking, no re-embedding.** Average
  ticket is 5–10 KB / 1.5–3 K tokens; Haiku's 200 K window absorbs it
  with ~2 % utilization. Re-chunking would add a second cosine layer
  with the SAME failure mode we're trying to escape, plus new
  ingestion infra for zero measurable benefit at this corpus size.
* **Source = document file name.** Sprint 11's full-fidelity ingest
  stamps the entire source ticket on every chunk's ``metadata_json``
  via ``.update(ticket)``, AND each chunk joins to ``documents`` for
  the original file name. One SQL round-trip retrieves both — no
  new tables, no new index.
* **Failure-open.** If the JSON fetch fails or the LLM call errors,
  the caller falls through to today's chunk-based scoped retrieval
  path (``scoped_retrieval.retrieve_within_incident``). Zero
  regression risk to existing behaviour.
* **Search-KB / hybrid retrieval untouched.** This module is invoked
  ONLY when ``chat_sessions.scope_incident_id`` is set; the global
  ``/ask`` path is byte-identical.

Public API
----------
``fetch_full_ticket_json(scope_incident_id) ->
    Optional[FullTicketRecord]``

``synthesize_scoped_answer(question, record, *, generate_fn=None) ->
    Optional[str]``  — returns the LLM's natural-language answer or
None on any error.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import text as _sql_text

from backend.db.connection import engine


logger = logging.getLogger("acadia-log-iq")


@dataclass
class FullTicketRecord:
    """One ticket's complete JSON + the source-document attribution
    used to populate the response's ``sources[]`` array."""
    incident_number: str
    ticket_json: Dict[str, Any]
    document_name: str       # documents.name — the ingest file name
    document_id: str         # documents.id::text
    owner_id: Optional[str]  # documents.owner_id


# ─────────────────────────────────────────────────────────────
# SQL — single round-trip
# ─────────────────────────────────────────────────────────────
# We don't care which chunk we read the metadata_json from; Sprint 11's
# full-fidelity ingest copies the entire source ticket onto every
# chunk's metadata_json via .update(ticket), so any chunk for the
# scoped Incident_Number contains the complete JSON. LIMIT 1 keeps
# the query trivially cheap.
#
# `metadata_json->>'primary_id'` is the universal identifier (Sprint 4
# / Sprint 11); ``metadata_json->>'incident_number'`` is the legacy
# alias kept for back-compat. Mirrors `scoped_retrieval`'s WHERE shape
# so both paths see the same universe.
_SQL = """
SELECT
    c.metadata_json                         AS metadata_json,
    d.id::text                              AS file_id,
    d.name                                  AS source,
    d.owner_id                              AS owner_id
FROM chunks c
JOIN documents d
  ON d.id = c.document_id
JOIN document_versions dv
  ON dv.id = c.document_version_id
WHERE
    UPPER(COALESCE(c.metadata_json->>'primary_id',
                   c.metadata_json->>'incident_number'))
        = :scope_id_upper
    AND d.status = 'active'
    AND dv.is_active = TRUE
    AND d.current_version_id = dv.id
LIMIT 1
"""


def fetch_full_ticket_json(
    scope_incident_id: str,
) -> Optional[FullTicketRecord]:
    """One SQL round-trip → full ticket JSON + source attribution.
    Returns None when nothing matches OR on any DB error. Never
    raises."""
    sid = (scope_incident_id or "").strip()
    if not sid:
        return None
    try:
        with engine.connect() as conn:
            row = conn.execute(
                _sql_text(_SQL), {"scope_id_upper": sid.upper()},
            ).mappings().first()
    except Exception as exc:
        logger.warning(
            "[scoped_full_ticket] DB fetch failed scope=%s err=%s",
            sid, exc,
        )
        return None

    if not row:
        return None

    metadata = row.get("metadata_json")
    if not isinstance(metadata, dict):
        return None
    return FullTicketRecord(
        incident_number=sid,
        ticket_json=metadata,
        document_name=str(row.get("source") or "unknown_source"),
        document_id=str(row.get("file_id") or ""),
        owner_id=row.get("owner_id"),
    )


# ─────────────────────────────────────────────────────────────
# LLM synthesis — JSON-aware natural-language answer
# ─────────────────────────────────────────────────────────────
# Prompt design notes:
#   * Field-mapping rules are EXPLICIT — engineers don't speak in
#     underscored field names. We give the LLM a handful of
#     example mappings so it generalises confidently.
#   * "Do NOT invent values" is repeated and bolded because Haiku
#     is occasionally creative when the JSON omits something. The
#     polite "not found" fallback below catches the case.
#   * Output is plain prose — single answer, no structure. This is
#     what /ask's AnswerResponse.answer expects.
_SYSTEM_PROMPT = """You are a Tier-1 IT Service Management assistant. The user is asking about ONE specific historical incident ticket. The complete structured ticket data is provided below as JSON.

YOUR JOB: Answer the user's question naturally and concisely, mapping their natural-language phrasing to the JSON's structured fields semantically.

CRITICAL RULES:
1. Map natural language to JSON fields semantically. Engineers do NOT type underscored field names. Examples:
   - "affected assets" / "what got hit" / "impacted devices" → Metadata.Affected_Assets
   - "root cause" / "why did this happen" → Executive_Sharable_RCA.Root_Cause_Technical_High_Level
   - "fix" / "resolution" / "what they did" → Executive_Sharable_RCA.Resolution_Steps
   - "who fixed it" / "owner" → Key_Contributors[].Contributor or Forensic_Performance_Audit[].Contributor_Name
   - "how long" / "duration" / "TTR" → Metadata.TTL or Incident_Efficiency_Metrics.Total_Resolution_Time_Minutes
   - "fingerprints" / "log signatures" / "errors" → Metadata.Fingerprints
   - "timeline" / "what happened when" → Executive_Sharable_RCA.High_Level_Timeline or Forensic_Performance_Audit[].Key_Movements_Timeline
   - "5 whys" / "ITIL analysis" → ITIL_5_Why
   - "customer" → Metadata.customer_name
   - "priority" / "severity" → Metadata.priority
   - "team" / "group" / "who handled" → Metadata.Resolution_Groups or Engagement_Analysis.Team_Path
   - "products" / "vendor" → Engagement_Analysis.Products_Involved
   - "blast radius" / "impact" → Architecture_and_Blast_Radius
   - "preventative actions" / "follow-up" → Executive_Sharable_RCA.Corrective_Preventative_Actions

2. Form complete, natural sentences in a peer-engineer tone. Do NOT echo raw underscored field names. Do NOT dump JSON.

3. If the answer is a list, write natural prose: "The affected assets were NY4-CORE-RTR-01, BGP Neighbor 10.255.0.1, and BFD-Session-Gi0/0/1."

4. If the JSON does NOT contain the answer for the user's question, say so politely in ONE sentence and suggest they use Search KB for a wider lookup. Do NOT invent values to fill the gap.

5. NEVER fabricate IPs, hostnames, ASNs, ticket numbers, dates, or specific values not present in the JSON. This is a hard rule.

6. Reference the ticket id (Metadata.Incident_Number) inline when answering specifics, e.g., "In INC-PHOENIX-402, ..."

7. Keep responses concise: 1–3 sentences for simple lookups. Longer ONLY when the user asks for narrative content (RCA, timeline, 5-why)."""


def _render_payload(question: str, record: FullTicketRecord) -> str:
    """Compose the user-message payload (ticket JSON + question)."""
    try:
        json_dump = json.dumps(record.ticket_json, ensure_ascii=False, indent=2, default=str)
    except Exception:
        json_dump = str(record.ticket_json)
    return (
        f"TICKET ID: {record.incident_number}\n\n"
        f"TICKET DATA (JSON):\n{json_dump}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Your answer:"
    )


# Output budget — answer is plain prose. 1-3 sentence simple
# lookups need ~80 tokens; longer narrative answers (full RCA / 5-why)
# need ~400. Cap at 600 with safety headroom.
_MAX_OUTPUT_TOKENS = 600


def synthesize_scoped_answer(
    question: str,
    record: FullTicketRecord,
    *,
    generate_fn: Optional[Callable[[str, int], str]] = None,
) -> Optional[str]:
    """LLM call. Returns the natural-language answer string, or None
    on any failure. Failure-open by design — caller falls back to
    today's chunk-based scoped retrieval path."""
    q = (question or "").strip()
    if not q or not record:
        return None

    if generate_fn is None:
        try:
            from backend.api import safe_generate as generate_fn  # type: ignore
        except Exception as exc:
            logger.warning(
                "[scoped_full_ticket] safe_generate import failed (%s) — "
                "returning None so caller falls back to chunk path", exc,
            )
            return None

    prompt = f"{_SYSTEM_PROMPT}\n\n---\n\n{_render_payload(q, record)}"

    try:
        raw = generate_fn(prompt, _MAX_OUTPUT_TOKENS)
        out = (raw or "").strip()
    except Exception as exc:
        logger.warning(
            "[scoped_full_ticket] LLM call failed scope=%s err=%s",
            record.incident_number, exc,
        )
        return None

    # Strip an accidental code-fence the model occasionally adds.
    if out.startswith("```"):
        out = out.strip("`").lstrip("\n").lstrip()

    if not out:
        return None

    logger.info(
        "[scoped_full_ticket] scope=%s answered chars=%d "
        "(question=%r)",
        record.incident_number, len(out), q[:60],
    )
    return out


# ─────────────────────────────────────────────────────────────
# Source helper — frontend renders ``sources[]`` so we wire up one
# entry pointing at the originating ingest file. Field shape mirrors
# what `pgvector_search` and `retrieve_within_incident` already emit
# downstream so the chat message renderer needs zero changes.
# ─────────────────────────────────────────────────────────────
def build_source_for_full_ticket(
    record: FullTicketRecord,
) -> List[Dict[str, Any]]:
    """One source entry naming the ingest file + the incident."""
    return [{
        "file_id": record.document_id,
        "owner_id": record.owner_id,
        "source": record.document_name,
        "file_type": "ticket",
        "section_heading": f"Ticket {record.incident_number}",
        "chunk_type": "scoped_full_ticket",
        "summary": None,
        "labels_json": {},
        "metadata_json": record.ticket_json,
    }]
