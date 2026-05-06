"""
generate_architecture_docx.py

Generates ARCHITECTURE.docx — a polished, demo/KT/interview-grade Word
document describing the Acadia Log-IQ / Tier-1 Copilot end-to-end
architecture, data flows, and glossary.

Usage
-----
    pip install python-docx
    python generate_architecture_docx.py

Output
------
    ARCHITECTURE.docx (in the same folder as this script)

The output is a single self-contained .docx with:
  - Title page + version block
  - Auto-update-able Table of Contents (right-click -> Update Field in Word)
  - 14 sections covering executive summary, system architecture,
    Proactive/Reactive flows, Resolution Journey stages, chat round-trip,
    AWS services, database schema, authentication, sequence diagrams,
    demo talk track, interview Q&A, and a glossary.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, Inches, RGBColor

# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #

NAVY = RGBColor(0x0E, 0x2D, 0x4D)         # Acadia primary
NAVY_LIGHT = RGBColor(0x14, 0x3F, 0x69)
ACCENT = RGBColor(0x3D, 0xA2, 0xE0)        # Acadia secondary
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x6B, 0x72, 0x80)
CODE_BG = "F4F6F8"
TABLE_HEADER_BG = "0E2D4D"
TABLE_HEADER_INK = RGBColor(0xFF, 0xFF, 0xFF)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _set_cell_bg(cell, color_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tc_pr.append(shd)


def _set_cell_borders(cell, color_hex: str = "CCCCCC") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), color_hex)
        tcBorders.append(b)
    tc_pr.append(tcBorders)


def add_para(
    doc: Document,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    size: int = 11,
    color: RGBColor = INK,
    align=None,
    space_after: int = 6,
) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(space_after)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Calibri"
        run.font.color.rgb = NAVY if level <= 2 else NAVY_LIGHT
        if level == 1:
            run.font.size = Pt(22)
        elif level == 2:
            run.font.size = Pt(16)
        elif level == 3:
            run.font.size = Pt(13)
        else:
            run.font.size = Pt(12)
        run.bold = True


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)


def add_numbers(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(item, style="List Number")
        for run in p.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(11)


def add_code_block(doc: Document, code: str) -> None:
    """Render code as a single-cell table with light grey background."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = tbl.rows[0].cells[0]
    _set_cell_bg(cell, CODE_BG)
    _set_cell_borders(cell, "DDDDDD")
    cell.text = ""  # clear default paragraph
    p = cell.paragraphs[0]
    for line in code.split("\n"):
        run = p.add_run(line + "\n")
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        run.font.color.rgb = INK
    # add a trailing empty para for spacing
    doc.add_paragraph()


def add_table(
    doc: Document,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_widths_inches: list[float] | None = None,
) -> None:
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl.autofit = False

    # Header row
    for col_idx, header in enumerate(headers):
        cell = tbl.rows[0].cells[col_idx]
        _set_cell_bg(cell, TABLE_HEADER_BG)
        _set_cell_borders(cell, "0E2D4D")
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(header)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.bold = True
        run.font.color.rgb = TABLE_HEADER_INK
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Body rows
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, val in enumerate(row):
            cell = tbl.rows[row_idx].cells[col_idx]
            _set_cell_bg(cell, "FFFFFF" if row_idx % 2 == 1 else "F8FAFC")
            _set_cell_borders(cell, "DDDDDD")
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(10)
            run.font.color.rgb = INK
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Column widths
    if col_widths_inches:
        for row in tbl.rows:
            for cell, width_in in zip(row.cells, col_widths_inches):
                cell.width = Inches(width_in)
    doc.add_paragraph()


def add_toc_field(doc: Document) -> None:
    """Insert a Word TOC field. User must right-click -> Update Field after opening."""
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = r'TOC \o "1-3" \h \z \u'
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = (
        "Right-click here in Word and choose 'Update Field' to populate the table of contents."
    )
    fldChar3 = OxmlElement("w:fldChar")
    fldChar3.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(placeholder)
    run._r.append(fldChar3)


def add_callout(doc: Document, label: str, text: str, color_hex: str = "E8F1F8") -> None:
    """Sidebar-style callout box."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cell = tbl.rows[0].cells[0]
    _set_cell_bg(cell, color_hex)
    _set_cell_borders(cell, "B0C5D8")
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(f"{label}  ")
    run.font.name = "Calibri"
    run.font.size = Pt(10)
    run.font.color.rgb = NAVY
    run.bold = True
    run2 = p.add_run(text)
    run2.font.name = "Calibri"
    run2.font.size = Pt(10)
    run2.font.color.rgb = INK
    doc.add_paragraph()


# --------------------------------------------------------------------------- #
# Document content
# --------------------------------------------------------------------------- #

def build_document(out_path: Path) -> None:
    doc = Document()

    # Page setup — 1in margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # ---- Title page ----
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("\n\n\n")
    run = p.add_run("Acadia Log-IQ")
    run.font.name = "Calibri"
    run.font.size = Pt(36)
    run.font.color.rgb = NAVY
    run.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("End-to-End Architecture & Operating Guide")
    run.font.name = "Calibri"
    run.font.size = Pt(18)
    run.font.color.rgb = NAVY_LIGHT
    run.italic = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("\n\nAI Copilot for Tier-1 Network Operations Centres")
    run.font.size = Pt(13)
    run.font.color.rgb = MUTED

    today = _dt.date.today().isoformat()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"\n\n\n\nVersion 12 (post-Sprint-12)\nGenerated: {today}")
    run.font.size = Pt(11)
    run.font.color.rgb = MUTED

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "\n\nProprietary & Confidential — Internal Use Only"
    )
    run.font.size = Pt(10)
    run.italic = True
    run.font.color.rgb = MUTED

    doc.add_page_break()

    # ---- Table of contents ----
    add_heading(doc, "Table of Contents", level=1)
    add_toc_field(doc)
    doc.add_page_break()

    # ====================================================================== #
    # 1. Executive Summary
    # ====================================================================== #
    add_heading(doc, "1. Executive Summary", level=1)

    add_para(
        doc,
        "Acadia Log-IQ is an AI-powered copilot for Tier-1 Network Operations Centre engineers. "
        "When an alert fires (Proactive) or a customer reports an issue (Reactive), the engineer "
        "feeds the signal into the platform and Log-IQ returns the closest historical resolution, "
        "an 8-section structured troubleshooting playbook, and a five-stage guided journey that "
        "ends with an Operational Handoff package ready for Tier-2 escalation.",
    )
    add_para(
        doc,
        "The system grounds every answer in the customer's own ticket history, KBs, and SOPs — "
        "no off-corpus hallucination. It runs on AWS Bedrock for embeddings and LLM inference, "
        "PostgreSQL with pgvector for retrieval, and a React + FastAPI stack for the UX and APIs.",
    )

    add_heading(doc, "1.1 Why this matters", level=2)
    add_bullets(doc, [
        "Tier-1 engineers spend 60–80 % of their shift triaging repeat incidents that already "
        "have known resolutions in the corpus — Log-IQ surfaces those in under 3 seconds.",
        "Every interaction is captured as telemetry (stage_rendered, helpful_clicked, "
        "next_stage_clicked, kb_chat_engaged) so the organization gets reliable engagement and "
        "MTTR analytics out of the box.",
        "The Operational Handoff package replaces hand-typed escalation emails with a structured, "
        "evidence-backed payload — including per-stage time-spent — so Tier-2 starts from the "
        "Tier-1 worker's confirmed dead-ends, not zero.",
        "Sprint-layered feature flags let us roll any feature forward or back with a single env "
        "var change. No code rollback is required, ever.",
    ])

    add_heading(doc, "1.2 Key capabilities", level=2)
    add_table(
        doc,
        ["Capability", "What it does"],
        [
            ["Two intake surfaces", "Proactive structured form + Reactive paste-and-extract — both feed the same downstream retrieval"],
            ["Hybrid retrieval", "pgvector cosine + BM25 + Postgres FTS + metadata filters, fused via Reciprocal Rank Fusion"],
            ["Sprint-7 ranking boosts", "Recency, success_frequency, same_customer, same_asset_family"],
            ["Resolution Journey", "5 progressively-revealed Cards: Best Match → Related Incidents → Guided Workflow → KB Reference → Operational Handoff"],
            ["Chat ↔ Journey round-trip", "Stage 4 spins up a chat session pre-loaded with the alert; engineer can return to the journey without losing context"],
            ["Multi-agent troubleshooting", "Planner → Analyst → Composer pipeline (Sonnet + Haiku) for complex queries via /ask"],
            ["Confidence & grounding", "4-signal confidence score + fabrication detection + version-aware source flagging"],
            ["Telemetry-driven escalation", "Stage 5 escalation package includes the full traversal log with per-stage durations"],
        ],
        col_widths_inches=[2.0, 4.5],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 2. System architecture
    # ====================================================================== #
    add_heading(doc, "2. System Architecture", level=1)

    add_para(doc, "Three layers, two data stores, one streaming external service.")

    add_heading(doc, "2.1 Layered overview", level=2)
    add_code_block(doc, """\
+------------------------------------------------------------------+
|                      React 18 Frontend                           |
|             (Ant Design + Tailwind, CRA)                         |
|                                                                  |
|   LandingRouter   |   ChatArea     |   Tier1Workspace            |
|   - Proactive     |   ChatInput    |   ResolutionJourney         |
|   - Reactive      |   ChatMessage  |   Stage 0..5 cards          |
+------------------------------------------------------------------+
                            | REST/JSON + Clerk JWT
                            v
+------------------------------------------------------------------+
|                    FastAPI Backend                               |
|                                                                  |
|  /tier1/*  |  /tier1/journey/*  |  /intake/*  |  /ask, /upload   |
|                                                                  |
|  retrieval/orchestrator + tier1_copilot/retrieval                |
|  routing/model_router  +  agents/{planner,analyst,composer}      |
|  validation/{validator, confidence_scorer, grounding_checker}    |
+------------------------------------------------------------------+
        |                                          |
        v                                          v
+--------------------+               +----------------------------+
|  Postgres + pgv    |               |       AWS Bedrock          |
|  + chat tables     |               |  Titan / Mistral /         |
|  + tier1_journey_  |               |  Claude Haiku / Sonnet     |
|    events          |               +----------------------------+
+--------------------+
""")

    add_heading(doc, "2.2 Component responsibilities", level=2)
    add_table(
        doc,
        ["Component", "Responsibility"],
        [
            ["LandingRouter", "Default entry point. Mounts the Tier-1 intake form (post-Sprint-11). Listens for journey resume signal."],
            ["Tier1IntakeForm", "Hosts the Proactive | Reactive split layout (SourceAwareIntake)."],
            ["Tier1Workspace", "Wraps the Resolution Journey + answer card + chips + diagnostics overlays."],
            ["ResolutionJourney", "Orchestrates the 5-stage progressive disclosure with parallel /initial + /resume-state fetch."],
            ["FastAPI /tier1 router", "Sprint-6+ Copilot APIs: analyze, feedback, session lifecycle."],
            ["FastAPI /tier1/journey router", "Sprint-10+ Journey APIs: stage data, telemetry events, search-KB handoff, resume-state."],
            ["FastAPI /intake router", "Sprint-9+ Reactive intake: extract candidates, record card-pick feedback."],
            ["retrieval.orchestrator", "/ask hybrid retrieval (vector + BM25 + FTS + metadata)."],
            ["tier1_copilot.retrieval", "Tier-1-specific 2-stage retrieval (exact SQL → hybrid) + weighted rerank."],
            ["routing.model_router", "Picks Haiku / Sonnet based on complexity classifier."],
            ["agents.{planner,analyst,composer}", "Multi-agent troubleshooting pipeline for complex /ask queries."],
            ["validation.validator", "Phase-6 grounding + confidence + fabrication checks."],
            ["services.contextual_ingestion_service", "Document upload → adaptive parser → chunking → embedding → storage."],
            ["AWS Bedrock", "Hosts Titan embed, Mistral 7B, Claude Haiku 4.5, Claude Sonnet 4.6."],
            ["Postgres + pgvector", "Single source of truth: chunks, vectors, sessions, telemetry, caches."],
        ],
        col_widths_inches=[2.0, 4.5],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 3. The two intake flows
    # ====================================================================== #
    add_heading(doc, "3. The Two Intake Flows", level=1)
    add_para(
        doc,
        "Both Proactive and Reactive flows feed the same downstream POST /tier1/analyze endpoint "
        "and produce the same Resolution Journey. The split layout was introduced in Sprint 11 to "
        "let an engineer choose the entry that fits the trigger.",
    )

    add_heading(doc, "3.1 Proactive flow (10 steps)", level=2)
    add_numbers(doc, [
        "Engineer fills the Proactive form (severity, asset, alert type required; six optional fields).",
        "Click 'Analyze alert' → frontend calls analyzeAlert(payload) from tier1Api.js.",
        "POST /tier1/analyze → handled by backend/tier1_copilot/routes.py:analyze.",
        "normalize_alert(req, alias_dict) → produces signature_hash from canonical alert fields.",
        "tier1_answer_cache lookup keyed by signature_hash. On cache miss:",
        "retrieve_top_matches() runs Stage 1 (exact SQL ILIKE on alert_signature / fingerprints_text / component_category) — short-circuits if score ≥ 0.80.",
        "If Stage 1 weak: Stage 2 hybrid (pgvector + ts_vector), candidates fused; weighted rerank applied (Jaccard + vector_similarity + resolution_quality + Sprint-7 boosts).",
        "prompt_builder.build_prompt() → Bedrock Mistral. parse_answer() decodes into Tier1AnswerSection. Falls back to template_fallback() on parse failure.",
        "tier1_sessions row created with current_match_index, top_5_match_ids, started_at. Result cached.",
        "Frontend stores result; Tier1Workspace mounts ResolutionJourney with sessionId.",
    ])

    add_callout(
        doc,
        "Why the two-stage retrieval?",
        "Stage 1 catches the 70 % of alerts where an exact field match is the right hit (e.g. "
        "BGP-5-ADJCHANGE on edge-rtr-04). When Stage 1 is weak (< 0.80) we fall back to a hybrid "
        "semantic search that also scores success_frequency and recency, so the user always gets "
        "the highest-quality past resolution rather than the most recent one.",
    )

    add_heading(doc, "3.2 Reactive flow (8 steps)", level=2)
    add_numbers(doc, [
        "Engineer pastes raw text (email, phone notes, portal ticket) into the right column.",
        "Click 'Extract & Suggest' → POST /intake/extract.",
        "backend/tier1_copilot/intake/routes.py validates length, loads catalogs, invokes Bedrock LLM via extractor.py.",
        "Each candidate is run through validate_candidate() — substring grounding against raw_text rejects hallucinated values.",
        "diversifier.diversify() caps candidates at 4 distinct interpretations.",
        "audit.log_extraction() writes intake_extractions row; response includes extraction_id + candidates[].",
        "Frontend renders SuggestionCarousel; engineer picks a card → Proactive form populated via prefill prop.",
        "Engineer reviews and clicks 'Analyze alert' → flow merges with Proactive Step 3.",
    ])

    add_callout(
        doc,
        "Anti-hallucination",
        "validate_candidate() requires every extracted field value to appear as a substring in the "
        "pasted text. This deterministically blocks the LLM from inventing customer names, asset "
        "IDs, or fingerprints that weren't actually present.",
    )

    doc.add_page_break()

    # ====================================================================== #
    # 4. Resolution Journey deep-dive
    # ====================================================================== #
    add_heading(doc, "4. Resolution Journey — Stage by Stage", level=1)

    add_para(
        doc,
        "After /tier1/analyze returns a session_id, ResolutionJourney mounts and progressively "
        "reveals five Cards. Each Card has its own backend endpoint and is gated behind a "
        "'Reveal next stage' click — minimizing cognitive load and giving Tier-2 a clear traversal "
        "log if escalation happens.",
    )

    add_heading(doc, "4.1 Stage 0 — Best Historical Match & Recommended Resolution", level=2)
    add_para(doc, "Backend: backend/tier1_copilot/journey/stage0_confidence.py")
    add_bullets(doc, [
        "Loads the cohort (top-5 ranked tickets from the Tier-1 retrieval).",
        "Sorts by Resolution_Quality_Score DESC and surfaces the single best ticket's Primary_Fix and Resolution_Steps.",
        "Aggregates corpus_size and platform_median_minutes for that profile to set engineer expectations.",
        "Sparse-fallback: when the cohort is empty, renders a placeholder banner instructing the engineer to use the playbook below.",
        "No 'Reveal next stage' button — Stage 0 + Pivot Insights always render together.",
    ])

    add_heading(doc, "4.2 Pivot Insights (Stage 1A + Stage 1B merged)", level=2)
    add_para(doc, "Backend: stage1_smoking_gun.py + stage1_do_not_chase.py — rendered by PivotInsightsPanel.js on the frontend")
    add_bullets(doc, [
        "Smoking Gun — the strongest single pivot signal across the cohort (e.g. 'BFD timer mismatch in 78 % of past matches').",
        "Do Not Chase — paths that historically dead-ended; surfaced inline so the engineer doesn't waste time on them.",
        "Sprint 10.2 merged 1A+1B into one Card to reduce vertical scroll.",
        "Engagement: helpful_clicked telemetry but no 'Reveal next stage' (always rendered).",
    ])

    add_heading(doc, "4.3 Stage 2 — Related Incidents & Probable Causes", level=2)
    add_para(doc, "Backend: stage2_historical.py — frontend: Stage2HistoricalMatches.js")
    add_bullets(doc, [
        "Renders up to 5 HistoricalMatchCards mapped from cohort tickets.",
        "Card fields drawn from Incident_Summary.INCIDENT, Executive_Sharable_RCA.*, Operational_SOP.signal_identification.human_symptom.",
        "First card expanded by default; rest collapsed.",
        "Sprint 11 — 'View more matches' reveal when backend returns more useful cards than the visible cap.",
        "Reveal click POSTs next_stage_clicked → /tier1/journey/{sid}/event.",
    ])

    add_heading(doc, "4.4 Stage 3 — Guided Troubleshooting Workflow", level=2)
    add_para(doc, "Backend: stage3_troubleshooting.py — frontend: Stage3TroubleshootingApproach.js")
    add_bullets(doc, [
        "Collects diagnostic steps from 6 source paths in the gold ticket schema.",
        "Normalises action text, deduplicates, and sequences by avg ordinal across the cohort.",
        "Caps at 8 consolidated steps; flags Alt A / Alt B / Primary branches.",
        "Each step links to 'Ask in chat' (uses useChatHandoff hook).",
        "Reveal click → /stage-4.",
    ])

    add_heading(doc, "4.5 Stage 4 — Knowledge Base & SOP Reference", level=2)
    add_para(doc, "Backend: stage4_kb_handoff.py + stage4_search_kb_handoff.py — frontend: Stage4SearchKBHandoff.js")
    add_bullets(doc, [
        "Builds a prefilled chat message that includes the alert + key cohort context.",
        "On 'Open chat' click — POST /tier1/journey/{sid}/search-kb-handoff:",
        "    • mints a chat_sessions row,",
        "    • inserts the user turn with metadata.journey_session_id = sid,",
        "    • returns chat_session_id.",
        "Frontend dispatches SET_MODE('troubleshooting'), getSession() → SET_SESSION, then askQuestion(text, chat_session_id) → /ask.",
        "kb_chat_engaged telemetry posted after the assistant turn returns (engagement signal for Stage 5).",
        "Sprint-10.6 deletion: empty-corpus guard and allowed_doc_kinds filter were removed because uploaded PDFs go through generic ingestion (no doc_kind stamp) and regular /ask retrieves them perfectly.",
    ])

    add_heading(doc, "4.6 Stage 5 — Operational Handoff", level=2)
    add_para(doc, "Backend: stage5_escalation.py — frontend: Stage5EscalationPackage.js")
    add_bullets(doc, [
        "Wraps Sprint-7's escalation_package.build_package and APPENDS the engineer's stage-traversal log.",
        "fetch_traversal_log() reads tier1_journey_events for the session and rolls up per-stage:",
        "    • viewed (any stage_rendered)",
        "    • marked Helpful (any helpful_clicked)",
        "    • opened KB chat (N exchanges) (Stage 4 only — kb_chat_engaged count)",
        "    • advanced at <UTC timestamp> (when next_stage_clicked fired)",
        "    • / <duration> — Sprint-12 per-stage time-spent (e.g. / 1m 3s)",
        "Output package fields: Summary, Priority, Affected assets, Suggested owner, Escalation path, What has been tried, Recommended next action, Relevant tickets, Recommended contacts.",
        "Recommended contacts come from escalation_directory.py (Vendor TAC, Internal Tier-2, etc.).",
    ])

    add_callout(
        doc,
        "Sprint 12 contribution",
        "The traversal log used to show only timestamps, which made post-mortem analysis painful "
        "(was the engineer stuck on Stage 3 for 30 seconds or 30 minutes?). The Sprint-12 update "
        "appends a / <duration> suffix to every traversal entry, computed as next_stage_clicked - "
        "first_seen, or last_event - first_seen for stages that weren't advanced past.",
    )

    doc.add_page_break()

    # ====================================================================== #
    # 5. Chat ↔ Journey round-trip
    # ====================================================================== #
    add_heading(doc, "5. Chat ↔ Journey Round-Trip", level=1)
    add_para(
        doc,
        "Stage 4 is the bridge between the structured journey and the free-form chat. The engineer "
        "can hop into chat for a one-off follow-up question and come back to the journey without "
        "losing context. Sprint-11 introduced a clean state machine for this round-trip; "
        "Sprint-12 hardened the New Chat button so a fresh empty chat opens correctly even after "
        "the LandingRouter default flipped to Tier-1.",
    )

    add_heading(doc, "5.1 Outbound (Journey → Chat)", level=2)
    add_numbers(doc, [
        "Stage 4 click → useChatHandoff(sid).askInChat(prefilledMessage).",
        "POST /tier1/journey/{sid}/search-kb-handoff → backend mints chat_sessions row, inserts user turn with journey_session_id metadata, returns chat_session_id.",
        "Frontend: SET_MODE('troubleshooting') → AppLayout swaps from LandingRouter to ChatArea.",
        "getSession(chat_session_id) → SET_SESSION dispatches with metadata.journey_session_id captured into state.sessionMetadata.",
        "askQuestion(text, chat_session_id) → POST /ask → ADD_ASSISTANT_MESSAGE.",
        "ChatArea renders a 'Back to Resolution Journey' banner because sessionMetadata.journey_session_id is present.",
    ])

    add_heading(doc, "5.2 Return (Chat → Journey)", level=2)
    add_numbers(doc, [
        "Engineer clicks 'Return to Stages' or 'Escalate to Tier 2' inside JourneyMessageActions.",
        "Pre-navigation telemetry: escalation_initiated_from_chat fired BEFORE dispatch (Sprint 10.5 §3.2 fix — events were getting lost on unmount).",
        "RESUME_JOURNEY({journeySessionId}) reducer:",
        "    • sets state.journeyResumeSessionId,",
        "    • clears state.selectedMode (so AppLayout falls back to LandingRouter).",
        "AppLayout re-renders LandingRouter; effect picks up journeyResumeSessionId, sets screen='tier1', stores synthesized result blob with the sid, dispatches CLEAR_JOURNEY_RESUME.",
        "Tier1Workspace mounts; ResolutionJourney calls /resume-state which reads stage_advanced events and restores the engineer's position.",
    ])

    add_heading(doc, "5.3 Re-entry from chat history", level=2)
    add_numbers(doc, [
        "User clicks a chat in the sidebar from anywhere (including Tier-1 Workspace).",
        "handleSelectSession(id) → getSession(id) → SET_SESSION.",
        "SET_SESSION reducer (Sprint-12 hardened): if backend returns selected_mode it wins; if missing AND messages are present, default to 'troubleshooting' so AppLayout routes to ChatArea instead of leaving the user on LandingRouter.",
        "If that chat was journey-originated, the 'Back to Resolution Journey' banner is restored from sessionMetadata.journey_session_id.",
    ])

    doc.add_page_break()

    # ====================================================================== #
    # 6. AWS / external services
    # ====================================================================== #
    add_heading(doc, "6. AWS Services & External Dependencies", level=1)

    add_table(
        doc,
        ["Service", "Used by", "Purpose"],
        [
            ["Bedrock — Titan Embed V2", "embedding_service.py", "1024-dim cosine embeddings for chunks and queries"],
            ["Bedrock — Mistral 7B", "Tier-1 prompt + reranker", "Tier-1 8-section answer + LLM reranking"],
            ["Bedrock — Claude Haiku 4.5", "ingestion + default /ask", "Cost-efficient default LLM and ingestion metadata extractor"],
            ["Bedrock — Claude Sonnet 4.6", "model_router escalated path", "Multi-step / complex reasoning, planner agent"],
            ["S3", "storage/s3_storage.py + bulk_ingest.py", "Optional file storage when STORAGE_TYPE=s3"],
            ["SES", "api.send_feedback_email_async", "Sends feedback notification emails"],
            ["Clerk", "clerk_auth.py + AuthGate.js", "JWT auth (RS256), optional"],
            ["EC2", "deployment", "Application hosting via docker compose"],
            ["RDS PostgreSQL 15", "all relational data", "Primary persistence layer"],
        ],
        col_widths_inches=[2.0, 1.8, 2.7],
    )

    add_heading(doc, "6.1 Bedrock model IDs in current use", level=2)
    add_code_block(doc, """\
BEDROCK_EMBED_MODEL  = amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL    = mistral.mistral-7b-instruct-v0:2
BEDROCK_HAIKU_MODEL  = us.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_SONNET_MODEL = us.anthropic.claude-sonnet-4-6
""")

    add_heading(doc, "6.2 Cost profile (1000 queries / day, indicative)", level=2)
    add_table(
        doc,
        ["Component", "Volume", "Model", "Monthly cost (USD)"],
        [
            ["Embeddings", "30 k", "Titan Embed V2", "~$3"],
            ["Default LLM (~85%)", "25.5 k", "Claude Haiku", "~$25"],
            ["Escalated LLM (~15%)", "4.5 k", "Claude Sonnet", "~$45"],
            ["Reranking", "30 k", "Mistral 7B", "~$3"],
            ["Total", "—", "—", "~$76"],
        ],
        col_widths_inches=[2.0, 1.2, 1.7, 1.5],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 7. Database schema
    # ====================================================================== #
    add_heading(doc, "7. Database Schema", level=1)
    add_para(
        doc,
        "PostgreSQL is the single source of truth. Migrations are SQL files in "
        "backend/db/migrations/, applied in numeric order by backend/db/migrate.py.",
    )

    add_heading(doc, "7.1 Core tables", level=2)
    add_table(
        doc,
        ["Table", "Purpose"],
        [
            ["documents", "Logical document record (name, owner, version family, ingestion_status, doc_kind)"],
            ["chunks", "Text chunks with rich metadata_json (Fingerprints, alert_signature, fingerprints_text, component_category, asset_family, cached_expert_answer)"],
            ["embeddings", "pgvector 1024-dim vectors keyed to chunk_id"],
            ["users", "Clerk-mapped user profiles (Phase 3)"],
            ["chat_sessions", "Chat history rows; carries selected_mode, entered_via, original_fingerprint, _session_metadata.journey_session_id"],
            ["chat_messages", "User/assistant turns + feedback, context_stats, semantic_cache_id"],
            ["learned_vocabulary", "Auto-learned identifier / enum tokens with canonical_form"],
            ["semantic_answer_cache", "Brief 5 cross-user answer cache (vector-keyed)"],
            ["pattern_analytics_cache", "Sprint 2 topic-keyed pattern stats"],
            ["organization_schemas", "Per-org column-name remappings"],
            ["logiq_sessions", "Generic session/user/org tracking"],
        ],
        col_widths_inches=[2.0, 4.5],
    )

    add_heading(doc, "7.2 Tier-1 specific tables", level=2)
    add_table(
        doc,
        ["Table", "Purpose"],
        [
            ["tier1_answer_cache", "Sprint 6 — signature-hash-keyed cached 8-section answers (Sprint 10.3 added top_5_match_ids TEXT[])"],
            ["tier1_sessions", "Sprint 7 — per-alert state (current_match_index, thumbs_down_count, escalated, what_tried, started_at)"],
            ["tier1_journey_events", "Sprint 10 — append-only telemetry (id BIGSERIAL, session_id, stage, event_type, payload_json JSONB, created_at)"],
            ["intake_extractions", "Sprint 9 — universal intake audit (raw_text hash, candidates, picked_index, was_rejected)"],
        ],
        col_widths_inches=[2.0, 4.5],
    )

    add_heading(doc, "7.3 Notable migrations", level=2)
    add_table(
        doc,
        ["File", "What it adds"],
        [
            ["001_phase1_foundation.sql", "vector / pgcrypto extensions, base schema"],
            ["002_phase2_contextual_ingestion.sql", "Versioning columns on documents"],
            ["003_phase3_multi_user.sql", "users table, owner-scoped indexes"],
            ["034_add_doc_kind.sql", "doc_kind classification (ticket/sop/kb/contact_*/vendor_case)"],
            ["035_fingerprint_gin_indexes.sql", "GIN index on chunks.metadata_json -> Metadata -> Fingerprints"],
            ["036_session_fingerprint_state.sql", "entered_via, original_fingerprint on chat_sessions"],
            ["038_tier1_copilot.sql", "Denormalized alert_signature, fingerprints_text, component_category + tier1_answer_cache"],
            ["039_tier1_progressive.sql", "chunks.asset_family + tier1_sessions"],
            ["040_tier1_journey_events.sql", "tier1_journey_events"],
            ["041_universal_intake.sql", "intake_extractions"],
            ["041_tier1_answer_cache_top5.sql", "Adds top_5_match_ids TEXT[] (note: shares 041_ prefix)"],
        ],
        col_widths_inches=[2.5, 4.0],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 8. Authentication
    # ====================================================================== #
    add_heading(doc, "8. Authentication (Clerk)", level=1)
    add_para(
        doc,
        "When CLERK_ENABLED=true, every authenticated route requires a Clerk-issued JWT. "
        "backend/clerk_auth.py uses PyJWKClient to fetch issuer public keys, validates RS256, exp, "
        "iss, and optionally azp. Frontend wraps the app in <ClerkProvider>; <AuthGate> blocks "
        "unauthenticated traffic; useAuthInterceptor injects the active token into every axios "
        "call via setTokenGetter. useAutoRegister POSTs /auth/register-or-login on first auth so "
        "the user gets a row in the users table.",
    )
    add_para(
        doc,
        "Setting CLERK_ENABLED=false falls back to optional API-key auth (API_KEY / UI_API_KEY env "
        "vars) for local dev or single-tenant deployments.",
    )

    doc.add_page_break()

    # ====================================================================== #
    # 9. Tech stack matrix
    # ====================================================================== #
    add_heading(doc, "9. Tech Stack Matrix", level=1)
    add_table(
        doc,
        ["Layer", "Technology", "Notes"],
        [
            ["Frontend framework", "React 18 (CRA)", "Functional components + hooks. State via useReducer + Context (ChatContext). No Redux."],
            ["UI kit", "Ant Design 5", "Forms, modals, layout"],
            ["Styling", "Tailwind + CSS variables", "Acadia theme tokens in theme/acadiaTheme.js"],
            ["HTTP", "axios", "JWT-aware via setTokenGetter"],
            ["Auth", "Clerk (RS256 JWT)", "Optional; falls back to API key when off"],
            ["Backend", "FastAPI (Python 3.11)", "Single app; routers mounted by feature flag"],
            ["DB", "PostgreSQL 15 + pgvector", "1024-dim cosine; rich metadata_json on chunks"],
            ["Embeddings", "Titan Embed V2 (Bedrock)", "amazon.titan-embed-text-v2:0"],
            ["Default LLM", "Claude Haiku 4.5 (Bedrock)", "Fast + cheap"],
            ["Complex LLM", "Claude Sonnet 4.6 (Bedrock)", "Routed via complexity classifier"],
            ["Tier-1 LLM", "Mistral 7B (Bedrock)", "8-section parser; deterministic fallback if parse fails"],
            ["Reranker", "Mistral 7B (Bedrock)", "Modular — pluggable cross-encoder"],
            ["Storage", "Local FS / S3", "STORAGE_TYPE switch"],
            ["Email", "AWS SES", "Feedback notifications only"],
            ["Containers", "Docker + Compose", "Two compose files: local + EC2"],
            ["CI", "GitHub Actions", "Note: docker-build.yml is stale (Streamlit-era)"],
        ],
        col_widths_inches=[1.5, 2.0, 3.0],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 10. Sequence diagrams
    # ====================================================================== #
    add_heading(doc, "10. Sequence Diagrams", level=1)

    add_heading(doc, "10.1 Proactive analyze", level=2)
    add_code_block(doc, """\
Engineer  Frontend            FastAPI                  Bedrock        Postgres
   |  fill form |                |                       |              |
   |---Analyze->|                |                       |              |
   |            |--POST /tier1/->|                       |              |
   |            |     analyze   |                       |              |
   |            |                |--SQL ILIKE ----------------------->|
   |            |                |<--exact matches -------------------|
   |            |                |                                    |
   |            |                |  (if weak: Stage 2)                |
   |            |                |--vector + ts_vector -------------->|
   |            |                |<--candidates --------------------- |
   |            |                |                                    |
   |            |                |--prompt + invoke --> Mistral       |
   |            |                |<--8-section JSON ---               |
   |            |                |                                    |
   |            |                |--INSERT tier1_session ------------>|
   |            |                |--cache result --------------------->|
   |            |<--response ----|                                    |
   |            |                                                     |
   |            |--render Tier1Workspace                              |
   |            |--GET /tier1/journey/{sid}/initial ----------------->|
   |            |<--Stage 0 + Pivot Insights -------------------------|
""")

    add_heading(doc, "10.2 Stage 4 → Chat handoff", level=2)
    add_code_block(doc, """\
Engineer Frontend          FastAPI                    Bedrock     Postgres
   |  click Open chat |        |                         |            |
   |------------------->|        |                         |            |
   |  useChatHandoff   |--POST /tier1/journey/{sid}/      |            |
   |                   |   search-kb-handoff             |            |
   |                   |--mint chat_sessions row -------------------->|
   |                   |--insert user turn  (journey_session_id) ---->|
   |                   |<--{chat_session_id} ----                    |
   |                   |--SET_MODE("troubleshooting")                |
   |                   |--getSession(chat_session_id) --------------->|
   |                   |<--SET_SESSION (sessionMetadata) ------------|
   |                   |--POST /ask  -->                             |
   |                   |       (retrieval + LLM + validation) -->Bedrock
   |                   |<--answer + sources ------                   |
   |                   |--ADD_ASSISTANT_MESSAGE                      |
   |                   |--POST /tier1/journey/{sid}/event            |
   |                   |   (event_type=kb_chat_engaged) ------------>|
""")

    doc.add_page_break()

    # ====================================================================== #
    # 11. Demo talk track
    # ====================================================================== #
    add_heading(doc, "11. Demo Talk Track", level=1)

    add_para(
        doc,
        "A 7-minute end-to-end demo for a stakeholder who has never seen Log-IQ. Each beat below "
        "is what to say + what to click, in order.",
        italic=True,
    )

    add_heading(doc, "Beat 1 — The problem (45s)", level=2)
    add_para(
        doc,
        "“Tier-1 NOC engineers spend 60–80 % of their shift triaging repeat incidents that "
        "already have known resolutions. Every shift turnover loses context. Every escalation "
        "starts from zero. Log-IQ is the AI copilot that closes that loop — same alert, same "
        "fix, ten times faster.”"
    )

    add_heading(doc, "Beat 2 — Landing page (30s)", level=2)
    add_para(
        doc,
        "[Click] open the app. [Show] the Proactive | Reactive split. [Say]: “Two ways in. "
        "Proactive is the form on the left — the alert just fired in your monitoring tool. "
        "Reactive is the paste box on the right — a customer emailed or called.”"
    )

    add_heading(doc, "Beat 3 — Reactive intake (60s)", level=2)
    add_para(
        doc,
        "[Paste] a sample customer email into the Reactive box. [Click] Extract & Suggest. "
        "[Show] the four candidate cards. [Say]: “LLM extracted four interpretations, validated "
        "every field as a substring of the source — no hallucinations. Engineer picks one.” "
        "[Click] a card. [Show] the Proactive form populated."
    )

    add_heading(doc, "Beat 4 — Analyze (30s)", level=2)
    add_para(
        doc,
        "[Click] Analyze alert. [While loading]: “Behind this button — pgvector semantic search, "
        "BM25 keyword, Postgres FTS, four-way fusion, weighted rerank including recency, "
        "success-rate, same-customer boost, and same-asset-family. Then a Mistral prompt that "
        "produces an 8-section playbook.”"
    )

    add_heading(doc, "Beat 5 — Resolution Journey (2 min)", level=2)
    add_para(
        doc,
        "[Show] Stage 0 — Best Historical Match. [Say]: “The single highest-quality past "
        "resolution. The Primary_Fix that worked. The Resolution_Steps the engineer last took.” "
        "[Show] Pivot Insights. [Say]: “Smoking gun + do-not-chase. Tells the engineer what to "
        "investigate AND what to ignore.” [Click] Reveal next stage. [Show] Stage 2 — five "
        "related incidents. [Click] Reveal. [Show] Stage 3 — guided steps. [Click] Reveal. "
        "[Show] Stage 4 — Knowledge Base. [Click] Open chat → walk through asking a "
        "follow-up question."
    )

    add_heading(doc, "Beat 6 — Operational Handoff (90s)", level=2)
    add_para(
        doc,
        "[Click] Return to Stages. [Say]: “Round-trip preserved — chat session linked back to "
        "the journey.” [Click] Reveal Stage 5. [Show] the escalation package. [Say]: “Summary, "
        "priority, owner, escalation path. The full traversal log including per-stage time spent. "
        "Tier-2 starts from confirmed dead-ends, not zero. Cisco TAC contact, internal Tier-2 "
        "channel — pulled from the escalation directory.”"
    )

    add_heading(doc, "Beat 7 — The numbers (45s)", level=2)
    add_para(
        doc,
        "“For 1000 queries a day we run on roughly 76 dollars a month of Bedrock spend, sub-second "
        "latency on cached signatures, three-second median on cold queries. Sprint-by-sprint "
        "feature flags mean every capability you just saw can be rolled forward or back without a "
        "code deploy. Questions?”"
    )

    doc.add_page_break()

    # ====================================================================== #
    # 12. Interview prep
    # ====================================================================== #
    add_heading(doc, "12. Interview Q&A — Architecture & Design", level=1)

    qa_items = [
        ("Why pgvector and not Pinecone / Chroma / Weaviate?",
         "Single source of truth. We already needed Postgres for sessions, telemetry, users, and "
         "ticket metadata. Adding a separate vector DB doubles the operational surface and forces "
         "us to keep two stores in sync (chunks vs. vectors). pgvector cosine on a 1024-dim "
         "Titan embedding gives us sub-100ms retrieval at our corpus size. Chroma was the "
         "original choice (CHROMA_PERSIST_DIR is still in env files) but most retrieval paths "
         "now query pgvector + ts_vector against chunks directly."),

        ("Why Bedrock and not OpenAI?",
         "Three reasons. (1) Data policy — AWS contractually does not use customer data for "
         "model training, which matters because the corpus contains internal incident records. "
         "(2) Multi-model in one bill — Titan, Mistral, Haiku, Sonnet are all Bedrock APIs with "
         "the same auth and IAM. (3) AWS-native deployment — IAM Roles on EC2 mean we never ship "
         "credentials to the runtime."),

        ("How do you prevent hallucination?",
         "Three layers. (a) Reactive intake validates every extracted field as a substring of "
         "the source paste — deterministic, not LLM-judged. (b) /ask uses a 4-signal confidence "
         "score (retrieval strength, query coverage, grounding overlap, consistency) and falls "
         "back to a safe 'insufficient evidence' answer when confidence < 0.35. (c) Tier-1 has a "
         "deterministic template_fallback() that produces a valid 8-section answer when the "
         "Mistral output fails to parse."),

        ("Why a separate Tier-1 retrieval pipeline if /ask already exists?",
         "Different optimization targets. /ask is a general document QA pipeline tuned for "
         "natural-language questions across heterogeneous corpora. Tier-1's retrieval is tuned "
         "for one specific shape — alert → ticket — and exploits denormalized columns "
         "(alert_signature, fingerprints_text, component_category) added in migration 038 for "
         "millisecond exact-match. It also applies success-rate / recency / same-customer boosts "
         "that don't make sense for general document search."),

        ("What if Bedrock is down?",
         "Tier-1's parse step has a deterministic template_fallback that produces a valid "
         "8-section answer from the top match's metadata alone — no LLM call required. "
         "/ask returns its safe 'insufficient evidence' fallback when the LLM fails. Both paths "
         "log the failure to the eval harness. The retrieval layer (pgvector + BM25 + ts_vector) "
         "is fully local and unaffected by Bedrock outages."),

        ("How do you know whether the engineer actually used the journey?",
         "Every interaction posts to tier1_journey_events: stage_rendered, helpful_clicked, "
         "next_stage_clicked, kb_chat_engaged, escalation_initiated_from_chat, abandoned. "
         "Sprint 5's escalation package surfaces the per-stage traversal log AND the per-stage "
         "duration, so a Tier-2 reader can see exactly where Tier-1 spent time and where they "
         "skipped."),

        ("Why feature flags everywhere?",
         "Operational risk. The product evolved across 12 sprints; some sprints rewrote the "
         "intake form, others added new endpoints, others changed the retrieval ranking. "
         "Flag-off must be byte-identical to the previous sprint. That invariant lets us ship "
         "every day to production without code rollbacks — if something regresses, flip the flag, "
         "ship in 10 seconds."),

        ("How is the chat ↔ journey round-trip implemented?",
         "Three pieces. (a) Stage 4 mints a chat_sessions row with journey_session_id stamped on "
         "_session_metadata. (b) When the engineer clicks 'Return to Stages' inside chat, "
         "JourneyMessageActions dispatches RESUME_JOURNEY which sets journeyResumeSessionId and "
         "clears selectedMode — AppLayout falls back to LandingRouter which picks up the resume "
         "signal and mounts Tier1Workspace. (c) Re-entry from sidebar history works because "
         "SET_SESSION captures metadata.journey_session_id back into state, so the 'Back to "
         "Resolution Journey' banner re-renders."),

        ("What's stored where?",
         "Postgres holds everything relational + the embeddings (pgvector). Files (PDFs, DOCXes "
         "uploaded by admins) live in S3 when STORAGE_TYPE=s3 or local FS otherwise. No data "
         "leaves the AWS perimeter except for Bedrock API calls — and Bedrock data does not "
         "train models per AWS policy."),

        ("Why does the journey have 6 'stages' if it's named the 5-stage journey?",
         "Stage 1A and Stage 1B were merged into Pivot Insights in Sprint 10.2 — they're a "
         "single Card visually. STAGE_ORDER in the frontend is "
         "['stage_0', 'pivot_insights', 'stage_2', 'stage_3', 'stage_4', 'stage_5'] — six entries "
         "but five Card surfaces."),
    ]
    for q, a in qa_items:
        add_heading(doc, f"Q. {q}", level=3)
        add_para(doc, a)

    doc.add_page_break()

    # ====================================================================== #
    # 13. KT checklist
    # ====================================================================== #
    add_heading(doc, "13. Knowledge Transfer Checklist", level=1)
    add_para(
        doc,
        "If you're handing this codebase to a new engineer, walk through this checklist with "
        "them. Each item points at the file or directory they should read first.",
        italic=True,
    )

    add_table(
        doc,
        ["Topic", "Read this first", "Why"],
        [
            ["Repo overview", "README.md", "High-level layout + flag list"],
            ["Backend entrypoint", "backend/api.py (lifespan + router mounts)", "Shows how feature flags gate routers"],
            ["Tier-1 retrieval", "backend/tier1_copilot/retrieval.py", "Two-stage retrieval + weighted rerank logic"],
            ["8-section answer", "backend/tier1_copilot/prompt_builder.py + schemas.py", "Prompt + parser + fallback"],
            ["Resolution Journey", "backend/tier1_copilot/journey/routes.py + stage*.py", "All 5 stages + telemetry"],
            ["Reactive intake", "backend/tier1_copilot/intake/{routes.py, extractor.py, validator.py}", "How paste-and-extract works"],
            ["Stage 4 chat handoff", "stage4_search_kb_handoff.py + frontend useChatHandoff.js", "Chat session minting + state flips"],
            ["Frontend routing", "frontend/src/components/LandingRouter.js + App.js", "Conditional landing vs chat"],
            ["State management", "frontend/src/hooks/ChatContext.js", "All reducers — NEW_CHAT, SET_SESSION, RESUME_JOURNEY"],
            ["Stage UI", "frontend/src/components/Tier1Copilot/journey/ResolutionJourney.js + Stage*.js", "Mount order + reveal logic"],
            ["Theme", "frontend/src/theme/acadiaTheme.js", "Colour tokens — navy palette"],
            ["Database", "backend/db/migrations/*.sql in numeric order", "Schema evolution by sprint"],
            ["Auth", "backend/clerk_auth.py + frontend/AuthGate.js", "How Clerk JWT flows through"],
            ["Deployment", "docker-compose.ec2.yml + backend/Dockerfile + frontend/Dockerfile", "Two-container production layout"],
            ["Tests", "backend/tier1_copilot/tests + backend/tests/tier1_copilot/journey", "Most coverage lives here"],
        ],
        col_widths_inches=[1.5, 2.5, 2.5],
    )

    doc.add_page_break()

    # ====================================================================== #
    # 14. Glossary
    # ====================================================================== #
    add_heading(doc, "14. Glossary", level=1)

    add_table(
        doc,
        ["Term", "Definition"],
        [
            ["Alert", "A single fault signal — either machine-generated (Proactive) or human-reported (Reactive)."],
            ["Alert signature / signature_hash", "A deterministic hash of the canonical alert fields used for cache lookup."],
            ["Asset family", "A grouping of similar assets (e.g. all 'core-rtr' devices) used for the same_asset_family ranking boost."],
            ["BFD", "Bidirectional Forwarding Detection — a network protocol layer often involved in BGP fault troubleshooting."],
            ["BGP", "Border Gateway Protocol. A common alert source in NOC contexts."],
            ["BM25", "A classic information retrieval ranking function. We use it as one of four candidate channels in /ask retrieval."],
            ["Cohort", "The top-N tickets returned by Tier-1 retrieval for a single alert. Drives Stages 0, 2, 3."],
            ["Confidence band", "High / Medium / Low / None classification based on the top match's score."],
            ["Component category", "A denormalized field on chunks (added in migration 038) for fast exact-match SQL."],
            ["Do Not Chase", "Patterns observed in the cohort that historically dead-ended; surfaced inline so the engineer skips them."],
            ["Fingerprint", "A specific code identifying an alert pattern (e.g. BGP-5-ADJCHANGE). Stored as JSONB on chunks.metadata_json."],
            ["FTS", "Full-Text Search. We use Postgres ts_vector + plainto_tsquery as one retrieval channel."],
            ["Gold ticket", "A historical ticket curated to the gold schema (Incident_Summary, Operational_SOP, Executive_Sharable_RCA, Resolution_Quality_Score, ...)."],
            ["Journey", "The 5-stage Resolution Journey rendered after /tier1/analyze."],
            ["KB / SOP", "Knowledge Base / Standard Operating Procedure. Both stored as documents in Postgres."],
            ["Operational Handoff", "Stage 5 — the final escalation package handed to Tier-2."],
            ["Pivot Insights", "Stage 1A + 1B merged. Contains Smoking Gun + Do Not Chase."],
            ["Proactive", "Intake mode for monitoring/alert-triggered events. Left column on the landing page."],
            ["Profile match", "A short tag describing the matched cohort (e.g. 'BGP / 4400 / customer-A')."],
            ["Reactive", "Intake mode for customer-reported issues. Right column on the landing page."],
            ["Resolution Quality Score", "A numeric quality grade attached to each gold ticket. Drives Stage 0 sorting."],
            ["RRF", "Reciprocal Rank Fusion — strategy-aware weighted merge of retrieval channels."],
            ["Smoking Gun", "The single strongest pivot signal in a cohort."],
            ["Stage", "A single Card in the Resolution Journey UI. Six logical stages, five rendered Cards."],
            ["Telemetry", "Append-only event log in tier1_journey_events. Drives /resume-state and the Stage 5 traversal log."],
            ["Top-5 match IDs", "The five highest-ranked candidate tickets from the cohort. Stored on tier1_sessions and surfaced via the Tier1AnswerCard pagination."],
            ["Traversal log", "The roll-up of stage events into 'viewed', 'marked Helpful', 'advanced at...', '/ <duration>' entries shown in the escalation package."],
        ],
        col_widths_inches=[2.0, 4.5],
    )

    doc.add_page_break()

    # ---- Footer note ----
    add_para(
        doc,
        "End of document.",
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        color=MUTED,
    )

    # ---- Save ----
    doc.save(out_path)
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = here / "ARCHITECTURE.docx"
    build_document(out)
    print("Open ARCHITECTURE.docx in Word.")
    print("Tip: right-click the Table of Contents and choose 'Update Field' to populate it.")
