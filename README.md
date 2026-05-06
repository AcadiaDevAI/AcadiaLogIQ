# Acadia Log-IQ — AI Copilot for Network Operations

> **AI-powered Tier-1 NOC copilot.** Engineers paste an alert (or a customer email), and the system returns the closest historical resolution, a structured 8-section troubleshooting playbook, a 5-stage guided journey, and a built-in escalation handoff — grounded in your own ticket corpus, KBs, and SOPs.

[![Backend](https://img.shields.io/badge/backend-FastAPI%20%7C%20Python%203.11-009688)](backend/api.py)
[![Frontend](https://img.shields.io/badge/frontend-React%2018%20%7C%20AntD-1677ff)](frontend/package.json)
[![LLM](https://img.shields.io/badge/LLM-AWS%20Bedrock-FF9900)](#aws--external-services)
[![Vector](https://img.shields.io/badge/vector-Postgres%20%2B%20pgvector-336791)](backend/db/migrations)
[![Auth](https://img.shields.io/badge/auth-Clerk-6c47ff)](backend/clerk_auth.py)
[![Deploy](https://img.shields.io/badge/deploy-Docker%20%7C%20EC2-2496ED)](docker-compose.ec2.yml)

---

## Table of Contents

1. [What this is](#what-this-is)
2. [System architecture at a glance](#system-architecture-at-a-glance)
3. [Tech stack](#tech-stack)
4. [The two intake flows — Proactive & Reactive](#the-two-intake-flows--proactive--reactive)
5. [The Resolution Journey](#the-resolution-journey)
6. [Chat ↔ Journey round-trip](#chat--journey-round-trip)
7. [Repository layout](#repository-layout)
8. [Quickstart — local development](#quickstart--local-development)
9. [Environment variables](#environment-variables)
10. [Feature flags (Sprint-layered)](#feature-flags-sprint-layered)
11. [API surface](#api-surface)
12. [Database](#database)
13. [Document ingestion pipeline](#document-ingestion-pipeline)
14. [Retrieval pipeline](#retrieval-pipeline)
15. [Authentication (Clerk)](#authentication-clerk)
16. [AWS / external services](#aws--external-services)
17. [Deployment (Docker + EC2)](#deployment-docker--ec2)
18. [Testing](#testing)
19. [Observability](#observability)
20. [Contributing & development conventions](#contributing--development-conventions)
21. [Troubleshooting](#troubleshooting)
22. [Roadmap](#roadmap)

---

## What this is

Acadia Log-IQ is an end-to-end production system that helps **Tier-1 NOC engineers resolve incidents faster** by surfacing the closest historical resolution from a corpus of past tickets, KBs, and SOPs — and then guiding them through a 5-stage resolution journey (Best Match → Related Incidents → Guided Workflow → KB Reference → Operational Handoff).

The platform has two complementary surfaces sharing a single retrieval backbone:

| Surface | Audience | Trigger | Output |
|---|---|---|---|
| **Tier-1 Copilot** | NOC engineer with a live alert or customer-reported issue | Structured form fill **or** raw text paste | 8-section troubleshooting answer + 5-stage Resolution Journey |
| **Document QA chat** | Anyone who wants to query the document corpus conversationally | Free-form question | Grounded answer with sources, optional multi-agent troubleshooting, confidence-scored |

Both share the same Bedrock Titan embeddings, hybrid retrieval pipeline (pgvector + BM25 + full-text + metadata filtering), Postgres-backed chat sessions, and Clerk auth.

The codebase has evolved across **12 sprints** of incremental, flag-gated delivery — every Sprint feature is gated behind a paired backend (`LOGIQ_*_BACKEND`) and frontend (`REACT_APP_LOGIQ_*_FRONTEND`) flag so the platform can be rolled forward or back without code changes.

---

## System architecture at a glance

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              React 18 Frontend                                 │
│                          (Ant Design + Tailwind + CRA)                         │
│                                                                                │
│   ┌─────────────────┐   ┌─────────────────┐   ┌────────────────────────────┐   │
│   │  LandingRouter  │   │   ChatArea +    │   │    Tier1Workspace          │   │
│   │  (entry point)  │──▶│   ChatInput     │   │    └─ ResolutionJourney    │   │
│   │                 │   │   ChatMessage   │   │       (5 stage cards)      │   │
│   │  ┌─Proactive──┐ │   │   Sidebar       │   │                            │   │
│   │  └─Reactive───┘ │   └─────────────────┘   └────────────────────────────┘   │
│   └─────────────────┘                                                          │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ REST / JSON (axios + Clerk JWT)
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Backend                                   │
│                                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │  /tier1/*    │  │ /tier1/      │  │  /intake/*   │  │  /ask, /upload, │    │
│  │  Sprint 6    │  │  journey/*   │  │  Sprint 9    │  │  /chat/*,       │    │
│  │  Copilot     │  │  Sprint 10   │  │  Universal   │  │  /fingerprint/* │    │
│  │  (analyze,   │  │  (5-stage    │  │  Intake      │  │  /feedback/*    │    │
│  │   match,     │  │   journey,   │  │  (extract +  │  │  Phases 1-6     │    │
│  │   feedback)  │  │   telemetry) │  │   suggest)   │  │                 │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘    │
│         │                 │                 │                   │             │
│         ▼                 ▼                 ▼                   ▼             │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │       Shared retrieval / LLM / validation / agents subsystems        │    │
│  │                                                                      │    │
│  │  retrieval/orchestrator  →  query_classifier → keyword_search +      │    │
│  │  pgvector + BM25 + ts_vector → fusion (RRF) → reranker → top-N       │    │
│  │                                                                      │    │
│  │  routing/model_router  →  complexity_classifier → Haiku / Sonnet     │    │
│  │  agents/{planner, analyst, composer}  →  multi-agent troubleshooting │    │
│  │  validation/{validator, confidence_scorer, grounding_checker}        │    │
│  │  tier1_copilot/retrieval, prompt_builder, aggregator (Tier-1 path)   │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────┬──────────────────────────────┬──────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐      ┌────────────────────────────────────────────────┐
│ PostgreSQL+pgvector  │      │              AWS Bedrock                       │
│ (RDS or self-hosted) │      │   • amazon.titan-embed-text-v2:0  (1024-d)    │
│                      │      │   • mistral.mistral-7b-instruct-v0:2 (rerank) │
│ • documents          │      │   • anthropic.claude-haiku-4-5  (default LLM) │
│ • chunks (+ vectors) │      │   • anthropic.claude-sonnet-4-6 (complex Q)   │
│ • embeddings         │      └────────────────────────────────────────────────┘
│ • chat_sessions      │
│ • chat_messages      │      ┌────────────────────────────────────────────────┐
│ • tier1_sessions     │      │       Other AWS / external services            │
│ • tier1_journey_     │      │   • S3 (optional file storage)                 │
│   events             │      │   • SES (feedback email notifications)         │
│ • intake_extractions │      │   • Clerk (JWT auth, optional)                 │
│ • semantic_answer_   │      └────────────────────────────────────────────────┘
│   cache              │
│ • tier1_answer_cache │
└──────────────────────┘
```

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| **Frontend framework** | React 18 (CRA) | Functional components + hooks, no Redux (uses `useReducer` + Context) |
| **UI kit** | Ant Design 5 | Forms, modals, layout primitives |
| **Styling** | Tailwind CSS + custom theme tokens | `frontend/src/theme/acadiaTheme.js` defines `MODERN_TOKENS` (gradient navy palette) |
| **HTTP client** | axios | `frontend/src/services/api.js` with Clerk JWT interceptor |
| **Backend framework** | FastAPI (Python 3.11) | `backend/api.py` is the entry point; `lifespan` hook builds BM25 index + glossary at startup |
| **Auth** | Clerk (JWT, RS256) | Backend verifies via `PyJWKClient` in `backend/clerk_auth.py`; frontend uses `@clerk/clerk-react` |
| **Vector DB** | PostgreSQL + pgvector | 1024-dim cosine; `chunks` + `embeddings` tables |
| **Keyword retrieval** | PostgreSQL `to_tsvector` + ILIKE | Plus an in-memory BM25 index rebuilt at startup from `chunks.content` |
| **Reranker** | Mistral 7B (Bedrock) | Modular — pluggable cross-encoder support exists |
| **LLM (default)** | Claude Haiku 4.5 (Bedrock) | Cost-efficient, sub-second latency |
| **LLM (complex)** | Claude Sonnet 4.6 (Bedrock) | Routed to via complexity classifier (signals: multi-step, reasoning, context size, retrieval confidence, multi-document span) |
| **LLM (legacy / Tier-1 fallback)** | Mistral 7B (Bedrock) | Used by Tier-1 Copilot for the 8-section answer prompt; deterministic `template_fallback` if parse fails |
| **Embeddings** | Amazon Titan Embed V2 | 1024-dim, batched ingest |
| **File storage** | Local FS or AWS S3 | `STORAGE_TYPE` env switch |
| **Email** | AWS SES | Feedback submissions only |
| **Containerization** | Docker + Docker Compose | Two compose files: local dev + EC2 |
| **CI** | GitHub Actions | `.github/workflows/docker-build.yml` (note: stale Streamlit workflow — see roadmap) |

---

## The two intake flows — Proactive & Reactive

The Tier-1 Copilot landing page (post-Sprint-11) renders a **two-column split**: Proactive on the left, Reactive on the right. Both columns feed the same downstream `/tier1/analyze` endpoint and produce the same Resolution Journey.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Tier-1 Copilot Landing                             │
├──────────────────────────────────────┬──────────────────────────────────────┤
│            PROACTIVE                  │            REACTIVE                  │
│   Monitoring / alert-triggered        │   Customer-reported via email,      │
│                                       │   phone, portal, chat or note       │
│                                       │                                      │
│   What's happening?                   │   Tell us what's happening           │
│                                       │                                      │
│   ┌────────────────────────┐          │   ┌────────────────────────────┐    │
│   │ Severity: P1 P2 P3 P4  │          │   │ [paste raw alert /         │    │
│   │ Asset or system: ___   │          │   │  customer message here]    │    │
│   │ What's the alert: ___  │          │   │                            │    │
│   │ + 6 optional fields    │          │   │ 0/10000 chars              │    │
│   │                        │          │   │       [Extract & Suggest]  │    │
│   │       [Analyze alert]  │          │   └────────────────────────────┘    │
│   └────────────────────────┘          │              │                       │
│           │                           │              ▼                       │
│           │                           │   POST /intake/extract               │
│           │                           │   → up to 4 candidate "interpret-    │
│           │                           │     ation cards" (severity +         │
│           │                           │     asset + alert_type + customer    │
│           │                           │     + location)                      │
│           │                           │              │                       │
│           │     ◀─── pick a card ─────┘              ▼                       │
│           │                              SuggestionCarousel                  │
│           │                              (engineer picks one)                │
│           │     pre-fills Proactive form on the left ◀───┘                   │
│           ▼                                                                  │
│   POST /tier1/analyze                                                        │
│           │                                                                  │
└───────────┼──────────────────────────────────────────────────────────────────┘
            ▼
       Tier1Workspace → ResolutionJourney (5 stages)
```

### Proactive flow (10 steps)

1. Engineer fills the structured form (Severity, Asset, Alert type — required; 6 optional fields).
2. Click **Analyze alert** → frontend calls `analyzeAlert(payload)` from `frontend/src/components/Tier1Copilot/tier1Api.js`.
3. `POST /tier1/analyze` → handled by `backend/tier1_copilot/routes.py:analyze`.
4. Alert is normalized via `normalize_alert(req, alias_dict)` → produces a `signature_hash`.
5. Cache check (`tier1_answer_cache` keyed by signature). On miss:
6. `retrieve_top_matches()` runs:
   - **Stage 1** — exact SQL ILIKE against denormalized `chunks.alert_signature`, `fingerprints_text`, `component_category` (added in migration 038).
   - **Stage 2** — hybrid pgvector cosine + `to_tsvector` plainto_tsquery (each LIMIT 30), candidates fused.
   - **Weighted rerank** — Jaccard overlap (alert_type / asset / fingerprint / technology) + vector_similarity + resolution_quality. Sprint-7 boosts: recency, success_frequency, same_customer_boost, same_asset_family.
7. `prompt_builder.build_prompt()` constructs the Mistral prompt with the top match + compact context. On parse failure, `template_fallback()` returns a deterministic 8-section answer.
8. `parse_answer(raw)` decodes into `Tier1AnswerSection` (issue_understanding, historical_match, most_likely_cause, recommended_first_checks, most_likely_fix, validation, escalate_if, follow_up_question).
9. A row is created in `tier1_sessions` (state: `current_match_index`, `top_5_match_ids`, `started_at`); response cached.
10. Frontend stores the result and renders `<Tier1Workspace>` → `<ResolutionJourney>` (when `TIER1_JOURNEY_ON` and a `session_id` is present).

### Reactive flow (8 steps)

1. Engineer pastes raw text (email, phone notes, portal ticket dump) into the right column.
2. Click **Extract & Suggest** → `POST /intake/extract`.
3. `backend/tier1_copilot/intake/routes.py` validates length, loads catalogs (severity / asset / alert-type / customer derived from the corpus), invokes Bedrock LLM via `extractor.py`.
4. Each candidate is run through `validate_candidate()` (substring grounding against the raw text — rejects hallucinations).
5. `diversifier.diversify()` caps to 4 distinct candidate cards.
6. `audit.log_extraction()` writes an `intake_extractions` row; response includes `extraction_id` + candidates.
7. Frontend renders `<SuggestionCarousel>`; engineer picks a card → `onCardPicked(filled)` → the Proactive form on the left is populated via `prefill` prop + `useEffect`.
8. Engineer reviews/edits and clicks **Analyze alert** → from here the flow merges with Step 3 of Proactive.

---

## The Resolution Journey

Once `/tier1/analyze` returns a session_id, `ResolutionJourney` renders a 5-stage progressive disclosure. Each stage is a Card with its own backend endpoint and is revealed only when the engineer clicks "Reveal next stage" — minimizing cognitive load and giving Tier-2 a clear traversal log if escalation happens.

| Stage | UI title (current) | Backend module | Purpose |
|---|---|---|---|
| **Stage 0** | Best Historical Match & Recommended Resolution | `journey/stage0_confidence.py` (Best-Ticket Distillation) | Surfaces the single highest-quality past resolution with the Primary_Fix and Resolution_Steps from that ticket. |
| **Pivot Insights** | Smoking Gun + Do Not Chase (merged) | `journey/stage1_smoking_gun.py` + `stage1_do_not_chase.py` | Pulls the strongest pivot signal across the cohort and the noisy paths *not* to chase. |
| **Stage 2** | Related Incidents & Probable Causes | `journey/stage2_historical.py` | Up to 5 related incidents (`HistoricalMatchCard`) drawn from the ranked cohort. |
| **Stage 3** | Guided Troubleshooting Workflow | `journey/stage3_troubleshooting.py` | Up to 8 consolidated diagnostic steps (deduplicated, sequenced, with Alt A / Alt B / Primary branches). |
| **Stage 4** | Knowledge Base & SOP Reference | `journey/stage4_kb_handoff.py` + `stage4_search_kb_handoff.py` | Spawns a chat session pre-loaded with the alert; engineer can ask follow-ups against full corpus. |
| **Stage 5** | Operational Handoff | `journey/stage5_escalation.py` | Builds an escalation package (what's tried, recommended owner / next action / contacts) including the engineer's stage-traversal log with **per-stage time-spent**. |

### Reveal mechanics

- `STAGE_ORDER = ["stage_0", "pivot_insights", "stage_2", "stage_3", "stage_4", "stage_5"]`
- On mount: `GET /tier1/journey/{sid}/initial` and `GET /tier1/journey/{sid}/resume-state` run in parallel. Stage 0 + Pivot Insights are always revealed; the rest unlock on click.
- Every reveal click POSTs `next_stage_clicked` to `/tier1/journey/{sid}/event` (telemetry → `tier1_journey_events`).
- `resume-state` reads `event_type='stage_advanced'` rows to restore "where the engineer left off" on remount.

### Telemetry

`backend/tier1_copilot/journey/telemetry.py` validates each event against an allowlist:

| event_type | Fired when |
|---|---|
| `stage_rendered` | Stage Card mounted |
| `helpful_clicked` | Engineer marked a stage as helpful |
| `next_stage_clicked` | Engineer advanced to the next stage |
| `stage_advanced` | Resume marker (used by `/resume-state`) |
| `kb_chat_engaged` | Stage 4 → chat handoff actually returned an answer |
| `escalation_initiated_from_chat` | Engineer clicked "Escalate" inside the chat (Sprint 10.5) |
| `abandoned` | Tab closed / session timeout |

All inserts are fire-and-forget — telemetry failures are logged, never raised.

---

## Chat ↔ Journey round-trip

Stage 4 is the bridge between the structured journey and the free-form chat surface. The user can hop into chat for an open-ended follow-up and come back to the journey without losing context.

```
ResolutionJourney (Stage 4)
   │  click "Open chat / Search KB"
   │  useChatHandoff(sid).askInChat(prefilledMessage)
   ▼
POST /tier1/journey/{sid}/search-kb-handoff
   │  • mints a chat_sessions row
   │  • inserts the user turn with metadata.journey_session_id = sid
   │  • returns { chat_session_id }
   ▼
SET_MODE("troubleshooting")   ──▶  AppLayout swaps to ChatArea
SET_SESSION (hydrate)         ──▶  state.sessionMetadata.journey_session_id set
askQuestion(text, chat_session_id) → /ask → ADD_ASSISTANT_MESSAGE
   │
   │  Engineer can now keep chatting normally.
   │  ChatArea renders a "Back to Resolution Journey" banner because
   │  sessionMetadata.journey_session_id is set.
   ▼
Engineer clicks "Return to Stages" or "Escalate to Tier 2"
   │  JourneyMessageActions dispatches RESUME_JOURNEY({ journeySessionId })
   ▼
ChatContext: journeyResumeSessionId set, selectedMode cleared
   │
   ▼
AppLayout falls back to LandingRouter → mounts Tier1Workspace
with the journey session restored.
```

The round-trip also supports re-entry from the **chat history sidebar**: clicking a past chat dispatches `SET_SESSION` with the persisted `selected_mode` ("troubleshooting"); AppLayout flips back into ChatArea — and if that chat was journey-originated, the "Back to Resolution Journey" banner is still there.

> **Sprint 12 fix.** The "New Chat" button used to reset `selectedMode` to `null`, which after the Sprint-11 LandingRouter changes started bouncing engineers to the Tier-1 intake form instead of opening a fresh chat. The reducer now keeps `selectedMode = "troubleshooting"` so a fresh chat opens cleanly.

---

## Repository layout

```
AICode_Chatbot/
├── backend/
│   ├── api.py                                # FastAPI app, /ask pipeline, /upload, /chat/*, /fingerprint/*
│   ├── config.py                             # Pydantic settings; LOGIQ_* feature flags
│   ├── clerk_auth.py                         # Clerk JWT verification (PyJWKClient)
│   ├── vector_store.py                       # Postgres CRUD + BM25 index + sessions persistence
│   ├── tier1_copilot/                        # Sprint 6+: structured Tier-1 Copilot
│   │   ├── routes.py                         # /tier1/* endpoints
│   │   ├── retrieval.py                      # Two-stage retrieval + weighted reranking
│   │   ├── normalizer.py                     # Alert → signature_hash + alias expansion
│   │   ├── prompt_builder.py                 # Mistral prompt + 8-section parser + fallback
│   │   ├── aggregator.py                     # Match list assembly
│   │   ├── cache.py                          # tier1_answer_cache accessors
│   │   ├── feedback.py                       # Thumbs / follow-up actions
│   │   ├── alias_dictionary.py
│   │   ├── schemas.py                        # Pydantic request/response models
│   │   ├── session_state/tier1_session.py    # tier1_sessions row + state machine
│   │   ├── intake/                           # Sprint 9: universal intake (paste flow)
│   │   │   ├── routes.py                     # /intake/* endpoints
│   │   │   ├── extractor.py
│   │   │   ├── validator.py                  # Substring grounding (anti-hallucination)
│   │   │   ├── diversifier.py                # Caps candidates at 4
│   │   │   └── catalogs.py                   # Severity/asset/alert/customer catalogs
│   │   ├── journey/                          # Sprint 10: Resolution Journey
│   │   │   ├── routes.py                     # /tier1/journey/* endpoints
│   │   │   ├── stage0_confidence.py          # Best-ticket distillation
│   │   │   ├── stage1_smoking_gun.py         # Pivot signal
│   │   │   ├── stage1_do_not_chase.py
│   │   │   ├── stage2_historical.py          # 5 related incidents
│   │   │   ├── stage3_troubleshooting.py     # Consolidated steps
│   │   │   ├── stage4_kb_handoff.py
│   │   │   ├── stage4_search_kb_handoff.py   # Chat session minting
│   │   │   ├── stage5_escalation.py          # Operational handoff package
│   │   │   ├── telemetry.py                  # tier1_journey_events writer
│   │   │   └── ticket_loader.py
│   │   └── diagnostics/                      # Sprint 7+: deeper analysis cards
│   │       ├── deeper_diagnostics.py
│   │       ├── stuck_detector.py
│   │       ├── explain_recommendation.py
│   │       ├── escalation_package.py
│   │       └── escalation_directory.py
│   ├── retrieval/                            # /ask hybrid retrieval (Phase 3)
│   │   ├── orchestrator.py                   # Coordinator: classify → retrieve → fuse → rerank
│   │   ├── query_classifier.py               # keyword / semantic / mixed
│   │   ├── keyword_search.py
│   │   ├── fusion.py                         # Reciprocal Rank Fusion
│   │   └── reranker.py                       # Modular (LLM / cross-encoder / none)
│   ├── routing/                              # Phase 4 model routing
│   │   ├── model_router.py
│   │   ├── complexity_classifier.py
│   │   └── context_builder.py
│   ├── agents/                               # Phase 5 multi-agent troubleshooting
│   │   ├── orchestrator.py                   # Escalation gate
│   │   ├── planner.py                        # Sonnet — query decomposition
│   │   ├── analyst.py                        # Haiku — per-step
│   │   ├── composer.py                       # Haiku — synthesis
│   │   ├── expert_copilot_template.py        # Sprint 5 gold-schema renderer
│   │   └── base.py                           # Token budget, LLM invoker
│   ├── validation/                           # Phase 6 guardrails
│   │   ├── validator.py
│   │   ├── confidence_scorer.py              # 4-signal blend (retrieval/coverage/grounding/consistency)
│   │   ├── grounding_checker.py              # Fabrication detection
│   │   └── eval_harness.py
│   ├── ingestion/
│   │   ├── structured_parser.py              # 3-strategy adaptive heading detection
│   │   └── prompt_templates.py
│   ├── services/
│   │   ├── contextual_ingestion_service.py   # process_document — main ingestion entry
│   │   ├── embedding_service.py              # Bedrock Titan v2 client
│   │   ├── bedrock_haiku.py                  # Anthropic Haiku JSON invoke
│   │   ├── semantic_cache.py                 # Brief 5 cross-user answer cache
│   │   ├── session_mode_state.py
│   │   └── journey_retrieval_context.py
│   ├── storage/
│   │   ├── local_storage.py                  # Filesystem provider
│   │   └── s3_storage.py                     # S3 provider (boto3)
│   ├── db/
│   │   ├── connection.py                     # SQLAlchemy engine + SessionLocal
│   │   ├── migrate.py                        # Numeric-ordered SQL runner
│   │   └── migrations/                       # 001_*.sql … 041_*.sql (see Database section)
│   ├── tests/                                # Backend tests
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.js                            # ThemeProvider → ConfigProvider → AuthGate → ChatProvider → AppLayout
│   │   ├── index.js                          # ClerkProvider mount
│   │   ├── components/
│   │   │   ├── LandingRouter.js              # Default = tier1; legacy fingerprint screen disabled
│   │   │   ├── LandingPage.js                # Pre-Sprint-4 mode picker
│   │   │   ├── ChatArea.js
│   │   │   ├── ChatInput.js
│   │   │   ├── ChatMessage.js
│   │   │   ├── Sidebar.js                    # History + new-chat + admin upload
│   │   │   ├── AuthGate.js
│   │   │   ├── FingerprintInputScreen.js     # (currently disabled in LandingRouter)
│   │   │   ├── Tier1Copilot/
│   │   │   │   ├── Tier1IntakeForm.js        # Proactive | Reactive split (SourceAwareIntake)
│   │   │   │   ├── Tier1Workspace.js
│   │   │   │   ├── Tier1AnswerCard.js
│   │   │   │   ├── Tier1FollowupChips.js
│   │   │   │   ├── tier1Constants.js         # All UI labels, feature flags, severity options
│   │   │   │   ├── tier1Api.js
│   │   │   │   ├── intake/
│   │   │   │   │   ├── UniversalIntakePanel.js
│   │   │   │   │   ├── SuggestionCarousel.js
│   │   │   │   │   └── intakeApi.js
│   │   │   │   └── journey/
│   │   │   │       ├── ResolutionJourney.js
│   │   │   │       ├── Stage0BestTicketDistillation.js
│   │   │   │       ├── Stage0ConfidenceLead.js   # legacy Sprint-10 path
│   │   │   │       ├── Stage1aSmokingGun.js
│   │   │   │       ├── Stage1bDoNotChase.js
│   │   │   │       ├── PivotInsightsPanel.js
│   │   │   │       ├── Stage2HistoricalMatches.js
│   │   │   │       ├── Stage3TroubleshootingApproach.js
│   │   │   │       ├── Stage4SearchKBHandoff.js
│   │   │   │       ├── Stage5EscalationPackage.js
│   │   │   │       ├── EscalateButton.js
│   │   │   │       ├── HelpfulButton.js
│   │   │   │       ├── useChatHandoff.js     # Shared "Ask in chat" plumbing
│   │   │   │       └── journeyApi.js
│   │   │   └── journey-chat/
│   │   │       └── JourneyMessageActions.js  # Return-to-Stages / Escalate-to-Tier-2 buttons
│   │   ├── hooks/
│   │   │   ├── ChatContext.js                # useReducer state — sessions, messages, mode, journey resume
│   │   │   ├── ThemeContext.js
│   │   │   ├── useAuthInterceptor.js         # axios JWT injector
│   │   │   └── useAutoRegister.js
│   │   ├── services/api.js                   # Axios + setTokenGetter; all REST helpers
│   │   ├── theme/
│   │   │   ├── acadiaTheme.js                # MODERN_TOKENS (navy gradient palette)
│   │   │   └── ThemeProvider.js
│   │   ├── config/clientSettings.js          # GUIDED_WORKFLOW_ENABLED + Sprint flags
│   │   └── index.css                         # Tailwind + CSS variables
│   ├── package.json
│   ├── nginx.conf
│   ├── Dockerfile
│   └── .env / .env.example
├── docker-compose.yml                        # Local dev
├── docker-compose.ec2.yml                    # EC2 deployment (api on 8000, ui on 8501→80)
├── .github/workflows/docker-build.yml        # CI (note: stale Streamlit workflow)
├── README.md                                 # This file
└── ARCHITECTURE.docx                         # Detailed end-to-end flow doc (generated)
```

---

## Quickstart — local development

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15 with the `pgvector` extension
- AWS credentials with Bedrock access (or skip Bedrock and run with stubs)
- (Optional) Clerk account for auth

### 1. Database

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

Migrations run automatically on first backend startup via `backend/db/migrate.py` (numeric-ordered SQL files in `backend/db/migrations/`).

### 2. Backend

```bash
cd backend
cp .env.example .env       # edit DATABASE_URL, AWS_*, BEDROCK_*, CLERK_* etc.
pip install -r requirements.txt

# From the repo root:
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `curl http://localhost:8000/health`

### 3. Frontend

```bash
cd frontend
cp .env.example .env       # set REACT_APP_API_BASE=http://localhost:8000
npm install --legacy-peer-deps   # Clerk 5.x requires this flag
npm start
```

Open `http://localhost:3000`. The LandingRouter mounts the Tier-1 Copilot intake by default (Sprint 11+).

### 4. Useful logs

```powershell
# Windows / PowerShell — capture backend logs to file:
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload 2>&1 |
    Tee-Object logs\backend.log
```

---

## Environment variables

### Backend (`backend/.env`)

```env
# ─── AWS / Bedrock ──────────────────────────────────────────
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=                   # blank on EC2 (use IAM Role)
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=

BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL=mistral.mistral-7b-instruct-v0:2
BEDROCK_HAIKU_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_SONNET_MODEL=us.anthropic.claude-sonnet-4-6

# ─── Database ───────────────────────────────────────────────
DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname

# ─── Storage ────────────────────────────────────────────────
UPLOAD_DIR=/app/uploads
# STORAGE_TYPE=s3
# S3_BUCKET=acadia-logiq-uploads

# ─── Vector store / collection ──────────────────────────────
CHROMA_PERSIST_DIR=/app/data/chroma  # legacy — most retrieval uses pgvector
COLLECTION_NAME=logs_titan_v2_1024

# ─── Tuning ─────────────────────────────────────────────────
CHUNK_MAX_CHARS=6000
CHUNK_MIN_CHARS=200
CHUNK_BATCH_SIZE=6
ENABLE_LLM_CHUNK_FALLBACK=true
ENABLE_METADATA_EXTRACTION=true

# ─── Routing / agents / validation ──────────────────────────
ENABLE_MODEL_ROUTING=true
ROUTING_DEFAULT_MODEL=haiku
ENABLE_AGENT_MODE=true
AGENT_COMPLEXITY_THRESHOLD=0.65
ENABLE_ANSWER_VALIDATION=true
ENABLE_EVAL_LOGGING=true

# ─── Auth (Clerk) ───────────────────────────────────────────
CLERK_ENABLED=false
CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=                    # NEVER commit

# ─── Email (SES) ────────────────────────────────────────────
SES_ENABLED=false
SES_SENDER_EMAIL=noreply@yourdomain.com
SES_FEEDBACK_RECIPIENT=team@yourdomain.com

# ─── Server ─────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
API_KEY=                             # optional API-key auth fallback when Clerk off
UI_API_KEY=                          # used by frontend when CLERK_ENABLED=false
```

### Frontend (`frontend/.env`)

```env
REACT_APP_API_BASE=http://localhost:8000
REACT_APP_CLERK_PUBLISHABLE_KEY=     # leave blank to disable auth UI
REACT_APP_BUILD_TIMESTAMP=dev        # surfaced in the bottom-right BuildStamp

# Sprint feature flags (every flag you flip here also has a backend twin)
REACT_APP_GUIDED_WORKFLOW_ENABLED=true
REACT_APP_LOGIQ_SPRINT2_FRONTEND=true
REACT_APP_LOGIQ_SPRINT3A_FRONTEND=true
REACT_APP_LOGIQ_SPRINT3B_FRONTEND=true
REACT_APP_LOGIQ_SPRINT3C_FRONTEND=true
REACT_APP_LOGIQ_SPRINT3D_FRONTEND=true
REACT_APP_LOGIQ_SPRINT3E_FRONTEND=true
REACT_APP_LOGIQ_SPRINT4_FRONTEND=true
REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND=true
REACT_APP_LOGIQ_TIER1_PROGRESSIVE_FRONTEND=true
REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND=true
REACT_APP_LOGIQ_TIER1_MODERN_THEME=true
REACT_APP_LOGIQ_TIER1_DOWNLOAD_DEMO=true
REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND=true
REACT_APP_LOGIQ_TIER1_INSIGHTS_FRONTEND=true
REACT_APP_LOGIQ_UNIFIED_TIER1_UX_FRONTEND=true
REACT_APP_LOGIQ_TIER1_JOURNEY_FRONTEND=true
```

---

## Feature flags (Sprint-layered)

Every Sprint feature is gated behind a paired backend + frontend flag. Routers in `backend/api.py` are mounted **only when their flag is on** so older sprint behavior remains byte-identical when flags are off — making rollback a single-env-var change.

| Flag (frontend) | Flag (backend) | What it gates |
|---|---|---|
| `REACT_APP_LOGIQ_SPRINT2_FRONTEND` | — | Context-break detection + pattern response cards |
| `REACT_APP_LOGIQ_SPRINT3A_FRONTEND` | — | Mode-aware prompts + post-thumbs-up action chips |
| `REACT_APP_LOGIQ_SPRINT3B_FRONTEND` | — | Thumbs-down KB pivot + confidence banner |
| `REACT_APP_LOGIQ_SPRINT3C/D/E_FRONTEND` | — | Mode-specific contact-card / ticket / vendor chips |
| `REACT_APP_LOGIQ_SPRINT4_FRONTEND` | `LOGIQ_SPRINT4_BACKEND` | Mounts `LandingRouter` instead of legacy `LandingPage` |
| `REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND` | `LOGIQ_TIER1_COPILOT_BACKEND` | Sprint 6 — Tier-1 Copilot intake form & `/tier1/*` routes |
| `REACT_APP_LOGIQ_TIER1_PROGRESSIVE_FRONTEND` | `LOGIQ_TIER1_PROGRESSIVE_BACKEND` | Sprint 7 — progressive Workspace + ranking boosts |
| `REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND` | — | Sprint 8 Track A — confidence labels v2, severity chips, skeletons |
| `REACT_APP_LOGIQ_TIER1_MODERN_THEME` | — | Sprint 8 Track B — modern navy theme tokens |
| `REACT_APP_LOGIQ_UNIVERSAL_INTAKE_FRONTEND` | `LOGIQ_UNIVERSAL_INTAKE_BACKEND` | Sprint 9 — Reactive paste flow + `/intake/*` routes |
| `REACT_APP_LOGIQ_TIER1_JOURNEY_FRONTEND` | `LOGIQ_TIER1_JOURNEY_BACKEND` | Sprint 10 — 5-stage Resolution Journey + `/tier1/journey/*` routes |
| `REACT_APP_LOGIQ_UNIFIED_TIER1_UX_FRONTEND` | — | Sprint 11 — Proactive ⏐ Reactive split layout |

---

## API surface

### Top-level (`backend/api.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + chunk count + model status |
| GET | `/me` | Current Clerk user (or anonymous) |
| POST | `/auth/register-or-login` | Idempotent first-time user creation |
| POST | `/upload` | File upload + ingestion job |
| GET | `/upload_status/{job_id}` | Ingestion progress |
| GET | `/files` | List documents |
| DELETE | `/files/{file_id}` | Delete a document |
| POST | `/ask` | Main RAG endpoint (~1.5k LOC orchestrator) |
| GET | `/chat/sessions` | List the user's chat sessions |
| GET | `/chat/sessions/{id}` | Full session with messages |
| DELETE | `/chat/sessions/{id}` | Delete a session |
| GET / POST | `/chat/sessions/{id}/mode` | Read/set the guided-workflow mode |
| POST | `/chat/sessions/{id}/context/reset` | Clear mode + form data |
| POST | `/fingerprint/lookup` | Sprint 4 fingerprint exact-match (still wired but disabled in LandingRouter) |
| POST | `/feedback/state` | Save like/dislike on a single message |
| POST | `/feedback/submit` | Submit free-form feedback (triggers SES email) |

### Tier-1 Copilot (`backend/tier1_copilot/routes.py`, prefix `/tier1`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/tier1/analyze` | Main Tier-1 entry (Proactive form payload) |
| POST | `/tier1/feedback` | Per-response thumbs / follow-up |
| POST | `/tier1/session` | Create a Tier-1 session (Sprint 7) |
| GET | `/tier1/session/{id}/status` | Session state |
| POST | `/tier1/session/{id}/match-index` | Move match-card pagination |
| POST | `/tier1/session/{id}/match/{index}` | Lock in a specific match |
| POST | `/tier1/session/{id}/deeper-diagnostics` | Sprint 7 card |
| POST | `/tier1/session/{id}/escalation-package` | Sprint 7 card |
| POST | `/tier1/session/{id}/explain-recommendation` | Sprint 7 card |
| GET | `/tier1/health` | Tier-1 router health |

### Resolution Journey (`backend/tier1_copilot/journey/routes.py`, prefix `/tier1/journey`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/{sid}/initial` | Stage 0 + Pivot Insights payload (parallel paint) |
| GET | `/{sid}/pivot-insights` | Just the pivot panel |
| GET | `/{sid}/stage-2` | Up to 5 historical matches |
| GET | `/{sid}/stage-3` | Consolidated troubleshooting steps |
| GET | `/{sid}/stage-4` | KB handoff prefilled message |
| GET | `/{sid}/stage-5` | Operational handoff package |
| POST | `/{sid}/search-kb-handoff` | Mints a chat session pre-loaded with the alert |
| POST | `/{sid}/event` | Append a telemetry event (`stage_rendered`, `next_stage_clicked`, etc.) |
| GET | `/{sid}/resume-state` | Restore "where the engineer left off" on remount |

### Universal Intake (`backend/tier1_copilot/intake/routes.py`, prefix `/intake`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/intake/extract` | Reactive paste → up to 4 candidate cards |
| POST | `/intake/extraction/{id}/feedback` | Records which card was picked / rejected |
| GET | `/intake/health` | Intake router health |

---

## Database

PostgreSQL is the single source of truth. Migrations are SQL files in `backend/db/migrations/`, applied in numeric order by `backend/db/migrate.py` (each wrapped in `engine.begin()` and split on `;`).

### Core tables

| Table | Purpose |
|---|---|
| `documents` | Logical document record (name, owner, version family, ingestion_status, doc_kind) |
| `chunks` | Text chunks with rich `metadata_json` (Fingerprints, alert_signature, fingerprints_text, component_category, asset_family, cached_expert_answer) |
| `embeddings` | pgvector 1024-dim vectors keyed to `chunk_id` |
| `users` | Clerk-mapped user profiles (Phase 3 multi-user) |
| `chat_sessions` | Chat history rows; carries `selected_mode`, `entered_via`, `original_fingerprint`, `_session_metadata.journey_session_id` |
| `chat_messages` | User/assistant turns + `feedback`, `context_stats`, `semantic_cache_id` |
| `learned_vocabulary` | Auto-learned identifier / enum tokens with `canonical_form` |
| `semantic_answer_cache` | Brief 5 cross-user answer cache, vector-keyed |
| `pattern_analytics_cache` | Sprint 2 pattern stats (topic-keyed) |
| `organization_schemas` | Per-org column-name remappings for ingestion |
| `logiq_sessions` | Generic session/user/org tracking |

### Tier-1 specific

| Table | Purpose |
|---|---|
| `tier1_answer_cache` | Sprint 6 — signature-hash-keyed cached 8-section answers (extended in Sprint 10.3 with `top_5_match_ids TEXT[]`) |
| `tier1_sessions` | Sprint 7 — per-alert session state (current_match_index, thumbs_down_count, escalated, what_tried, started_at) |
| `tier1_journey_events` | Sprint 10 — append-only telemetry log (id BIGSERIAL, session_id, stage, event_type, payload_json JSONB, created_at) |
| `intake_extractions` | Sprint 9 — universal intake audit (raw_text hash, candidates, picked_index, was_rejected) |

### Notable migrations

| File | What it adds |
|---|---|
| `001_phase1_foundation.sql` | `vector`/`pgcrypto` extensions, base schema |
| `002_phase2_contextual_ingestion.sql` | Versioning columns on `documents` |
| `003_phase3_multi_user.sql` | `users`, owner-scoped indexes |
| `034_add_doc_kind.sql` | `doc_kind` (ticket/sop/kb/contact_*/vendor_case) classification |
| `035_fingerprint_gin_indexes.sql` | GIN index on `chunks.metadata_json -> 'Metadata' -> 'Fingerprints'` |
| `036_session_fingerprint_state.sql` | `entered_via`, `original_fingerprint` on `chat_sessions` |
| `038_tier1_copilot.sql` | Denormalized `alert_signature`, `fingerprints_text`, `component_category` columns + `tier1_answer_cache` |
| `039_tier1_progressive.sql` | `chunks.asset_family`; `tier1_sessions` |
| `040_tier1_journey_events.sql` | `tier1_journey_events` |
| `041_universal_intake.sql` | `intake_extractions` |
| `041_tier1_answer_cache_top5.sql` | Adds `top_5_match_ids TEXT[]` |

> **Note:** Two files share the `041_` prefix — `migrate.py` orders alphabetically so this is deterministic in practice but a future renumber would tidy it up.

### Fingerprints

Fingerprints (e.g. `BGP-5-ADJCHANGE`) are stored as a JSONB array at `chunks.metadata_json -> 'Metadata' -> 'Fingerprints'`. Migration 035 adds a GIN index for sub-millisecond exact-match lookups via `POST /fingerprint/lookup` (the legacy Sprint-4 entry point — currently disabled in `LandingRouter` but the endpoint still functions).

---

## Document ingestion pipeline

`backend/services/contextual_ingestion_service.py:process_document` is the entry called from `POST /upload`.

```
File uploaded
   │
   ▼
1. SHA-256 fingerprint   ──▶  exact_duplicate? skip
   │
   ▼
2. Adaptive parser (3 strategies):
   (a) DOCX heading styles
   (b) Content-pattern regex (Scenario/Chapter/Step etc.)
   (c) Haiku LLM section discovery (≈ $0.002, only if a/b fail)
   │
   ▼
3. Scenario-aware chunking (one section = one chunk)
   │
   ▼
4. Per-chunk metadata extraction (Haiku, concurrent, batched)
   │
   ▼
5. Titan Embed v2 → 1024-dim vectors (concurrent)
   │
   ▼
6. INSERT chunks + embeddings + metadata into Postgres / pgvector
   │
   ▼
7. Update in-memory BM25 index
   │
   ▼
8. Version detection: new_document / new_version / exact_duplicate
   (supersedes old version when same family)
```

Gold-schema JSON tickets follow a special path (`_ingest_gold_ticket_json`): one chunk per ticket with rich `metadata_json` carrying `Fingerprints`, `Incident_Summary`, `Operational_SOP`, `Executive_Sharable_RCA`, `Resolution_Quality_Score`, etc. — these are the ticket bodies the Tier-1 retrieval queries.

---

## Retrieval pipeline

### `/ask` (general document QA)

`backend/retrieval/orchestrator.py` runs four candidate channels in parallel and fuses them:

```
Query
  │  Titan embed
  ▼
┌─────────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ pgvector cosine │  │  BM25    │  │  PG FTS +    │  │  Metadata    │
│   (LIMIT 25)    │  │ (LIMIT   │  │  ILIKE fallb.│  │  JSONB filter│
│                 │  │   20)    │  │   (LIMIT 15) │  │   (LIMIT 10) │
└────────┬────────┘  └────┬─────┘  └──────┬───────┘  └──────┬───────┘
         │                │               │                 │
         └────────────┬───┴───────────────┴─────────────────┘
                      ▼
               Reciprocal Rank Fusion (strategy-aware weights)
                      │
                      ▼
              Modular reranker (LLM / cross-encoder / none)
                      │
                      ▼
                Top-N chunks → context builder
                      │
                      ▼
       Complexity classifier → Haiku (default) / Sonnet (complex)
                      │
                      ▼
              Answer + Phase 6 grounding/confidence checks
```

### Tier-1 Copilot

`backend/tier1_copilot/retrieval.py:retrieve_top_matches` is a separate, narrower path optimized for alert→ticket matching:

1. **Stage 1 — exact SQL** (ILIKE on denormalized columns) — short-circuits if score ≥ 0.80.
2. **Stage 2 — hybrid** (pgvector + ts_vector) when Stage 1 is weak.
3. **Weighted rerank** combining Jaccard overlap (alert_type / asset / fingerprint / technology) + vector similarity + resolution_quality, plus Sprint-7 boosts (recency, success_frequency, same_customer, same_asset_family).

Confidence band: ≥ 0.85 High · ≥ 0.60 Medium · ≥ 0.40 Low · else None.

---

## Authentication (Clerk)

When `CLERK_ENABLED=true` the backend requires a Clerk-issued JWT on every authenticated route. `backend/clerk_auth.py` uses `PyJWKClient` to fetch the issuer's public keys, validates RS256, exp, iss (`https://<frontend>.clerk.accounts.dev`), and optionally `azp` (frontend origin).

The frontend wraps the app in `<ClerkProvider>` (`frontend/src/index.js`); `<AuthGate>` blocks unauthenticated traffic; `useAuthInterceptor` injects the active token into every axios call via `setTokenGetter`. `useAutoRegister` POSTs `/auth/register-or-login` on first auth so the user gets a row in the `users` table.

Set `CLERK_ENABLED=false` to fall back to optional API-key auth (`API_KEY` / `UI_API_KEY` env vars) — useful for local dev or single-tenant deployments.

---

## AWS / external services

| Service | Used by | Notes |
|---|---|---|
| **Bedrock — Titan Embed V2** | `backend/services/embedding_service.py` | 1024-dim cosine embeddings |
| **Bedrock — Mistral 7B** | `backend/api.py` (Tier-1 prompt) + `backend/retrieval/reranker.py` | Tier-1 8-section answer + reranking |
| **Bedrock — Claude Haiku 4.5** | `backend/services/bedrock_haiku.py` (ingestion metadata + chunking fallback) + `backend/routing/model_router.py` (default `/ask` LLM) | Cost-efficient, sub-second answer |
| **Bedrock — Claude Sonnet 4.6** | `backend/routing/model_router.py` (escalated path) + `backend/agents/planner.py` | Multi-step / complex reasoning |
| **S3** | `backend/storage/s3_storage.py` (when `STORAGE_TYPE=s3`); `backend/scripts/bulk_ingest.py` | Optional file storage |
| **SES** | `backend/api.py:send_feedback_email_async` | Sends a SES email when `/feedback/submit` is called |
| **Clerk** | `backend/clerk_auth.py`, `frontend/src/components/AuthGate.js` | Optional |

### IAM (least-privilege example)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        "arn:aws:bedrock:us-east-1::foundation-model/mistral.mistral-7b-instruct-v0:2",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5*",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-6*"
      ]
    },
    { "Sid": "S3Storage", "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:DeleteObject"],
      "Resource": "arn:aws:s3:::acadia-logiq-uploads/*" },
    { "Sid": "SESEmail", "Effect": "Allow",
      "Action": "ses:SendEmail", "Resource": "*",
      "Condition": { "StringEquals": { "ses:FromAddress": "noreply@yourdomain.com" } } }
  ]
}
```

---

## Deployment (Docker + EC2)

### Dockerfiles

- **`backend/Dockerfile`** — `python:3.11-slim`, installs `build-essential libpq-dev poppler-utils`, runs as non-root `appuser`, exposes 8000, healthcheck on `/health`.
- **`frontend/Dockerfile`** — multi-stage. Build stage: `node:18-alpine`, `npm install --legacy-peer-deps`, accepts every `REACT_APP_*` flag as a build ARG, runs `npm run build`. Serve stage: `nginx:alpine` with a custom `nginx.conf`, exposes port 80.

### Compose

`docker-compose.ec2.yml` defines:

```
service: api    → built from ./backend, port 8000:8000, env_file backend/.env, named volume "uploads"
service: ui     → built from ./frontend, depends_on api: service_healthy, port 8501:80
network: app-network
```

Frontend is built with `REACT_APP_API_BASE=/api` so nginx proxies API calls to the backend container.

### EC2 deployment (one-shot)

```bash
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/your-org/AICode_Chatbot.git
cd AICode_Chatbot
cp backend/.env.example backend/.env
nano backend/.env  # set DATABASE_URL, AWS_REGION, model IDs, CLERK_*

export REACT_APP_CLERK_PUBLISHABLE_KEY=pk_live_xxx
docker compose -f docker-compose.ec2.yml up --build -d
docker compose -f docker-compose.ec2.yml logs -f
```

Verify: `curl http://localhost:8000/health` → `{"status": "ok", ...}`.

> **Use IAM Roles on EC2**, not hardcoded AWS keys. Leave `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` blank — boto3 picks up the instance role automatically.

---

## Testing

### Backend

```bash
cd backend
pytest -v                                              # all tests
pytest tier1_copilot/tests -v                          # Tier-1 unit tests
pytest tests/tier1_copilot/journey -v                  # Journey integration tests
pytest tier1_copilot/intake/tests -v                   # Universal intake tests
pytest tests/test_validation.py -v                     # Phase 6 validation
```

Approximate coverage: ~50 backend test files. Notable suites:

- `backend/tier1_copilot/tests/` — normalizer, alias_dictionary, retrieval, ranking_boosts, prompt_builder, cache, escalation_package, deeper_diagnostics, explain_recommendation, stuck_detector, session_state, match_by_index_endpoint.
- `backend/tests/tier1_copilot/journey/` — stage1 do-not-chase, stage1 smoking-gun, stage2 historical, stage3 troubleshooting, stage3 per-ticket-details, stage4 kb-handoff, stage5 escalation, journey-initial endpoint, pivot-insights, resume-state, telemetry-invariants, search-kb-handoff.

### Frontend

```bash
cd frontend
npm test
```

Two suites today:

- `frontend/src/components/Tier1Copilot/journey/__tests__/ResolutionJourney.test.js`
- `frontend/src/components/journey-chat/__tests__/JourneyMessageActions.test.js`

---

## Observability

### Structured logging (logger: `acadia-log-iq`)

```
2026-05-05 19:43:54 - acadia-log-iq - INFO - Retrieval complete:
    strategy=mixed vector=25 bm25=20 kw=15 → fused=15 → reranked=6 (2648ms)
2026-05-05 19:43:54 - acadia-log-iq - INFO - Model routing: model=haiku (score=0.092)
2026-05-05 19:43:57 - acadia-log-iq - INFO - Generation complete: model=haiku, 3129ms, 12563 prompt chars
2026-05-05 19:43:57 - acadia-log-iq - INFO - Confidence: 0.652 (ret=0.80 cov=0.71 gnd=0.58 con=1.00) → PASS
```

### JSONL eval records

Every `/ask` request emits an `EVAL_RECORD: {...}` JSONL line — useful for offline quality regression analysis.

### Response diagnostics

`/ask` and `/tier1/analyze` return rich `context_stats` blocks with retrieval breakdown, model used, complexity score, agents path (if escalated), validation outcome, and per-stage timing.

### Journey telemetry

Every stage interaction posts to `tier1_journey_events`. Stage 5's escalation package now includes the engineer's stage-traversal log with **per-stage time-spent** (Sprint 12) — useful both for the Tier-2 reader and for engagement analytics.

---

## Contributing & development conventions

### Sprint flag discipline

Every new feature lands behind a paired backend + frontend flag. Flag-off must be byte-identical to the previous sprint. This is the project's most important invariant — it lets us ship daily without rolling back code.

### Commenting style

Every non-trivial branch carries a comment that explains **why**, **what sprint introduced it**, and **what fallback exists when the flag is off**. Read top-of-file comments in:

- `frontend/src/components/LandingRouter.js`
- `frontend/src/components/Tier1Copilot/Tier1IntakeForm.js`
- `frontend/src/components/Tier1Copilot/journey/useChatHandoff.js`
- `frontend/src/components/Tier1Copilot/tier1Constants.js`
- `backend/tier1_copilot/journey/stage5_escalation.py`
- `backend/tier1_copilot/journey/telemetry.py`

### Adding a stage label

`STAGE_LABELS` in `frontend/src/components/Tier1Copilot/tier1Constants.js` is the single source of truth for both the stage Card titles **and** the "Reveal next stage" button labels in `Stage1aSmokingGun`, `Stage1bDoNotChase`, `Stage2HistoricalMatches`, `Stage3TroubleshootingApproach`, and `PivotInsightsPanel`.

### Adding an `/ask` endpoint feature

`/ask` is ~1.5k LOC of orchestration. Reuse the existing context_stats / pattern detector / clarification flow — don't fork the handler.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Settings object has no attribute X` | `backend/config.py` missing a Phase 6 / Sprint setting. Pull latest and clear `__pycache__`. |
| `Python-dotenv could not parse statement at line N` | `.env` has a comment without `#` prefix on that line. |
| `Foreign key violation on /reset` | `UPDATE documents SET current_version_id = NULL` before deleting `document_versions`. |
| `Haiku returned invalid JSON` warnings | Normal — Haiku occasionally hits `max_tokens`. Retry logic handles it. Increase `HAIKU_MAX_TOKENS` if persistent. |
| Low retrieval accuracy | Re-index documents after parser changes. Confirm `CHUNK_MAX_CHARS=6000` and `ROUTING_DEFAULT_MODEL=haiku`. |
| EC2 deploy doesn't reflect changes | Docker layer cache. `docker compose -f docker-compose.ec2.yml up --build --force-recreate -d`. |
| New Chat opens the Tier-1 intake form instead of an empty chat | Sprint 12 fix in `ChatContext.js` — ensure `selectedMode = "troubleshooting"` in the `NEW_CHAT` reducer case. |
| Stage labels still show "Stage N — ..." | Sprint 12 rename — `STAGE_LABELS` was updated in `tier1Constants.js`; clear browser cache and `Ctrl+Shift+R`. |

---

## Roadmap

| Item | Status |
|---|---|
| Cohere Embed v3 (telecom-aware embeddings) | Planned |
| Cross-encoder reranker (faster than LLM) | Planned |
| Streaming `/ask` responses (SSE) | Planned |
| Multi-tenant (per-org isolation) | Planned |
| Alembic migrations | Planned |
| Prometheus metrics + Grafana | Planned |
| Replace stale `docker-build.yml` Streamlit workflow | Pending |
| Renumber the duplicate `041_` migration | Pending |

---

## License

Internal — © Acadia Consultants. All rights reserved.

---

## Maintainers

- Engineering: maruthiphani75@gmail.com
- Operations: dev@acadiaconsultants.com

---

> 📄 **For the deep end-to-end walkthrough** (data flows with sequence diagrams, full glossary, demo / KT / interview prep), see **`ARCHITECTURE.docx`** at the project root.
