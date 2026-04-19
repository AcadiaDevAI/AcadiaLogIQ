# Acadia Log IQ — Refactor Summary

## Status snapshot

| Fix | Description | Status |
|-----|-------------|--------|
| 1 | Gold JSON fast-path ingestion | **DONE** |
| 2 | Ticket-ID exact-match retrieval | **DONE** |
| 3 | `mode_selector` hybrid-first default | **DONE** |
| 4 | Clarifier skip for specific queries | **DONE** |
| 5 | Follow-up context inheritance before retrieval | **DONE** |
| 6 | SQL-over-metadata path for aggregation queries | **DONE** |
| 7 | Tiered escalation + clean "not found" for edge cases | **DONE** |

All seven fixes compile and import cleanly. Fixes 6 and 7 (a) now have their
upstream signals wired — Fix 1 populates the ticket-native `metadata_json`
fields that Fix 6 aggregates over, and Fix 2 emits the `ticket_id_not_found`
search mode that Fix 7 (a) short-circuits on.

---

## Fix 1 — Gold JSON fast-path ingestion

**Files touched**
- `backend/services/contextual_ingestion_service.py` — added `_is_gold_ticket_json`, `_render_gold_ticket_body`, `_ingest_gold_ticket_json`; routed at the top of `process_document`.

**Rationale**
When a gold-ticket JSON is uploaded, every field we'd ask Haiku to guess
(`incident_number`, `customer_name`, `priority`, `component_category`, SLA,
`resolution_quality_score`) is already in the source. The detector looks for
a list whose first object has `Metadata.Incident_Number` AND either
`Executive_Sharable_RCA` or `ITIL_5_Why` — enough to disambiguate our gold
format from arbitrary JSON. On a hit we build one chunk per ticket, header-
first (`TICKET: {id} | CUSTOMER: ... | PRIORITY: ... | COMPONENT: ...`),
then flatten non-empty sections into prose. `chunk.metadata_json` is populated
directly from the source. Zero Haiku calls — log line
`[ingest] gold-ticket fast-path: N tickets, 0 Haiku calls`. The function returns
the same dict shape the generic path does, so the api.py Phase 2/3 (embedding,
BM25, insert) runs unchanged.

---

## Fix 2 — Ticket-ID exact-match retrieval

**Files touched**
- `backend/retrieval/keyword_search.py` — added `ticket_id_exact_search(incident_numbers, allowed_file_ids, n_results)`.
- `backend/retrieval/orchestrator.py` — added `_extract_ticket_ids(query)` + short-circuit at the top of `retrieve(...)`.

**Rationale**
Embedding distance treats `INC-10000` and `INC-10035` as near-neighbours
because their narratives overlap, so "root cause of INC-10000" retrieves the
wrong ticket. When the query names any `INC-\d+` ID we skip the four
parallel channels and run a single indexed JSONB lookup
(`UPPER(metadata_json->>'incident_number') = ANY(:ids)`). Hit →
`search_mode="ticket_id_exact"`; miss → `search_mode="ticket_id_not_found"`
with `stats.requested_ticket_ids` and `stats.ticket_id` populated. The miss
signal is what Fix 7 (a) catches to emit the canned "ticket not found"
message instead of hallucinating. Because `step_retriever.build_step_retriever`
calls `retrieve(...)` directly, the short-circuit propagates into per-step
agent retrieval without duplicating the logic.

---

## Fix 3 — `mode_selector` hybrid-first default

**Files touched**
- `backend/agents/mode_selector.py` — rewrote `resolve_mode`; imports `_AGENT_ELIGIBLE_PATTERNS` from `agents/orchestrator.py`.

**Rationale**
`MODE_AUTO` was agent-first, so every trivial lookup escalated to the 4-stage
pipeline. New auto flow:
1. `source_count < 1` → hybrid.
2. Query matches any agent-eligible pattern (`compare`, `walk me through`,
   `across all`, root-cause + recommend, end-to-end) → agents.
3. `complexity_tier == "complex" AND score >= AGENT_COMPLEXITY_THRESHOLD` → agents.
4. Otherwise → hybrid.

`MODE_HYBRID` and `MODE_MULTI_AGENT` overrides are preserved verbatim. The
reason string now makes the outcome explicit, e.g.
`mode=auto → hybrid (no agent-eligible pattern, tier=simple)` or
`mode=auto → agents (pattern matched: 'Compare')`.

Smoke-tested acceptance cases:
- "What's the root cause of INC-10000?" → hybrid.
- "Compare INC-10005 and INC-10006" → agents.
- "Walk me through how to fix WiFi SSID issues" → agents.

---

## Fix 4 — Clarifier skip for specific queries

**Files touched**
- `backend/agents/clarifier.py`

**Rationale**
The clarifier was interrupting queries that were already unambiguous —
ticket IDs (`INC-10000`), named customers (`Nebula-Corp`), and direct
factual asks (`List all open incidents`) — asking the user to restate them.
`_looks_trivially_clear` now returns True on three targeted patterns and
emits `[clarifier] skipped — specific query pattern: {reason}` when any
fires. Existing trivial-clarity and domain-lock logic are untouched, so
"It's down" still goes through the LLM clarifier.

---

## Fix 5 — Follow-up context inheritance before retrieval

**Files touched**
- `backend/api.py` — added `_enrich_query_with_history` helper and wired it before query expansion in `/ask`.

**Rationale**
A bare follow-up like "what QA gaps did it have?" carries no ticket ID, so
embedding and BM25 drift to a different incident. The helper scans the last
3 assistant turns, extracts `INC-\d+` / `Enterprise-\d+` / `Nebula-Corp`
matches, and prepends them to the retrieval-only query when the current
query names none of them. The LLM prompt and chat persistence still see
`req.q` — the user never sees the mangled version.

---

## Fix 6 — SQL-over-metadata path for aggregation queries

**Files touched**
- `backend/retrieval/metadata_sql.py` (new) — `detect_aggregation_intent`, `run_aggregation`, `AggIntent`, `AggResult`.
- `backend/api.py` — runs the fast-path before embedding/retrieval; returns the prose summary and skips the agent pipeline on a hit.

**Rationale**
"How many Nebula-Corp tickets?" and "which tickets missed SLA?" are set
operations, not semantic search. The new module detects count/list intent,
extracts customer/priority/SLA/component filters, and runs a single
parameterised JSONB query against `chunks.metadata_json`. All binds use
SQLAlchemy `text()` + `bindparam(expanding=True)`. If the detector misses or
SQL returns zero rows, the pipeline falls through to normal RAG. With Fix 1
now populating `incident_number`, `customer_name`, `priority`,
`sla_target_met` on ticket chunks, this path fires as designed on gold-JSON
uploads.

---

## Fix 7 — Tiered escalation + clean "not found" for edge cases

**Files touched**
- `backend/api.py` — short-circuit on `retrieval.stats["search_mode"] == "ticket_id_not_found"` returning a canned message; passes new `stage` / `unresolved_count` kwargs to the agent pipeline; passes `retrieval_stats` to the validator.
- `backend/agents/orchestrator.py` — `run_agent_pipeline` accepts `stage` / `unresolved_count`; when both indicate repeated unresolved docs-stage follow-ups, a synthetic "search runbooks and KBs" step is prepended to the plan so the Analyst's per-step retriever routes across non-ticket chunks.
- `backend/validation/validator.py` — accepts `retrieval_stats`; when `search_mode == "ticket_id_not_found"`, returns a `passed=True, was_modified=False` result so the canonical message passes through untouched.

**Rationale**
Three edge cases were leaking: (a) asking about a non-existent ticket
produced a hallucinated fallback; (b) two "still not resolved" follow-ups
in docs stage left the user stuck on the same ticket content; (c) the
validator's false-refusal guard risked rewriting the deterministic "not
found" message. With Fix 2 now emitting `ticket_id_not_found`, (a) is live;
(b) and (c) were already independent of Fixes 1/2.

---

## How to verify

Run the API locally and issue these queries in order. Each category's
expected behaviour is listed.

### 1. Single-Ticket Lookup
- **Query:** `What's the root cause of INC-10000?`
- **Expected:** Clarifier skipped (log: `[clarifier] skipped — specific query pattern: ticket-id`). Mode selector chooses hybrid. Retrieval short-circuits via `ticket_id_exact` and returns only chunks whose `metadata_json.incident_number == "INC-10000"`.

### 2. Cross-Ticket Analytical
- **Query A:** `How many Nebula-Corp tickets?`
  - **Expected:** SQL fast-path fires (log: `[aggregation] SQL fast-path: N results`), response is `- Nebula-Corp has N tickets: INC-...`, `context_stats.aggregation_fast_path = true`, sub-200ms, no LLM call.
- **Query B:** `Compare INC-10005 and INC-10006`
  - **Expected:** Clarifier skipped (two ticket IDs present). Mode selector escalates to agents (pattern matched: 'Compare'). Retrieval runs two ticket-ID exact lookups (one per ID, via the same orchestrator short-circuit) for the Analyst's per-step retriever.

### 3. Follow-Up Conversational
- **Turn 1:** `Tell me about INC-10015` → answer about INC-10015. Clarifier skipped. Retrieval via `ticket_id_exact`.
- **Turn 2:** `What QA gaps?` → log line `[followup] enriched query with entities: ['INC-10015']`; retrieval runs against `INC-10015 What QA gaps?` while the LLM and chat history still show `What QA gaps?`.

### 4. Edge Cases
- **Query A:** `What happened on INC-99999?`
  - **Expected:** Retrieval returns `search_mode=ticket_id_not_found`; `/ask` returns the canned `- Ticket INC-99999 was not found ...` message in under 100ms, skipping LLM and validator.
- **Query B:** `Tell me about INC-10041` (a partial ticket with empty SOP)
  - **Expected:** Clarifier skipped. Retrieval returns whatever sections do exist; Analyst answers from those; no generic fallback. If the user signals "still not resolved" twice in docs stage, the next turn's Analyst first step is `Search runbooks and KBs for the originally-reported issue ...` and the response's `source_names` include runbook/KB files, not just the ticket.

### 5. Ingestion
- **Action:** Re-upload the gold ticket JSON.
- **Expected:** Log line `[ingest] gold-ticket fast-path: 45 tickets, 0 Haiku calls`. Database chunk count matches ticket count; every row has `metadata_json->>'doc_kind' = 'ticket'` and a non-null `incident_number`.

---

## Conversational refactor — session 2

End-to-end testing with the gold JSON exposed four behaviour gaps that the
7-fix series didn't cover: responses that came back as rigid bullet-point
reports, false-refusal messages on tickets whose answers were clearly in the
chunks, aggregation queries that bypassed the SQL fast-path, and the
ticket-not-found short-circuit being overridden by the false-refusal guard
downstream. This pass addresses all five goals from `CONVERSATIONAL_REFACTOR_BRIEF.md`.

### Goal 1 — Conversational prompts
**Files:** `backend/routing/model_router.py`, `backend/agents/composer.py`.
Rewrote the Haiku/Sonnet system prompt in `_invoke_claude` and the composer's
synthesis prompt in `run_composer`. Both now frame the model as a senior
operations engineer talking to a trainee: length proportional to the question,
natural prose by default, bullets only when the question calls for a list
(e.g. "list …", "compare …", "what are the steps"), no "based on the
documents" meta-phrasing, no section headers unless the user asked for a
structured breakdown. Expected effect: "root cause of INC-10000" returns a
2–4 sentence answer, while "compare INC-10005 and INC-10006" still produces
structured output.

### Goal 2 — Kill false refusals
**Files:** `backend/routing/context_builder.py`, `backend/validation/validator.py`,
`backend/api.py`. `_GROUNDING_RULES` rewritten: dropped the "every answer MUST
be in bullet-point format" directive, added explicit anti-hedge guidance
("Do NOT use phrases like 'insufficient evidence' or 'I cannot extract'…").
`_NOT_FOUND_PHRASES` tightened from 9 phrases to 5 — confident partial
answers ("do not contain specific timeline details, but the root cause was
X") no longer trip the guard. New `retry_fn` parameter on `validate_answer`:
when a false refusal is detected on a deterministic retrieval mode
(`ticket_id_exact` or hybrid_phase3) the validator calls the retry once with
a stronger-extraction directive before falling through to the canned message
(which now also logs `[validator] false refusal survived retry`). `/ask`
passes a bound `_retry_generate` closure that re-invokes `route_and_generate`
with an "The answer IS in the documents below" preamble.

### Goal 3 — Aggregation regex
**File:** `backend/retrieval/metadata_sql.py`. Broadened `detect_aggregation_intent`
to catch "how many / count / number of" + customer, "which customer had the
most incidents", "list / show me all / which tickets where", ranking queries
("which ticket had the highest resolution quality score"), and "how many
tickets missed / met / breached / failed / passed SLA". Added `ranking_field`
and `ranking_direction` to `AggIntent`. `run_aggregation` now dispatches
`operation == "rank"` → `_run_ranking` (ORDER BY on a whitelisted numeric
`metadata_json` field, LIMIT 5) and `operation == "group_by_customer"` →
`_run_group_by_customer` (`GROUP BY customer_name` with `COUNT(DISTINCT
incident_number)` ranked). When no intent matches, the detector now logs one
INFO line so operators can see why the SQL path didn't fire.

### Goal 4 — Ticket-not-found ordering
**File:** `backend/api.py`. The `search_mode == "ticket_id_not_found"`
short-circuit in `/ask` was already positioned before validator/agents/model
routing — rewrote the response body to the brief's wording ("`<ids>` was not
found in the indexed documents. If you expected this ticket to be available,
please upload the corresponding data.") and switched to the plural
`requested_ticket_ids` from the retrieval stats. Added a belt-and-suspenders
gate on the aggregation fast-path: skips detection when the query contains
`INC-\d+`, so a per-ticket lookup can never be mis-classified as a bulk
aggregation.

### Goal 5 — Consistency diagnostics
**File:** `backend/retrieval/orchestrator.py`. After every `ticket_id_exact`
short-circuit (hit or miss) the orchestrator now logs two INFO lines:
`[ticket_id_exact] ids=… rows_returned=… top_chunk_incident=…` and
`[ticket_id_exact] top_chunk_length=… sections_present=[…]` where the second
line reports which of `ROOT CAUSE:`, `RESOLUTION DETAIL:`, `ITIL 5-WHY ROOT
CAUSE:`, `SOP EXECUTION STEPS:`, `QA AUDITOR GAPS:` appear in the top chunk.
Purpose: if INC-10001 fails and INC-10000 succeeds, the presence/absence of
expected section labels in the logs will pinpoint whether the renderer in
`contextual_ingestion_service.py` is emitting different shapes for the two
tickets.

### Expected behaviour after this session
- `What was the root cause of INC-10000?` → 2-4 sentences of natural prose, no bullet dump.
- `Which teams were involved in resolving INC-10001?` → conversational Resolution_Groups list.
- `How many incidents were filed for Nebula-Corp?` → "- Nebula-Corp has 9 tickets: …" via SQL fast-path, sub-200ms, no LLM call.
- `Which customer had the most incidents?` → "- Nebula-Corp has the most, with 9 tickets. Next: …".
- `How many tickets missed SLA targets?` → SQL count with `sla_target_met = false`.
- `Which ticket had the highest resolution quality score?` → "- INC-XX has the highest resolution quality score at 5.".
- `What is the resolution for INC-10002?` → canned ticket-not-found message in under 100ms, zero LLM calls.
- `Compare INC-10006 and INC-10005` → still routes to agents, still structured output because the question asks for a comparison.

## Production architectural refactor — session 3

Session 3 generalises the gold-ticket special case into a schema-agnostic
architecture: structured ingestion, identifier short-circuit, follow-up
enrichment, and the agent token budget all now work for any record type
with a primary identifier. The gold-ticket fast-path is preserved as one
registered schema among many.

### Goal 1 — Schema-agnostic ingestion registry
**File:** `backend/services/contextual_ingestion_service.py`. Introduced a
`STRUCTURED_SCHEMAS` registry containing two classes: `GoldTicketSchema`
(wraps the existing gold-ticket JSON fast-path) and `GenericArraySchema`
(detects any list-of-dicts — at the root or under `records`/`items`/`data`/
`tickets`/`entries`/`results` — where records carry an ID-like field). The
field-detection logic prefers a curated candidate list (`id`, `uuid`,
`primary_id`, `key`, `issue_key`, `ticket_number`, `case_id`, `kb_id`,
`article_id`, …) and falls back to regex `^[A-Za-z]*(?:id|key|number|no)$`
for schemas we haven't seen before. Each chunk row now carries
`primary_id` + `id_type` in `metadata_json`; for gold tickets,
`incident_number` is preserved as a legacy alias so old callers still work.
`process_document` iterates the registry and falls through to the legacy
parse+chunk+Haiku pipeline on any detect-miss. Adding a new schema = adding
one class to the registry; no caller changes.

### Goal 2 — Config-driven identifier extraction
**Files:** `backend/config.py`, `backend/retrieval/orchestrator.py`,
`backend/retrieval/keyword_search.py`, `backend/api.py`,
`backend/validation/validator.py`. `config.py` now exposes two dicts:
`IDENTIFIER_PATTERNS` (id_type → regex with one capture group) and
`IDENTIFIER_CANONICAL_FORMAT` (id_type → str.format template). Ships with
`ticket_number` (`INC[-\s]?(\d{3,})`) and `issue_key` (`[A-Za-z][A-Za-z0-9]+-\d+`)
out of the box — adding a new pattern is a two-line config change.
`_extract_ticket_ids` is rewritten as `_extract_identifiers` returning
`List[Tuple[str, str]]`; patterns run in declaration order and earlier
matches own their span, so `INC-10005` can't be re-tagged as an issue_key.
The legacy `_extract_ticket_ids` shim filters to `ticket_number` for
backwards compatibility. `keyword_search.ticket_id_exact_search` is replaced
by `identifier_exact_search`, whose SQL now matches on `metadata_json->>'primary_id'`
with `COALESCE(..., metadata_json->>'incident_number')` so pre-refactor
chunks keep working without a re-upload. The old function remains as a
thin alias. The retrieval short-circuit emits `search_mode="identifier_exact"`
/ `"identifier_not_found"`; api.py and validator.py accept both the new
names and the legacy `ticket_id_*` spellings.

### Goal 3 — Smart follow-up enrichment
**File:** `backend/api.py`. Added `_classify_followup_intent(query)` which
classifies each turn as `self_contained` (has an identifier — any schema —
or a tracked named entity like Enterprise-N / Nebula-Corp), `cross_cutting`
(contains `across` / `all` / `every` / `each` / `common` / `patterns` /
`trends` / `compare` / `how many` / `list all` / `show all` / `rank` /
`which X had`), or `narrow_followup` (everything else). `_enrich_query_with_history`
now enriches only `narrow_followup` and logs `[followup] intent=<kind>, ...`
so regressions are visible from the log stream. Fixes the WiFi/root-cause
regression where cross-ticket aggregate questions were being biased by one
carried ticket ID from the prior turn.

### Goal 4 — Dynamic agent token budget
**Files:** `backend/config.py`, `backend/agents/orchestrator.py`,
`backend/agents/analyst.py`. Six new budget knobs: `AGENT_BUDGET_PER_STEP_TOKENS`,
`AGENT_BUDGET_PLANNER_TOKENS`, `AGENT_BUDGET_COMPOSER_TOKENS`,
`AGENT_BUDGET_CLARIFIER_TOKENS`, `AGENT_BUDGET_HARD_CAP_TOKENS`, and
`ENABLE_DYNAMIC_AGENT_BUDGET`. After the Planner returns its step list,
the orchestrator recomputes `max_total = CLARIFIER + PLANNER + N*PER_STEP + COMPOSER`
and clamps to `HARD_CAP_TOKENS` (so a runaway 50-step plan can't explode
cost). The Analyst now checks `budget.remaining > AGENT_BUDGET_COMPOSER_TOKENS`
before each step and stops early otherwise, logging `[agents] analyst stopped early
at step N/M to reserve Composer tokens`. 4-step comparison queries now get a
15,000-token envelope; the Composer always runs.

### Diagnostics & deliverables
- `[retrieve] extracted ticket_ids=…` renamed to `[retrieve] extracted identifiers=…`.
- `[ticket_id_exact] …` log prefix renamed to `[identifier_exact] …`.
- Gold-ticket ingestion log renamed from `[ingest] gold-ticket fast-path:
  N tickets, 0 Haiku calls` to `[ingest] structured fast-path: N records,
  schema=GoldTicketSchema` — same format fires for `GenericArraySchema`.
- `backend/config.py::IDENTIFIER_PATTERNS` is flagged as "config-only
  extensibility — add new schemas here without code changes" in its docstring.
- Not modified: retrieval fusion, reranker, validator's false-refusal guard
  logic, Clarifier LLM logic, `model_router` prompts, or any frontend file.
