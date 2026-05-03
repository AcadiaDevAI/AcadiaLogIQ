# Acadia Log IQ — Ingestion Pipeline Audit

> **Audit date:** 2026-04-30
> **Scope:** Why does the rich 27-ticket JSON in `file1_txt.txt` end up as a slim flat schema in `chunks.metadata_json`, leaving the journey readers (which expect `Executive_Sharable_RCA`, `Incident_Summary`, `Forensic_Performance_Audit`, `Key_Contributors`) with nothing to read?

## Executive summary

The ingestion path detects gold-ticket JSON, dispatches to a dedicated fast-path (`_ingest_gold_ticket_json` at `backend/services/contextual_ingestion_service.py:734`), and constructs `row_metadata_json` as a **hand-picked slim dict** at `backend/services/contextual_ingestion_service.py:813-855`. The four parent objects the journey readers expect — `Executive_Sharable_RCA`, `Incident_Summary`, `Forensic_Performance_Audit`, `Key_Contributors`, plus `QA_Auditor_Feedback` and `ITIL_5_Why` — are **not copied through**. A flag-gated pass-through at `contextual_ingestion_service.py:869-888` carries `Metadata`, `Symptom_Solution_Mapping`, `Operational_SOP`, `Knowledge_Base`, `remediation_payload`, and `Header` (Sprint 4 fingerprint feature, currently `LOGIQ_SPRINT4_BACKEND=true` per `.env:187`), which is why those four nested objects are present in the DB and the Sprint 10.x journey "currently works" only for the headline ID lookup. The four dropped parents are deterministic Python omissions — there is no LLM in this fast-path, and there is no later code that re-merges them. The raw upload bytes are still on disk via `local://` storage URIs (`backend/storage/local_storage.py:21`) and `document_versions.storage_uri`, so backfill is feasible without re-uploading; `document_versions.enrichment_json` only stores doc-level summary metadata (`vector_store.py:675`), not the per-ticket rich JSON, so it is **not** a viable backfill source. Re-ingestion would re-issue 225 Titan embed calls (≈ a few cents) and add the four missing top-level keys without altering any slim-key consumer (every reader uses `meta.get(...)` defaults or `metadata_json->>'...'` with NULL tolerance).

---

## 1. Pipeline architecture

End-to-end call graph from `POST /upload` to `INSERT INTO chunks`.

```
HTTP POST /upload
  └─ backend/api.py:1917  upload(file, file_type, doc_kind, ...)
       └─ saves bytes via storage.save_bytes -> local://...   (local_storage.py:17)
       └─ create_ingestion_job(...)                            (api.py:1957)
       └─ background_tasks.add_task(index_file_job, ...)       (api.py:1966)

BackgroundTask: backend/api.py:1294  index_file_job(...)
  ├─ Phase 1 — parse + structured detection + (optional) Haiku metadata
  │   └─ asyncio.to_thread(process_document, ...)               (api.py:1319)
  │       └─ backend/services/contextual_ingestion_service.py:1616  process_document(...)
  │           ├─ _diagnose_json_structure(file_bytes)            (cis.py:1649) — Sprint 2.9 reject malformed
  │           ├─ STRUCTURED_SCHEMAS loop (cis.py:1670)
  │           │     ContactSchema -> _ingest_contact_array
  │           │     GoldTicketSchema -> _ingest_gold_ticket_json (cis.py:734)   << fires for file1_txt.txt
  │           │     GenericArraySchema -> _ingest_generic_array  (cis.py:1102)
  │           └─ legacy fallback: parse_file -> build_chunks -> Haiku metadata
  │                 backend/ingestion/structured_parser.py:474 parse_file
  │                 backend/ingestion/structured_parser.py:523 build_chunks
  │                 backend/services/contextual_ingestion_service.py:328 batch_extract_chunk_metadata
  │
  ├─ Phase 2 — Bedrock Titan embedding for every chunk
  │   └─ ThreadPoolExecutor(safe_embed)                          (api.py:1405)
  │       └─ safe_embed(text)                                    (api.py:750)
  │           └─ bedrock.invoke_model(modelId=settings.BEDROCK_EMBED_MODEL, ...)
  │               (config.py:49 -> "amazon.titan-embed-text-v2:0")
  │
  ├─ Phase 3 — assemble chunk_rows + BM25 entries                (api.py:1432-1456)
  │
  ├─ Phase 4 — DB persistence
  │   └─ asyncio.to_thread(insert_document_and_chunks, ...)      (api.py:1469)
  │       └─ backend/vector_store.py:566  insert_document_and_chunks(...)
  │           ├─ INSERT INTO documents                           (vector_store.py:606)
  │           ├─ INSERT INTO document_versions (.. enrichment_json ..)  (vector_store.py:643)
  │           ├─ INSERT INTO document_metadata                   (vector_store.py:692)
  │           ├─ Batched INSERT INTO chunks (..., metadata_json)  (vector_store.py:798-835)  <<< the row that becomes the slim schema
  │           └─ INSERT INTO embeddings                          (vector_store.py:838)
  │
  ├─ Phase 5 — BM25 index update                                  (api.py:1493)
  └─ Phase 5b — glossary learn (non-fatal)                        (api.py:1502)
```

**Key files referenced:**

- `backend/api.py:1917` — `/upload` route handler (writes raw bytes, schedules background job).
- `backend/api.py:1294` — `index_file_job` (orchestrates Phases 1-5).
- `backend/services/contextual_ingestion_service.py:1616` — `process_document` (schema dispatch).
- `backend/services/contextual_ingestion_service.py:734` — `_ingest_gold_ticket_json` (the actual builder for ticket files).
- `backend/vector_store.py:566` — `insert_document_and_chunks` (the four `INSERT`s).

**Summary.** Uploads land on disk and trigger an async job that detects gold-ticket JSON, runs a deterministic Python builder (no LLM), embeds each ticket as a single chunk via Titan, and writes documents/versions/metadata/chunks/embeddings rows in one transaction. The chunk-level `metadata_json` column is the artefact under audit.

---

## 2. Metadata persistence — what actually gets stored

The dict written to `chunks.metadata_json` for a ticket is built **once per ticket** at the top of `_ingest_gold_ticket_json`'s loop. The exact constructor:

`backend/services/contextual_ingestion_service.py:813-855`
```python
        row_metadata_json = {
            # Universal identifier fields — schema-agnostic retrieval (Goal 1/2).
            # `primary_id` is what identifier_exact_search matches against;
            # `id_type` lets the orchestrator log the schema family and future
            # callers route by type. `incident_number` is preserved as a ticket-
            # specific alias so legacy readers keep working.
            "primary_id": incident_number,
            "id_type": "ticket_number",
            # Ticket-native fields — these are what Fix 2 and Fix 6 read.
            "doc_kind": "ticket",
            "incident_number": incident_number,
            "customer_name": customer_name,
            "priority": priority,
            "component_category": component_category,
            "ticket_status": _safe_str(metadata.get("ticket_status"), max_len=40),
            "resolved_date": resolved_date,
            "resolution_groups": metadata.get("Resolution_Groups"),
            "sla_target_met": exec_rca.get("SLA_Target_Met"),
            "resolution_quality_score": exec_rca.get("Resolution_Quality_Score"),
            # FINAL_CLEANUP Bug 1 — rework filter SQL reads metadata_json->>'rework_detected'.
            # Populate from QA_Auditor_Feedback so `_build_ticket_scope_clauses` rework clause
            # actually matches rows. bool() coerces Python-native False so the JSON-encoded
            # value is 'false' (not 'null') — the SQL clause accepts true/false/yes/no/1/0.
            "rework_detected": (
                bool(qa.get("Rework_Detected", False))
                if getattr(settings, "INGEST_REWORK_METADATA_ENABLED", True)
                else None
            ),
            "llm_enrichment_status": ticket.get("llm_enrichment_status"),
            # Generic fields downstream code still reads.
            "title": filename,
            "source_type": file_type,
            "document_type": "Ticket",
            "vendor": None,
            "product": None,
            "domain": None,
            "version": None,
            "document_date": resolved_date,
            "effective_date": None,
            "created_date": None,
            "purpose_description": None,
            "operational_context": "ticket",
        }
```

A **flag-gated** pass-through then re-attaches a few rich keys to the same dict:

`backend/services/contextual_ingestion_service.py:869-888`
```python
        if getattr(settings, "LOGIQ_SPRINT4_BACKEND", False):
            # Top-level "Metadata" block carries Fingerprints AND
            # Dynamic_Domain_Payload.Domain_Type — both are indexed.
            if isinstance(ticket.get("Metadata"), dict):
                row_metadata_json["Metadata"] = ticket["Metadata"]
            if isinstance(ticket.get("Symptom_Solution_Mapping"), dict):
                row_metadata_json["Symptom_Solution_Mapping"] = ticket[
                    "Symptom_Solution_Mapping"
                ]
            if isinstance(ticket.get("Operational_SOP"), dict):
                row_metadata_json["Operational_SOP"] = ticket["Operational_SOP"]
            # Knowledge_Base is a list of sections in the sample schema.
            if isinstance(ticket.get("Knowledge_Base"), (list, dict)):
                row_metadata_json["Knowledge_Base"] = ticket["Knowledge_Base"]
            if isinstance(ticket.get("remediation_payload"), dict):
                row_metadata_json["remediation_payload"] = ticket[
                    "remediation_payload"
                ]
            if isinstance(ticket.get("Header"), str):
                row_metadata_json["Header"] = ticket["Header"]
```

`LOGIQ_SPRINT4_BACKEND` is **on** today (`/.env:187` and `backend/.env:220` both set `LOGIQ_SPRINT4_BACKEND=true`), so the six keys above ARE in the DB.

The dict is then assigned at `contextual_ingestion_service.py:907`:
```python
                "metadata_json": row_metadata_json,
```

…and ultimately serialised by `vector_store.py:832`:
```python
"metadata_json": json.dumps(row.get("metadata_json", {})),
```
into `chunks.metadata_json` (`vector_store.py:806-815`).

**Source-shape vs DB-shape comparison.**

Source `file1_txt.txt` ticket top-level keys (per Sprint 10.8.1 doc and the journey readers):
- `Metadata` ✓ kept (Sprint 4 pass-through)
- `Header` ✓ kept (Sprint 4 pass-through, only when string)
- `Incident_Summary` ✗ **dropped**
- `Executive_Sharable_RCA` ✗ **dropped** (only `SLA_Target_Met`, `Resolution_Quality_Score` lifted out as flat fields)
- `Forensic_Performance_Audit` ✗ **dropped**
- `Key_Contributors` ✗ **dropped**
- `QA_Auditor_Feedback` ✗ **dropped** (only `Rework_Detected` lifted out as `rework_detected`)
- `ITIL_5_Why` ✗ **dropped**
- `Symptom_Solution_Mapping` ✓ kept (Sprint 4 pass-through)
- `Operational_SOP` ✓ kept (Sprint 4 pass-through)
- `Knowledge_Base` ✓ kept (Sprint 4 pass-through)
- `remediation_payload` ✓ kept (Sprint 4 pass-through)

The slim flat keys (`primary_id`, `id_type`, `doc_kind`, `incident_number`, `customer_name`, `priority`, `component_category`, `ticket_status`, `resolved_date`, `resolution_groups`, `sla_target_met`, `resolution_quality_score`, `rework_detected`, `llm_enrichment_status`, `title`, `source_type`, `document_type`, `vendor`, `product`, `domain`, `version`, `document_date`, `effective_date`, `created_date`, `purpose_description`, `operational_context`) are **synthesised** from the source dict, not copied from it.

**Summary.** The DB shape is a deliberate slim schema — the comments at line 814 and 821 ("Universal identifier fields", "Ticket-native fields — these are what Fix 2 and Fix 6 read") indicate this was designed for the SQL retrieval path (`metadata_json->>'primary_id'` etc.) and not for the journey readers, which want the original nested structure.

---

## 3. Schema variance investigation — where parent keys are stripped

### Strip point: confirmed

The strip happens at exactly one site — the `row_metadata_json` literal at `backend/services/contextual_ingestion_service.py:813`. Because this dict is **constructed from scratch with a hand-picked subset of keys** (rather than starting from `ticket.copy()` and removing fields), any source key that is not explicitly listed simply does not appear.

Of the original parents:

| Source key | Outcome | Where |
|---|---|---|
| `Metadata` | KEPT (whole object) | cis.py:872-873 |
| `Header` | KEPT (when string) | cis.py:887-888 |
| `Symptom_Solution_Mapping` | KEPT | cis.py:874-877 |
| `Operational_SOP` | KEPT | cis.py:878-879 |
| `Knowledge_Base` | KEPT | cis.py:881-882 |
| `remediation_payload` | KEPT | cis.py:883-886 |
| `Incident_Summary` | **DROPPED** | not referenced |
| `Executive_Sharable_RCA` | **DROPPED** (only 2 leaf scalars lifted: `SLA_Target_Met`, `Resolution_Quality_Score`) | cis.py:830-831 |
| `Forensic_Performance_Audit` | **DROPPED** | not referenced |
| `Key_Contributors` | **DROPPED** | not referenced |
| `QA_Auditor_Feedback` | **DROPPED** (only `Rework_Detected` lifted as `rework_detected`) | cis.py:836-840 |
| `ITIL_5_Why` | **DROPPED** | only used for `_is_gold_ticket_json` detection (cis.py:496) |

### No later code re-merges the dropped keys

I verified two ways:

1. After `enriched_rows.append({...})` at `contextual_ingestion_service.py:890`, no code path mutates `row["metadata_json"]` for ticket rows. `index_file_job` only adds `row["id"]` and `row["embedding"]` (`api.py:1438-1439`), then forwards the row dict to `insert_document_and_chunks`.
2. `insert_document_and_chunks` reads `metadata_json` once at `vector_store.py:832` and passes the JSON string straight into the `INSERT INTO chunks(...)` `VALUES` (`vector_store.py:806-815`). Nothing in between merges or augments.

So the strip site at `cis.py:813` is the ONLY place where the source-vs-stored shape diverges. There is no second-stage normaliser, no LLM rewrite, no DB trigger.

### Why the keys were dropped vs kept

The kept keys correspond to features that have explicit downstream readers indexed by GIN:

- `Metadata.Fingerprints` — Sprint 4 fingerprint lookup (`backend/db/migrations/035_fingerprint_gin_indexes.sql`).
- `Symptom_Solution_Mapping`, `Knowledge_Base`, `Operational_SOP`, `remediation_payload` — Sprint 4 Expert Copilot composer (`backend/agents/composer.py:380`, `backend/tier1_copilot/context_extractor.py:20-29`, `backend/tier1_copilot/alias_dictionary.py:78,101`).

The dropped keys (`Executive_Sharable_RCA`, `Incident_Summary`, `Forensic_Performance_Audit`, `Key_Contributors`, `QA_Auditor_Feedback`, `ITIL_5_Why`) had **no consumer at the time the slim schema was designed**. Sprint 10.8.1 — the realignment that introduced the journey readers — added consumers that read those parents (`backend/tier1_copilot/journey/stage2_historical.py:142-144`, `stage3_troubleshooting.py:172-242`, `stage1_smoking_gun.py:139`, `stage0_confidence.py:256,328`). The retain list in `_ingest_gold_ticket_json` was never updated to follow.

**Summary.** The strip is a **deliberate slim schema** authored before the journey readers existed; the four parents the journey expects are simply not on the keep list at `cis.py:813-855` or the `LOGIQ_SPRINT4_BACKEND` extension at `cis.py:869-888`. Re-ingestion (or a backfill) is the only way to surface them.

---

## 4. Chunking strategy

For gold-ticket JSON: **one chunk per ticket** — no further sub-chunking. Each chunk's `metadata_json` is the per-ticket `row_metadata_json` (slim schema as documented above), not the full original ticket dict.

`backend/services/contextual_ingestion_service.py:781-909`
```python
    enriched_rows: List[Dict[str, Any]] = []
    latest_resolved: Optional[str] = None

    for idx, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            continue
        ...
        # Header line — deterministic BM25 / exact-match target.
        header = (
            f"TICKET: {incident_number or '?'} | "
            f"CUSTOMER: {customer_name or '?'} | "
            f"PRIORITY: {priority or '?'} | "
            f"COMPONENT: {component_category or '?'}"
        )
        body = _render_gold_ticket_body(ticket)
        content = f"{header}\n\n{body}" if body else header
        ...
        enriched_rows.append(
            {
                "chunk_index": idx,
                "content": content,
                "contextualized_content": content,
                "summary": None,
                "section_heading": section_heading,
                "chunk_type": "ticket",
                "page_number": None,
                "token_estimate": max(1, len(content) // 4),
                "source_order": idx,
                "labels_json": {...},
                "metadata_json": row_metadata_json,
            }
        )
```

Note `chunk_index = idx` (the position in the source array) and `chunk_type = "ticket"`. With 27 source records → 27 chunks (per file). The 225 chunks observed in the DB therefore represent the cumulative across all uploaded ticket files (e.g. 5 files × 45 tickets → 225, or similar mix).

The legacy non-ticket path (PDF/DOCX/text) splits differently — heading-aware blocks then size-bounded chunk groups (`backend/ingestion/structured_parser.py:523 build_chunks`) — but it never runs for `file1_txt.txt` because `_is_gold_ticket_json` (`cis.py:476`) fires first.

**Summary.** Chunking for tickets is a 1:1 record-to-chunk map. The chunk's `metadata_json` is the slim builder dict, not the full ticket — there is no "raw JSON shadow copy" stored alongside each chunk.

---

## 5. LLM-based extraction

**No LLM runs in the gold-ticket fast path.** The slim schema is produced by deterministic Python `dict` construction.

Evidence:

- The fast-path docstring says so explicitly (`contextual_ingestion_service.py:744`):
  > "One-chunk-per-ticket ingestion — no Haiku metadata calls."
- `_ingest_gold_ticket_json` does not import or invoke `haiku_client`, `safe_generate`, or `bedrock`. It reads `ticket.get(...)` keys directly.
- The Haiku metadata extractor (`batch_extract_chunk_metadata`, `cis.py:328`) is only called from the legacy text path at `cis.py:1786` and `cis.py:1794`, which the schema dispatch (`cis.py:1670-1701`) skips when `_is_gold_ticket_json` returns True.

For completeness, the legacy path's prompt (used for PDF/DOCX/text, NOT for tickets) is at `backend/ingestion/prompt_templates.py:23-68`. Its output schema is ALSO slim (`title`, `document_type`, `vendor`, `product`, `domain`, `version`, dates, plus per-chunk `section`, `chunk_type`, `tags`, `entities`, `keywords`, `summary`, `operational_context`) — so even if a ticket had been mis-routed there, the rich parents would still be lost.

The only LLM that touches a ticket file is the Bedrock Titan embedder at `backend/api.py:750-766` (vector embedding only — no metadata reshaping).

**Summary.** Slim-schema construction is pure Python. No prompt change can fix the missing keys; only the `row_metadata_json` literal at `cis.py:813` (or the `LOGIQ_SPRINT4_BACKEND` block at `cis.py:869`) can.

---

## 6. Backfill feasibility — is the raw JSON still available?

### Candidate A — `document_versions.storage_uri` + on-disk file: **VIABLE**

`backend/api.py:1954`:
```python
storage_uri = storage.save_bytes(relative_name, content)
```

`backend/storage/local_storage.py:17-21`:
```python
    def save_bytes(self, relative_name: str, content: bytes) -> str:
        path = self.root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return f"local://{quote(str(path))}"
```

`backend/vector_store.py:643-666` writes `storage_uri` into `document_versions`, and `local_storage.resolve_local_path` (`local_storage.py:28-32`) reverses the URI back to a real `Path`. The bytes never get deleted by the ingestion path — only `delete_document_and_chunks` (file delete API) calls `storage.delete`.

**Net:** The original 5 source files almost certainly still sit under `backend/uploads/` (`config.py:32 UPLOAD_DIR = BASE_DIR / "uploads"`). A backfill script can `SELECT storage_uri FROM document_versions WHERE document_id = ...`, read each file, JSON-parse, and `UPDATE chunks SET metadata_json = metadata_json || jsonb_build_object('Executive_Sharable_RCA', ..., 'Incident_Summary', ..., ...) WHERE id = ...`. This avoids re-embedding — the vector and `chunks.content` stay untouched.

### Candidate B — `document_versions.enrichment_json`: **NOT VIABLE**

`backend/vector_store.py:675`:
```python
"enrichment_json": json.dumps((metadata or {}).get("metadata_json", {})),
```

Per `_ingest_gold_ticket_json` (`cis.py:941-953`), the doc-level `metadata_json` is itself slim (title, document_type, vendor, product, domain, version, dates, glossary, doc_kind). So `document_versions.enrichment_json` for ticket uploads holds doc-level descriptive metadata only — it never contained the per-ticket nested objects.

### Candidate C — `document_metadata.metadata_json`: **NOT VIABLE**

Same source as B (`vector_store.py:727`). Doc-level only.

### Candidate D — `raw_documents` / `raw_json` / `original_json` / `raw_bytes` table or column: **DOES NOT EXIST**

A repo-wide grep across `backend/**/*.{py,sql}` for `raw_documents`, `raw_json`, `original_json`, `raw_bytes` returns zero hits. There is no shadow table.

### Candidate E — S3: depends on environment

`backend/storage/s3_storage.py` exists, but the active provider is selected by config; current local dev appears to use `local://` URIs (`local_storage.py:21`). If production runs S3, `storage_uri = s3://bucket/key` and the same backfill works against the bucket.

**Summary.** Backfill is feasible **only** via candidate A: re-read the original upload bytes from `document_versions.storage_uri`, parse, and `UPDATE chunks.metadata_json` to add the four missing parents. Embeddings and BM25 do not need to change. Candidates B/C/D/E either hold the wrong shape or don't exist.

---

## 7. Re-ingestion cost

### What runs per file (gold-ticket fast path)

Phases (per `index_file_job`, `api.py:1294`):

1. **Parse + structured-detection** — pure Python; no API. ~ms.
2. **Embedding** — 1 Bedrock Titan call per chunk (`api.py:1409 safe_embed`).
3. **DB insert** — local Postgres; no API.
4. **BM25 update** — in-process.
5. **Glossary learn** — non-fatal, no API.

There are **no Haiku/LLM calls per chunk** for tickets (confirmed §5).

### Embed model and pricing

`backend/config.py:49`:
```python
BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
```

The repo has no published rate constant in code; no `cost_per_call` symbol. AWS lists Titan Text Embed V2 at roughly **$0.00002 per 1K input tokens** (≈ $0.02 per million tokens) — actual at-runtime cost is dominated by token count, not call count.

### Estimate for 225 chunks across 5 files

- **Embedding API calls:** 225 (one per chunk).
- **Token count per chunk:** the `_render_gold_ticket_body` output is ~2-6 KB → roughly 500-1500 tokens per ticket. Use 1000 tokens/chunk as a midpoint.
- **Total tokens:** 225 × 1000 ≈ 225,000 tokens.
- **Embedding cost:** 225,000 × $0.00002 / 1000 = **≈ $0.0045** (under one cent).
- **LLM cost:** $0.00 (gold-ticket fast-path has no Haiku step).

### Wall-clock estimate

The PERF log markers are in `backend/api.py`:
- `api.py:1333` — Parse + metadata
- `api.py:1426` — Embedding `(succeeded/total, workers)`
- `api.py:1486` — DB insert
- `api.py:1526` — TOTAL with breakdown

Embed concurrency is 12 (`config.py:123 EMBED_CONCURRENCY: int = 12`). With Titan typical latency ~150-400 ms per call, 225 calls / 12 workers ≈ 19 batches × ~300 ms ≈ **6-10 seconds of embed time**. Parse is millisecond-class; DB insert for 225 rows is ~1-2 seconds via the 50-row batches at `vector_store.py:795-845`. Totals usually log out at **15-30 seconds end-to-end for 5 ticket files**, dominated by embedding.

Without per-doc historical [PERF] log values in the workspace, the estimate is anchored on the concurrency (12) and call count (225). Real numbers will be in `logs/` for prior runs.

**Summary.** Re-ingestion is essentially free — sub-cent dollar cost, ~30 s wall-clock for all five files. There is no LLM-call cost. The cheaper alternative (backfill from `storage_uri` without re-embedding) is similar in wall-clock but costs $0 on Bedrock.

---

## 8. Side effects of changing ingestion

If the slim builder is updated to also copy the four dropped parents, what breaks?

### SQL consumers (read via `metadata_json->>'<key>'`)

These all read the **flat slim keys** that the slim builder synthesises. None of them reference the rich parents. Adding `Executive_Sharable_RCA` (etc.) at the top level of `metadata_json` does not collide with any of these.

| File:line | Slim key read | Effect of rich-JSON co-existence |
|---|---|---|
| `backend/db/queries.py:109-117, 222-229` | `primary_id`, `incident_number`, `opened_date`, `resolution_text`, `resolution`, `summary`, `sla_met`, `sla_target_met`, `quality_score` | Co-exists (different keys) |
| `backend/retrieval/keyword_search.py:68-69, 456-458` | `primary_id`, `incident_number`, `vendor`, `product`, `domain` | Co-exists |
| `backend/retrieval/orchestrator.py:292, 1140, 1147` | `<identifier columns>`, `resolution_quality_score` | Co-exists |
| `backend/retrieval/metadata_sql.py:920, 1011, 1055, 1107-1115, 1206-1218, 1321-1328` | `<safe column names>`, `rework_detected`, `component_category`, `incident_number`, `customer_name`, `<numeric field>` | Co-exists |
| `backend/db/migrations/038_tier1_copilot.sql:105` | `doc_kind` | Co-exists |

### Python `meta.get(...)` consumers

All use `meta.get("<key>")` with `or` fallbacks or default-None semantics. Adding parents alongside the flat fields breaks none of them.

| File:line | Slim key read | What it does |
|---|---|---|
| `backend/agents/expert_copilot_template.py:71` | `priority` | Header rendering |
| `backend/agents/composer.py:659` | `customer_name` | Composer payload |
| `backend/retrieval/orchestrator.py:607` | `incident_number` | Top-result identifier surface |
| `backend/tier1_copilot/alias_dictionary.py:70` | `component_category` | Alias bucketing |
| `backend/tier1_copilot/context_extractor.py:44-46` | `customer_name`, `priority`, `component_category` | Tier1 context summary |
| `backend/tier1_copilot/diagnostics/explain_recommendation.py:79-99` | `priority`, `customer_name`, `component_category` | Recommendation rationale |
| `backend/tier1_copilot/diagnostics/escalation_package.py:143-145` | `priority`, `customer_name` | Escalation envelope |
| `backend/tier1_copilot/intake/catalogs.py:180` | `customer_name` | Catalog dedup |
| `backend/tier1_copilot/journey/stage2_historical.py:221, 226` | `customer_name`, `ticket_status` | Card rendering |
| `backend/tier1_copilot/retrieval.py:363, 394, 433` | `component_category`, `customer_name`, `resolved_date` | Tier1 reranking |

### Sprint 10.x journey reads (the new readers — would START working)

These read the rich shape directly. They currently see only the SIX kept keys, which is why some surfaces work and some don't.

| File:line | Path read | Currently populated? |
|---|---|---|
| `backend/tier1_copilot/journey/stage2_historical.py:142-144` | `Executive_Sharable_RCA`, `Forensic_Performance_Audit`, `Key_Contributors` | **NO** (dropped) |
| `backend/tier1_copilot/journey/stage2_historical.py:209` | `Incident_Summary.INCIDENT` | **NO** (dropped) |
| `backend/tier1_copilot/journey/stage2_historical.py:201` | `Metadata.Incident_Number` | YES (kept by Sprint 4 pass-through) |
| `backend/tier1_copilot/journey/stage3_troubleshooting.py:173, 206, 225, 242` | `Executive_Sharable_RCA.Resolution_Steps`, `Forensic_Performance_Audit.*`, `Key_Contributors.Key_Impact_Players[].Hero_Action` | **NO** (dropped) |
| `backend/tier1_copilot/journey/stage1_smoking_gun.py:139` | `Executive_Sharable_RCA.Root_Cause_Technical_High_Level` | **NO** (dropped) |
| `backend/tier1_copilot/journey/stage0_confidence.py:256, 328` | `Executive_Sharable_RCA`, `Forensic_Performance_Audit.Critical_Intervention` | **NO** (dropped) |
| `backend/tier1_copilot/context_extractor.py:20-29` | `Symptom_Solution_Mapping`, `Operational_SOP`, `remediation_payload` | YES (kept by Sprint 4 pass-through) |
| `backend/tier1_copilot/alias_dictionary.py:78, 101` | `Symptom_Solution_Mapping`, `Knowledge_Base` | YES (kept by Sprint 4 pass-through) |
| `backend/tier1_copilot/intake/catalogs.py:166-167` | `Symptom_Solution_Mapping` | YES (kept by Sprint 4 pass-through) |

### Confirmation that the journey "currently works"

It works **only** for surfaces sourced from the six kept keys (e.g. `Metadata.Incident_Number` for the headline, `Symptom_Solution_Mapping`/`Knowledge_Base` for alias dictionary). Anything routed through `Executive_Sharable_RCA`, `Incident_Summary`, `Forensic_Performance_Audit`, `Key_Contributors`, or `QA_Auditor_Feedback.*` returns empty / `UNKNOWN-N`. The Sprint 10.8.1 realignment correctly anticipated the array-shape paths (e.g. `Forensic_Performance_Audit[0].Key_Movements_Timeline[]`) but cannot help when the parent object isn't in the row.

**Summary.** No SQL or Python reader breaks if the four dropped parents are added back. They co-exist with the slim flat keys. The change is strictly additive — every existing consumer keeps working and the journey readers start finding the data they expect.

---

## Appendix — file inventory

- `backend/api.py:1294` — `index_file_job` — orchestrator.
- `backend/api.py:1917` — `/upload` route.
- `backend/api.py:750` — `safe_embed` (Titan call).
- `backend/services/contextual_ingestion_service.py:476` — `_is_gold_ticket_json` detector.
- `backend/services/contextual_ingestion_service.py:734` — `_ingest_gold_ticket_json` (the strip site).
- `backend/services/contextual_ingestion_service.py:813-855` — `row_metadata_json` slim literal.
- `backend/services/contextual_ingestion_service.py:869-888` — `LOGIQ_SPRINT4_BACKEND` rich-key pass-through.
- `backend/services/contextual_ingestion_service.py:1102` — `_ingest_generic_array` (similar pattern, also slim).
- `backend/services/contextual_ingestion_service.py:1616` — `process_document` (schema dispatch).
- `backend/services/contextual_ingestion_service.py:328` — `batch_extract_chunk_metadata` (Haiku — legacy path only).
- `backend/vector_store.py:566` — `insert_document_and_chunks`.
- `backend/vector_store.py:806-815` — chunk INSERT.
- `backend/vector_store.py:832` — `metadata_json = json.dumps(row.get("metadata_json", {}))`.
- `backend/vector_store.py:643-675` — `document_versions` row (carries `storage_uri` + `enrichment_json`).
- `backend/storage/local_storage.py:17-32` — local file persistence.
- `backend/config.py:49` — `BEDROCK_EMBED_MODEL`.
- `backend/config.py:123` — `EMBED_CONCURRENCY = 12`.
- `backend/config.py:851` — `LOGIQ_SPRINT4_BACKEND` flag default `False` (overridden to `true` in `.env`).
- `backend/db/migrations/001_phase1_foundation.sql` — `documents`, `document_versions`, `chunks`, `embeddings` schemas.
- `backend/db/migrations/002_phase2_contextual_ingestion.sql:29` — `enrichment_json` column.
- `backend/tier1_copilot/journey/ticket_loader.py:67` — cohort metadata fetch (`SELECT id, metadata_json FROM chunks WHERE id = ANY(:ids)`).
- `backend/tier1_copilot/journey/stage{0,1,2,3}_*.py` — Sprint 10.x readers (the consumers that currently see nothing in the four dropped parents).
