# AICode Chatbot — Updated Architecture (Multi-Agent-First)

This README reflects the current state of the system **after** the multi-agent-first refactor. All queries now route through the agent pipeline by default; hybrid retrieval is used **inside** the agents as a per-step retrieval step.

---

## 1. Core principle — agent-first, hybrid retrieval inside

| Previous | Current |
|---|---|
| `/ask` ran hybrid RAG; agents fired only when the 5-gate complexity check passed | `/ask` routes through the agent pipeline by default; hybrid path runs only when `mode=hybrid` is explicitly requested, or as a safety-net fallback |
| Agents consumed a pre-assembled `doc_context` string for every plan step | Agents own their retrieval — each plan step triggers its own BM25 + pgvector + RRF + rerank pass via `step_retriever` |
| One model routing decision per query (`route_and_generate`) | Planner (Sonnet) + Analyst (Haiku × N steps) + Composer (Haiku) per query, with token budget + wall-clock timeout |

The hybrid stack (BM25, pgvector, RRF fusion, reranker, query expansion, variant fallback, `assemble_context`, `has_sufficient_document_support`) is **preserved in full** — it is now called from two places:

1. Up front in `/ask` to populate the initial ranked-chunk list (feeds the Planner's preview).
2. Inside the Analyst, per step, through `build_step_retriever(...)` — the same `orchestrator_retrieve`, same `expand_query`, same `assemble_context`.

Nothing was deleted. Every previous module is still importable with the same signatures.

---

## 2. End-to-end request flow (`/ask`)

```
POST /ask { q, session_id?, mode?, stage? }
  │
  ├── 1. Save user message                       (services + repositories)
  │
  ├── 2. Trivial short-circuit                   (routing/trivial_detector)
  │        "hi", "thanks", "bye" → canned reply, no LLM
  │
  ├── 3. Input guard                             (routing/input_guard)
  │        prompt-injection / oversize / unsafe phrasing → canned refusal
  │
  ├── 4. Active file ids + answer cache lookup   (routing/answer_cache)
  │        hit → return cached payload
  │
  ├── 5. Query expansion                         (retrieval/query_expansion)
  │        acronyms, normalization, variants
  │
  ├── 6. Embed (Titan) + hybrid retrieval        (vector + retrieval/orchestrator)
  │        BM25 + pgvector + RRF + rerank
  │        variant-query fallback on weak support
  │
  ├── 7. has_sufficient_document_support         (retrieval)
  │        no support → canned "not in documents" reply
  │
  ├── 8. Chunk limiter                           (routing/chunk_limiter)
  │        cap to DEFAULT_MAX_CHUNKS (=8), preserves order
  │
  ├── 9. Intent detection                        (routing/intent_detector)
  │        not_resolved / check_kb / show_sop / search_docs / what_next
  │
  ├── 10. Stage enforcement                      (routing/stage_enforcer)
  │         stage=tickets → filter ranked chunks to ticket sources
  │         stage=docs + repeat "not resolved" → escalate to multi_agent
  │
  ├── 11. assemble_context                       (retrieval)
  │
  ├── 12. Complexity classifier                  (routing/complexity_classifier)
  │         tier + score, purely heuristic, no LLM
  │
  ├── 13. Mode resolution                        (agents/mode_selector)
  │         auto        → agents  (agent-first default)
  │         hybrid      → standard RAG (explicit override)
  │         multi_agent → agents (forced)
  │         intent + stage may also escalate auto → multi_agent
  │
  ├── 14. Pipeline execution
  │         if should_agent:
  │             build_step_retriever(...)        (agents/step_retriever)
  │             Planner  (Sonnet) → plan steps   (agents/planner)
  │             Analyst  (Haiku × N) per-step
  │                   ├── step_retriever(step)   ← hybrid retrieval per step
  │                   └── Haiku findings
  │             Composer (Haiku) → final answer  (agents/composer)
  │             empty-answer safety net → route_and_generate (hybrid)
  │         else:
  │             route_and_generate(...)          (routing/model_router)
  │
  ├── 15. validate_answer                        (validation)
  │
  ├── 16. Evidence check                         (routing/evidence_checker)
  │         short_answer / low_overlap / low_confidence / no_sources
  │
  ├── 17. Save assistant message
  ├── 18. answer_cache.put(...)                  (routing/answer_cache)
  └── 19. Return AnswerResponse { answer, sources, confidence, processing_time_ms, session_id, context_stats }
```

All guards (`trivial`, `input_guard`, `stage`, `evidence`, cache get/put) **fail-open**: any internal error logs a warning and the request continues.

---

## 3. Module map

### `backend/agents/` — the new default path
| File | Purpose |
|---|---|
| `orchestrator.py` | 5-gate `should_escalate_to_agents` (kept for telemetry) + `run_agent_pipeline` — Planner → Analyst → Composer |
| `base.py` | `TokenBudget`, `AgentStepResult`, `AgentPipelineResult`, `invoke_llm` (routes Mistral / Haiku / Sonnet via Bedrock) |
| `planner.py` | Sonnet — decomposes query into ≤ `AGENT_MAX_STEPS` JSON steps |
| `analyst.py` | Haiku × steps — per-step findings; **consumes `step_retriever_fn`** for per-step hybrid retrieval |
| `composer.py` | Haiku — synthesizes findings into a bullet-point answer |
| `step_retriever.py` | **(new)** builds a closure wrapping `expand_query` + `safe_embed` + `orchestrator_retrieve` + `assemble_context` so the Analyst fetches fresh, step-specific chunks |
| `mode_selector.py` | `resolve_mode`: `auto` → agents, `hybrid` → standard RAG, `multi_agent` → forced agents. Also calls the classical 5-gate check for telemetry |

### `backend/routing/` — guards, cache, routing
| File | Purpose |
|---|---|
| `trivial_detector.py` | canned replies for greetings / thanks / ack / bye |
| `input_guard.py` | prompt-injection + oversize + unsafe phrasing filter; fail-open |
| `intent_detector.py` | explicit user intents (`not_resolved`, `check_kb`, `show_sop`, `search_docs`, `what_next`) with optional mode-upgrade hints |
| `stage_enforcer.py` | per-stage policy (`tickets` filters chunks to ticket sources; `docs` tracks repeated `"not resolved"` per session and escalates to `multi_agent` at threshold) |
| `chunk_limiter.py` | caps ranked chunks to `DEFAULT_MAX_CHUNKS=8` pre-context |
| `answer_cache.py` | thread-safe LRU + TTL cache keyed on `(normalized_query, owner_id, sorted_file_ids)`; fail-safe gets/puts |
| `complexity_classifier.py` | heuristic `tier + score`, no LLM |
| `model_router.py` | `route_and_generate` for the hybrid override path (Haiku default, Sonnet on complex) |
| `evidence_checker.py` | post-generation weak-evidence detection; metadata only, does not mutate the answer |
| `context_builder.py` | hybrid-path prompt builder |

### `backend/retrieval/` — hybrid stack (shared by both paths)
- `orchestrator.retrieve` (exported as `orchestrator_retrieve`) — BM25 + pgvector + RRF + rerank
- `query_expansion.expand_query` — acronyms, normalization, variant queries
- `fusion`, `reranker`, `keyword_search`, `query_classifier`
- `assemble_context`, `has_sufficient_document_support`

These modules are called from **both** `/ask` (up-front retrieval) and the `step_retriever` (per-step retrieval inside agents).

### `backend/vector/` + `vector_store.py`
Titan embeddings (`BEDROCK_EMBED_MODEL`) via `safe_embed`; pgvector read/write.

### `backend/ingestion/`
Document parsing, chunking, metadata extraction, index writes. Runs once per upload; feeds both retrieval paths equally.

### `backend/metadata/`
Per-chunk metadata (file_type, source_type, section, page, owner, ticket-vs-doc). Already carried in every ranked-chunk tuple and consulted by `stage_enforcer._looks_like_ticket`. Not yet surfaced inside agent prompt headers — a planned enhancement.

### `backend/validation/`
`validate_answer` — version-warning detection, coherence, confidence scoring, optional rewrite. Runs for both paths.

### `backend/services/` + `backend/repositories/` + `backend/db/`
Session persistence, file access control, Clerk auth glue, feedback, migrations. Unchanged.

### `backend/agents/README.md`
Original multi-agent design doc — still accurate for Planner/Analyst/Composer internals; this file layers the agent-first flow and step-retriever on top.

---

## 4. Request model

```python
class Question(BaseModel):
    q: str                                         # the user query
    session_id: Optional[str] = None               # for continuity + cache + unresolved counter
    mode: Optional[str] = "auto"                   # 'auto' (default → agents) | 'hybrid' | 'multi_agent'
    stage: Optional[str] = "general"               # 'general' | 'tickets' | 'docs'
```

`mode` and `stage` are both additive — older clients that omit them keep working.

---

## 5. Response — `context_stats` fields

All new metadata lives in `context_stats`. Top-level response fields (`answer`, `sources`, `confidence`, `processing_time_ms`, `session_id`) are unchanged for frontend compatibility.

Key fields introduced in each phase:

| Phase | Fields |
|---|---|
| Mode selector | `mode_requested`, `mode_effective` |
| Intent detector | `intent`, `intent_matched`, `intent_phrase`, `intent_upgraded_mode` |
| Trivial / cache / chunk limiter | `trivial_short_circuit`, `trivial_kind`, `cache_hit`, `cache_key`, `cache_stored`, `chunk_original_count`, `chunk_limited_count`, `chunk_limit_applied`, `chunk_max_allowed` |
| Input guard / stage / evidence | `input_guard_flagged`, `input_guard_reason`, `input_guard_phrase`, `stage`, `stage_filter_applied`, `stage_original_chunks`, `stage_filtered_chunks`, `stage_enforced_mode`, `stage_escalate_reason`, `unresolved_count`, `weak_evidence`, `weak_reasons`, `evidence_answer_chars`, `evidence_overlap_tokens`, `evidence_source_count` |
| Agent-first + step retriever | `hybrid_path_used`, `step_retrieval_applied_count`, `step_retrieval_total_steps`, `agent_mode`, `agent_reason`, `agent_steps`, `agent_tokens`, `agent_ms` |

---

## 6. Models (Bedrock)

```
BEDROCK_EMBED_MODEL   = "amazon.titan-embed-text-v2:0"
BEDROCK_LLM_MODEL     = "mistral.mistral-7b-instruct-v0:2"   # internal tools only
BEDROCK_HAIKU_MODEL   = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_SONNET_MODEL  = "us.anthropic.claude-sonnet-4-6"
```

Usage:
- **Agent-first path (default):** Sonnet (Planner) + Haiku × steps (Analyst) + Haiku (Composer).
- **Hybrid override path (`mode=hybrid`):** Haiku by default; Sonnet when complexity tier is `"complex"`.
- **Embeddings:** Titan v2 (same for both paths).
- **Mistral:** retained as a last-resort fallback in `model_router._invoke_mistral`.

---

## 7. Cost + safety controls

| Control | Value | Purpose |
|---|---|---|
| `AGENT_MAX_TOTAL_TOKENS` | 8,000 | Hard ceiling across Planner + Analyst + Composer per request |
| `AGENT_TIMEOUT_SECONDS` | 45 | Wall-clock timeout for the whole agent pipeline |
| `AGENT_MAX_STEPS` | 4 | Max plan steps the Planner can emit |
| `DEFAULT_MAX_CHUNKS` | 8 | Chunk limiter cap before context / agents |
| Answer cache | TTL 600 s, LRU 500 | Skips retrieval + LLM on repeats |
| Trivial short-circuit | ≤ 40 chars | Skips retrieval + LLM for chit-chat |
| Input guard | regex patterns | Blocks prompt-injection and unsafe phrasing |
| Evidence checker | per-request | Flags weak answers (metadata only) |
| Fail-open semantics | every guard | Any internal error continues the request |

---

## 8. Running

```bash
cd AICode_Chatbot
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Environment: `.env` with AWS Bedrock creds + region, Postgres URL, Clerk keys (optional).

Quick smoke:
```bash
python -c "from backend.agents.step_retriever import build_step_retriever; print('ok')"
curl -sX POST http://localhost:8000/ask -H 'Content-Type: application/json' -d '{"q":"what is an alert threshold?"}'
```

Expected for the curl above: `context_stats.agent_mode == true`, `context_stats.hybrid_path_used == false`, `context_stats.step_retrieval_total_steps >= 1`.

---

## 9. Non-goals / deliberate omissions

- No code deletions. `should_escalate_to_agents`, `route_and_generate`, `_invoke_mistral`, the 5-gate escalation patterns, and the classical hybrid path are all still live and reachable.
- `context_stats` extensions only; no breaking changes to top-level response fields.
- No DB migrations required by any phase so far. Per-session unresolved counters and the answer cache live in-memory — persistence is a planned follow-up.

---

## 10. Known next-step wins (not yet implemented)

1. Inject chunk metadata (`source`, `section`, `page`, `file_type`) directly into agent prompt headers for stronger citations.
2. Step-finding cache (DB-backed) so repeated agent runs reuse prior Analyst output.
3. Persist `stage_enforcer` unresolved counters and the reasoning trace (`AgentPipelineResult.reasoning_summary`) in `repositories/`.
4. Per-step `validate_answer` (currently only the Composer output is validated).
5. Feedback-aware Planner — use prior thumbs-down signals to re-plan differently.


How to run: backend

 python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload 2>&1 | Tee-Object logs\backend.log  

 Forntend:
 Cd frontend

 npm install

 npm run build

 nmp start

 db
  psql -h logiq-db.c6vow688a3co.us-east-1.rds.amazonaws.com -U postgres -d logiq_dev -p 5432