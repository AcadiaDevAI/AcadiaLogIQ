# Sprint 5 — Template-First Expert Copilot + Answer Cache (JSON-Only)

> **Objective:** Reduce Sprint 4 fingerprint-lookup latency from ~55s
> to ~5s (first hit) / <200ms (cached hit) by:
>
> 1. **Rendering the deterministic parts of the Expert Copilot answer
>    directly from the gold-schema JSON fields** (Phase 2 branching
>    diagnostics, Phase 3 remediation steps, RaC snippets, KB
>    citations), and
> 2. **Caching the full rendered answer per ticket** so repeat lookups
>    bypass the LLM entirely.
>
> **Scope boundaries (critical):**
> - Sprint 5 applies **ONLY** to retrievals that return a gold-schema
>   JSON ticket (`doc_kind='ticket'` AND contains `Operational_SOP`,
>   `Symptom_Solution_Mapping`, and `remediation_payload`). PDFs, Word
>   docs, KB articles, SOPs ingested as text, contact directories, and
>   any free-text documents continue to use the existing LLM-synthesis
>   pipeline unchanged.
> - Does NOT modify Sprint 3A-3E mode-aware composer voices for chat
>   queries — those continue using the full LLM path.
> - Does NOT modify the Sprint 4 `/fingerprint/lookup` URL, request
>   shape, or response shape. The optimization is internal.
> - Does NOT remove any JSON field during ingestion. Full ticket JSON
>   remains in storage untouched.
>
> **Motivation:** At 10,000+ tickets with many fingerprints queried
> repeatedly, the current Sprint 4 architecture sends 40KB of JSON per
> query to Haiku and waits for ~55s of token generation. But ~80% of
> the Expert Copilot output is deterministic — it's iterating over
> `diagnostic_logic_chunks` and `execution_steps` arrays and formatting
> them as bullet lists. The LLM is doing translation work, not
> synthesis work. Templates handle translation; LLMs handle synthesis.
> Sprint 5 puts each job where it belongs.
>
> **Definition of Done:**
> 1. Gold-schema JSON ticket fingerprint lookup (first hit): p95 < 8s.
> 2. Gold-schema JSON ticket fingerprint lookup (cached hit): p95 < 300ms.
> 3. Expert Copilot output structure (Phase 1/2/Expert Pivot/Phase 3)
>    unchanged from Sprint 4 — visually identical to the user.
> 4. Non-JSON document retrievals (PDFs, Word, KBs, contacts) execute
>    the existing Sprint 4 / Sprint 3 path with zero latency change.
> 5. Cache is invalidated automatically when a ticket is re-uploaded
>    (existing ingestion flow handles this via chunk-delete-and-reinsert).
> 6. Master flag `LOGIQ_SPRINT5_BACKEND=False`. Flag-off = byte-
>    identical Sprint 4 behavior (every lookup hits LLM).

---

## 1. Files To Touch

| # | File | Action | Purpose |
|---|---|---|---|
| 1 | `backend/config.py` | MODIFY | Add `LOGIQ_SPRINT5_BACKEND` flag + `EXPERT_COPILOT_CACHE_TTL_DAYS: int = 30` |
| 2 | `backend/db/migrations/037_expert_answer_cache.sql` | NEW | `ADD COLUMN cached_expert_answer TEXT + cached_at TIMESTAMP` on chunks table |
| 3 | `backend/agents/expert_copilot_template.py` | NEW | Pure-Python template renderer for gold-schema JSON |
| 4 | `backend/retrieval/orchestrator.py` | MODIFY | `retrieve_by_fingerprint` returns structured dict including `is_gold_schema` flag so caller can route |
| 5 | `backend/api.py` | MODIFY | `/fingerprint/lookup` branches on `is_gold_schema` + cache check; cache write on LLM completion |
| 6 | `backend/agents/composer.py` | MODIFY | Add `run_hybrid_expert_pipeline` that takes pre-rendered template + asks LLM only for Phase 1 narrative + Expert Pivot (~2KB payload vs 40KB) |
| 7 | `backend/vector_store.py` | MODIFY | Helpers: `get_cached_expert_answer(chunk_id)` + `set_cached_expert_answer(chunk_id, answer)` |
| 8 | `backend/tests/test_sprint5_expert_template.py` | NEW | Template-render unit tests + cache roundtrip |

**Net: 5 modified + 3 new files.** Zero frontend changes — the optimization is entirely server-side and the response shape is unchanged.

---

## 2. Behavior Preservation Contract

DO NOT touch:
- Sprints 1, 2, 2.5–2.9, 3A–3E composer voices and retrieval
- Sprint 4 `/fingerprint/lookup` URL or request/response contract
- Any PDF, DOCX, CSV, or contact ingestion path
- Any Sprint 3B KB pivot logic
- Ingestion of gold tickets (Sprint 2.9's `_ingest_gold_ticket_json`) —
  full JSON stays untouched
- The GIN indexes on fingerprints (Sprint 4)
- The Expert Copilot composer voice string (still used for the narrative
  synthesis portion, just fed a smaller payload)

---

## 3. Implementation

### 3.1 Cache column migration

```sql
-- 037_expert_answer_cache.sql
BEGIN;

ALTER TABLE chunks
  ADD COLUMN IF NOT EXISTS cached_expert_answer TEXT DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS cached_expert_answer_at TIMESTAMP DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_cached_expert_answer_at
  ON chunks(cached_expert_answer_at);

COMMIT;
```

> **Verify before write:** confirm the chunk-per-ticket ingestion stores
> one chunk per ticket (Sprint 2.9 `_ingest_gold_ticket_json` does this
> explicitly). If ingestion were changed to multi-chunk per ticket,
> cache keying would need revision. Expected: unchanged, one row per
> ticket.

### 3.2 Gold-schema detector (strict)

Before any optimization applies, the retrieved record must be a
gold-schema JSON ticket. Detection is conservative — if any required
section is missing, fall back to the full LLM pipeline (current
behavior).

```python
# backend/agents/expert_copilot_template.py

_REQUIRED_GOLD_SECTIONS = (
    "Symptom_Solution_Mapping",
    "Operational_SOP",
    "remediation_payload",
)


def is_gold_schema_ticket(metadata_json: Dict[str, Any]) -> bool:
    """Sprint 5 — strict detector for gold-schema JSON tickets.

    Returns True ONLY if the record has all three required top-level
    sections AND the Fingerprints array. Anything else (PDFs, KBs,
    contacts, partial tickets) returns False and bypasses the
    optimization — existing LLM pipeline runs.
    """
    if not isinstance(metadata_json, dict):
        return False
    meta = metadata_json.get("Metadata") or {}
    if not isinstance(meta, dict):
        return False
    if not meta.get("Fingerprints"):
        return False
    for key in _REQUIRED_GOLD_SECTIONS:
        if key not in metadata_json:
            return False
        if not isinstance(metadata_json[key], (dict, list)):
            return False
    return True
```

### 3.3 Template renderer (pure Python, no LLM)

```python
# backend/agents/expert_copilot_template.py

def render_phase_2_branching(json_ticket: Dict) -> str:
    """Render Phase 2 Branching Diagnostics from diagnostic_logic_chunks.

    Iterates the array; each step becomes a numbered sub-section with
    action, command (if present), and branching logic. Deterministic
    output — no LLM call.
    """
    sop = json_ticket.get("Operational_SOP") or {}
    chunks = sop.get("diagnostic_logic_chunks") or []
    if not chunks:
        return ""

    out_lines = ["## Phase 2: Branching Diagnostics", ""]
    for chunk in chunks:
        step_id = chunk.get("step_id", "")
        action = chunk.get("action", "")
        command = chunk.get("command")
        branching = chunk.get("branching_logic", "")

        out_lines.append(f"### {step_id}: {action}".strip())
        if command:
            out_lines.append(f"- **Command:** `{command}`")
        if branching:
            out_lines.append(f"- **Branching Logic:** {branching}")
        out_lines.append("")

    return "\n".join(out_lines).rstrip()


def render_phase_3_remediation(json_ticket: Dict) -> str:
    """Render Phase 3 Validated Fix from Symptom_Solution_Mapping +
    remediation_payload. No LLM call."""
    ssm = json_ticket.get("Symptom_Solution_Mapping") or {}
    remed = (json_ticket.get("Operational_SOP") or {}).get("remediation_payload") \
         or json_ticket.get("remediation_payload") \
         or {}

    out = ["## Phase 3: Validated Fix", ""]
    if ssm.get("Primary_Fix"):
        out.append(f"**Primary Fix:** {ssm['Primary_Fix']}")
    if ssm.get("Primary_Fix_Confidence_Interval"):
        out.append(f"**Confidence:** {ssm['Primary_Fix_Confidence_Interval']}")
    if ssm.get("Validation_Metric"):
        out.append(f"**Validation Metric:** {ssm['Validation_Metric']}")

    steps = remed.get("execution_steps") or []
    if steps:
        out.append("")
        out.append("**Remediation Steps:**")
        for i, step in enumerate(steps, 1):
            task = step.get("task", "")
            action = step.get("action", "")
            out.append(f"{i}. **{task}:** `{action}`")

    rac = remed.get("Remediation_As_Code") or {}
    if rac.get("Executable_Snippet"):
        lang = (rac.get("IaC_Language") or "").lower()
        out.append("")
        out.append(f"**Remediation as Code ({lang or 'snippet'}):**")
        out.append(f"```{lang}")
        out.append(rac["Executable_Snippet"])
        out.append("```")

    return "\n".join(out)


def render_header_and_fingerprints(json_ticket: Dict) -> str:
    """Render the opening header block. No LLM call."""
    header = json_ticket.get("Header", "")
    meta = json_ticket.get("Metadata") or {}
    inc = meta.get("Incident_Number", "unknown")
    priority = meta.get("priority", meta.get("Priority", "n/a"))
    fps = meta.get("Fingerprints") or []

    out = [
        f"# Troubleshooting Guide: {header}",
        f"**Incident ID:** {inc} | **Priority:** {priority}",
        "",
        "---",
        "",
        "**Detected Fingerprints:**",
    ]
    for fp in fps:
        out.append(f"- `{fp}`")
    return "\n".join(out)


def render_kb_citations(json_ticket: Dict) -> str:
    """Render KB citations list. No LLM call."""
    kb_list = json_ticket.get("Knowledge_Base") or []
    ids = []
    for kb in kb_list:
        unit = kb.get("semantic_unit_educational") or {}
        kid = unit.get("knowledge_id")
        if kid:
            ids.append(kid)
    if not ids:
        return ""
    return "**Referenced KB:** " + ", ".join(f"`{k}`" for k in ids)
```

### 3.4 Hybrid LLM pipeline — Phase 1 narrative + Expert Pivot only

```python
# backend/agents/composer.py (additions)

def run_hybrid_expert_pipeline(
    *,
    json_ticket: Dict,
    pre_rendered_sections: Dict[str, str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
) -> str:
    """Sprint 5 — hybrid pipeline for gold-schema JSON.

    Templates render Phase 2, Phase 3, Header, Fingerprints, KB
    citations deterministically. LLM only generates the two sections
    that need genuine synthesis:
      - Phase 1: Forensic Triage (narrative + dependency analysis)
      - Expert Pivot (mental pivot + red herrings + invisible triggers)

    LLM payload shrinks from ~40KB to ~8KB. Latency drops ~5x.
    """
    # Build a small, focused prompt — only sections the LLM needs to
    # synthesize Phase 1 + Expert Pivot. Nothing else.
    ssm = json_ticket.get("Symptom_Solution_Mapping") or {}
    meta = json_ticket.get("Metadata") or {}
    kb_list = json_ticket.get("Knowledge_Base") or []
    kb_excerpt = kb_list[0] if kb_list else {}

    focused_context = {
        "header": json_ticket.get("Header"),
        "incident_number": meta.get("Incident_Number"),
        "target_service": meta.get("Target_Service"),
        "affected_assets": meta.get("Affected_Assets"),
        "customer_name": meta.get("customer_name"),
        "detected_symptom": ssm.get("Detected_Symptom"),
        "origin_event": ssm.get("Origin_Event"),
        "fingerprints": meta.get("Fingerprints"),
        "knowledge_base_primary": kb_excerpt,
    }

    prompt = f"""You are the Expert Troubleshooting Copilot. Generate ONLY two sections of a troubleshooting guide:

1. **Phase 1: Forensic Triage** — 3-4 sentences establishing what the fingerprint(s) technically imply, plus a 'First-look checklist' with 3-5 initial verification commands.

2. **Expert Pivot** — 2-3 paragraphs capturing the single highest-confidence hypothesis (the 'mental pivot') and 2-4 verification steps to confirm it.

Do NOT generate Phase 2, Phase 3, headers, fingerprint lists, or remediation code — those are rendered separately.

Context (JSON):
{json.dumps(focused_context, indent=2)}

Output ONLY the two sections requested, with their markdown headers. Nothing else."""

    step_result = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_COMPOSER_MODEL,
        max_tokens=1200,  # enough for Phase 1 + Expert Pivot, not the whole thing
        budget=budget,
        agent_name="expert_copilot_hybrid",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )

    llm_output = step_result.output.strip() if step_result.success else ""

    # Stitch final markdown: header, fingerprints, LLM-generated
    # Phase 1 + Expert Pivot, then template-rendered Phase 2 + Phase 3.
    parts = [
        pre_rendered_sections["header"],
        "",
        "---",
        "",
        llm_output,                              # Phase 1 + Expert Pivot (LLM)
        "",
        "---",
        "",
        pre_rendered_sections["phase_2"],        # Template
        "",
        "---",
        "",
        pre_rendered_sections["phase_3"],        # Template
    ]
    if pre_rendered_sections.get("kb_citations"):
        parts.extend(["", "---", "", pre_rendered_sections["kb_citations"]])

    return "\n".join(p for p in parts if p is not None)
```

### 3.5 `/fingerprint/lookup` integration

In `backend/api.py`, the existing handler changes to:

```python
@app.post("/fingerprint/lookup")
async def fingerprint_lookup(body: FingerprintLookupRequest):
    if not settings.LOGIQ_SPRINT4_BACKEND:
        raise HTTPException(404, "Not available")

    fp = (body.fingerprint or "").strip()
    if not fp:
        raise HTTPException(422, detail={"error": "fingerprint cannot be empty"})

    patch_session_mode(body.session_id, {
        "entered_via": "fingerprint",
        "original_fingerprint": fp,
    })

    # Retrieve — returns (json_ticket, chunk_id) or (None, None)
    result = retrieve_by_fingerprint(fp, return_chunk_id=True)
    if result is None or result[0] is None:
        return {"match": False, "fingerprint": fp,
                "message": f"No match found for fingerprint '{fp}'."}

    json_ticket, chunk_id = result

    # Sprint 5 — only optimize gold-schema JSON
    sprint5_on = getattr(settings, "LOGIQ_SPRINT5_BACKEND", False)
    gold_schema = is_gold_schema_ticket(json_ticket)

    if sprint5_on and gold_schema:
        # Cache check (fast path)
        cached = get_cached_expert_answer(chunk_id)
        if cached:
            logger.info("[sprint5] cache_hit chunk_id=%s fp=%s", chunk_id, fp)
            return {"match": True, "fingerprint": fp,
                    "source_incident": json_ticket.get("Metadata", {}).get("Incident_Number"),
                    "answer": cached, "response_type": "expert_copilot",
                    "cache_hit": True}

        # Hybrid pipeline (LLM only for Phase 1 + Expert Pivot)
        pre_rendered = {
            "header": render_header_and_fingerprints(json_ticket),
            "phase_2": render_phase_2_branching(json_ticket),
            "phase_3": render_phase_3_remediation(json_ticket),
            "kb_citations": render_kb_citations(json_ticket),
        }
        answer = run_hybrid_expert_pipeline(
            json_ticket=json_ticket,
            pre_rendered_sections=pre_rendered,
            budget=make_budget(analytical=True),
            generate_fn=generate_fn, bedrock_client=bedrock_client,
        )
        set_cached_expert_answer(chunk_id, answer)
        logger.info("[sprint5] cache_miss chunk_id=%s fp=%s cached_answer_chars=%d",
                    chunk_id, fp, len(answer))

        return {"match": True, "fingerprint": fp,
                "source_incident": json_ticket.get("Metadata", {}).get("Incident_Number"),
                "answer": answer, "response_type": "expert_copilot",
                "cache_hit": False}

    # Non-gold-schema OR flag off → Sprint 4 behavior (full LLM)
    compose_result = run_composer(
        query=f"Expert troubleshooting guide for fingerprint {fp}",
        findings=[json.dumps(json_ticket, indent=2)],
        source_names=[json_ticket.get("Metadata", {}).get("Incident_Number", "unknown")],
        budget=make_budget(analytical=True),
        generate_fn=generate_fn, bedrock_client=bedrock_client,
        session_mode=get_session_mode(body.session_id),
        voice_override="expert_copilot",
    )
    return {"match": True, "fingerprint": fp,
            "source_incident": json_ticket.get("Metadata", {}).get("Incident_Number"),
            "answer": compose_result.output, "response_type": "expert_copilot",
            "cache_hit": False}
```

### 3.6 Cache helpers

```python
# backend/vector_store.py

def get_cached_expert_answer(chunk_id: str) -> Optional[str]:
    """Return the cached Expert Copilot answer for a chunk, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""SELECT cached_expert_answer
                    FROM chunks
                    WHERE id = :cid
                      AND cached_expert_answer IS NOT NULL"""),
            {"cid": chunk_id},
        ).first()
    return row[0] if row else None


def set_cached_expert_answer(chunk_id: str, answer: str) -> None:
    """Write the rendered Expert Copilot answer to the chunk cache."""
    with engine.begin() as conn:
        conn.execute(
            text("""UPDATE chunks
                    SET cached_expert_answer = :a,
                        cached_expert_answer_at = NOW()
                    WHERE id = :cid"""),
            {"a": answer, "cid": chunk_id},
        )
```

Cache invalidation is automatic: Sprint 2.9's gold-ticket ingestion
DELETEs and re-INSERTs chunks on re-upload, so the cache columns reset
to NULL naturally. No explicit invalidation logic needed.

### 3.7 Retrieval signature update

`retrieve_by_fingerprint` currently returns `Optional[Dict]`. Add
`return_chunk_id: bool = False` kwarg; when True, return
`Tuple[Dict, str]` so the handler can key the cache. Default False
preserves the existing contract for any other callers.

---

## 4. Acceptance Tests

### 4.1 Flag-off regression (mandatory)
With `LOGIQ_SPRINT5_BACKEND=False`: every lookup calls the full LLM
pipeline exactly as Sprint 4. No cache reads, no cache writes.

### 4.2 Unit tests
```python
# Gold-schema detection
assert is_gold_schema_ticket(FULL_JSON) is True
assert is_gold_schema_ticket({"Metadata": {}}) is False
assert is_gold_schema_ticket({"Metadata": {"Fingerprints": ["X"]}}) is False  # missing sections

# Template rendering
out = render_phase_2_branching(FULL_JSON)
assert "## Phase 2: Branching Diagnostics" in out
assert "Verify if BFD is software or hardware" in out  # from PHOENIX-402

out = render_phase_3_remediation(FULL_JSON)
assert "bfd interval 100 min_rx 100 multiplier 3" in out  # RaC verbatim

# Cache roundtrip
set_cached_expert_answer("chunk-id-1", "answer text")
assert get_cached_expert_answer("chunk-id-1") == "answer text"
```

### 4.3 Runtime acceptance
| # | Action | Expected |
|---|---|---|
| T1 | Flag on, lookup `%BGP-5-ADJCHANGE` first time | ~5–8s response. Log: `[sprint5] cache_miss`. Answer has all 4 sections. |
| T2 | Same lookup immediately again | <300ms response. Log: `[sprint5] cache_hit`. Answer visually identical. |
| T3 | Re-upload the same ticket JSON | Cache cleared (chunk DELETE/INSERT). Next lookup → cache_miss again. |
| T4 | Lookup a fingerprint on a PDF document | Full LLM pipeline runs (no Sprint 5 log line). Latency unchanged from Sprint 4. |
| T5 | Lookup with flag off | Full LLM pipeline, no cache reads/writes. |
| T6 | Sprints 2.5–2.9, 3A–3E regression | Unchanged. Sprint 5 only touches `/fingerprint/lookup` internals. |

---

## 5. Rollback

Flip `LOGIQ_SPRINT5_BACKEND=False`:
- `/fingerprint/lookup` skips cache and hybrid pipeline
- Every request hits the full LLM path
- Existing cached rows remain in DB (harmless)
- No migration rollback needed

To drop the cache columns entirely:
```sql
ALTER TABLE chunks
  DROP COLUMN IF EXISTS cached_expert_answer,
  DROP COLUMN IF EXISTS cached_expert_answer_at;
DROP INDEX IF EXISTS idx_chunks_cached_expert_answer_at;
```

---

## 6. Claude Code Execution Prompt

```

```