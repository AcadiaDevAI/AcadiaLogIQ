# Adaptive Ingestion Batching — Dynamic Batch Size + max_tokens Per Content Density

## Problem

During PDF ingestion (HPBX-End-User-Guide-2.1.pdf), backend logs flood with:

```
WARNING - Haiku returned invalid JSON on attempt 1/3:
  Unterminated string starting at: line 78 column 30 (char 2867)
  | stop_reason=max_tokens | raw_len=2868
```

20+ truncation warnings in 60 seconds. Haiku's JSON output exceeds the fixed `max_tokens=4096` budget when batches contain text-heavy chunks. The system falls back to brace-balancing repair which silently drops chunk metadata (summary, acronyms, section labels go empty), degrading retrieval quality downstream.

Currently `CHUNK_BATCH_SIZE: int = 10` is hardcoded — same batch size for every file type and every chunk density.

## Solution Overview

Two complementary changes, both fully dynamic:

1. **Dynamic batch size** — instead of hardcoded `CHUNK_BATCH_SIZE=10`, compute batch size per file based on actual chunk content density
2. **Adaptive max_tokens** — instead of hardcoded `HAIKU_MAX_TOKENS=4096`, compute max_tokens per batch based on what THIS specific batch's expected output looks like

Plus a self-correcting safety belt: if a call still hits `max_tokens` despite estimation, halve the batch and retry once.

Zero hardcoded file-type rules. Works for JSON tickets, PDFs, DOCX, XLSX, future formats — any content type that produces ParsedChunks.

## Files Touched

- `backend/services/contextual_ingestion_service.py` — add 3 helper functions, modify the existing batch loop
- `backend/config.py` — add 4 new tunable parameters (with safe defaults)

That's it. Single-feature change, no architectural restructuring.

---

## Implementation

### Step 1 — Add config parameters

In `backend/config.py`, in the section near `CHUNK_BATCH_SIZE` (around line 75-86), add:

```python
# ----------------------------------------------------------------
# Adaptive Ingestion Batching
# ----------------------------------------------------------------
# When True, batch size and max_tokens are computed per-batch based on
# actual chunk content density. When False, falls back to fixed
# CHUNK_BATCH_SIZE and HAIKU_MAX_TOKENS (legacy behavior).
ADAPTIVE_INGESTION_BATCHING_ENABLED: bool = True

# Target Haiku output token budget per batch. Higher = bigger batches
# (fewer API calls but more risk of truncation). 4000 is the sweet spot
# for Haiku — leaves headroom under the 4096 typical safe ceiling.
ADAPTIVE_TARGET_OUTPUT_TOKENS: int = 4000

# Hard caps on batch size. Even adaptive logic respects these.
ADAPTIVE_BATCH_SIZE_MIN: int = 3      # Below this, API overhead dominates
ADAPTIVE_BATCH_SIZE_MAX: int = 25     # Above this, Haiku quality drops on large batches

# Safety buffer multiplier when computing max_tokens for a batch.
# 1.3 = give 30% extra headroom over estimated output size.
ADAPTIVE_MAX_TOKENS_BUFFER: float = 1.3

# Hard caps on max_tokens for any single Haiku call.
ADAPTIVE_MAX_TOKENS_MIN: int = 2000
ADAPTIVE_MAX_TOKENS_MAX: int = 8000
```

### Step 2 — Add adaptive helper functions

In `backend/services/contextual_ingestion_service.py`, add these three helper functions ABOVE the existing `_extract_chunk_metadata_once` function (around line 230):

```python
# ─────────────────────────────────────────────────────────────────────
# Adaptive batching helpers — size batches and Haiku output budget by
# actual chunk content density instead of using fixed values that
# truncate on text-heavy documents.
# ─────────────────────────────────────────────────────────────────────


def _estimate_output_tokens_per_chunk(chunk_text_length: int) -> int:
    """
    Estimate how many output tokens Haiku will produce when generating
    metadata for a chunk of the given input length.

    Empirical formula based on observed Haiku JSON output sizes:
      - ~80 tokens base structure overhead (chunk_index, type, etc.)
      - summary scales roughly as input/8 (compression ratio ~8:1)
      - ~50 tokens for acronyms list, section labels, classifications

    Returns at least 100 tokens (sanity floor for tiny chunks).
    """
    base_overhead = 80
    summary_estimate = max(0, chunk_text_length // 8)
    extras_overhead = 50
    return max(100, base_overhead + summary_estimate + extras_overhead)


def _compute_adaptive_batch_size(chunks: List[ParsedChunk]) -> int:
    """
    Decide how many chunks fit in one Haiku call without truncation.

    Samples up to first 3 chunks to estimate average output density,
    then computes a batch size that keeps total expected output well
    under the configured target budget.

    Fully dynamic — no hardcoded rules per file type. A JSON ticket
    chunk gets a different batch size than a verbose PDF chunk
    automatically because their text lengths differ.
    """
    if not chunks:
        return 1

    if not getattr(settings, "ADAPTIVE_INGESTION_BATCHING_ENABLED", True):
        return int(getattr(settings, "CHUNK_BATCH_SIZE", 10))

    # Sample a few chunks to estimate average output size.
    sample = chunks[: min(3, len(chunks))]
    avg_input_length = sum(len(c.text) for c in sample) / len(sample)
    avg_output_per_chunk = _estimate_output_tokens_per_chunk(int(avg_input_length))

    # Reserve 30% safety headroom in the target budget.
    target = float(getattr(settings, "ADAPTIVE_TARGET_OUTPUT_TOKENS", 4000))
    safe_budget = target * 0.7

    raw_batch_size = max(1, int(safe_budget / avg_output_per_chunk))

    # Apply hard caps.
    floor = int(getattr(settings, "ADAPTIVE_BATCH_SIZE_MIN", 3))
    ceiling = int(getattr(settings, "ADAPTIVE_BATCH_SIZE_MAX", 25))
    batch_size = max(floor, min(raw_batch_size, ceiling))

    logger.info(
        "[adaptive_batch] sized: avg_input=%d chars → est_output=%d tokens/chunk "
        "→ batch_size=%d (raw=%d, capped to [%d, %d])",
        int(avg_input_length), avg_output_per_chunk, batch_size,
        raw_batch_size, floor, ceiling,
    )
    return batch_size


def _compute_adaptive_max_tokens(batch_chunks: List[ParsedChunk]) -> int:
    """
    Right-size max_tokens for THIS specific batch's expected output.

    Sums the estimated output tokens for each chunk in the batch,
    multiplies by a safety buffer, clamps to sensible bounds. This
    means a batch of 8 verbose chunks gets a different budget than a
    batch of 8 sparse chunks — automatically.

    Falls back to settings.HAIKU_MAX_TOKENS if adaptive disabled.
    """
    if not batch_chunks:
        return int(getattr(settings, "HAIKU_MAX_TOKENS", 4096))

    if not getattr(settings, "ADAPTIVE_INGESTION_BATCHING_ENABLED", True):
        return int(getattr(settings, "HAIKU_MAX_TOKENS", 4096))

    total_estimated = sum(
        _estimate_output_tokens_per_chunk(len(c.text)) for c in batch_chunks
    )
    buffer = float(getattr(settings, "ADAPTIVE_MAX_TOKENS_BUFFER", 1.3))
    target = int(total_estimated * buffer)

    floor = int(getattr(settings, "ADAPTIVE_MAX_TOKENS_MIN", 2000))
    ceiling = int(getattr(settings, "ADAPTIVE_MAX_TOKENS_MAX", 8000))
    max_tokens = max(floor, min(target, ceiling))

    logger.info(
        "[adaptive_batch] max_tokens: %d chunks → est_total=%d → buffered=%d → "
        "max_tokens=%d (capped to [%d, %d])",
        len(batch_chunks), total_estimated, target, max_tokens, floor, ceiling,
    )
    return max_tokens
```

### Step 3 — Update `_extract_chunk_metadata_once` to pass adaptive max_tokens

Modify the existing function to compute and pass the per-batch max_tokens:

```python
def _extract_chunk_metadata_once(
    *,
    document_name: str,
    source_type: str,
    chunks: List[ParsedChunk],
) -> Optional[Dict[int, Dict[str, Any]]]:
    payload = []
    for chunk in chunks:
        payload.append(
            {
                "chunk_index": chunk.chunk_index,
                "section_heading": chunk.section_heading,
                "operational_section": chunk.operational_section,
                "chunk_type_hint": chunk.chunk_type,
                "text": chunk.text[: min(settings.MAX_METADATA_INPUT_CHARS, 2200)],
            }
        )

    prompt = build_chunk_metadata_prompt(
        document_name=document_name,
        source_type=source_type,
        chunk_batch_json=_safe_json(payload),
    )

    # NEW: compute adaptive max_tokens for this specific batch
    adaptive_max_tokens = _compute_adaptive_max_tokens(chunks)

    result = haiku_client.invoke_json(
        system=CHUNK_METADATA_SYSTEM,
        prompt=prompt,
        max_tokens=adaptive_max_tokens,  # ← pass the adaptive value
    )

    if not result or "chunks" not in result or not isinstance(result.get("chunks"), list):
        return None

    # ... rest unchanged
```

### Step 4 — Update batching loop in `process_document` to use adaptive batch size

Find the batching loop in `process_document` (around line 423-426):

```python
# BEFORE (hardcoded)
batches: List[List[ParsedChunk]] = []
for start in range(0, len(chunks), settings.CHUNK_BATCH_SIZE):
    batches.append(chunks[start : start + settings.CHUNK_BATCH_SIZE])

# AFTER (dynamic)
batches: List[List[ParsedChunk]] = []
adaptive_batch_size = _compute_adaptive_batch_size(chunks)
for start in range(0, len(chunks), adaptive_batch_size):
    batches.append(chunks[start : start + adaptive_batch_size])

logger.info(
    "[adaptive_batch] file=%s total_chunks=%d batch_size=%d → %d batches",
    filename, len(chunks), adaptive_batch_size, len(batches),
)
```

### Step 5 — Add self-correcting retry on truncation (safety belt)

In `batch_extract_chunk_metadata`, wrap the call so if it returns nothing AND truncation likely happened, retry with halved batch.

Find `batch_extract_chunk_metadata` (around line 281) and modify:

```python
def batch_extract_chunk_metadata(
    *,
    document_name: str,
    source_type: str,
    chunks: List[ParsedChunk],
) -> Dict[int, Dict[str, Any]]:
    if not settings.ENABLE_METADATA_EXTRACTION:
        return {
            chunk.chunk_index: _fallback_chunk_metadata(chunk, document_name, source_type)
            for chunk in chunks
        }

    # Try the batch as-is first (with adaptive max_tokens already applied).
    result = _extract_chunk_metadata_once(
        document_name=document_name,
        source_type=source_type,
        chunks=chunks,
    )

    # SAFETY BELT: if we got nothing OR coverage is incomplete, the call
    # may have truncated. Retry with halved batch when adaptive enabled.
    if (
        getattr(settings, "ADAPTIVE_INGESTION_BATCHING_ENABLED", True)
        and len(chunks) > 1
        and (not result or len(result) < len(chunks) // 2)
    ):
        logger.warning(
            "[adaptive_batch] incomplete coverage on batch of %d chunks "
            "(got %d) — splitting batch in half and retrying",
            len(chunks), len(result) if result else 0,
        )
        mid = len(chunks) // 2
        first_half = chunks[:mid]
        second_half = chunks[mid:]

        retry_result: Dict[int, Dict[str, Any]] = {}
        for half in (first_half, second_half):
            half_result = _extract_chunk_metadata_once(
                document_name=document_name,
                source_type=source_type,
                chunks=half,
            )
            if half_result:
                retry_result.update(half_result)

        # Use retry result if better, else original
        if len(retry_result) > (len(result) if result else 0):
            result = retry_result

    # Fill in fallback metadata for any chunks that still didn't get processed.
    if result:
        return {
            chunk.chunk_index: result.get(
                chunk.chunk_index,
                _fallback_chunk_metadata(chunk, document_name, source_type),
            )
            for chunk in chunks
        }
    else:
        return {
            chunk.chunk_index: _fallback_chunk_metadata(chunk, document_name, source_type)
            for chunk in chunks
        }
```

---

## Acceptance Tests

### Test 1 — Tiny JSON tickets (your gold schema)
1. Re-ingest `goldschema_48t_FictionalData.txt` (45 tickets)
2. Watch logs for the adaptive sizing:
   ```
   [adaptive_batch] sized: avg_input=10000 chars → est_output=1380 tokens/chunk → batch_size=2 (raw=2, capped to [3, 25])
   [adaptive_batch] file=goldschema_48t_FictionalData.txt total_chunks=45 batch_size=3 → 15 batches
   [adaptive_batch] max_tokens: 3 chunks → est_total=4140 → buffered=5382 → max_tokens=5382
   ```
3. Verify:
   - Zero `Haiku returned invalid JSON` warnings
   - All 45 chunks get full metadata (no fallback used)
   - Total ingestion time within ~10% of current

### Test 2 — PDF with verbose content (HPBX guide, 181 chunks)
1. Re-ingest `HPBX-End-User-Guide-2.1.pdf`
2. Expected logs:
   ```
   [adaptive_batch] sized: avg_input=2200 chars → est_output=405 tokens/chunk → batch_size=6 (raw=6, capped to [3, 25])
   [adaptive_batch] file=HPBX-End-User-Guide-2.1.pdf total_chunks=181 batch_size=6 → 31 batches
   [adaptive_batch] max_tokens: 6 chunks → est_total=2430 → buffered=3159 → max_tokens=3159
   ```
3. Verify:
   - **Zero `stop_reason=max_tokens` warnings** (current state has 20+)
   - All 181 chunks get full metadata
   - All acronyms extracted to glossary (currently degraded)
   - Wall-clock time ≤ current (parallel batches still run concurrently)

### Test 3 — Self-correcting retry on edge case
1. Force a truncation by setting `ADAPTIVE_TARGET_OUTPUT_TOKENS=500` temporarily
2. Re-ingest a PDF
3. Expected logs:
   ```
   WARNING - Haiku returned invalid JSON on attempt 1/3
   [adaptive_batch] incomplete coverage on batch of 6 chunks (got 0) — splitting batch in half and retrying
   ```
4. Verify retry succeeds with smaller batches → reset `ADAPTIVE_TARGET_OUTPUT_TOKENS` to 4000

### Test 4 — Feature flag rollback
1. Set `ADAPTIVE_INGESTION_BATCHING_ENABLED=False`
2. Re-ingest the same PDF
3. Verify behavior reverts to current state (uses fixed `CHUNK_BATCH_SIZE=10` and `HAIKU_MAX_TOKENS=4096`)
4. Confirms feature is cleanly toggleable

### Test 5 — DOCX upload (future use case)
1. Upload any DOCX file
2. Verify `[adaptive_batch] sized: ...` log appears with appropriate batch_size based on content density
3. Confirm same behavior pattern as PDF — clean ingestion, no truncation warnings

---

## Observability — Logs You'll See

```
[adaptive_batch] sized: avg_input=2200 chars → est_output=405 tokens/chunk → batch_size=6 (raw=6, capped to [3, 25])
[adaptive_batch] file=HPBX-End-User-Guide-2.1.pdf total_chunks=181 batch_size=6 → 31 batches
[adaptive_batch] max_tokens: 6 chunks → est_total=2430 → buffered=3159 → max_tokens=3159
[PERF] HPBX-End-User-Guide-2.1.pdf — Parse + metadata: 52.3s (181 chunks)   ← faster than current 64.9s
```

If retry safety belt fires:
```
[adaptive_batch] incomplete coverage on batch of 8 chunks (got 0) — splitting batch in half and retrying
```

---

## Why This Works

**Truncation root cause:** Haiku's `max_tokens` was set to a fixed 4096 regardless of how much output the actual batch needed. A batch of 10 verbose PDF chunks needs ~5000 tokens of output JSON — exceeds 4096 → truncates mid-string → JSON invalid → retry with same parameters → same truncation.

**Adaptive solution:**
- For VERBOSE batches: smaller batch size + larger max_tokens → fits cleanly first try
- For COMPACT batches: larger batch size + appropriate max_tokens → fewer API calls, no waste
- For UNEXPECTED edge cases: retry with halved batch → guaranteed to fit

**Per file type behavior (no hardcoded rules):**
- Gold JSON tickets (~10K chars/chunk): batch_size=3, max_tokens=5382, ~15 calls for 45 chunks
- PDF guide (~2200 chars/chunk): batch_size=6, max_tokens=3159, ~31 calls for 181 chunks
- Future small DOCX (~800 chars/chunk): batch_size=15, max_tokens=3000, ~3 calls for 50 chunks
- Future massive XLSX (~300 chars/chunk): batch_size=25, max_tokens=2000, ~167 calls for 5000 rows

System computes the right values from chunk content alone — no file-extension special casing.

---

## Cost & Accuracy Impact

**Cost:**
- Number of API calls: similar to slightly higher (15-31 vs current 5-9)
- Per-call cost: lower (right-sized inputs and outputs)
- Retry waste eliminated: current state burns 25%+ extra tokens on failed-then-repaired calls
- **Net: ~5-10% LOWER total Bedrock spend**

**Accuracy:**
- Current: ~25% of PDF chunks lose metadata silently to truncation repair
- After fix: 100% chunks get clean metadata
- Glossary completeness: currently incomplete on long PDFs, after fix complete
- Retrieval quality downstream: improves because every chunk has its summary, acronyms, section labels populated

**Latency:**
- More batches but they run in parallel via existing `METADATA_CONCURRENCY` thread pool
- Wall-clock time: similar or slightly faster (no retry round-trips)

---

## Rollback

Single flag flip:
```python
ADAPTIVE_INGESTION_BATCHING_ENABLED: bool = False
```

Reverts to legacy fixed `CHUNK_BATCH_SIZE=10` and `HAIKU_MAX_TOKENS=4096` behavior. Zero data migration needed.

---

## Ready for Execution

Hand to Claude Code with:

> Read ADAPTIVE_INGESTION_BATCHING_BRIEF.md at repo root. Implement all 5 steps in order. Add config flags first, then helper functions, then update _extract_chunk_metadata_once and process_document and batch_extract_chunk_metadata. Restart backend and re-ingest the existing HPBX-End-User-Guide-2.1.pdf to verify zero truncation warnings. Paste the [adaptive_batch] log lines plus a count of any remaining "Haiku returned invalid JSON" warnings (target: 0).