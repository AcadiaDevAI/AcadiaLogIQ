"""
Phase 2 prompt templates — optimized for speed.

Changes from original:
- Leaner chunk metadata schema (fewer per-chunk fields)
- Document-level fields extracted once, not repeated per chunk
- Shorter system prompt to reduce input tokens
- Explicit instruction to keep output compact
- Dates must be full YYYY-MM-DD (partial date fix)
"""

from textwrap import dedent


CHUNK_METADATA_SYSTEM = dedent(
    """
    Extract retrieval metadata from document chunks. Return strict JSON only.
    No markdown fences. No commentary. Be concise. Prefer null over guessing.
    """
).strip()


def build_chunk_metadata_prompt(*, document_name: str, source_type: str, chunk_batch_json: str) -> str:
    return dedent(
        f"""
        Return strict JSON matching this schema exactly:
        {{
          "document": {{
            "title": "string|null",
            "document_type": "SOP|Runbook|KB|Vendor doc|Incident doc|Unknown",
            "vendor": "string|null",
            "product": "string|null",
            "domain": "string|null",
            "version": "string|null",
            "document_date": "YYYY-MM-DD|null",
            "effective_date": "YYYY-MM-DD|null",
            "created_date": "YYYY-MM-DD|null",
            "tags": ["max 4 strings"],
            "keywords": ["max 4 strings"]
          }},
          "chunks": [
            {{
              "chunk_index": 0,
              "section": "string|null",
              "chunk_type": "symptom_chunk|diagnostic_chunk|resolution_chunk|escalation_chunk|command_chunk|validation_chunk|summary_chunk|reference_chunk|classification_chunk|error_chunk|workaround_chunk|root_cause_chunk|general_chunk",
              "tags": ["max 3"],
              "entities": [{{"text":"string","label":"string"}}],
              "keywords": ["max 3"],
              "summary": "<=80 chars",
              "operational_context": "<=60 chars"
            }}
          ]
        }}

        Rules:
        - Dates must be full YYYY-MM-DD format. If only year-month is known, use first of month (e.g. 2024-10 -> 2024-10-01). If only year, use YYYY-01-01.
        - Use null or [] if unknown. Do not guess.
        - entities: max 3 per chunk.
        - Do not add extra keys.
        - Keep output compact — short values only.

        Document: {document_name}
        Source: {source_type}

        Chunks:
        {chunk_batch_json}
        """
    ).strip()


VERSION_DECISION_SYSTEM = dedent(
    """
    Detect duplicates and versions for operational documents.
    Return strict JSON only. Prefer conservative decisions.
    """
).strip()


def build_version_decision_prompt(*, incoming_json: str, candidates_json: str) -> str:
    return dedent(
        f"""
        Classify the incoming document as exact_duplicate, new_version, or new_document.

        Return strict JSON:
        {{
          "decision": "exact_duplicate|new_version|new_document",
          "matched_document_id": "string|null",
          "reason": "string",
          "confidence": 0.0,
          "normalized_name": "string",
          "version_family_key": "string",
          "version_label": "string|null",
          "version_rank": 0.0,
          "document_date": "YYYY-MM-DD|null",
          "effective_date": "YYYY-MM-DD|null",
          "created_date": "YYYY-MM-DD|null"
        }}

        Rules:
        - Dates must be full YYYY-MM-DD. Pad partial dates (e.g. 2024-10 -> 2024-10-01).
        - exact_duplicate only if content/fingerprint match.
        - new_version only if title/metadata strongly indicate same document family.
        - If uncertain, return new_document.

        Incoming:
        {incoming_json}

        Candidates:
        {candidates_json}
        """
    ).strip()