"""
Phase 2 prompt templates.

What this file adds:
- strict JSON metadata extraction prompts
- structured operational labeling prompts
- version support decision prompts
"""

from textwrap import dedent


CHUNK_METADATA_SYSTEM = dedent(
    """
    You extract operational retrieval metadata from document chunks.
    Always return strict JSON only.
    Never include markdown fences or commentary.
    Keep outputs concise, grounded, and directly supported by the text.
    """
).strip()


def build_chunk_metadata_prompt(*, document_name: str, source_type: str, chunk_batch_json: str) -> str:
    return dedent(
        f"""
        Analyze the following document chunks and return strict JSON with this schema:

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
            "tags": ["string"],
            "keywords": ["string"]
          }},
          "chunks": [
            {{
              "chunk_index": 0,
              "section": "string|null",
              "chunk_type": "symptom_chunk|diagnostic_chunk|resolution_chunk|escalation_chunk|command_chunk|validation_chunk|summary_chunk|reference_chunk|classification_chunk|error_chunk|workaround_chunk|root_cause_chunk|general_chunk",
              "document_type": "string|null",
              "vendor": "string|null",
              "product": "string|null",
              "domain": "string|null",
              "version": "string|null",
              "date": "YYYY-MM-DD|null",
              "tags": ["string"],
              "entities": [{{"text":"string","label":"string"}}],
              "keywords": ["string"],
              "summary": "string|null",
              "purpose_description": "string|null",
              "operational_context": "string|null"
            }}
          ]
        }}

        Rules:
        - Use only evidence from the text.
        - Prefer operational meanings over generic labels.
        - summary must be <= 280 chars.
        - purpose_description should help retrieval precision.
        - If a value is not clear, use null or an empty list.
        - Entities can use labels like PRODUCT, SERVICE, VENDOR, COMMAND, TEAM, DATE, ERROR_CODE, API, ENVIRONMENT.

        Document name: {document_name}
        Source type: {source_type}

        Chunk batch:
        {chunk_batch_json}
        """
    ).strip()


VERSION_DECISION_SYSTEM = dedent(
    """
    You support duplicate and version detection for operational documents.
    Always return strict JSON only.
    Prefer conservative decisions when uncertain.
    """
).strip()


def build_version_decision_prompt(*, incoming_json: str, candidates_json: str) -> str:
    return dedent(
        f"""
        Determine whether the incoming document is:
        - exact_duplicate
        - new_version
        - new_document

        Return strict JSON in this exact shape:
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

        Decision rules:
        - exact_duplicate only if content/fingerprint match or extremely clear same-content evidence.
        - new_version only if title/name/metadata strongly indicate same logical document family.
        - version ordering priority is explicit version, then effective/document dates, then upload timestamp fallback.
        - If uncertain, return new_document.

        Incoming document:
        {incoming_json}

        Existing candidates:
        {candidates_json}
        """
    ).strip()