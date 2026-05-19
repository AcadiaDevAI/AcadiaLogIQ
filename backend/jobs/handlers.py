"""
Job dispatch handlers — one function per job kind.

The worker (``backend.jobs.worker``) reads a claimed row, looks up
its ``kind`` in ``HANDLERS``, and calls the resolved function with
``(incident_number, payload)``. Each handler runs ONE LLM call,
saves the result to ``report_cache`` (migration 043), and returns
``(report_kind, incident_number)`` — the pointer that lands in the
job row's ``result_kind`` / ``result_incident`` columns.

Failure model
-------------
* Ticket-not-found → raise ``NonRetryableJobError``; worker marks
  the job permanently failed (no retries).
* LLM transient error → raise the underlying exception; worker
  marks the job soft-failed and re-queues with backoff.
* LLM returns empty markdown → raise ``NonRetryableJobError``
  (model misbehaving on this specific prompt; retrying won't help).

Why this module is intentionally thin
-------------------------------------
The actual LLM + cache-save logic already lives inside the existing
``rca/routes.py`` and ``gap_analysis/routes.py`` ``_cached_or_llm``
helpers. We deliberately do NOT duplicate it here — we import and
reuse. Phase 1 is "move the call out of the request thread", not
"rewrite the LLM pipeline".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Tuple

from backend.tier1_copilot._shared.report_cache import (
    REPORT_KIND_GAP_ANALYSIS,
    REPORT_KIND_POST_MORTEM,
    REPORT_KIND_RCA_CUSTOMER,
    REPORT_KIND_RCA_INTERNAL,
)
from backend.jobs.ingestion_queue import JOB_KIND_INGEST_DOCUMENT

logger = logging.getLogger("acadia-log-iq")


class NonRetryableJobError(Exception):
    """Raise from a handler to skip the retry loop and mark failed."""


# ─────────────────────────────────────────────────────────────────
# RCA handlers — delegate to the existing _cached_or_llm helper
# ─────────────────────────────────────────────────────────────────


def _run_rca_panel(
    *,
    report_kind: str,
    template_attr: str,
    max_tokens_attr: str,
    incident_number: str,
    payload: Dict[str, Any],
) -> Tuple[str, str]:
    """Shared body for both RCA handlers — only the prompt template and
    token budget differ between panels.
    """
    from backend.tier1_copilot.rca import routes as rca_routes
    from backend.tier1_copilot.rca.ticket_lookup import find_ticket_by_incident_number
    from backend.tier1_copilot.rca.prompts import (
        CUSTOMER_FACING_PROMPT,
        INTERNAL_INCIDENT_PROMPT,
    )

    templates = {
        "CUSTOMER_FACING_PROMPT": CUSTOMER_FACING_PROMPT,
        "INTERNAL_INCIDENT_PROMPT": INTERNAL_INCIDENT_PROMPT,
    }
    template = templates[template_attr]
    max_tokens = getattr(rca_routes, max_tokens_attr)

    ticket = find_ticket_by_incident_number(incident_number)
    if ticket is None:
        raise NonRetryableJobError(f"ticket_not_found: {incident_number}")

    try:
        ticket_json = json.dumps(ticket, ensure_ascii=False, default=str)
    except Exception as exc:
        raise NonRetryableJobError(f"ticket_serialisation_failed: {exc}")

    md, _came_from_cache = rca_routes._cached_or_llm(
        report_kind=report_kind,
        incident_number=incident_number,
        template=template,
        ticket_json=ticket_json,
        max_tokens=max_tokens,
        regenerate=bool(payload.get("regenerate", False)),
    )
    if not md or not md.strip():
        raise NonRetryableJobError("empty_llm_output")
    return (report_kind, incident_number)


def _handle_rca_customer(incident_number: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    return _run_rca_panel(
        report_kind=REPORT_KIND_RCA_CUSTOMER,
        template_attr="CUSTOMER_FACING_PROMPT",
        max_tokens_attr="_MAX_TOKENS_CUSTOMER",
        incident_number=incident_number,
        payload=payload,
    )


def _handle_rca_internal(incident_number: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    return _run_rca_panel(
        report_kind=REPORT_KIND_RCA_INTERNAL,
        template_attr="INTERNAL_INCIDENT_PROMPT",
        max_tokens_attr="_MAX_TOKENS_INTERNAL",
        incident_number=incident_number,
        payload=payload,
    )


# ─────────────────────────────────────────────────────────────────
# Gap Analysis handlers
# ─────────────────────────────────────────────────────────────────


def _run_gap_panel(
    *,
    report_kind: str,
    template_attr: str,
    max_tokens_attr: str,
    incident_number: str,
    payload: Dict[str, Any],
) -> Tuple[str, str]:
    from backend.tier1_copilot.gap_analysis import routes as gap_routes
    from backend.tier1_copilot.gap_analysis.ticket_lookup import find_ticket_by_incident_number
    from backend.tier1_copilot.gap_analysis.prompts import (
        BLAMELESS_POSTMORTEM_PROMPT,
        GAP_ANALYSIS_MASTER_PROMPT,
    )

    templates = {
        "GAP_ANALYSIS_MASTER_PROMPT": GAP_ANALYSIS_MASTER_PROMPT,
        "BLAMELESS_POSTMORTEM_PROMPT": BLAMELESS_POSTMORTEM_PROMPT,
    }
    template = templates[template_attr]
    max_tokens = getattr(gap_routes, max_tokens_attr)

    ticket = find_ticket_by_incident_number(incident_number)
    if ticket is None:
        raise NonRetryableJobError(f"ticket_not_found: {incident_number}")

    try:
        ticket_json = json.dumps(ticket, ensure_ascii=False, default=str)
    except Exception as exc:
        raise NonRetryableJobError(f"ticket_serialisation_failed: {exc}")

    md, _came_from_cache = gap_routes._cached_or_llm(
        report_kind=report_kind,
        incident_number=incident_number,
        template=template,
        ticket_json=ticket_json,
        max_tokens=max_tokens,
        regenerate=bool(payload.get("regenerate", False)),
    )
    if not md or not md.strip():
        raise NonRetryableJobError("empty_llm_output")
    return (report_kind, incident_number)


def _handle_gap_analysis(incident_number: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    return _run_gap_panel(
        report_kind=REPORT_KIND_GAP_ANALYSIS,
        template_attr="GAP_ANALYSIS_MASTER_PROMPT",
        max_tokens_attr="_MAX_TOKENS_GAP_ANALYSIS",
        incident_number=incident_number,
        payload=payload,
    )


def _handle_post_mortem(incident_number: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    return _run_gap_panel(
        report_kind=REPORT_KIND_POST_MORTEM,
        template_attr="BLAMELESS_POSTMORTEM_PROMPT",
        max_tokens_attr="_MAX_TOKENS_POST_MORTEM",
        incident_number=incident_number,
        payload=payload,
    )


# ─────────────────────────────────────────────────────────────────
# Ingestion handler — moves document ingestion out of the API
# process onto the worker fleet.
#
# Implementation is a thin shim around the existing async
# ``index_file_job`` in backend.api. We deliberately do NOT
# duplicate the ingestion pipeline here — Phase 1's rule applies:
# move the call site, don't rewrite the logic.
#
# Trade-off accepted: ``from backend.api import index_file_job``
# pulls in the FastAPI app object. In the worker process this is
# harmless (no port bound) and costs ~1 s of additional cold-start
# import time. The alternative (extracting index_file_job into its
# own module) would touch hundreds of lines of api.py to thread
# through the bedrock client / storage / s3_upload_provider globals
# — exactly the "disturb the core code" we agreed to avoid.
# ─────────────────────────────────────────────────────────────────


def _handle_ingest_document(
    file_id: str,
    payload: Dict[str, Any],
) -> Tuple[str, str]:
    """Run one document through the existing ingestion pipeline.

    ``file_id`` is the canonical key for ingestion jobs (the
    queue's idempotency column). The actual work uses the parameter
    bundle inside ``payload`` — see ``ingestion_queue.enqueue_ingestion_job``
    for the shape contract.

    Returns ``("ingest_document", job_id)`` for the worker's
    ``mark_done`` bookkeeping — the work product itself lives in
    the ``chunks`` + ``embeddings`` tables, not in ``report_cache``.
    """
    import asyncio

    # Lazy import — pulls in backend.api which constructs the FastAPI
    # app + module-level globals (bedrock client, storage providers,
    # SES). In the worker process there's no inbound port so the FastAPI
    # app stays inert; only the global side effects matter.
    from backend.api import index_file_job

    job_id = payload.get("job_id")
    if not job_id:
        raise NonRetryableJobError("ingest_document payload missing job_id")

    storage_uri = payload.get("storage_uri") or ""
    filename = payload.get("filename") or ""
    file_type = payload.get("file_type") or "kb"
    payload_file_id = payload.get("file_id") or file_id
    owner_id = payload.get("owner_id")
    file_size_mb = float(payload.get("file_size_mb") or 0.0)
    doc_kind = payload.get("doc_kind") or "ticket"

    # ``index_file_job`` is async (uses asyncio.to_thread internally
    # for the parse + embed batches). We run it in a fresh event
    # loop here — the worker process is single-threaded so a per-job
    # asyncio.run is safe.
    asyncio.run(
        index_file_job(
            job_id,
            storage_uri,
            filename,
            file_type,
            payload_file_id,
            owner_id,
            file_size_mb,
            doc_kind,
        )
    )

    return (JOB_KIND_INGEST_DOCUMENT, job_id)


# ─────────────────────────────────────────────────────────────────
# Registry — single point of truth for worker dispatch.
#
# Adding a new job kind = add a handler above and an entry here.
# No other code change needed in the queue or worker.
# ─────────────────────────────────────────────────────────────────
HANDLERS: Dict[str, Callable[[str, Dict[str, Any]], Tuple[str, str]]] = {
    REPORT_KIND_RCA_CUSTOMER:    _handle_rca_customer,
    REPORT_KIND_RCA_INTERNAL:    _handle_rca_internal,
    REPORT_KIND_GAP_ANALYSIS:    _handle_gap_analysis,
    REPORT_KIND_POST_MORTEM:     _handle_post_mortem,
    JOB_KIND_INGEST_DOCUMENT:    _handle_ingest_document,
}
