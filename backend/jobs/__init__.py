"""
backend.jobs — async job queue (Postgres-backed) for long-running
LLM generations.

Subpackage layout
-----------------
* ``queue.py``        — enqueue / claim / complete / fail. The data
                         contract callers depend on.
* ``handlers.py``     — registry of ``kind → callable`` mappings. The
                         worker dispatches a claimed row to the right
                         handler.
* ``worker.py``       — main loop that the worker container runs.
                         Imports queue + handlers, polls in a loop.

Importing this package has no side effects; modules are pulled in
lazily by callers (api.py imports queue.enqueue + queue.get; the
worker container imports worker.run).
"""

from .queue import (
    enqueue,
    get_job,
    claim_next,
    mark_running,
    mark_done,
    mark_failed,
    JobStatus,
    JOB_KIND_RCA_CUSTOMER,
    JOB_KIND_RCA_INTERNAL,
    JOB_KIND_GAP_ANALYSIS,
    JOB_KIND_POST_MORTEM,
)

__all__ = [
    "enqueue",
    "get_job",
    "claim_next",
    "mark_running",
    "mark_done",
    "mark_failed",
    "JobStatus",
    "JOB_KIND_RCA_CUSTOMER",
    "JOB_KIND_RCA_INTERNAL",
    "JOB_KIND_GAP_ANALYSIS",
    "JOB_KIND_POST_MORTEM",
]
