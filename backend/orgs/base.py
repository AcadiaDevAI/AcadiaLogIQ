"""OrgProfile — the base org profile.

Defaults on this base class == current shared (Acadia) behavior. Subclass per
org and override ONLY what differs; everything you don't override keeps
inheriting the shared core, so shared fixes/features apply to every org
automatically. This is the seam that replaces per-org `if` branches.

Phase 1 carries public config (display_name / theme / enabled_flows /
feature_flags — surfaced via GET /orgs/me/config). Behavior hooks
(system_prompt, historic_matcher, …) are added here as override points and
wired into shared routes incrementally in later phases.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Set

# Every org gets these flows unless it overrides `enabled_flows`.
DEFAULT_FLOWS: frozenset = frozenset(
    {"tier1", "chat", "rca", "gap", "ticket_filter", "escalation"}
)


class OrgProfile:
    """Base = Acadia / shared-core defaults. Override per org in subclasses."""

    # Normalized slug this profile registers under (see registry._norm).
    slug: str = "default"
    display_name: str = "Acadia LogIQ"
    theme: Dict[str, Any] = {}

    # KB retrieval policy. When True, a KB query that mentions an identifier
    # the exact-match lookup can't find still falls through to the full
    # hybrid search (vector + BM25 + keyword across json/pdf/excel/…) instead
    # of returning "identifier not found". Default False = shared behavior
    # (only natural-language-shaped queries fall through). See
    # backend/retrieval/orchestrator.py::retrieve.
    kb_search_all_on_identifier_miss: bool = False

    # Extra doc_kinds to FOLD INTO a doc-kind-scoped KB search. Stage 4
    # "Search KB / SOP" scopes retrieval to ("sop", "kb"); an org can widen
    # that so KB search ALSO surfaces other corpora. US Pharma adds "ticket"
    # so JSON ticket documents show up in KB search (JSON auto-classifies as
    # doc_kind="ticket"). Default () = shared behavior: KB search stays SOP/KB
    # only. Only applied when the search is already doc-kind-scoped; an
    # unrestricted (None) search already sees every kind. See
    # backend/api.py /ask KB retrieval branch.
    kb_search_extra_doc_kinds: tuple = ()

    # Retrieval concurrency policy. The hybrid orchestrator runs its search
    # channels (vector / BM25 / keyword / metadata) in parallel threads. On
    # the DEFAULT path they all share ONE copied request context via
    # ctx.run — but a contextvars.Context can only be entered by one thread
    # at a time, so concurrent channels raise "cannot enter context: ... is
    # already entered" and silently drop BM25 + keyword (only vector, which
    # enters first, survives). When True, each channel gets its OWN copy of
    # the request context, so all channels run without contending — restoring
    # true hybrid retrieval. Default False = legacy shared-context behavior.
    # See backend/retrieval/orchestrator.py.
    retrieval_per_channel_context: bool = False

    # Tier-1 intake policy. When True, the tier-1 analyze flow requires a
    # store_id and hard-scopes historic matches to that store (US Pharma).
    # Default False = shared behavior (Acadia's asset/alert intake, no store
    # scoping). See backend/tier1_copilot/routes.py + retrieval.py.
    tier1_requires_store_id: bool = False

    def __init__(self, org_id: Optional[uuid.UUID] = None, org_slug: Optional[str] = None):
        self.org_id = org_id
        # Prefer the real per-request slug; fall back to the class slug.
        self.org_slug = org_slug or self.slug

    # ---- public config (safe to expose to the frontend) -------------------
    @property
    def enabled_flows(self) -> Set[str]:
        return set(DEFAULT_FLOWS)

    @property
    def feature_flags(self) -> Dict[str, bool]:
        return {}

    def public_config(self) -> Dict[str, Any]:
        """Shape returned by GET /orgs/me/config — the single source of truth
        the frontend reads for org display name, theme, and enabled flows."""
        return {
            "org_id": str(self.org_id) if self.org_id else None,
            "slug": self.org_slug,
            "display_name": self.display_name,
            "theme": dict(self.theme or {}),
            "enabled_flows": sorted(self.enabled_flows),
            "feature_flags": dict(self.feature_flags),
        }

    # ---- behavior override points (Phase 2+) ------------------------------
    # Default implementations return None / shared-core sentinels; shared code
    # treats None as "use the existing default". Override per org to diverge.
    def is_flow_enabled(self, flow: str) -> bool:
        return flow in self.enabled_flows

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(slug={self.org_slug!r})"
