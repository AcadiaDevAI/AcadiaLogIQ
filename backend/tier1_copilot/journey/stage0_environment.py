"""Sprint 12.4 — Environment Context & Tech Component Profile.

Per spec:
  Define the master technology blast radius by aggregating the
  environment data from all cohort tickets. Deduplicate the list of
  vendor products, hardware/software components, and logical entities.
  Surface one comprehensive "Environment Context & Tech Stack Profile"
  indicating all systems involved in the historical cluster.

Source field families (per cohort ticket):
  - Metadata.Dynamic_Domain_Payload.Domain_Type           (string)
  - Metadata.component_category                           (string)
  - RAG_Potency_Metadata.Synaptic_Cluster_ID              (string)
  - Engagement_Analysis.Products_Involved                 (List[str])
  - Metadata.technical_entities                           (List[str])

Aggregation rules:
  - Deduplicate by case-insensitive normalisation, but emit the
    first-seen original casing/spelling. Preserves the cluster's
    canonical naming (e.g. ``Cisco IOS-XR``, not ``cisco ios-xr``).
  - Preserve **first-seen order across the cohort** — rank-1 ticket's
    values lead, rank-2's new additions follow, etc. The cohort comes
    in retrieval-similarity order from `ticket_loader.load_cohort_metadata`,
    so this ordering surfaces the dominant tech for the matched cluster
    on top.
  - For ``Synaptic_Cluster_ID`` only: skip the sentinel value ``"Isolated"``
    (which means "not in any named cluster" — surfacing it as a tag
    would mislead the reader).

Logging:
  Emits one INFO line per call summarising cohort coverage:
  ``[stage0_env] cohort=N tickets_with_data=M domains=X components=Y
  clusters=Z products=P entities=E`` so prod monitoring can spot
  ingestion regressions (a batch of tickets without populated
  environment data shows up as a coverage drop).

Pure function — no DB, no LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import EnvironmentProfile


logger = logging.getLogger("acadia-log-iq")


_ISOLATED_SENTINEL = "isolated"


def _safe_get(d: Any, *path: str) -> Any:
    """Walk a dotted path through nested dicts; return value or None."""
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _coerce_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _coerce_list(v: Any) -> List[str]:
    """Tolerate single-string fields that should have been lists.

    The gold schema typically writes ``Products_Involved`` as a list,
    but a hand-edited or legacy ticket may carry a single string. Both
    shapes are flattened to ``List[str]`` here so the dedupe loop is
    uniform.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if x is not None and str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    # Any other shape (dict, int) is silently ignored — environment
    # fields are documented as scalar/list-of-scalar; richer shapes
    # would require domain-specific rendering we can't infer.
    return []


def _add(seen: Dict[str, str], ordered: List[str], value: str, *, drop_isolated: bool = False) -> None:
    """Insert `value` into `ordered` if its case-folded form is new.

    `seen` maps case-folded → original-casing-first-seen so we never
    emit "Cisco IOS-XR" twice as "cisco ios-xr" + "Cisco IOS-XR". When
    ``drop_isolated`` is True, the literal sentinel ``"Isolated"`` is
    silently skipped (used for Synaptic_Cluster_ID only).
    """
    if not value:
        return
    key = value.casefold()
    if drop_isolated and key == _ISOLATED_SENTINEL:
        return
    if key in seen:
        return
    seen[key] = value
    ordered.append(value)


def build_environment_profile(cohort: List[Dict[str, Any]]) -> EnvironmentProfile:
    """See module docstring."""
    n = len(cohort or [])
    if n == 0:
        return EnvironmentProfile(
            cohort_size=0,
            tickets_with_data=0,
            empty=True,
        )

    domain_types: List[str] = []
    component_categories: List[str] = []
    synaptic_cluster_ids: List[str] = []
    products_involved: List[str] = []
    technical_entities: List[str] = []

    seen_domains: Dict[str, str] = {}
    seen_components: Dict[str, str] = {}
    seen_clusters: Dict[str, str] = {}
    seen_products: Dict[str, str] = {}
    seen_entities: Dict[str, str] = {}

    tickets_with_data = 0

    for ticket in cohort:
        if not isinstance(ticket, dict):
            continue

        contributed = False

        domain = _coerce_str(_safe_get(
            ticket, "Metadata", "Dynamic_Domain_Payload", "Domain_Type",
        ))
        if domain:
            before = len(seen_domains)
            _add(seen_domains, domain_types, domain)
            contributed = contributed or (len(seen_domains) > before) or (domain.casefold() in seen_domains)

        component = _coerce_str(_safe_get(ticket, "Metadata", "component_category"))
        if component:
            before = len(seen_components)
            _add(seen_components, component_categories, component)
            contributed = contributed or (len(seen_components) > before) or (component.casefold() in seen_components)

        cluster = _coerce_str(_safe_get(
            ticket, "RAG_Potency_Metadata", "Synaptic_Cluster_ID",
        ))
        if cluster:
            before = len(seen_clusters)
            _add(seen_clusters, synaptic_cluster_ids, cluster, drop_isolated=True)
            # `_add` skips "Isolated"; only count contribution when the
            # cluster actually landed in the deduped list, OR when the
            # ticket carried any other field below. The bool below
            # rolls up across all fields, so this is just a no-op in
            # the isolated-only-cluster case.
            contributed = contributed or (len(seen_clusters) > before)

        for p in _coerce_list(_safe_get(ticket, "Engagement_Analysis", "Products_Involved")):
            before = len(seen_products)
            _add(seen_products, products_involved, p)
            contributed = contributed or (len(seen_products) > before)

        for e in _coerce_list(_safe_get(ticket, "Metadata", "technical_entities")):
            before = len(seen_entities)
            _add(seen_entities, technical_entities, e)
            contributed = contributed or (len(seen_entities) > before)

        # Even if a ticket only re-stated an already-seen value, it's
        # still a ticket that *carried* environment data — count it.
        # We re-derive contribution from "did the ticket have any
        # non-empty source field" rather than the dedupe deltas above.
        any_field = any([
            _coerce_str(_safe_get(ticket, "Metadata", "Dynamic_Domain_Payload", "Domain_Type")),
            _coerce_str(_safe_get(ticket, "Metadata", "component_category")),
            _coerce_str(_safe_get(ticket, "RAG_Potency_Metadata", "Synaptic_Cluster_ID")),
            bool(_coerce_list(_safe_get(ticket, "Engagement_Analysis", "Products_Involved"))),
            bool(_coerce_list(_safe_get(ticket, "Metadata", "technical_entities"))),
        ])
        if any_field:
            tickets_with_data += 1

    empty = not (
        domain_types or component_categories or synaptic_cluster_ids
        or products_involved or technical_entities
    )

    coverage_pct = round(100.0 * tickets_with_data / n) if n else 0
    logger.info(
        "[stage0_env] cohort=%d tickets_with_data=%d (%d%%) "
        "domains=%d components=%d clusters=%d products=%d entities=%d "
        "empty=%s",
        n, tickets_with_data, coverage_pct,
        len(domain_types), len(component_categories), len(synaptic_cluster_ids),
        len(products_involved), len(technical_entities),
        empty,
    )

    return EnvironmentProfile(
        domain_types=domain_types,
        component_categories=component_categories,
        synaptic_cluster_ids=synaptic_cluster_ids,
        products_involved=products_involved,
        technical_entities=technical_entities,
        cohort_size=n,
        tickets_with_data=tickets_with_data,
        empty=empty,
    )
