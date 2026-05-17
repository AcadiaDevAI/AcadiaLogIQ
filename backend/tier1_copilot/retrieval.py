"""Two-stage retrieval + weighted ranking for Tier-1 Copilot.

Stage 1 (exact): SQL ILIKE / token-match against the three denormalized
columns added by migration 038 (alert_signature, fingerprints_text,
component_category). If a candidate scores >= 0.80 we short-circuit
and return immediately — no embedding / LLM cost.

Stage 2 (hybrid): invoked only when Stage 1 is weak. Runs pgvector
cosine similarity (reusing the existing `embeddings` join) and BM25
over chunks.content, merges candidates, then applies a weighted
per-field scoring pass per §7 of the sprint spec.

Confidence banding is a pure function of the final score:
  >= 0.85 → High
  0.60-0.84 → Medium
  0.40-0.59 → Low
  < 0.40 → None (caller returns graceful no-match)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Sprint 6 baseline weights — fallback if the Sprint 7 settings dict
# isn't available for any reason.
_WEIGHTS = {
    "alert_type_match": 0.30,
    "asset_match": 0.20,
    "fingerprint_match": 0.20,
    "technology_match": 0.10,
    "vector_similarity": 0.10,
    "resolution_quality": 0.10,
}


def _active_weights() -> Dict[str, float]:
    """Return the Sprint 7 ranking weights from settings (with fallback).

    Sprint 7 shifts weight into `recency`, `success_frequency`,
    `same_customer_boost`, and `same_asset_family` (§3 of the sprint
    spec)."""
    return getattr(settings, "TIER1_RANKING_WEIGHTS", _WEIGHTS)


# ─────────────────────────────────────────────────────────────
# Sprint 7 — asset family prefix extraction
# ─────────────────────────────────────────────────────────────
def derive_asset_family(asset_name: Optional[str]) -> Optional[str]:
    """Extract asset family prefix from an asset name.

    Examples:
      NY4-CORE-RTR-01      → ny4-core-rtr
      SFO-EDGE-FW-02       → sfo-edge-fw
      V-Desktop-Session-17 → v-desktop-session

    Returns None if the input is blank. Returns the lowercased input
    (trailing numeric segment stripped, bounding hyphens/underscores
    trimmed) otherwise. Intentionally conservative — the boost weight
    is only 0.075 so occasional over-matches have bounded impact.
    """
    if not asset_name:
        return None
    stripped = asset_name.strip()
    if not stripped:
        return None
    m = re.match(r"^(.+?)[-_]?\d+$", stripped)
    if m:
        return m.group(1).strip("-_").lower() or None
    return stripped.lower()

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return _TOKEN_RE.findall(s.lower())


def confidence_band(score: float) -> str:
    if score >= settings.TIER1_HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if score >= settings.TIER1_MIN_CONFIDENCE_THRESHOLD:
        return "Medium"
    if score >= 0.40:
        return "Low"
    return "None"


def retrieve_top_matches(
    *,
    normalized: Dict[str, Any],
    alert_input: Dict[str, Any],
    engine: Any,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a list of candidate dicts, each with:
      chunk_id, metadata_json, final_score, confidence, component, fingerprints

    Never raises — any DB or embed failure degrades to an empty list so
    the route handler can return a graceful "no match" payload.
    """
    k = top_k or int(getattr(settings, "TIER1_TOP_K", 5))

    # ---- Stage 1: exact ----------------------------------------------------
    exact_hits = _exact_lookup(normalized, alert_input, engine)
    if exact_hits and exact_hits[0]["final_score"] >= 0.80:
        ranked = sorted(exact_hits, key=lambda r: r["final_score"], reverse=True)
        logger.info(
            "[tier1_copilot] stage1 hit score=%.3f chunk_id=%s",
            ranked[0]["final_score"], ranked[0].get("chunk_id"),
        )
        return ranked[:k]

    # ---- Stage 2: hybrid ---------------------------------------------------
    try:
        candidates = _hybrid_candidates(
            normalized, alert_input, engine, embed_fn,
        )
    except Exception as exc:
        logger.warning("[tier1_copilot] hybrid candidates failed: %s", exc)
        candidates = exact_hits

    ranked = _weighted_rank(candidates, alert_input, normalized)
    if ranked:
        logger.info(
            "[tier1_copilot] stage2 top score=%.3f chunk_id=%s total_cands=%d",
            ranked[0]["final_score"], ranked[0].get("chunk_id"), len(candidates),
        )
    return ranked[:k]


# ─────────────────────────────────────────────────────────────
# Stage 1 — exact SQL lookup against denormalized columns
# ─────────────────────────────────────────────────────────────
def _exact_lookup(
    normalized: Dict[str, Any],
    alert_input: Dict[str, Any],
    engine: Any,
) -> List[Dict[str, Any]]:
    try:
        from sqlalchemy import text
    except Exception:
        return []

    asset = (alert_input.get("asset_name") or "").lower()
    alert_type = (alert_input.get("alert_type") or "").lower()
    signature = normalized.get("alert_signature") or ""

    sql = text(
        """
        SELECT
            id,
            metadata_json,
            alert_signature,
            fingerprints_text,
            component_category,
            CASE
                WHEN alert_signature = :sig THEN 1.0
                WHEN alert_signature ILIKE '%' || :asset || '%'
                     AND alert_signature ILIKE '%' || :alert_type || '%'
                     THEN 0.85
                WHEN fingerprints_text ILIKE '%' || :alert_type || '%'
                     THEN 0.70
                ELSE 0.60
            END AS score
        FROM chunks
        WHERE (metadata_json->>'doc_kind' IS NULL
               OR metadata_json->>'doc_kind' = 'ticket')
          AND (
              alert_signature ILIKE '%' || :asset || '%'
              OR fingerprints_text ILIKE '%' || :alert_type || '%'
              OR alert_signature = :sig
          )
        ORDER BY score DESC,
                 COALESCE(
                     (metadata_json->'Metadata'->>'Resolution_Quality_Score')::int,
                     0
                 ) DESC
        LIMIT 10
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sql,
                {"sig": signature, "asset": asset, "alert_type": alert_type},
            ).mappings().all()
    except Exception as exc:
        logger.warning("[tier1_copilot] stage1 SQL failed: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        md = row["metadata_json"] or {}
        if isinstance(md, str):
            import json
            try:
                md = json.loads(md)
            except Exception:
                md = {}
        out.append({
            "chunk_id": row["id"],
            "metadata_json": md,
            "final_score": float(row["score"] or 0.0),
            "confidence": confidence_band(float(row["score"] or 0.0)),
            "component": row.get("component_category"),
            "fingerprints_text": row.get("fingerprints_text"),
        })
    return out


# ─────────────────────────────────────────────────────────────
# Stage 2 — hybrid candidates via pgvector (if embed_fn available)
#           + a BM25-ish tsvector rank over chunks.content
# ─────────────────────────────────────────────────────────────
def _hybrid_candidates(
    normalized: Dict[str, Any],
    alert_input: Dict[str, Any],
    engine: Any,
    embed_fn: Optional[Callable[[str], List[float]]],
) -> List[Dict[str, Any]]:
    try:
        from sqlalchemy import text
    except Exception:
        return []

    search_text = normalized.get("search_text") or ""
    candidates: Dict[str, Dict[str, Any]] = {}

    # ---- pgvector cosine (optional: only if we can embed) -----------------
    if embed_fn is not None and search_text:
        try:
            query_vec = embed_fn(search_text)
        except Exception as exc:
            logger.warning("[tier1_copilot] embed failed: %s", exc)
            query_vec = None
        if query_vec:
            _vec_literal = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
            vec_sql = text(
                """
                SELECT
                    c.id AS id,
                    c.metadata_json AS metadata_json,
                    c.alert_signature AS alert_signature,
                    c.fingerprints_text AS fingerprints_text,
                    c.component_category AS component_category,
                    (e.embedding <=> CAST(:qv AS vector)) AS distance
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE (c.metadata_json->>'doc_kind' IS NULL
                       OR c.metadata_json->>'doc_kind' = 'ticket')
                ORDER BY e.embedding <=> CAST(:qv AS vector)
                LIMIT 30
                """
            )
            try:
                with engine.connect() as conn:
                    vec_rows = conn.execute(vec_sql, {"qv": _vec_literal}).mappings().all()
                for r in vec_rows:
                    sim = max(0.0, 1.0 - float(r["distance"] or 1.0))
                    candidates.setdefault(r["id"], {
                        "chunk_id": r["id"],
                        "metadata_json": r["metadata_json"] or {},
                        "alert_signature": r.get("alert_signature"),
                        "fingerprints_text": r.get("fingerprints_text"),
                        "component_category": r.get("component_category"),
                        "_vector_sim": sim,
                    })
            except Exception as exc:
                logger.warning("[tier1_copilot] pgvector stage failed: %s", exc)

    # ---- tsvector rank on chunks.content ---------------------------------
    if search_text:
        ts_sql = text(
            """
            SELECT
                c.id AS id,
                c.metadata_json AS metadata_json,
                c.alert_signature AS alert_signature,
                c.fingerprints_text AS fingerprints_text,
                c.component_category AS component_category,
                ts_rank(
                    to_tsvector('english', COALESCE(c.content, '')),
                    plainto_tsquery('english', :q)
                ) AS rank
            FROM chunks c
            WHERE (c.metadata_json->>'doc_kind' IS NULL
                   OR c.metadata_json->>'doc_kind' = 'ticket')
              AND to_tsvector('english', COALESCE(c.content, ''))
                  @@ plainto_tsquery('english', :q)
            ORDER BY rank DESC
            LIMIT 30
            """
        )
        try:
            with engine.connect() as conn:
                ts_rows = conn.execute(ts_sql, {"q": search_text}).mappings().all()
            for r in ts_rows:
                entry = candidates.setdefault(r["id"], {
                    "chunk_id": r["id"],
                    "metadata_json": r["metadata_json"] or {},
                    "alert_signature": r.get("alert_signature"),
                    "fingerprints_text": r.get("fingerprints_text"),
                    "component_category": r.get("component_category"),
                    "_vector_sim": 0.0,
                })
                entry["_ts_rank"] = float(r["rank"] or 0.0)
        except Exception as exc:
            logger.warning("[tier1_copilot] tsvector stage failed: %s", exc)

    return list(candidates.values())


# ─────────────────────────────────────────────────────────────
# Weighted rerank — pure Python, deterministic, test-friendly
# ─────────────────────────────────────────────────────────────
def _weighted_rank(
    candidates: List[Dict[str, Any]],
    alert_input: Dict[str, Any],
    normalized: Dict[str, Any],
) -> List[Dict[str, Any]]:
    weights = _active_weights()

    alert_type_toks = set(_tokens(alert_input.get("alert_type")))
    asset_toks = set(_tokens(alert_input.get("asset_name")))
    tech_toks = set(_tokens(alert_input.get("technology")))
    fp_toks = set(_tokens(alert_input.get("error_code")))

    alert_customer = (alert_input.get("customer") or "").strip().lower()
    alert_asset_family = derive_asset_family(alert_input.get("asset_name"))

    now = datetime.now(timezone.utc)

    scored: List[Dict[str, Any]] = []
    for c in candidates:
        md = c.get("metadata_json") or {}
        meta = md.get("Metadata") if isinstance(md, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ssm = md.get("Symptom_Solution_Mapping") if isinstance(md, dict) else {}
        ssm = ssm if isinstance(ssm, dict) else {}

        ticket_alert_toks = set(
            _tokens((ssm.get("Detected_Symptom") or ""))
        ) | set(_tokens(c.get("fingerprints_text")))
        ticket_asset_toks = set(_tokens(meta.get("Target_Service"))) | set(
            _tokens(" ".join(str(a) for a in (meta.get("Affected_Assets") or [])))
        )
        ticket_component_toks = set(_tokens(c.get("component_category"))) \
            | set(_tokens(meta.get("component_category")))
        ticket_fp_toks = set(_tokens(c.get("fingerprints_text")))

        comps: Dict[str, float] = {}
        comps["alert_type_match"] = _overlap(alert_type_toks, ticket_alert_toks)
        comps["asset_match"] = _overlap(asset_toks, ticket_asset_toks)
        comps["fingerprint_match"] = _overlap(fp_toks, ticket_fp_toks)
        comps["technology_match"] = _overlap(tech_toks, ticket_component_toks)
        comps["vector_similarity"] = max(
            0.0, min(1.0, float(c.get("_vector_sim", 0.0)))
        )

        q = meta.get("Resolution_Quality_Score") or 0
        try:
            q_norm = max(0.0, min(1.0, float(q) / 5.0))
        except Exception:
            q_norm = 0.0
        comps["resolution_quality"] = q_norm

        # Sprint 7 — recency / success-frequency / customer / asset-family.
        # Recency — linear decay over 365 days. Missing timestamps
        # score 0.0 (neutral — do not penalize tickets without a
        # parseable created_at).
        comps["recency"] = _recency_score(meta, now)
        # Success frequency — fraction of sibling tickets (same
        # primary_fix) tagged as resolved. If we don't have sibling
        # telemetry, treat quality_score as a proxy.
        comps["success_frequency"] = q_norm
        # Customer boost — 1.0 iff the alert's customer field matches
        # the ticket's customer_name. Case-insensitive exact compare.
        ticket_customer = (
            (meta.get("customer_name") or meta.get("Customer_Name") or "")
            .strip()
            .lower()
        )
        comps["same_customer_boost"] = (
            1.0 if alert_customer and alert_customer == ticket_customer else 0.0
        )
        # Asset-family boost — prefix-match via derive_asset_family.
        ticket_family = (c.get("asset_family") or "").strip().lower() or (
            derive_asset_family(
                (meta.get("Affected_Assets") or [None])[0]
                if isinstance(meta.get("Affected_Assets"), list)
                else None
            )
            or ""
        )
        comps["same_asset_family"] = (
            1.0
            if alert_asset_family
            and ticket_family
            and alert_asset_family == ticket_family
            else 0.0
        )

        score = 0.0
        for key, w in weights.items():
            score += w * comps.get(key, 0.0)

        c["final_score"] = round(min(1.0, max(0.0, score)), 4)
        c["confidence"] = confidence_band(c["final_score"])
        c["_score_components"] = comps
        scored.append(c)

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    return scored


def _recency_score(meta: Dict[str, Any], now: datetime) -> float:
    raw = (
        meta.get("resolved_date")
        or meta.get("Resolved_Date")
        or meta.get("created_date")
        or meta.get("Created_Date")
    )
    if not raw:
        return 0.0
    try:
        s = str(raw)
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s[: len(fmt)].replace("Z", ""), fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return 0.0
        age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
        # Linear decay to 0 at 365 days, floor at 0.
        return max(0.0, min(1.0, 1.0 - (age_days / 365.0)))
    except Exception:
        return 0.0


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    # Jaccard-like, capped at 1.0
    return min(1.0, inter / max(1, len(a)))
