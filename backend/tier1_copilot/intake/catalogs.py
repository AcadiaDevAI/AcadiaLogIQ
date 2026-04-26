"""Sprint 9 — IntakeCatalogs builder.

Loads four in-memory catalogs from `chunks.metadata_json`:
  - severities (always {"P1","P2","P3","P4"} — fixed enum)
  - asset_families (Sprint 7 derive_asset_family — strip trailing digits)
  - alert_types (normalised Detected_Symptom strings)
  - customers (normalised customer names)

The validator consumes these to reject hallucinated LLM outputs and
the prompt builder samples them as `hints` to keep the LLM grounded
in the corpus's known vocabulary.

Catalog construction is **lazy** — the singleton initialises on first
`get_intake_catalogs()` call and stays in memory for the process. An
admin reindex can call `IntakeCatalogs.rebuild_from_db(engine)` to
refresh after bulk ingestion.

NO new dependency: difflib's `SequenceMatcher` is the fuzzy-match
engine, used by validator.py.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Reuse the Sprint 7 prefix-strip semantics so asset families align
# across modules without duplicating logic in two places.
_ASSET_FAMILY_RE = re.compile(r"^(.+?)[-_]?\d+$")


def derive_asset_family(asset_name: Optional[str]) -> Optional[str]:
    if not asset_name:
        return None
    s = str(asset_name).strip()
    if not s:
        return None
    m = _ASSET_FAMILY_RE.match(s)
    if m:
        return m.group(1).strip("-_").lower() or None
    return s.lower()


_CUSTOMER_SUFFIXES = (
    " inc", " inc.", " corp", " corp.", " corporation",
    " llc", " llc.", " ltd", " ltd.", " limited", " co", " co.",
    " gmbh", " plc", " s.a.", " s.a", " ag",
)


def normalise_customer(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip().lower()
    if not s:
        return None
    # strip common legal-entity suffixes for catalog comparison
    for suffix in _CUSTOMER_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].rstrip(",.;: ")
            break
    return s or None


_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalise_alert_type(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    s = _PUNCT_RE.sub(" ", str(text).lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


# ─────────────────────────────────────────────────────────────
# IntakeCatalogs container
# ─────────────────────────────────────────────────────────────
class IntakeCatalogs:
    """In-memory catalogs derived once from the ticket corpus."""

    def __init__(self) -> None:
        self.severities: Set[str] = {"P1", "P2", "P3", "P4"}
        self.asset_families: Set[str] = set()
        self.alert_types: Set[str] = set()
        self.customers: Set[str] = set()
        self.asset_family_index: Dict[str, str] = {}
        self.alert_type_index: Dict[str, str] = {}
        self.customer_index: Dict[str, str] = {}
        # Frequency counters for prompt-hint sampling — most common first.
        self._asset_family_counts: Counter = Counter()
        self._alert_type_counts: Counter = Counter()
        self._customer_counts: Counter = Counter()
        self._lock = threading.RLock()
        self._built = False

    # ── readers ──────────────────────────────────────────────
    def is_built(self) -> bool:
        return self._built

    def top_asset_families(self, n: int = 30) -> List[str]:
        return [t for t, _ in self._asset_family_counts.most_common(n)]

    def top_alert_types(self, n: int = 30) -> List[str]:
        return [t for t, _ in self._alert_type_counts.most_common(n)]

    def top_customers(self, n: int = 30) -> List[str]:
        return [t for t, _ in self._customer_counts.most_common(n)]

    # ── builders ────────────────────────────────────────────
    def rebuild_from_rows(self, rows: Iterable[dict]) -> None:
        """Build catalogs from an iterable of {"metadata_json": {...}}.

        Defensive: missing keys / non-dict shapes are skipped silently.
        """
        max_terms = int(getattr(
            settings, "INTAKE_CATALOG_MAX_TERMS_PER_TYPE", 10000,
        ))
        af_counts: Counter = Counter()
        at_counts: Counter = Counter()
        c_counts: Counter = Counter()
        af_index: Dict[str, str] = {}
        at_index: Dict[str, str] = {}
        c_index: Dict[str, str] = {}

        for row in rows:
            md = row.get("metadata_json") if isinstance(row, dict) else None
            if not isinstance(md, dict):
                continue
            meta = md.get("Metadata") if isinstance(md.get("Metadata"), dict) else {}

            # Asset families — Affected_Assets[] (JSONB array per Sprint 7
            # correction #1) plus Target_Service as a fallback string.
            assets = meta.get("Affected_Assets")
            if isinstance(assets, list):
                for a in assets:
                    fam = derive_asset_family(a if isinstance(a, str) else str(a))
                    if fam:
                        af_counts[fam] += 1
                        af_index.setdefault(fam, fam)
                        # Also index the raw lowercase form as alias.
                        if isinstance(a, str):
                            af_index.setdefault(a.strip().lower(), fam)
            elif isinstance(assets, str):
                fam = derive_asset_family(assets)
                if fam:
                    af_counts[fam] += 1
                    af_index.setdefault(fam, fam)

            target = meta.get("Target_Service")
            if isinstance(target, str):
                fam = derive_asset_family(target)
                if fam:
                    af_counts[fam] += 1
                    af_index.setdefault(fam, fam)
                    af_index.setdefault(target.strip().lower(), fam)

            # Alert types — Detected_Symptom (lower-priority: Origin_Event).
            ssm = md.get("Symptom_Solution_Mapping") \
                if isinstance(md.get("Symptom_Solution_Mapping"), dict) \
                else {}
            for src in (ssm.get("Detected_Symptom"), ssm.get("Origin_Event")):
                norm = normalise_alert_type(src)
                if norm:
                    at_counts[norm] += 1
                    at_index.setdefault(norm, norm)
                    if isinstance(src, str):
                        at_index.setdefault(src.strip().lower(), norm)

            # Customer — primary path Metadata.customer_name; fallbacks
            # Metadata.Customer_Name then Metadata.Customer (spec §3).
            cust_raw = (
                meta.get("customer_name")
                or meta.get("Customer_Name")
                or meta.get("Customer")
            )
            cust = normalise_customer(cust_raw)
            if cust:
                c_counts[cust] += 1
                c_index.setdefault(cust, cust)
                if isinstance(cust_raw, str):
                    c_index.setdefault(cust_raw.strip().lower(), cust)

        # Apply LRU-like cap — keep the most common N terms per catalog.
        af_kept = {t for t, _ in af_counts.most_common(max_terms)}
        at_kept = {t for t, _ in at_counts.most_common(max_terms)}
        c_kept = {t for t, _ in c_counts.most_common(max_terms)}

        with self._lock:
            self.asset_families = af_kept
            self.alert_types = at_kept
            self.customers = c_kept
            self._asset_family_counts = af_counts
            self._alert_type_counts = at_counts
            self._customer_counts = c_counts
            # Trim aliases that point at evicted canonical values.
            self.asset_family_index = {
                k: v for k, v in af_index.items() if v in af_kept
            }
            self.alert_type_index = {
                k: v for k, v in at_index.items() if v in at_kept
            }
            self.customer_index = {
                k: v for k, v in c_index.items() if v in c_kept
            }
            self._built = True

        logger.info(
            "[intake] catalogs built: assets=%d alert_types=%d customers=%d",
            len(af_kept), len(at_kept), len(c_kept),
        )

    def rebuild_from_db(self, engine: Any) -> None:
        """Pull ticket metadata_json rows from the DB and rebuild.
        Graceful: any DB failure leaves the prior catalogs intact."""
        rows: List[dict] = []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(
                    """
                    SELECT metadata_json
                    FROM chunks
                    WHERE metadata_json->>'doc_kind' = 'ticket'
                       OR metadata_json->'Metadata' IS NOT NULL
                    """
                )).mappings().all()
                rows = [dict(r) for r in result]
        except Exception as exc:
            logger.warning(
                "[intake] catalog DB rebuild failed (keeping prior state): %s",
                exc,
            )
            return
        self.rebuild_from_rows(rows)

    def health_snapshot(self) -> Tuple[int, int, int, int]:
        """Return (severities, asset_families, alert_types, customers)
        sizes — used by /intake/health."""
        with self._lock:
            return (
                len(self.severities),
                len(self.asset_families),
                len(self.alert_types),
                len(self.customers),
            )


# ─────────────────────────────────────────────────────────────
# Module-level singleton with lazy first-build
# ─────────────────────────────────────────────────────────────
_SINGLETON: Optional[IntakeCatalogs] = None
_SINGLETON_LOCK = threading.Lock()


def get_intake_catalogs(*, engine: Any = None, lazy_build: bool = True) -> IntakeCatalogs:
    """Return the process-wide IntakeCatalogs singleton.

    On first call, if `engine` is provided (or can be lazily imported),
    runs `rebuild_from_db` so subsequent route handlers see a populated
    catalog. Subsequent callers just get the in-memory singleton.
    """
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = IntakeCatalogs()
                if lazy_build:
                    eng = engine
                    if eng is None:
                        try:
                            from backend.db.connection import engine as _db_engine
                            eng = _db_engine
                        except Exception:
                            eng = None
                    if eng is not None:
                        _SINGLETON.rebuild_from_db(eng)
    return _SINGLETON


def reset_singleton_for_tests() -> None:
    """Test helper — discards the singleton so the next get rebuilds."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        _SINGLETON = None
