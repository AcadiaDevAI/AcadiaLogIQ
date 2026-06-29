"""
Dynamic Vocabulary Learner — schema-agnostic token preservation.

Scans ingested content for three categories of structured tokens and
persists them to learned_vocabulary so query normalization can preserve
them verbatim:
    1. identifier  — INC-NEBULA-772, INC_546, 2019020632366
    2. field_name  — Resolution_Quality_Score, SLA_Target_Met
    3. enum_value  — P1, CLOSED, ESCALATED

Design principle: zero hardcoded customer-specific patterns. New customer
upload -> patterns auto-detected -> queries work with no code changes.

All operations are fail-safe: any DB or parse error is logged and the
caller proceeds without the vocabulary update. This must never block
ingestion or query execution.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import engine

logger = logging.getLogger("acadia-log-iq")


# Heuristic patterns — deliberately broad to catch new customer formats;
# min-occurrence filter rejects noise.
_IDENTIFIER_RE = re.compile(
    r"\b[A-Z]{2,}[-_][A-Z0-9]+(?:[-_][A-Z0-9]+)*\b|\b\d{10,}\b"
)
_FIELD_NAME_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z]+)+\b")
_ENUM_VALUE_RE = re.compile(r"\b[A-Z]{1,6}\d{0,3}\b")

# v2 Bug #2: strict identifier-VALUE pattern anchored to full token.
# Used by the JSON tree walker to classify string values structurally.
_IDENTIFIER_VALUE_RE = re.compile(
    r"^[A-Z]{2,}[-_][A-Z0-9]+(?:[-_][A-Z0-9]+)*$|^\d{10,}$"
)

_DEFAULT_MIN_OCCURRENCE = 3
_DEFAULT_ENUM_MIN_OCCURRENCE = 3


_cache_lock = threading.Lock()
_vocab_cache: Set[str] = set()
_vocab_cache_loaded = False
_type_cache: Dict[str, str] = {}
_type_cache_loaded = False
# Canonical-form alias index: canonical_form → sorted list of learned aliases.
_canonical_alias_cache: Dict[str, List[str]] = {}


# Regexes for camelCase → UPPER_SNAKE canonicalization.
# First pass inserts an underscore before an uppercase letter that starts a
# new word (e.g. 'RAG_PotencyMetadata' from 'RAGPotencyMetadata'); the
# second pass handles the lowercase→uppercase transition.
_CAMEL_SPLIT_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_SPLIT_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def canonicalize_token(token: str) -> str:
    """Collapse camelCase / PascalCase / snake_case / lowercase variants to
    a shared UPPER_SNAKE canonical form.

    Examples:
        RAGPotencyMetadata    → RAG_POTENCY_METADATA
        RAG_Potency_Metadata  → RAG_POTENCY_METADATA
        rag_potency_metadata  → RAG_POTENCY_METADATA
        Resolution_Quality_Score → RESOLUTION_QUALITY_SCORE
        INC-NEBULA-772        → INC-NEBULA-772  (non-alpha separators preserved)

    Identifier-style tokens that already contain hyphens (INC-NEBULA-772) are
    upper-cased but otherwise left alone so ticket IDs survive the round trip.
    """
    if not token:
        return ""
    s = token.strip()
    # Identifier-style: contains a hyphen, no need to snake-split.
    if "-" in s:
        return s.upper()
    # camelCase / PascalCase → insert underscores at word boundaries. When the
    # input already contains underscores (snake_case / Title_Snake) the pass-1
    # regex injects a second underscore next to the existing one; collapse
    # runs of underscores afterward so all three variants converge.
    s = _CAMEL_SPLIT_RE_1.sub(r"\1_\2", s)
    s = _CAMEL_SPLIT_RE_2.sub(r"\1_\2", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _min_occurrence() -> int:
    return int(getattr(settings, "HOTFIX_VOCAB_MIN_OCCURRENCE", _DEFAULT_MIN_OCCURRENCE))


def _enum_min_occurrence() -> int:
    return int(getattr(settings, "VOCABULARY_ENUM_MIN_OCCURRENCE", _DEFAULT_ENUM_MIN_OCCURRENCE))


def _walk_json_tree(
    node: Any,
    field_names: Counter,
    identifiers: Counter,
    enum_candidates: Counter,
) -> None:
    """Recursively walk a JSON tree and classify tokens by structural position.

    Dict keys → field_names.
    String values matching _IDENTIFIER_VALUE_RE → identifiers.
    Short uppercase string values → enum candidates (promoted later based
    on occurrence count).
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key:
                field_names[key] += 1
            _walk_json_tree(value, field_names, identifiers, enum_candidates)
    elif isinstance(node, list):
        for item in node:
            _walk_json_tree(item, field_names, identifiers, enum_candidates)
    elif isinstance(node, str):
        token = node.strip()
        if not token:
            return
        if _IDENTIFIER_VALUE_RE.match(token):
            identifiers[token] += 1
        elif token.isupper() and 1 < len(token) <= 20 and token.isascii():
            enum_candidates[token] += 1


def _parse_as_jsonl(content: str) -> List[Any]:
    """Recover JSON objects from a JSONL or concatenated-object blob."""
    out: List[Any] = []
    # Line-by-line JSONL
    for line in content.splitlines():
        line = line.strip().rstrip(",")
        if not line or line[0] not in "{[":
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    if out:
        return out
    # Last-ditch: split on `}{` boundaries
    parts = re.split(r"\}\s*\{", content)
    for i, part in enumerate(parts):
        if i > 0:
            part = "{" + part
        if i < len(parts) - 1:
            part = part + "}"
        try:
            out.append(json.loads(part))
        except Exception:
            continue
    return out


def learn_vocabulary_from_json(
    json_content: str, file_id: str
) -> Optional[Dict[str, List[str]]]:
    """JSON-tree-aware vocabulary learning (v2 Bug #2).

    Returns the same shape as learn_from_content on success, or None when
    the content isn't parseable as JSON (caller falls back to regex).
    """
    if not json_content:
        return None
    try:
        data = json.loads(json_content)
    except Exception:
        data = _parse_as_jsonl(json_content)
        if not data:
            return None

    field_names: Counter = Counter()
    identifiers: Counter = Counter()
    enum_candidates: Counter = Counter()
    try:
        _walk_json_tree(data, field_names, identifiers, enum_candidates)
    except Exception as exc:
        logger.warning("[vocab_learner] JSON walk failed for %s: %s", file_id, exc)
        return None

    enum_threshold = _enum_min_occurrence()
    result = {
        "identifiers": list(identifiers.keys()),
        "field_names": list(field_names.keys()),
        "enum_values": [t for t, c in enum_candidates.items() if c >= enum_threshold],
    }
    logger.info(
        "[vocab_learner] file=%s JSON-walk learned %d ids, %d fields, %d enums",
        file_id,
        len(result["identifiers"]),
        len(result["field_names"]),
        len(result["enum_values"]),
    )
    return result


def learn_from_content(content: str, file_id: str) -> Dict[str, List[str]]:
    """Scan a document blob, return tokens above the occurrence threshold.

    When VOCABULARY_JSON_STRUCTURAL_PARSING is True and the content parses
    as JSON/JSONL, the tree walker is preferred so keys don't get mislabeled
    as identifiers. Falls through to the regex path on non-JSON content.
    """
    if not content:
        return {"identifiers": [], "field_names": [], "enum_values": []}

    if getattr(settings, "VOCABULARY_JSON_STRUCTURAL_PARSING", False):
        structural = learn_vocabulary_from_json(content, file_id)
        if structural is not None:
            return structural

    threshold = _min_occurrence()
    try:
        ids = Counter(_IDENTIFIER_RE.findall(content))
        fields = Counter(_FIELD_NAME_RE.findall(content))
        enums = Counter(_ENUM_VALUE_RE.findall(content))
    except Exception as exc:
        logger.warning("[vocab_learner] regex scan failed for %s: %s", file_id, exc)
        return {"identifiers": [], "field_names": [], "enum_values": []}

    result = {
        "identifiers":  [t for t, c in ids.items()    if c >= threshold],
        "field_names":  [t for t, c in fields.items() if c >= threshold],
        "enum_values":  [t for t, c in enums.items()  if c >= threshold],
    }
    logger.info(
        "[vocab_learner] file=%s regex-learned %d ids, %d fields, %d enums",
        file_id, len(result["identifiers"]), len(result["field_names"]), len(result["enum_values"]),
    )
    return result


def persist(learned: Dict[str, List[str]], file_id: str) -> int:
    """Upsert learned tokens into learned_vocabulary. Returns rows touched.

    Each row also stores canonical_form so casing variants of the same field
    (RAGPotencyMetadata / RAG_Potency_Metadata / rag_potency_metadata) share
    a single canonical key and can be cross-matched at query time.
    """
    type_map = {
        "identifiers": "identifier",
        "field_names": "field_name",
        "enum_values": "enum_value",
    }

    # Flatten all learned tokens, deduping by token. ON CONFLICT DO UPDATE
    # cannot affect the same (organization_id, token) twice within ONE
    # multi-row INSERT, and organization_id is constant per call — so a
    # token appearing in two types collapses to one row. First type wins,
    # which matches the original behavior: the old per-row upsert set
    # token_type only on insert and never overwrote it on conflict.
    seen: set = set()
    items = []  # (token, token_type, canonical_form)
    for plural, singular in type_map.items():
        for token in learned.get(plural, []):
            if token in seen:
                continue
            seen.add(token)
            items.append((token, singular, canonicalize_token(token)))

    if not items:
        return 0

    # Migration 060 widened the PK to (organization_id, token); org flows in
    # via current_setting('app.current_org') with a zero-uuid fallback.
    _ORG_EXPR = (
        "COALESCE(CAST(current_setting('app.current_org', true) AS uuid), "
        "'00000000-0000-0000-0000-000000000000'::uuid)"
    )
    # Multi-row INSERT in batches — collapses ~N network round-trips to RDS
    # (the dominant cost) into ~N/BATCH. 200 rows * 3 params = 600 binds,
    # far under Postgres' 65535 limit.
    BATCH = 200
    rows = 0
    try:
        with engine.begin() as conn:
            for start in range(0, len(items), BATCH):
                batch = items[start : start + BATCH]
                values_clauses = []
                params = {"fid": file_id}
                for j, (tok, typ, canon) in enumerate(batch):
                    values_clauses.append(
                        f"({_ORG_EXPR}, :tok_{j}, :type_{j}, :canon_{j}, :fid)"
                    )
                    params[f"tok_{j}"] = tok
                    params[f"type_{j}"] = typ
                    params[f"canon_{j}"] = canon
                conn.execute(
                    text(
                        "INSERT INTO learned_vocabulary "
                        "(organization_id, token, token_type, canonical_form, first_seen_file) "
                        "VALUES " + ", ".join(values_clauses) +
                        " ON CONFLICT (organization_id, token) DO UPDATE SET "
                        "occurrence_count = learned_vocabulary.occurrence_count + 1, "
                        "canonical_form = COALESCE("
                        "learned_vocabulary.canonical_form, EXCLUDED.canonical_form), "
                        "last_seen_at = NOW()"
                    ),
                    params,
                )
                rows += len(batch)
    except Exception as exc:
        logger.warning("[vocab_learner] persist failed for %s: %s", file_id, exc)
        return 0
    return rows


def reload_cache() -> int:
    """Force-refresh the in-memory vocab cache. Returns token count loaded.

    Also populates the canonical-form alias index so is_known and
    get_aliases can match across casing variants.
    """
    global _vocab_cache, _vocab_cache_loaded, _canonical_alias_cache
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT token, canonical_form FROM learned_vocabulary")
            ).fetchall()
        new_cache: Set[str] = set()
        new_aliases: Dict[str, List[str]] = {}
        for r in rows:
            tok = r[0]
            canon = r[1] or canonicalize_token(tok)
            new_cache.add(tok)
            new_aliases.setdefault(canon, []).append(tok)
        for canon in new_aliases:
            new_aliases[canon] = sorted(set(new_aliases[canon]))
    except Exception as exc:
        logger.warning("[vocab_cache] reload failed: %s - keeping previous cache", exc)
        return len(_vocab_cache)
    with _cache_lock:
        _vocab_cache = new_cache
        _canonical_alias_cache = new_aliases
        _vocab_cache_loaded = True
    logger.info(
        "[vocab_cache] loaded %d tokens across %d canonical forms",
        len(_vocab_cache), len(_canonical_alias_cache),
    )
    return len(_vocab_cache)


def _ensure_loaded() -> None:
    with _cache_lock:
        already = _vocab_cache_loaded
    if not already:
        reload_cache()


def is_known(token: str) -> bool:
    """Return True if `token` (or any casing variant) was learned.

    Matches against the raw token set first (fast path), then falls back to
    canonical-form comparison so users can type `rag_potency_metadata`
    and match a learned `RAGPotencyMetadata` or `RAG_Potency_Metadata`.
    """
    if not token:
        return False
    _ensure_loaded()
    with _cache_lock:
        if token in _vocab_cache:
            return True
        canon = canonicalize_token(token)
        return canon in _canonical_alias_cache


def get_aliases(token: str) -> List[str]:
    """Return all learned aliases that share `token`'s canonical form.

    Empty list when the token has no canonical match. The caller can OR
    these aliases into a query so whichever casing appears in the source
    data is matched.
    """
    if not token:
        return []
    _ensure_loaded()
    canon = canonicalize_token(token)
    with _cache_lock:
        return list(_canonical_alias_cache.get(canon, []))


def is_identifier_type(token: str) -> bool:
    """Return True iff token is in vocabulary AND classified as 'identifier'."""
    if not token:
        return False
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT 1 FROM learned_vocabulary "
                    "WHERE token = :tok AND token_type = 'identifier' LIMIT 1"
                ),
                {"tok": token},
            ).first()
        return row is not None
    except Exception:
        return False


def get_token_type(token: str) -> str:
    """Return the learned token_type for `token`.

    Returns 'identifier' | 'field_name' | 'enum_value' | 'unknown'.
    Fails closed — any DB error yields 'unknown' so callers fall back to
    their regex path instead of surfacing an error.
    """
    if not token:
        return "unknown"
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT token_type FROM learned_vocabulary "
                    "WHERE token = :tok LIMIT 1"
                ),
                {"tok": token},
            ).first()
        if row is None:
            return "unknown"
        return str(row[0])
    except Exception:
        return "unknown"
