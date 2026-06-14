"""
Document-Aware Query Expansion — learns vocabulary from documents, no hardcoded lists.

HOW IT WORKS:
=============
1. During INGESTION: extracts glossary/abbreviation tables from document content
   and stores them in chunk metadata_json as {"glossary": {...}}
2. During STARTUP: rebuilds a GlossaryStore from all active document metadata
3. During QUERY: expands user queries using document-extracted vocabulary

This means:
- If a document has "DHCP = Dynamic Host Configuration Protocol" in a glossary table,
  the system learns it automatically
- If tomorrow a new document has "XYZP = Some New Protocol", the system learns that too
- Zero manual maintenance required
- Works for ANY document type, ANY domain

INTEGRATION POINTS:
===================
1. contextual_ingestion_service.py → extract_glossary_from_chunks() during ingestion
2. api.py → lifespan startup → rebuild_glossary_from_postgres()
3. api.py → ask() → expand_query() before retrieval
4. api.py → has_sufficient_document_support() → use expanded keywords
"""

import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Sprint 2.5 hotfix — per-token normalization helper. Applied only to
# tokens NOT in the learned vocabulary. Byte-identical to the legacy
# full-string normalization when run on a single token.
def _hotfix_normalize_token(tok: str) -> str:
    t = tok.replace("_", " ")
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    t = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", t)
    return t


# Strip surrounding punctuation for vocab lookup without losing it on the
# rendered output.
_HOTFIX_TOKEN_STRIP_RE = re.compile(r'^([\"\'(\[]*)(.*?)([\"\'\].,!?;:]*)$')


# Sprint 2.7 Bug B/E — English stopwords + generic NOC nouns that pollute
# the canonical-alias OR-injection path. These tokens enter the learned
# vocabulary because they appear in nearly every ingested ticket; when the
# canonical-form expansion fires on them it injects uppercase aliases
# ("the THE", "and AND", "time TIME") into the embedding query, degrading
# retrieval quality and inflating token counts for Haiku.
_HOTFIX_STOPWORDS = frozenset({
    # articles / determiners
    "the", "a", "an",
    # conjunctions
    "and", "or", "but", "nor",
    # prepositions / common connectors
    "of", "at", "in", "on", "to", "for", "with", "by", "from", "as",
    # aux verbs / copulas
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "has", "have", "had",
    # demonstratives
    "this", "that", "these", "those",
    # pronouns / possessives
    "it", "its", "they", "them", "their",
    "we", "our", "us",
    "you", "your",
    "i", "me", "my",
    # generic NOC nouns — ingested into vocab but hurt retrieval when
    # uppercase-aliased into the embedded query (see §4.3 of brief).
    "time", "cause", "root", "impact", "resolution",
})


def _is_vocab_eligible(token: str) -> bool:
    """Sprint 2.7 Bug B/E — guard the canonical-form expansion path against
    English stopwords and generic NOC nouns that pollute embedding queries
    with duplicated uppercase aliases."""
    t = (token or "").strip().lower()
    if not t:
        return False
    if t in _HOTFIX_STOPWORDS:
        return False
    # Require at least one digit OR one uppercase letter OR a dash/underscore —
    # these are the signals that separate identifier-like tokens from prose.
    has_digit = any(c.isdigit() for c in token)
    has_upper = any(c.isupper() for c in token)
    has_sep = "-" in token or "_" in token
    if has_digit or has_upper or has_sep:
        return True
    # Pure lowercase word with no separators — reject from vocab path.
    return False


# ============================================================================
# GLOSSARY STORE — populated entirely from document content
# ============================================================================
class GlossaryStore:
    """
    Maintains acronym ↔ full-form mappings extracted from uploaded documents.
    No hardcoded lists — everything comes from the documents themselves.
    """

    def __init__(self):
        self._acronym_to_full: Dict[str, Set[str]] = defaultdict(set)
        self._full_to_acronym: Dict[str, str] = {}
        self._doc_glossaries: Dict[str, Dict[str, str]] = {}

    def add_document_glossary(self, doc_id: str, glossary: Dict[str, str]):
        if not glossary:
            return
        self._doc_glossaries[doc_id] = glossary
        for abbr, full in glossary.items():
            abbr_lower = abbr.strip().lower()
            full_clean = full.strip()
            if abbr_lower and full_clean and len(abbr_lower) <= 15:
                self._acronym_to_full[abbr_lower].add(full_clean)
                self._full_to_acronym[full_clean.lower()] = abbr.strip().upper()
        logger.info(
            "Glossary registered for doc %s: %d entries (store total: %d acronyms)",
            doc_id[:12], len(glossary), len(self._acronym_to_full),
        )

    def remove_document_glossary(self, doc_id: str):
        self._doc_glossaries.pop(doc_id, None)
        self._rebuild_from_docs()

    def _rebuild_from_docs(self):
        self._acronym_to_full.clear()
        self._full_to_acronym.clear()
        for glossary in self._doc_glossaries.values():
            for abbr, full in glossary.items():
                abbr_lower = abbr.strip().lower()
                full_clean = full.strip()
                if abbr_lower and full_clean and len(abbr_lower) <= 15:
                    self._acronym_to_full[abbr_lower].add(full_clean)
                    self._full_to_acronym[full_clean.lower()] = abbr.strip().upper()

    def expand_acronym(self, term: str) -> List[str]:
        return list(self._acronym_to_full.get(term.lower(), []))

    def find_acronym(self, phrase: str) -> Optional[str]:
        return self._full_to_acronym.get(phrase.lower())

    @property
    def size(self) -> int:
        return len(self._acronym_to_full)

    def clear(self):
        self._doc_glossaries.clear()
        self._acronym_to_full.clear()
        self._full_to_acronym.clear()

    def get_all_entries(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._acronym_to_full.items()}


# Singleton
_glossary_store = GlossaryStore()


def get_glossary_store() -> GlossaryStore:
    return _glossary_store


# ============================================================================
# GLOSSARY EXTRACTION FROM DOCUMENT TEXT
# ============================================================================
def extract_glossary_from_text(text: str) -> Dict[str, str]:
    """
    Extract abbreviation→description mappings from document text.
    Handles multiple formats:
      - "DHCP    Dynamic Host Configuration Protocol"  (whitespace-separated)
      - "DHCP - Dynamic Host Configuration Protocol"   (dash-separated)
      - "DHCP: Dynamic Host Configuration Protocol"    (colon-separated)
      - "| DHCP | Dynamic Host Configuration Protocol |" (markdown table)
      - Inline definitions: "DHCP (Dynamic Host Configuration Protocol)"
    """
    glossary: Dict[str, str] = {}
    if not text:
        return glossary

    # === Strategy 1: Find a glossary/abbreviation section and parse it ===
    glossary_section = _find_glossary_section(text)
    if glossary_section:
        glossary.update(_parse_glossary_table(glossary_section))

    # === Strategy 2: Find inline definitions throughout the entire text ===
    glossary.update(_find_inline_definitions(text))

    # === Strategy 3: Parse any tabular abbreviation patterns in full text ===
    if not glossary_section:
        glossary.update(_parse_glossary_table(text))

    if glossary:
        logger.info("Extracted %d glossary entries from document text", len(glossary))
    return glossary


def _find_glossary_section(text: str) -> str:
    """Find and extract a glossary/abbreviation section from document text."""
    patterns = [
        # Markdown heading: ## Glossary / **Glossary**
        r"(?i)(?:^#{1,3}\s*|^\*\*\s*)(?:glossary|abbreviations?|acronyms?|definitions?|terminology)(?:\s*\*\*)?[ \t]*\n([\s\S]*?)(?=\n#{1,3}\s|\n\*\*[A-Z][a-z]{2,}|\n# |\Z)",
        # Plain text heading
        r"(?i)^(?:glossary|abbreviations?|acronyms?|definitions?)[ \t]*\n([\s\S]*?)(?=\n[A-Z][a-z]+ [A-Z][a-z]+\n|\n#{1,3}\s|\Z)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.MULTILINE)
        if match:
            section = match.group(1).strip()
            if len(section) > 20:  # Must have meaningful content
                return section
    return ""


def _parse_glossary_table(text: str) -> Dict[str, str]:
    """Parse glossary entries from tabular or list format."""
    glossary: Dict[str, str] = {}

    patterns = [
        # Markdown table: | ABBR | Description |
        r"\|\s*([A-Z][A-Za-z0-9/._-]{1,14})\s*\|\s*([A-Za-z][A-Za-z0-9\s,/().'-]{4,100}?)\s*\|",
        # Tab/space separated: ABBR    Description (2+ spaces/tabs)
        r"^[ \t]*([A-Z][A-Za-z0-9/._-]{1,14})[ \t]{2,}([A-Za-z][A-Za-z0-9\s,/().'-]{4,100}?)[ \t]*$",
        # Dash separated: ABBR - Description or ABBR — Description
        r"^[ \t]*([A-Z][A-Za-z0-9/._-]{1,14})\s+[-–—]\s+([A-Za-z][A-Za-z0-9\s,/().'-]{4,100}?)[ \t]*$",
        # Colon separated: ABBR: Description
        r"^[ \t]*([A-Z][A-Za-z0-9/._-]{1,14})\s*:\s*([A-Za-z][A-Za-z0-9\s,/().'-]{4,100}?)[ \t]*$",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            abbr = match.group(1).strip()
            desc = match.group(2).strip()
            if _is_valid_glossary_entry(abbr, desc):
                desc = re.sub(r"\s+", " ", desc).rstrip("|").strip()
                glossary[abbr] = desc

    return glossary


def _find_inline_definitions(text: str) -> Dict[str, str]:
    """
    Find inline abbreviation definitions like:
    - "DHCP (Dynamic Host Configuration Protocol)"
    - "Dynamic Host Configuration Protocol (DHCP)"
    """
    glossary: Dict[str, str] = {}

    # Pattern: ABBR (Full Form) — e.g., "DHCP (Dynamic Host Configuration Protocol)"
    for match in re.finditer(
        r"\b([A-Z][A-Z0-9]{1,9})\s*\(\s*([A-Z][A-Za-z\s]{5,80}?)\s*\)", text
    ):
        abbr, desc = match.group(1).strip(), match.group(2).strip()
        if _is_valid_glossary_entry(abbr, desc) and _looks_like_expansion(abbr, desc):
            glossary[abbr] = desc

    # Pattern: Full Form (ABBR) — e.g., "Dynamic Host Configuration Protocol (DHCP)"
    for match in re.finditer(
        r"([A-Z][A-Za-z\s]{5,80}?)\s*\(\s*([A-Z][A-Z0-9]{1,9})\s*\)", text
    ):
        desc, abbr = match.group(1).strip(), match.group(2).strip()
        if _is_valid_glossary_entry(abbr, desc) and _looks_like_expansion(abbr, desc):
            glossary[abbr] = desc

    return glossary


def _is_valid_glossary_entry(abbr: str, desc: str) -> bool:
    """Filter out false positives."""
    if len(abbr) < 2 or len(desc) < 4:
        return False
    if abbr.islower():
        return False
    # Description shouldn't be all uppercase (another abbreviation)
    if desc.isupper() and len(desc) < 12:
        return False
    # Skip if description is just numbers or too short words
    words = desc.split()
    if len(words) < 2 and len(abbr) > 2:
        return False
    return True


def _looks_like_expansion(abbr: str, desc: str) -> bool:
    """
    Check if desc looks like a plausible expansion of abbr.
    E.g., "DHCP" matches "Dynamic Host Configuration Protocol" because
    initials D-H-C-P match.
    """
    abbr_upper = abbr.upper()
    words = [w for w in desc.split() if len(w) > 1]
    if not words:
        return False

    # Check if initials of description words match the abbreviation
    initials = "".join(w[0].upper() for w in words if w[0].isalpha())
    if initials == abbr_upper:
        return True

    # Partial match (at least 60% of abbreviation letters match)
    if len(abbr_upper) >= 2:
        match_count = sum(1 for c in abbr_upper if c in initials)
        if match_count / len(abbr_upper) >= 0.6:
            return True

    # Allow if first letter matches and length is reasonable
    if words and words[0][0].upper() == abbr_upper[0] and len(words) >= len(abbr_upper) - 1:
        return True

    return False


# ============================================================================
# EXTRACT GLOSSARY FROM PARSED CHUNKS (called during ingestion)
# ============================================================================
def extract_glossary_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract glossary from a list of chunk dicts (as produced by contextual_ingestion_service).
    Scans chunk content AND section headings to find glossary-like content.
    """
    glossary: Dict[str, str] = {}

    # Prioritize chunks that look like glossary sections
    glossary_chunks = []
    other_chunks = []

    for chunk in chunks:
        content = chunk.get("content", "")
        heading = (chunk.get("section_heading") or "").lower()

        if any(kw in heading for kw in ("glossary", "abbreviat", "acronym", "definition", "terminolog")):
            glossary_chunks.append(content)
        elif any(kw in content[:200].lower() for kw in ("glossary", "abbreviat", "acronym")):
            glossary_chunks.append(content)
        else:
            other_chunks.append(content)

    # Parse glossary sections first (most reliable)
    for text in glossary_chunks:
        glossary.update(extract_glossary_from_text(text))

    # If we didn't find a glossary section, scan all chunks for inline definitions
    if not glossary:
        full_text = "\n".join(other_chunks[:20])  # First 20 chunks, avoid huge docs
        inline = _find_inline_definitions(full_text)
        glossary.update(inline)

    return glossary


# ============================================================================
# QUERY EXPANSION
# ============================================================================
class ExpandedQuery:
    """Result of query expansion."""

    def __init__(
        self,
        original: str,
        expanded_text: str,
        variants: List[str],
        expanded_keywords: Set[str],
        acronyms_found: Dict[str, List[str]],
    ):
        self.original = original
        self.expanded_text = expanded_text
        self.variants = variants
        self.expanded_keywords = expanded_keywords
        self.acronyms_found = acronyms_found

    def __repr__(self):
        return (
            f"ExpandedQuery(original={self.original!r}, "
            f"acronyms={list(self.acronyms_found.keys())})"
        )


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "how",
    "what", "when", "where", "why", "do", "does", "can", "i", "me", "my",
    "you", "your", "please", "tell", "about", "with", "from", "this", "that",
    "and", "or", "but", "not", "be", "been", "being", "have", "has", "had",
    "will", "would", "could", "should", "may", "might", "shall", "it", "its",
    "was", "were", "all", "each", "every", "any", "some", "no", "which",
    "there", "their", "they", "them", "than", "then", "so", "if", "as", "at",
    "by", "up", "out", "our", "us", "we", "he", "she", "him", "her", "his",
})


def expand_query(query: str, store: Optional[GlossaryStore] = None) -> ExpandedQuery:
    """
    Expand a user query using document-extracted vocabulary.

    If the glossary store has learned "DHCP = Dynamic Host Configuration Protocol"
    from the uploaded documents, this will:
    - Expand the embedding query to include the full form
    - Generate variant queries for fallback retrieval
    - Build an expanded keyword set for document support checking
    """
    if store is None:
        store = _glossary_store

    if not query or not query.strip():
        return ExpandedQuery(query, query, [query] if query else [], set(), {})

    # ── Query normalization ──────────────────────────────────────
    # Split concatenated words: "toDC" → "to DC", "fromDC" → "from DC"
    # Split camelCase: "DataCenter" → "Data Center"
    # Split underscore terms: "Consistent_High_Interface_Errors" → "Consistent High Interface Errors"
    # Sprint 2.5 #1/#2 — vocabulary-preserving normalization. Tokens
    # already seen in ingested content (ticket IDs, snake_case field
    # names, custom enums) pass through verbatim; everything else
    # runs through the same legacy transforms.
    #
    # v2 canonical-form matching: when a user token (e.g.
    # rag_potency_metadata) shares a canonical form with learned
    # aliases (RAGPotencyMetadata, RAG_Potency_Metadata), OR-inject
    # the aliases after the verbatim token so FTS/ILIKE matches
    # whichever casing appears in the source data.
    try:
        from backend.services.vocabulary_learner import (
            is_known as _vocab_is_known,
            get_aliases as _vocab_get_aliases,
        )
    except Exception:
        def _vocab_is_known(_t: str) -> bool:
            return False

        def _vocab_get_aliases(_t: str) -> List[str]:
            return []

    parts: List[str] = []
    for raw_tok in query.split():
        m = _HOTFIX_TOKEN_STRIP_RE.match(raw_tok)
        prefix = m.group(1) if m else ""
        core = m.group(2) if m else raw_tok
        suffix = m.group(3) if m else ""
        if core and _is_vocab_eligible(core) and _vocab_is_known(core):
            parts.append(raw_tok)  # verbatim
            # Alias OR-injection — add any learned variants that share
            # the same canonical form so casing mismatches don't hide
            # hits. Skip aliases equal to the user's token.
            try:
                aliases = [a for a in _vocab_get_aliases(core) if a != core]
            except Exception:
                aliases = []
            for alias in aliases:
                parts.append(alias)
        else:
            parts.append(prefix + _hotfix_normalize_token(core) + suffix)
    normalized = " ".join(parts)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized.lower() != query.lower():
        logger.info("Query normalized: '%s' -> '%s'", query, normalized)
        # Use normalized version for tokenization and embedding
        query_for_processing = normalized
    else:
        query_for_processing = query

    tokens = re.findall(r"\w+", query_for_processing)
    tokens_lower = [t.lower() for t in tokens]

    acronyms_found: Dict[str, List[str]] = {}
    expanded_parts = []
    keyword_set: Set[str] = set()

    for token in tokens:
        token_lower = token.lower()

        # Add to keywords if meaningful
        if len(token_lower) > 1 and token_lower not in _STOP_WORDS:
            keyword_set.add(token_lower)

        # Check if this token has an expansion in the glossary
        expansions = store.expand_acronym(token_lower)

        if expansions:
            acronyms_found[token.upper()] = expansions
            best = expansions[0]
            expanded_parts.append(f"{token} ({best})")
            # Add expansion words to keyword set
            for exp in expansions:
                for word in re.findall(r"\w+", exp.lower()):
                    if word not in _STOP_WORDS and len(word) > 1:
                        keyword_set.add(word)
        else:
            expanded_parts.append(token)

    expanded_text = " ".join(expanded_parts)

    # Build variants
    variants = [query]
    # Always include normalized form as a variant if different
    if normalized.lower() != query.lower() and normalized not in variants:
        variants.append(normalized)
    if acronyms_found:
        # Full-form variant
        full_parts = []
        for token in tokens:
            exps = store.expand_acronym(token.lower())
            full_parts.append(exps[0] if exps else token)
        full_form = " ".join(full_parts)
        if full_form.lower() != query.lower():
            variants.append(full_form)

        # Keyword-only variant
        key_terms = [t for t in tokens_lower if t not in _STOP_WORDS and len(t) > 1]
        extra = [k for k in keyword_set if k not in set(key_terms)][:4]
        kw_variant = " ".join(set(key_terms + extra))
        if kw_variant and kw_variant not in variants:
            variants.append(kw_variant)

    return ExpandedQuery(
        original=query,
        expanded_text=expanded_text,
        variants=variants,
        expanded_keywords=keyword_set,
        acronyms_found=acronyms_found,
    )


# ============================================================================
# ENHANCED DOCUMENT SUPPORT CHECK
# ============================================================================
def has_sufficient_document_support_v2(
    question: str,
    ranked: List[Tuple[str, str, Dict, float]],
    min_score: float = 0.08,
    min_keyword_hits: int = 1,
    expanded_keywords: Optional[Set[str]] = None,
) -> bool:
    """
    Enhanced document support check.

    Fixes (Phase 7):
    - Lowered min_score: 0.08 (was 0.12) — short precise queries like
      "QoS Trust Boundaries?" were being rejected despite relevant chunks
    - Score threshold is now a soft gate: if keywords match well, the
      chunk is accepted even with a lower vector score
    - Uses expanded keywords from glossary (DHCP also matches "Dynamic Host...")
    - Checks top 8 chunks (was 5) — broader window catches more relevant content
    - Also checks section headings and summaries
    - Strips trailing punctuation from query terms (e.g., "Boundaries?" → "boundaries")
    """
    if not ranked:
        return False

    top_score = float(ranked[0][3] or 0.0)

    # Build keyword set from question + expansions
    # Strip punctuation from tokens so "Boundaries?" becomes "boundaries"
    q_terms = {
        re.sub(r"[^\w]", "", t).lower()
        for t in re.findall(r"\w+", (question or "").lower())
        if len(t) > 1 and t.lower().rstrip("?.,!") not in _STOP_WORDS
    }
    q_terms.discard("")
    if expanded_keywords:
        q_terms = q_terms | expanded_keywords

    # Check content of top 8 chunks for keyword matches
    combined_text = " ".join((item[1] or "") for item in ranked[:8]).lower()
    content_hits = sum(1 for term in q_terms if term in combined_text)

    # Also check section headings and summaries
    meta_text = " ".join(
        ((item[2].get("section_heading") or "") + " " + (item[2].get("summary") or ""))
        for item in ranked[:8]
    ).lower()
    meta_hits = sum(1 for term in q_terms if term in meta_text)

    total_hits = content_hits + meta_hits
    # Deduplicated: count unique terms found (not double-count content+meta)
    unique_terms_found = sum(
        1 for term in q_terms if term in combined_text or term in meta_text
    )
    hit_ratio = unique_terms_found / max(len(q_terms), 1)

    logger.info(
        "Doc support check: top_score=%.4f, terms=%d, content_hits=%d, "
        "meta_hits=%d, unique_found=%d, ratio=%.2f, q_terms=%s",
        top_score, len(q_terms), content_hits, meta_hits,
        unique_terms_found, hit_ratio, q_terms,
    )

    # Hard floor: if score is extremely low AND no keywords match, reject
    if top_score < 0.04:
        logger.debug("Insufficient support: top_score=%.4f below hard floor 0.04", top_score)
        return False

    # Soft gate: allow low-score chunks through if keywords match well
    # This fixes short queries like "QoS Trust Boundaries?" where the
    # vector similarity is modest but the content clearly contains the answer
    if top_score < min_score:
        # Need stronger keyword evidence to compensate for low vector score
        if hit_ratio >= 0.4 or unique_terms_found >= 2:
            logger.info(
                "Low vector score (%.4f) but strong keyword match (ratio=%.2f, found=%d) — allowing",
                top_score, hit_ratio, unique_terms_found,
            )
            return True
        logger.debug(
            "Insufficient support: top_score=%.4f < min=%.4f and weak keywords (ratio=%.2f)",
            top_score, min_score, hit_ratio,
        )
        return False

    if not q_terms:
        return True

    return unique_terms_found >= min_keyword_hits or hit_ratio >= 0.15


# ============================================================================
# GLOSSARY REBUILD FROM POSTGRES (called at startup)
# ============================================================================
def rebuild_glossary_from_postgres() -> int:
    """
    Rebuild the glossary store from all active documents in PostgreSQL.
    Called once at startup after BM25 rebuild.

    Strategy:
    1. Check document_metadata.metadata_json for stored glossaries
    2. Scan chunk content for glossary/abbreviation sections
    3. Scan for inline abbreviation definitions
    """
    from backend.db.connection import SessionLocal
    from sqlalchemy import text as sql_text

    _glossary_store.clear()

    try:
        with SessionLocal() as db:
            # Strategy 1: Check stored metadata for glossaries
            meta_rows = db.execute(
                sql_text("""
                    SELECT dm.document_id::text, dm.metadata_json
                    FROM document_metadata dm
                    JOIN documents d ON d.id = dm.document_id
                    WHERE d.status = 'active'
                      AND dm.metadata_json IS NOT NULL
                """)
            ).mappings().all()

            for row in meta_rows:
                meta = row["metadata_json"]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(meta, dict):
                    glossary = meta.get("glossary") or meta.get("abbreviations") or {}
                    if glossary and isinstance(glossary, dict):
                        _glossary_store.add_document_glossary(row["document_id"], glossary)

            # Strategy 2: Scan chunks with glossary-like headings
            chunk_rows = db.execute(
                sql_text("""
                    SELECT c.content, d.id::text AS doc_id
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    JOIN document_versions dv ON dv.id = c.document_version_id
                    WHERE d.status = 'active'
                      AND dv.is_active = TRUE
                      AND (
                        LOWER(c.section_heading) LIKE '%glossary%'
                        OR LOWER(c.section_heading) LIKE '%abbreviat%'
                        OR LOWER(c.section_heading) LIKE '%acronym%'
                        OR LOWER(c.section_heading) LIKE '%terminolog%'
                        OR LOWER(c.section_heading) LIKE '%definition%'
                      )
                    LIMIT 50
                """)
            ).mappings().all()

            for row in chunk_rows:
                extracted = extract_glossary_from_text(row["content"])
                if extracted:
                    _glossary_store.add_document_glossary(row["doc_id"], extracted)

            # Strategy 3: If still empty, scan first few chunks for glossary content
            if _glossary_store.size == 0:
                content_rows = db.execute(
                    sql_text("""
                        SELECT c.content, d.id::text AS doc_id
                        FROM chunks c
                        JOIN documents d ON d.id = c.document_id
                        JOIN document_versions dv ON dv.id = c.document_version_id
                        WHERE d.status = 'active'
                          AND dv.is_active = TRUE
                          AND (
                            LOWER(c.content) LIKE '%glossary%'
                            OR LOWER(c.content) LIKE '%abbreviation%'
                          )
                        LIMIT 30
                    """)
                ).mappings().all()

                for row in content_rows:
                    extracted = extract_glossary_from_text(row["content"])
                    if extracted:
                        _glossary_store.add_document_glossary(row["doc_id"], extracted)

    except Exception as e:
        logger.warning("Glossary rebuild from DB failed (non-fatal): %s", e)

    total = _glossary_store.size
    logger.info("Glossary store rebuilt: %d unique acronyms from documents", total)
    return total