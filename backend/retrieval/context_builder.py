"""
Brief 4 / Opt 5 — Context compression for generation.

For single-ticket identifier_exact lookups, when the query targets a
specific labeled section (resolution / root cause / QA gaps / etc.),
trim the chunk down to the ticket header plus only those sections. Fully
regex-based; no LLM. Returns the original chunk unchanged when the query
is broad or the chunk is already small enough.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Keyword → section-header prefix(es). Every value must be uppercase +
# colon-suffix so the section-boundary heuristic lines up with the way
# chunks are emitted by contextual_ingestion_service.
SECTION_MAPPINGS: Dict[str, List[str]] = {
    "resolution": ["RESOLUTION DETAIL:", "RESOLUTION STEPS:"],
    "root cause": ["ROOT CAUSE:", "ITIL 5-WHY ROOT CAUSE:"],
    "5-why": ["ITIL 5-WHY ROOT CAUSE:"],
    "qa gaps": ["QA AUDITOR GAPS:"],
    "qa auditor": ["QA AUDITOR GAPS:"],
    "sop": ["SOP EXECUTION STEPS:"],
    "teams": ["RESOLUTION GROUPS:", "RESOLUTION DETAIL:"],
    "who": ["RESOLUTION GROUPS:", "RESOLUTION DETAIL:"],
    "customer": ["TICKET", "CUSTOMER:"],
    "priority": ["TICKET", "PRIORITY:"],
    "sla": ["SLA TARGET MET:"],
    "quality score": ["RESOLUTION QUALITY SCORE:"],
}


def _looks_like_section_header(line: str) -> bool:
    s = line.strip()
    if not s or not s.endswith(":"):
        return False
    # Strip the trailing colon for the uppercase check so "RESOLUTION DETAIL:"
    # still qualifies even if the case-check ever regresses on the colon.
    probe = s.rstrip(":").strip()
    return bool(probe) and probe == probe.upper()


def compress_chunk_for_query(chunk_content: str, query: str) -> Tuple[str, bool]:
    """
    If the query targets specific labeled sections, return only those
    sections (plus the first-line ticket header). Otherwise return the
    original content unchanged.

    Returns (possibly_compressed_content, was_compressed).
    """
    if not chunk_content:
        return chunk_content, False
    if not getattr(settings, "CONTEXT_COMPRESSION_ENABLED", False):
        return chunk_content, False
    min_chars = int(getattr(settings, "CONTEXT_COMPRESSION_MIN_CHUNK_CHARS", 5000))
    if len(chunk_content) < min_chars:
        return chunk_content, False

    q = (query or "").lower()
    target_sections: List[str] = []
    for keyword, sections in SECTION_MAPPINGS.items():
        if keyword in q:
            for sec in sections:
                if sec not in target_sections:
                    target_sections.append(sec)
    if not target_sections:
        return chunk_content, False

    lines = chunk_content.split("\n")
    header = lines[0] if lines else ""
    extracted: List[str] = [header, ""]
    in_target = False
    for line in lines[1:]:
        if any(line.startswith(sec) for sec in target_sections):
            in_target = True
            extracted.append("")
            extracted.append(line)
            continue
        if in_target:
            if _looks_like_section_header(line):
                if not any(line.startswith(sec) for sec in target_sections):
                    in_target = False
                    continue
            extracted.append(line)

    compressed = "\n".join(extracted).rstrip()
    if len(compressed) < len(chunk_content) * 0.7:
        return compressed, True
    return chunk_content, False
