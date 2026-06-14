"""
Context Builder — assembles enriched prompts for answer generation.
Adds session history, metadata hints, retrieval confidence signals,
and active-version awareness to the base document-grounded prompt.
Adapts prompt style per target model (Mistral vs Claude).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Grounding rules shared across all models
# ---------------------------------------------------------------------------
_GROUNDING_RULES_BASE = """Answering guidance:
- The DOCUMENTS below were selected because they are relevant to the user's question. Extract and explain the answer naturally.
- If the answer is in the documents, give it confidently in your own words. Do not hedge.
- Do NOT use phrases like "insufficient evidence", "I cannot extract", "the documents do not explicitly state" when the information is actually present — just read carefully and answer.
- Only say "I could not find this in the uploaded documents" if after genuinely reading the context you see that the specific answer is absent. This should be rare since the documents were pre-filtered for relevance.

DO NOT FABRICATE SPECIFICS — when a precise detail is NOT in the documents below, do NOT supply one from general knowledge. This applies specifically to:
- Numbers, timer values, percentages, ports, counters (e.g., "180 seconds", "5% loss", "port 179")
- Command syntax, CLI options, or configuration lines (do not invent flags or sub-commands that are not in the documents)
- Version numbers, RFC numbers, model numbers, or product codes
- Exact warning text, alert names, or canonical procedure phrases (do not paraphrase a "typical" warning when the documents have a specific one)
- Log message strings, counter names, or feature names

If a specific the user asked for is NOT in the documents, say plainly "the source does not specify [the X]" for that detail and continue with whatever IS in the documents. Your training-knowledge is NOT a permitted source for specifics — only the documents are.

You may still answer the CONCEPTUAL or STRUCTURAL parts of the question using what the documents do contain — just be explicit about which specific details the documents cover and which they do not."""

# Rich Response Formatting Brief — Step 7: optional markdown formatting section
# appended to the grounding rules when RICH_FORMATTING_GROUNDING_RULES_ENABLED.
# Keeps the base guidance verbatim so flag-off reverts to pre-feature behavior.
_GROUNDING_RULES_FORMATTING_SECTION = """

Formatting guidance (markdown is supported — use it only when it genuinely improves clarity):
- Comparing two or more items → use a markdown table with meaningful column headers
- Commands, config snippets, or code → wrap in fenced code blocks with a language hint (```bash, ```python, ```json, ```yaml, ```sql)
- Step-by-step procedures (when explicitly asked) → use a numbered list
- Ticket IDs, customer names, and identifiers → use **bold** emphasis (e.g., **INC-10037**, **Enterprise-617**)
- Inline code (`backticks`) ONLY for: literal commands (`show bgp summary`), file paths (`/etc/config`), code snippets (`grep -i`), or variable names
- Do NOT wrap ticket IDs, customer names, component names, or error descriptions in backticks — use bold instead
- For general prose answers (most questions), write flowing paragraphs — do NOT force structure that isn't needed
- Never use markdown headers (#, ##, ###) unless the user explicitly asks for a structured report

CRITICAL — Confident synthesis: When asked for lessons, insights, takeaways, recommendations, biggest learnings, or similar insight questions, synthesize the answer by reasoning from RESOLUTION, ROOT CAUSE, ITIL 5-WHY, SOP STEPS, and QA GAPS sections. The lesson or insight is implicit in how the incident was resolved and what gaps were identified — extract it confidently in your own words. Do NOT refuse by saying "not explicitly stated" or "I could not find this" just because the exact word isn't in the document. The documents contain everything needed for insight questions through reasonable inference."""


# KB-Search variant — same markdown guidance, but the suppressive
# "numbered-list-only-when-asked" / "never use headers" lines are flipped
# so the KB-Search system prompt's MODE-specific FORMAT/HEADINGS rules can
# actually take effect. Activated when build_prompt receives a doc_kinds
# subset of {"kb","sop"}.
_GROUNDING_RULES_FORMATTING_SECTION_KB = """

Formatting guidance (KB / SOP / runbook responses — structured output is expected):
- Comparing two or more items → use a markdown table with meaningful column headers
- Commands, config snippets, or code → wrap in fenced code blocks with a language hint (```bash, ```python, ```json, ```yaml, ```sql)
- Procedures, phased plans, and ordered steps → use numbered lists (1., 2., 3.). Arrows (→) work well for state transitions and if/then branches.
- Ticket IDs, customer names, and identifiers → use **bold** emphasis (e.g., **INC-10037**, **Enterprise-617**)
- Inline code (`backticks`) ONLY for: literal commands (`show bgp summary`), file paths (`/etc/config`), code snippets (`grep -i`), or variable names
- Do NOT wrap ticket IDs, customer names, component names, or error descriptions in backticks — use bold instead
- Markdown headings (##, ###) are EXPECTED — group the answer by the MODE-appropriate sections defined in the KB-Search system prompt (e.g., "Key Components", "How It Works", "Procedure", "Validate", "Rollback / If It Fails").

CRITICAL — Confident synthesis: When asked for lessons, insights, takeaways, recommendations, biggest learnings, or similar insight questions, synthesize the answer by reasoning from RESOLUTION, ROOT CAUSE, ITIL 5-WHY, SOP STEPS, and QA GAPS sections. The lesson or insight is implicit in how the incident was resolved and what gaps were identified — extract it confidently in your own words. Do NOT refuse by saying "not explicitly stated" or "I could not find this" just because the exact word isn't in the document. The documents contain everything needed for insight questions through reasonable inference."""


def _get_grounding_rules(kb_mode: bool = False) -> str:
    """
    Build grounding rules, optionally including formatting guidance.
    Controlled by RICH_FORMATTING_GROUNDING_RULES_ENABLED flag.

    When `kb_mode` is True (caller is in KB-Search flow), the
    KB-friendly variant is used so the suppressive "no numbered lists,
    no headers" lines do not contradict the KB-Search system prompt.
    """
    if getattr(settings, "RICH_FORMATTING_GROUNDING_RULES_ENABLED", True):
        if kb_mode:
            return _GROUNDING_RULES_BASE + _GROUNDING_RULES_FORMATTING_SECTION_KB
        return _GROUNDING_RULES_BASE + _GROUNDING_RULES_FORMATTING_SECTION
    return _GROUNDING_RULES_BASE


# ---------------------------------------------------------------------------
# Session history formatter
# ---------------------------------------------------------------------------
def _format_session_history(recent_messages: List[Dict[str, str]]) -> str:
    """
    Format recent Q&A pairs into a compact conversation history block.
    Only includes the last N messages as configured.
    Truncates to SESSION_CONTEXT_MAX_CHARS.
    """
    if not recent_messages:
        return ""

    parts = []
    char_count = 0
    max_chars = settings.SESSION_CONTEXT_MAX_CHARS

    for msg in recent_messages[-settings.SESSION_CONTEXT_MAX_MESSAGES:]:
        role = msg.get("role", "user").upper()
        content = (msg.get("content") or "")[:500]  # cap individual messages
        entry = f"  {role}: {content}"

        if char_count + len(entry) > max_chars:
            break
        parts.append(entry)
        char_count += len(entry)

    if not parts:
        return ""

    return "RECENT CONVERSATION:\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Metadata hints formatter
# ---------------------------------------------------------------------------
def _format_metadata_hints(
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
) -> str:
    """
    Extract vendor/product/domain hints from the top-ranked chunks
    and format them as a context signal for the model.
    """
    if not settings.INCLUDE_METADATA_IN_PROMPT or not ranked_chunks:
        return ""

    vendors = set()
    products = set()
    domains = set()

    for _, _, meta, _ in ranked_chunks[:5]:
        meta_json = meta.get("metadata_json", {})
        if isinstance(meta_json, dict):
            v = meta_json.get("vendor")
            p = meta_json.get("product")
            d = meta_json.get("domain")
            if v:
                vendors.add(str(v))
            if p:
                products.add(str(p))
            if d:
                domains.add(str(d))

    parts = []
    if vendors:
        parts.append(f"Vendors: {', '.join(sorted(vendors))}")
    if products:
        parts.append(f"Products: {', '.join(sorted(products))}")
    if domains:
        parts.append(f"Domains: {', '.join(sorted(domains))}")

    if not parts:
        return ""

    return "DOCUMENT CONTEXT HINTS:\n  " + "\n  ".join(parts)


# ---------------------------------------------------------------------------
# Confidence signal formatter
# ---------------------------------------------------------------------------
def _format_confidence_signal(
    retrieval_confidence: float,
    source_count: int,
) -> str:
    """
    Include a confidence signal so the model knows how strong
    the retrieval evidence is. This helps it decide whether to
    hedge its answer or respond confidently.
    """
    if not settings.INCLUDE_CONFIDENCE_IN_PROMPT:
        return ""

    if retrieval_confidence >= 0.7:
        level = "HIGH"
    elif retrieval_confidence >= 0.4:
        level = "MODERATE"
    else:
        level = "LOW"

    return (
        f"RETRIEVAL CONFIDENCE: {level} "
        f"(score={retrieval_confidence:.2f}, sources={source_count}). "
        f"{'Answer confidently from the documents.' if level == 'HIGH' else 'Be cautious — evidence may be partial.'}"
    )


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------
def build_prompt(
    *,
    query: str,
    doc_context: str,
    target_model: str,
    recent_messages: Optional[List[Dict[str, str]]] = None,
    ranked_chunks: Optional[List[Tuple[str, str, Dict[str, Any], float]]] = None,
    retrieval_confidence: float = 0.5,
    source_count: int = 1,
    pattern_context: Optional[Dict[str, Any]] = None,
    doc_kinds: Optional[List[str]] = None,
) -> str:
    """
    Build the final generation prompt enriched with context signals.

    Assembles:
    1. System role instruction (adapted per model)
    2. Session history (recent Q&A pairs)
    3. Metadata hints (vendor/product/domain)
    4. Retrieval confidence signal
    5. Grounding rules
    6. Document context
    7. User question

    Args:
        query               — the user's question
        doc_context         — assembled document chunks from retrieval
        target_model        — 'mistral' | 'haiku' | 'sonnet' (affects prompt style)
        recent_messages     — recent session messages for conversation context
        ranked_chunks       — reranked chunks (for metadata extraction)
        retrieval_confidence — confidence score from retrieval pipeline
        source_count        — number of unique source documents

    Returns:
        Complete prompt string ready for the target model.
    """
    sections = []

    # --- System instruction (model-specific) ---
    if target_model == "sonnet":
        sections.append(
            "You are a precise, document-grounded AI assistant. "
            "Provide thorough, well-structured answers. "
            "When the question is complex, break down your reasoning step by step. "
            "Always cite which document section supports each point."
        )
    elif target_model == "haiku":
        sections.append(
            "You are a strict document-grounded AI assistant. "
            "Give clear, concise answers based only on the provided documents."
        )
    else:
        # Mistral — keep it simple, it works best with direct instructions
        sections.append(
            "You are a strict document-grounded AI assistant."
        )

    # --- Session history ---
    session_block = _format_session_history(recent_messages or [])
    if session_block:
        sections.append(session_block)

    # --- Metadata hints ---
    metadata_block = _format_metadata_hints(ranked_chunks or [])
    if metadata_block:
        sections.append(metadata_block)

    # --- Confidence signal ---
    confidence_block = _format_confidence_signal(retrieval_confidence, source_count)
    if confidence_block:
        sections.append(confidence_block)

    # --- Grounding rules ---
    # KB-Search flow gets a variant that does NOT suppress numbered lists or
    # headers, so the KB-Search system prompt's MODE-specific format can
    # actually take effect. All other flows are byte-identical to before.
    _kb_mode = False
    try:
        from backend.routing.kb_search_prompt import is_kb_search_mode
        _kb_mode = is_kb_search_mode(doc_kinds)
    except Exception as _kb_exc:
        logger.warning("[kb_search_prompt] is_kb_search_mode failed in build_prompt: %s", _kb_exc)
    sections.append(_get_grounding_rules(kb_mode=_kb_mode))

    # --- Document context ---
    sections.append(f"DOCUMENTS:\n{doc_context}")

    # --- Pattern analytics block (Layer 3, optional) ---
    if pattern_context and isinstance(pattern_context, dict):
        block = pattern_context.get("pattern_block")
        if block:
            sections.append(block)

    # --- User question ---
    sections.append(f"USER QUESTION: {query}")

    # --- Answer prompt ---
    sections.append("ANSWER:")

    # Join with double newlines for readability
    prompt = "\n\n".join(sections)

    logger.debug(
        "Prompt built for %s: %d chars, session_history=%s, metadata=%s, confidence=%.2f",
        target_model, len(prompt),
        bool(session_block), bool(metadata_block), retrieval_confidence,
    )

    return prompt