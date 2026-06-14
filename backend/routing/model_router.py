"""
Model Router — Phase 6 accuracy fix.
Routes to Haiku (default), or Sonnet (complex queries).
Key fix: Mistral is no longer used for user-facing answer generation
because it cannot follow grounding instructions reliably.
Mistral remains available for internal tasks (reranking) only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.routing.complexity_classifier import ComplexityResult, classify_complexity
from backend.routing.context_builder import build_prompt

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RoutingResult:
    answer: str = ""
    model_used: str = "haiku"
    complexity: Optional[ComplexityResult] = None
    reason: str = ""
    generation_ms: int = 0
    prompt_chars: int = 0


# ---------------------------------------------------------------------------
# Routing policy — decides which model to use
# ---------------------------------------------------------------------------
def _select_model(complexity: ComplexityResult) -> Tuple[str, str]:
    """
    Apply routing policy based on complexity tier.

    Key change: Haiku is now the default for simple AND moderate queries.
    Mistral 7B is too weak for reliable document-grounded answers — it
    hallucinates, ignores instructions, and garbles technical content.

    Policy:
        simple   → Haiku   (cheap, reliable, good instruction-following)
        moderate → Haiku   (same — Haiku handles this well)
        complex  → Sonnet  (premium, multi-step reasoning, synthesis)
    """
    if not settings.ENABLE_MODEL_ROUTING:
        return settings.ROUTING_DEFAULT_MODEL, "routing disabled, using default"

    tier = complexity.tier
    score = complexity.score

    if tier == "complex":
        return "sonnet", f"complex query (score={score:.3f}): Sonnet for deep reasoning"

    # Both simple and moderate → Haiku (reliable grounding, low cost)
    return "haiku", f"{tier} query (score={score:.3f}): Haiku for reliable grounded answers"


# ---------------------------------------------------------------------------
# Model invocation functions
# ---------------------------------------------------------------------------
def _invoke_mistral(
    prompt: str,
    generate_fn: Callable[[str, int], str],
) -> str:
    """Invoke Mistral via safe_generate. Used only as last-resort fallback."""
    return generate_fn(prompt, settings.HAIKU_ANSWER_MAX_TOKENS)


def _build_claude_system_prompt(
    doc_kinds: Optional[List[str]] = None,
) -> str:
    """
    Build Claude system prompt with optional markdown formatting guidance.

    The conversational engineer voice (base_prompt) is ALWAYS included — it
    defines response style and is core to pre-feature behavior. The markdown
    formatting addendum is appended only when RICH_FORMATTING_PROMPT_ENABLED
    so the flag-off path is byte-identical to the pre-brief prompt.

    KB-search addendum (NEW, additive only)
    ---------------------------------------
    When `doc_kinds` is a non-empty subset of {"kb","sop"} — i.e. the
    request restricted retrieval to KB / SOP content — the full
    "RAG Knowledge Architect" prompt from
    `backend/routing/kb_search_prompt.py` is appended at the end. The
    base prompt and formatting addendum are NOT replaced or edited;
    the KB prompt is purely supplementary guidance for the KB-search
    chat flow. All other flows (general chat without doc_kinds,
    ticket-scoped chats, journey stages, RCA, escalation) get the
    exact same prompt they did before this change.
    """
    # Detect KB-Search mode early so we can relax the "no bullets / no headers"
    # guardrails ONLY for the KB/SOP path. All other flows (general chat,
    # ticket-scoped, RCA, journey stages, escalation) keep the conversational
    # voice exactly as before.
    try:
        from backend.routing.kb_search_prompt import is_kb_search_mode
        _kb_mode = is_kb_search_mode(doc_kinds)
    except Exception as _kb_exc:
        logger.warning("[kb_search_prompt] is_kb_search_mode failed: %s", _kb_exc)
        _kb_mode = False

    # Formatting-voice lines are the ONLY part of the base prompt that
    # differs between KB-Search mode and all other flows. In KB mode we
    # explicitly allow bullets, numbered steps (1., 2., 3.), arrows, and
    # markdown headings so the KB-Search system addendum below can actually
    # take effect. In every other flow we keep the original suppressive
    # rules verbatim.
    if _kb_mode:
        _voice_formatting = (
            "- Structure the answer with bullet points, numbered steps "
            "(1., 2., 3.) or arrows (→), and markdown headings (##, ###) — KB / "
            "SOP / runbook answers are EXPECTED to be structured. Follow the "
            "MODE-specific FORMAT, HEADINGS, and STYLE guidance from the "
            "KB-Search system prompt appended below.\n"
            "- Section headings such as \"Key Components\", \"How It Works\", "
            "\"Procedure\", \"Validate\", \"Rollback / If It Fails\" are "
            "encouraged whenever the MODE calls for them.\n"
        )
    else:
        _voice_formatting = (
            "- Use bullet points ONLY when the question genuinely calls for a list "
            "(e.g., \"list all X\", \"what are the steps\", \"compare\"). For everything "
            "else, answer in natural prose paragraphs.\n"
            "- Never produce section headers like \"Root Cause\", \"Resolution\", "
            "\"Key Findings\" unless the user explicitly asked for a structured breakdown.\n"
        )

    # Conversational-engineer voice. See CONVERSATIONAL_REFACTOR_BRIEF goal 1.1 —
    # bullets-only-when-asked + length-proportional + no "based on documents" phrasing.
    base_prompt = (
        "You are a senior operations engineer chatting with a trainee engineer. "
        "The DOCUMENTS section below contains pre-retrieved, highly relevant content "
        "from uploaded incident tickets, runbooks, and KBs. Your job is to understand "
        "the trainee's question and answer it in your own words, the way a human "
        "expert would in a real conversation.\n\n"
        "How to respond:\n"
        "- Read the documents, understand the answer, then explain it naturally. "
        "Do NOT copy-paste document text verbatim.\n"
        "- Match your response length to the question. A short specific question "
        "gets a short specific answer — two to four sentences. A broad question gets "
        "a fuller answer. A comparison gets a comparison. A how-to gets steps.\n"
        + _voice_formatting +
        "- The documents are your source of truth. If the answer is in them, give it "
        "confidently in your own words. Never say \"the documents show\" or \"based on "
        "the documents\" — just answer.\n"
        "- If the answer is genuinely not in the documents, say so plainly in one "
        "sentence. Do not hedge with \"insufficient evidence\" or \"I cannot extract.\"\n\n"
        "DO NOT FABRICATE SPECIFICS — when a precise technical detail is NOT in "
        "the DOCUMENTS, do NOT supply one from general knowledge:\n"
        "- Never invent numbers, timer values, percentages, ports, or counters "
        "(e.g., \"180 seconds\", \"port 179\") that are not in the documents.\n"
        "- Never invent command syntax, CLI flags, or sub-commands that are not "
        "literally shown in the documents. If the docs say one form, use that "
        "form — do NOT expand it into a more \"complete\" textbook version.\n"
        "- Never invent version numbers, RFC numbers, model numbers, or product codes.\n"
        "- Never paraphrase a \"typical\" warning, alert, or procedure when the "
        "documents contain a specific one — use the document's exact wording.\n"
        "- If a specific the user asks for is not in the documents, say "
        "\"the source does not specify\" for that detail and continue with what IS "
        "in the documents. Your training-knowledge is NOT a permitted source for "
        "specifics — only the documents are.\n"
        "- You may still synthesize CONCEPTUAL or STRUCTURAL answers (what the "
        "document teaches, the reasoning, the resolution pattern) confidently. "
        "The anti-fabrication rule applies to verbatim specifics, not to "
        "inferred concepts."
    )

    if not getattr(settings, "RICH_FORMATTING_PROMPT_ENABLED", True):
        return base_prompt

    # Same KB-mode swap as the base prompt — keep the markdown rules, but
    # flip the "numbered list ONLY when asked / no headers" lines so KB
    # answers can actually use the structure the KB-Search prompt requires.
    if _kb_mode:
        _addendum_structure = (
            "- Numbered lists (1., 2., 3.) are encouraged for procedures, "
            "phased plans, and ordered steps. Arrows (→) work well for "
            "state transitions, if/then branches, and short flows.\n"
            "- Markdown headings (##, ###) are EXPECTED — group the answer by "
            "the MODE-appropriate sections (e.g., \"Key Components\", "
            "\"Procedure\", \"Validate\", \"Rollback\") as defined in the "
            "KB-Search system prompt below.\n"
        )
    else:
        _addendum_structure = (
            "- Numbered list ONLY when the user explicitly asks for steps.\n"
            "- Default to flowing prose paragraphs for everything else. Do not use "
            "headers (# ## ###) unless asked for a formal structured report.\n"
        )

    formatting_addendum = (
        "\n\n"
        "Markdown formatting (use only when it genuinely helps — never forced):\n"
        "- Compare multiple items or datasets → markdown table.\n"
        "- Share commands, configs, or code → fenced code block with language hint "
        "(```bash, ```python, etc.).\n"
        "- Ticket IDs, customer names, product names → use **bold** "
        "(e.g., **INC-10037**, **Enterprise-617**, **ADTRAN 908E**).\n"
        "- Inline code (`backticks`) is ONLY for literal commands "
        "(`show bgp summary`), file paths (`/etc/config`), code variables "
        "(`getToken()`), or log snippets. Never for ticket IDs or identifiers.\n"
        "- Emphasize a key fact or metric → **bold**.\n"
        "- Call out a recommendation or important note → use a blockquote "
        "(> line) on its own paragraph.\n"
        + _addendum_structure +
        "\n"
        "CRITICAL — Confident synthesis: When asked for lessons, insights, "
        "takeaways, recommendations, or biggest learnings from a ticket or "
        "incident, synthesize the answer from RESOLUTION, ROOT CAUSE, SOP STEPS, "
        "and QA GAPS sections. The lesson is inferrable from how the incident "
        "was resolved and what went wrong — extract it confidently. Do NOT "
        "refuse just because the literal word 'lesson' isn't in the document. "
        "The documents contain what you need to answer through reasonable "
        "inference."
    )

    final_prompt = base_prompt + formatting_addendum

    # ── KB-Search system addendum (additive; off by default unless the
    # request's allowed_doc_kinds restricted retrieval to KB/SOP).
    # The check, normalization, and feature-flag gate all live in
    # kb_search_prompt.get_kb_search_addendum — when the path isn't
    # KB-search this returns "" and the system prompt is byte-identical
    # to the pre-change behaviour.
    try:
        from backend.routing.kb_search_prompt import get_kb_search_addendum
        kb_addendum = get_kb_search_addendum(doc_kinds)
    except Exception as exc:
        # Defensive: a bug in the KB-prompt module must not break the
        # main answer path. Log and continue with the unchanged prompt.
        logger.warning("[kb_search_prompt] get_kb_search_addendum failed: %s", exc)
        kb_addendum = ""

    if kb_addendum:
        final_prompt = final_prompt + "\n\n" + kb_addendum

    return final_prompt


def _invoke_claude(
    prompt: str,
    bedrock_client: Any,
    model_id: str,
    max_tokens: int,
    temperature: float,
    doc_kinds: Optional[List[str]] = None,
) -> str:
    """Invoke Claude (Haiku or Sonnet) via Bedrock Messages API.

    `doc_kinds` (optional) is forwarded to the system-prompt builder so
    the KB-search addendum (`backend/routing/kb_search_prompt.py`)
    activates ONLY when the request restricted retrieval to KB / SOP
    content. None or any other doc_kinds → unchanged prompt behaviour.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": _build_claude_system_prompt(doc_kinds=doc_kinds),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(response["body"].read().decode("utf-8"))
    content = payload.get("content", [])
    text_parts = [
        item.get("text", "")
        for item in content
        if item.get("type") == "text"
    ]
    result = "\n".join(text_parts).strip()

    if not result:
        stop_reason = payload.get("stop_reason", "unknown")
        logger.warning("Claude returned empty response | model=%s stop_reason=%s", model_id, stop_reason)
        return "No response generated."

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def route_and_generate(
    *,
    query: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
    retrieval_confidence: float,
    recent_messages: Optional[List[Dict[str, str]]] = None,
    generate_fn: Callable[[str, int], str],
    bedrock_client: Any,
    triage_context: Optional[Dict[str, Any]] = None,
    pattern_context: Optional[Dict[str, Any]] = None,
    doc_kinds: Optional[List[str]] = None,
) -> RoutingResult:
    """
    Main routing entry point. Called by the /ask endpoint.
    Classifies complexity → selects model → builds prompt → invokes → returns.
    """
    result = RoutingResult()

    # Brief 5 / Part 2 — response-class-driven max_tokens budget.
    # Classifier is zero-LLM (regex + optional merged-triage signal). When
    # disabled, fall back to a conservative default cap.
    _resp_class = None
    _resp_cap = settings.RESPONSE_TOKENS_DEFAULT
    if settings.RESPONSE_TOKEN_CAPS_ENABLED:
        try:
            from backend.services.response_class_router import (
                classify_response_class, get_token_cap,
            )
            _resp_class = classify_response_class(
                query, context={"merged_triage": triage_context} if triage_context else None,
            )
            _resp_cap = get_token_cap(_resp_class)
            logger.info(
                "[resp_class] class=%s max_tokens=%d query=%r",
                _resp_class.value, _resp_cap, (query or "")[:80],
            )
        except Exception as _rc_exc:
            logger.warning("[resp_class] classifier failed (%s) — default cap", _rc_exc)
            _resp_class = None
            _resp_cap = settings.RESPONSE_TOKENS_DEFAULT

    # Step 1: Classify complexity
    complexity = classify_complexity(
        query=query,
        ranked_chunks=ranked_chunks,
        retrieval_confidence=retrieval_confidence,
        source_count=len(set(source_names)),
        context_chars=len(doc_context),
    )
    result.complexity = complexity

    # Step 2: Select model
    model_name, reason = _select_model(complexity)
    result.model_used = model_name
    result.reason = reason

    logger.info(
        "Model routing: query='%.80s' -> model=%s | %s",
        query, model_name, reason,
    )

    # Step 3: Build enriched prompt
    # doc_kinds is threaded through so context_builder can apply the
    # KB-Search variant of grounding rules (numbered lists + headings
    # allowed) when the request is restricted to kb/sop content.
    prompt = build_prompt(
        query=query,
        doc_context=doc_context,
        target_model=model_name,
        recent_messages=recent_messages,
        ranked_chunks=ranked_chunks,
        retrieval_confidence=retrieval_confidence,
        source_count=len(set(source_names)),
        pattern_context=pattern_context,
        doc_kinds=doc_kinds,
    )
    result.prompt_chars = len(prompt)

    # Step 4: Invoke the selected model
    t_start = time.perf_counter()

    def _invoke_with_cap(cap: int) -> str:
        if model_name == "sonnet":
            return _invoke_claude(
                prompt=prompt,
                bedrock_client=bedrock_client,
                model_id=settings.BEDROCK_SONNET_MODEL,
                max_tokens=cap,
                temperature=settings.SONNET_TEMPERATURE,
                # KB-search addendum activates only when doc_kinds is a
                # non-empty subset of {"kb","sop"} (see
                # kb_search_prompt.is_kb_search_mode). Mistral path is
                # unaffected by design — KB prompt is Claude-only.
                doc_kinds=doc_kinds,
            )
        if model_name == "haiku":
            return _invoke_claude(
                prompt=prompt,
                bedrock_client=bedrock_client,
                model_id=settings.BEDROCK_HAIKU_MODEL,
                max_tokens=cap,
                temperature=settings.HAIKU_ANSWER_TEMPERATURE,
                doc_kinds=doc_kinds,
            )
        return _invoke_mistral(prompt, generate_fn)

    try:
        answer = _invoke_with_cap(_resp_cap)
        # Brief 5 / Part 2 — one-shot truncation retry at the next tier up.
        if (
            settings.RESPONSE_TOKEN_CAPS_ENABLED
            and _resp_class is not None
            and model_name in {"haiku", "sonnet"}
        ):
            from backend.services.response_class_router import is_truncated, next_tier, get_token_cap
            if is_truncated(answer, _resp_cap):
                nxt = next_tier(_resp_class)
                if nxt is not None:
                    retry_cap = get_token_cap(nxt)
                    logger.warning(
                        "[resp_class] answer appears truncated at %d tokens, retrying at %s (cap=%d)",
                        _resp_cap, nxt.value, retry_cap,
                    )
                    try:
                        answer = _invoke_with_cap(retry_cap)
                        _resp_class = nxt
                        _resp_cap = retry_cap
                    except Exception as _rt_exc:
                        logger.warning("[resp_class] retry failed (%s) — keeping truncated answer", _rt_exc)

    except Exception as exc:
        logger.warning("Model %s failed (%s), falling back to Mistral", model_name, exc)
        result.model_used = "mistral (fallback)"
        result.reason += f" | {model_name} failed: {exc}"
        try:
            answer = _invoke_mistral(prompt, generate_fn)
        except Exception as fallback_exc:
            logger.error("Mistral fallback also failed: %s", fallback_exc)
            answer = "Error generating response. Please try again."

    t_end = time.perf_counter()
    result.answer = answer
    result.generation_ms = int((t_end - t_start) * 1000)
    if _resp_class is not None:
        result.reason += f" | resp_class={_resp_class.value} max_tokens={_resp_cap}"

    logger.info(
        "Generation complete: model=%s, %dms, %d prompt chars, complexity=%.3f (%s)",
        result.model_used, result.generation_ms,
        result.prompt_chars, complexity.score, complexity.tier,
    )

    return result