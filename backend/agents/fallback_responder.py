"""
Fallback Responder — handles the "no document match" path.
Runs the Clarifier first so vague queries like "it's down" get asked back.
If the query is clear, emits a short, honest, support-engineer-style reply
that engages the user WITHOUT fabricating document content.

Used by /ask when retrieval returns nothing or insufficient support.
Fails safely: on any error, returns the legacy hard-coded "no support" text
so the existing behavior is preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm
from backend.agents.clarifier import run_clarifier, _format_history

logger = logging.getLogger("acadia-log-iq")


_LEGACY_NO_SUPPORT = (
    "- I could not find supporting information for that question in the currently uploaded files.\n"
    "- Please ask a question that is directly covered by the uploaded document content."
)


@dataclass
class FallbackResult:
    """Result of the no-doc-match fallback path."""
    answer: str = ""
    mode: str = "no_docs"              # 'clarify' | 'conversational' | 'legacy'
    clarifying: bool = False
    steps: List[AgentStepResult] = field(default_factory=list)
    reason: str = ""


def run_conversational_fallback(
    *,
    query: str,
    source_names: List[str],
    generate_fn: Callable,
    bedrock_client: Any,
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> FallbackResult:
    """
    Produce a helpful response when no supporting documents were retrieved.

    Flow:
      1. Run Clarifier (with conversation history). If it wants to ask a
         question, return that as the answer.
      2. Otherwise, generate a short, honest, support-engineer-style reply
         that acknowledges the lack of doc match and offers general guidance
         or asks for specifics — without inventing document content.

    Never raises. On error, returns the legacy hard-coded "no support" text
    under mode='legacy' so the caller's contract is preserved.
    """
    result = FallbackResult()
    budget = TokenBudget()

    try:
        # --- 1. Clarifier on the no-docs path (empty doc context) ---
        clar = run_clarifier(
            query=query,
            doc_context_preview="",
            source_names=source_names,
            budget=budget,
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
            prior_messages=prior_messages,
        )
        if clar.step is not None:
            result.steps.append(clar.step)

        if clar.needs_clarification and clar.questions:
            result.answer = clar.to_answer_text()
            result.mode = "clarify"
            result.clarifying = True
            result.reason = f"clarifier asked {len(clar.questions)} question(s)"
            return result

        # --- 2. Conversational support-engineer-style reply ---
        history_block = _format_history(prior_messages)
        history_section = (
            f"\nRecent conversation (oldest → newest):\n{history_block}\n"
            if history_block else ""
        )
        sources_hint = (
            f"Uploaded document sources on file: {', '.join(sorted(set(source_names))[:6])}."
            if source_names else
            "No documents are currently uploaded for this session."
        )

        prompt = f"""You are a calm, experienced support engineer replying in a chat.
The user's latest message could NOT be matched to any uploaded document content.
You must still engage helpfully — like a human support engineer would — without
fabricating information from non-existent documents.

Rules:
- Be brief and conversational (2–5 short bullet points).
- Do NOT invent document citations or pretend you found a source.
- If the message is a status report ("it's down", "still broken", "not working"),
  acknowledge it, suggest 1–2 generic first checks, and ask for the one most
  useful specific detail (e.g. service name, error text, timestamp) to narrow it down.
- If the message is a general question unrelated to uploaded docs, answer from
  common support knowledge in 1–3 bullets, and note that no matching document
  was found in case they expected one.
- Never refuse outright. Always move the conversation forward.
- Do NOT repeat a question the user has already answered in the recent conversation.

{sources_hint}
{history_section}
Latest user message: {query}

Reply:"""

        step = invoke_llm(
            prompt=prompt,
            model=settings.AGENT_COMPOSER_MODEL,
            max_tokens=min(600, settings.AGENT_COMPOSER_MAX_TOKENS),
            budget=budget,
            agent_name="fallback_responder",
            generate_fn=generate_fn,
            bedrock_client=bedrock_client,
        )
        result.steps.append(step)

        if step.success and step.output.strip():
            result.answer = step.output.strip()
            result.mode = "conversational"
            result.reason = "conversational no-doc reply"
            return result

        # LLM empty/failed → legacy text
        result.answer = _LEGACY_NO_SUPPORT
        result.mode = "legacy"
        result.reason = f"fallback LLM unusable: {step.error or 'empty output'}"
        logger.warning("Fallback responder falling back to legacy text: %s", result.reason)
        return result

    except Exception as exc:
        logger.exception("Fallback responder crashed, using legacy text: %s", exc)
        result.answer = _LEGACY_NO_SUPPORT
        result.mode = "legacy"
        result.reason = f"fallback crashed: {exc}"
        return result
