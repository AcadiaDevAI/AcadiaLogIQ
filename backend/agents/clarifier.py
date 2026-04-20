"""
Clarifier Agent — decides whether a query is ambiguous and, if so,
emits up to CLARIFIER_MAX_QUESTIONS short clarifying questions for the user.
Runs BEFORE the Planner in the multi-agent pipeline. Fails safely: if the
model call errors or the response can't be parsed, returns "no clarification
needed" so the pipeline continues to Planner → Analyst → Composer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings
from backend.agents.base import AgentStepResult, TokenBudget, invoke_llm

logger = logging.getLogger("acadia-log-iq")


@dataclass
class ClarifierResult:
    """Outcome of a clarifier run."""
    needs_clarification: bool = False
    questions: List[str] = field(default_factory=list)
    reason: str = ""
    step: Optional[AgentStepResult] = None

    def to_answer_text(self) -> str:
        """Format the clarifying questions as a user-facing answer."""
        if not self.questions:
            return ""
        lines = [
            "Before I can give you a grounded answer, could you clarify the following:",
            "",
        ]
        for i, q in enumerate(self.questions, start=1):
            lines.append(f"{i}. {q}")
        return "\n".join(lines)


_AMBIGUITY_HINTS = re.compile(
    r"\b(it|this|that|they|them|those|thing|issue|problem|error)\b",
    re.I,
)

# Domain detectors — used to enforce topic-locked clarifying questions.
_VPN_DOMAIN = re.compile(
    r"\b(vpn|tunnel|split[\-\s]?tunnel|anyconnect|pulse\s*secure|forticlient|"
    r"openvpn|ikev2|wireguard|globalprotect|remote\s+access)\b",
    re.I,
)
_AUTH_DOMAIN = re.compile(
    r"\b(auth(entic(ate|ation))?|login|log[\-\s]?in|sign[\-\s]?in|password|"
    r"credential|sso|mfa|2fa|otp|token)\b",
    re.I,
)
_TELEPHONY_DOMAIN = re.compile(
    r"\b(dial[\-\s]?tone|account\s*code|handset|hook[\-\s]?switch|voicemail|"
    r"extension|pbx|caller\s*id|call\s*forward)\b",
    re.I,
)

# Queries that explicitly ask for steps/how-to should be answered directly —
# the Planner/Analyst/Composer handle it; clarifier must not interject.
_DIRECT_ANSWER_PATTERNS = re.compile(
    r"\b(troubleshoot(ing)?\s+steps?|steps?\s+to\s+(troubleshoot|fix|resolve|"
    r"diagnose)|how\s+(do\s+i|to|can\s+i)\s+(fix|resolve|solve|troubleshoot)|"
    r"walk\s+me\s+through\s+(the\s+)?steps?)\b",
    re.I,
)

# Specific-target patterns — these pin down intent unambiguously, so the
# clarifier must not interrupt retrieval with a generic "can you clarify?".
_TICKET_ID_PATTERN = re.compile(r"\bINC-\d+\b", re.I)
_ENTITY_PATTERN = re.compile(r"\b(?:customer|enterprise|nebula)-?\w+\b", re.I)
_DIRECT_FACTUAL_PATTERN = re.compile(
    r"^(what|who|when|where|how\s+many|list|show)\b",
    re.I,
)


def _detect_domain(query: str) -> str:
    """Classify the user query into a coarse domain for topic-locking."""
    q = query or ""
    if _VPN_DOMAIN.search(q):
        return "vpn"
    if _AUTH_DOMAIN.search(q):
        return "auth"
    if _TELEPHONY_DOMAIN.search(q):
        return "telephony"
    return "general"


def _is_cross_domain(question: str, domain: str) -> bool:
    """True if `question` strays outside `domain`."""
    if domain == "vpn":
        return bool(_TELEPHONY_DOMAIN.search(question))
    if domain == "auth":
        return bool(_TELEPHONY_DOMAIN.search(question))
    if domain == "telephony":
        return bool(_VPN_DOMAIN.search(question) or _AUTH_DOMAIN.search(question))
    return False


def _format_history(prior_messages: Optional[List[Dict[str, str]]], max_turns: int = 4, max_chars: int = 1200) -> str:
    """Render recent turns as a compact block. Returns '' when no usable history."""
    if not prior_messages:
        return ""
    msgs = [m for m in prior_messages if isinstance(m, dict) and m.get("content")]
    msgs = msgs[-max_turns:]
    if not msgs:
        return ""
    lines: List[str] = []
    for m in msgs:
        role = str(m.get("role", "user")).strip().lower()
        label = "User" if role == "user" else ("Assistant" if role in ("assistant", "bot") else role.capitalize() or "User")
        content = str(m.get("content", "")).strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{label}: {content}")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[-max_chars:]
    return block


# ─────────────────────────────────────────────────────────────
# Agent pipeline performance — extended clarifier early-skip.
# Complements _looks_trivially_clear below. Adds extra guard patterns
# (greetings, simple factual, aggregation keywords) so the clarifier LLM
# call is avoided on queries whose intent is already obvious. Fully
# flag-gated via CLARIFIER_EARLY_SKIP_ENABLED — when False, returns
# (False, "flag_disabled") and the pipeline continues exactly as before.
# ─────────────────────────────────────────────────────────────

_CLARIFIER_SKIP_SIMPLE_PATTERNS = [
    re.compile(r"^(hi|hello|hey|thanks|thank you)\b", re.I),
    re.compile(r"^(what is|what are|who is|when did)\b.{3,40}$", re.I),
    re.compile(r"^(how many|count|list)\b", re.I),
]


def _should_skip_clarifier(
    query: str,
    extracted_identifiers: Optional[List[str]] = None,
    retrieval_confidence: float = 0.0,
) -> tuple[bool, str]:
    """
    Decide whether the clarifier step should be skipped entirely.

    Returns (should_skip, reason). Fails safely: if the feature flag is
    off, returns (False, "flag_disabled") so existing behavior is
    byte-identical.
    """
    if not getattr(settings, "CLARIFIER_EARLY_SKIP_ENABLED", True):
        return False, "flag_disabled"

    if extracted_identifiers and len(extracted_identifiers) >= 1:
        return True, "specific_identifier_present"

    if retrieval_confidence >= 0.90:
        return True, "high_retrieval_confidence"

    query_lower = (query or "").lower().strip()
    for pattern in _CLARIFIER_SKIP_SIMPLE_PATTERNS:
        if pattern.match(query_lower):
            return True, "simple_pattern"

    return False, "clarifier_needed"


def _looks_trivially_clear(query: str) -> Optional[str]:
    """
    Quick pre-check. Returns a short reason string when the clarifier LLM
    should be skipped, or None when the query is ambiguous enough to
    warrant clarification. The reason is surfaced in the caller's log line.
    """
    q = (query or "").strip()
    if len(q) < settings.CLARIFIER_MIN_QUERY_LEN:
        return "too-short"
    # Explicit "troubleshooting steps" / "how do I fix X" → answer directly.
    if _DIRECT_ANSWER_PATTERNS.search(q):
        return "direct-answer-request"
    # Long queries that cite specifics are usually clear enough to plan against.
    if len(q) > 160 and not _AMBIGUITY_HINTS.search(q):
        return "long-specific"
    # Specific targets should never be gated by the clarifier. ONE ticket ID
    # already fully disambiguates the query; so does TWO (e.g. "Compare
    # INC-10005 and INC-10006") — use findall so multi-ID queries skip too.
    if len(_TICKET_ID_PATTERN.findall(q)) >= 1:
        return "ticket-id"
    if _ENTITY_PATTERN.search(q):
        return "named-entity"
    if (
        _DIRECT_FACTUAL_PATTERN.search(q)
        and len(q) < 120
        and not _AMBIGUITY_HINTS.search(q)
    ):
        return "direct-factual"
    return None


def run_clarifier(
    *,
    query: str,
    doc_context_preview: str,
    source_names: List[str],
    budget: TokenBudget,
    generate_fn: Callable,
    bedrock_client: Any,
    prior_messages: Optional[List[Dict[str, str]]] = None,
) -> ClarifierResult:
    """
    Decide whether the query needs clarification before planning.

    Returns a ClarifierResult. On any failure (LLM error, parse error,
    disabled flag, budget exhausted) returns needs_clarification=False so
    the pipeline continues unchanged.
    """
    result = ClarifierResult()

    if not settings.ENABLE_CLARIFIER:
        result.reason = "clarifier disabled"
        return result

    history_block = _format_history(prior_messages)

    # Unconditional early return when the query is self-contained (ticket ID,
    # named entity, direct factual ask). Previously this skip was gated on
    # `not history_block`, which caused the LLM to still run after an earlier
    # assistant turn — including asking a clarifying question about a ticket
    # ID the user had already named explicitly. The skip MUST short-circuit
    # here so the log line "skipped — specific query pattern" is a promise,
    # not a suggestion.
    skip_reason = _looks_trivially_clear(query)
    if skip_reason:
        logger.info("[clarifier] skipped — specific query pattern: %s", skip_reason)
        result.reason = f"skipped: {skip_reason}"
        return result

    # Extended performance skip — greetings, simple factual, aggregation
    # patterns. Additive to _looks_trivially_clear; only fires when flag is on.
    extracted_ids = _TICKET_ID_PATTERN.findall(query or "")
    should_skip, skip_kind = _should_skip_clarifier(
        query, extracted_identifiers=extracted_ids, retrieval_confidence=0.0,
    )
    if should_skip:
        logger.info("[clarifier_skip] reason=%s", skip_kind)
        result.reason = f"skipped_early: {skip_kind}"
        return result

    if budget.exhausted:
        result.reason = "token budget exhausted before clarifier"
        return result

    sources_str = ", ".join(sorted(set(source_names))[:6]) if source_names else "unknown"
    domain = _detect_domain(query)
    # VPN / auth queries are capped to a single clarifying question to avoid
    # drift into unrelated sub-topics.
    max_q = max(1, int(settings.CLARIFIER_MAX_QUESTIONS))
    if domain in ("vpn", "auth"):
        max_q = 1

    domain_guidance = {
        "vpn": (
            "The user's query is about VPN / remote connectivity. Any clarifying "
            "question MUST be VPN-specific (e.g. client name/version, error "
            "message, tunnel type, MFA step, network location). NEVER ask about "
            "dial tone, account codes, handsets, voicemail, or other telephony "
            "topics. Ask at most ONE question."
        ),
        "auth": (
            "The user's query is about authentication / login failure. Any "
            "clarifying question MUST be auth-specific (e.g. which system, "
            "exact error, SSO vs local, MFA prompt seen). NEVER ask about "
            "dial tone, account codes, handsets, or other telephony topics. "
            "Ask at most ONE question, or answer directly if possible."
        ),
        "telephony": (
            "The user's query is about telephony / calling. Keep any clarifying "
            "question on-topic (handset, extension, dial tone, account code, "
            "voicemail). NEVER ask about VPN, SSO, or MFA tokens."
        ),
        "general": (
            "Stay strictly on the user's topic. Do not pivot to unrelated "
            "domains (e.g. do not ask telephony questions for a network issue, "
            "or network questions for a telephony issue)."
        ),
    }[domain]

    history_section = (
        f"\nRecent conversation (oldest → newest):\n{history_block}\n"
        if history_block else ""
    )

    prompt = f"""You are a clarification agent for a document-grounded assistant.
Decide whether the user's LATEST message is specific enough to answer directly
from the available documents, OR whether it is ambiguous / underspecified and
would benefit from 1-{max_q} short clarifying questions first.

Return ONLY a JSON object with this exact shape — no prose, no markdown:
{{"needs_clarification": true|false, "questions": ["...", "..."], "reason": "short reason"}}

Domain lock (CRITICAL):
{domain_guidance}

Rules:
- Interpret the latest message in light of the recent conversation. If an
  earlier assistant turn asked a question and the latest user message answers
  it (even tersely, e.g. "prod", "yes"), treat the query as CLEAR and set
  needs_clarification=false.
- Ask clarifying questions only when the query is genuinely ambiguous
  (vague pronouns with no antecedent in history, missing scope, unclear
  target, multiple plausible intents).
- Do NOT re-ask something the user has already answered in this conversation.
- Never ask more than {max_q} questions.
- Each question must be short (max ~20 words) and directly actionable.
- Each question MUST stay within the user's domain (see domain lock above);
  cross-domain questions are forbidden and will be discarded.
- Do NOT ask for information the documents are likely to contain — ask only
  about user intent, scope, or missing input.

Available document sources: {sources_str}

Document context preview (first 1200 chars):
{doc_context_preview[:1200]}
{history_section}
Latest user message: {query}

JSON:"""

    step = invoke_llm(
        prompt=prompt,
        model=settings.AGENT_CLARIFIER_MODEL,
        max_tokens=settings.AGENT_CLARIFIER_MAX_TOKENS,
        budget=budget,
        agent_name="clarifier",
        generate_fn=generate_fn,
        bedrock_client=bedrock_client,
    )
    result.step = step

    if not step.success or not step.output:
        result.reason = f"clarifier LLM failed: {step.error or 'empty output'}"
        logger.warning("Clarifier failed safely: %s", result.reason)
        return result

    raw = step.output.strip()
    if "```" in raw:
        try:
            raw = raw.split("```json")[-1].split("```")[0] if "```json" in raw else raw.split("```")[1]
        except IndexError:
            pass

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("no JSON object in clarifier output")
        parsed = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError) as exc:
        result.reason = f"clarifier parse error: {exc}"
        logger.warning("Clarifier parse failed safely: %s | raw=%r", exc, raw[:200])
        return result

    needs = bool(parsed.get("needs_clarification", False))
    questions_raw = parsed.get("questions", []) or []
    if not isinstance(questions_raw, list):
        questions_raw = []

    questions = [str(q).strip() for q in questions_raw if str(q).strip()]

    # Enforce the domain lock: drop any cross-domain question the LLM still
    # emitted (e.g. asking about dial tone / account codes for a VPN issue).
    dropped: List[str] = []
    kept: List[str] = []
    for q in questions:
        if _is_cross_domain(q, domain):
            dropped.append(q)
        else:
            kept.append(q)
    if dropped:
        logger.info(
            "Clarifier dropped %d cross-domain question(s) for domain=%s: %s",
            len(dropped), domain, dropped,
        )
    questions = kept[:max_q]

    if needs and questions:
        result.needs_clarification = True
        result.questions = questions
        result.reason = str(parsed.get("reason", "") or "ambiguous query")
        logger.info(
            "Clarifier wants %d question(s) [domain=%s]: %s",
            len(questions), domain, questions,
        )
    else:
        result.needs_clarification = False
        base_reason = str(parsed.get("reason", "") or "query is specific enough")
        if needs and not questions:
            # LLM wanted to ask, but every question was cross-domain → suppress.
            result.reason = f"{base_reason} (all questions dropped as cross-domain)"
        else:
            result.reason = base_reason
        logger.info(
            "Clarifier: no clarification needed [domain=%s] (%s)",
            domain, result.reason,
        )

    return result
