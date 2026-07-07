"""Thin Bedrock wrappers for the Escalation KB.

We deliberately do NOT reuse ``backend.services.embedding_service`` —
its current ``str(body)`` call shape is broken for Titan v2. Likewise
we keep our own Haiku invoker so prompt edits here can't perturb the
production chat pipeline.
"""

from __future__ import annotations

import json
import logging
import math
from typing import List, Optional

import boto3
from botocore.config import Config as BotoConfig

from backend.config import settings
from backend.services.token_usage import record_token_usage, extract_bedrock_usage


logger = logging.getLogger("acadia-log-iq")


def _make_client():
    boto_cfg = BotoConfig(
        retries={"max_attempts": 6, "mode": "adaptive"},
        read_timeout=60,
        connect_timeout=20,
        tcp_keepalive=True,
    )
    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
        "config": boto_cfg,
    }
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN
    return boto3.client(**kwargs)


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# ── Embeddings ──────────────────────────────────────────────────────


def embed_text(text: str) -> List[float]:
    body = json.dumps({"inputText": (text or "").strip() or " "})
    resp = _get_client().invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=body.encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))
    record_token_usage("embeddings", settings.BEDROCK_EMBED_MODEL, *extract_bedrock_usage(payload, resp))
    vector = payload.get("embedding") or []
    if not vector:
        raise RuntimeError("Titan returned an empty embedding")
    return [float(x) for x in vector]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ── Haiku grounded answer ───────────────────────────────────────────

_ANSWER_SYSTEM = (
    "You are LogIQ Assistance, a calm, senior-engineer escalation guide. "
    "Answer the user's question STRICTLY using the provided excerpts from the "
    "Escalation Procedures KB. Do not invent steps, names, or contact details "
    "that are not in the excerpts.\n\n"
    "Voice & formatting (follow exactly):\n"
    "- Open with a one-sentence direct answer, then the supporting detail.\n"
    "- MANDATORY BULLETING: if the answer contains 2 or more of any of the "
    "following, you MUST present them as hyphen bullets (\"- item\"), one per "
    "line — never as prose:\n"
    "    * conditions / criteria / triggers (a, b, c... or 1, 2, 3...)\n"
    "    * severities, tiers, or levels (Sev 1, Sev 2, Sev 3, Sev 4)\n"
    "    * steps, options, channels, or contact methods\n"
    "    * phone numbers, URLs, emails paired with a purpose\n"
    "  Each bullet stands alone — do NOT pack two severities or two channels "
    "into one bullet or one sentence.\n"
    "  Example of FORBIDDEN prose: \"Use the hotline for Sev 1 and Sev 2. "
    "For Sev 3 and Sev 4, open a web case.\"\n"
    "  Example of CORRECT bullets:\n"
    "    - Severity 1 — call the hotline (24x7)\n"
    "    - Severity 2 — call the hotline (24x7)\n"
    "    - Severity 3 — open a web case during business hours\n"
    "    - Severity 4 — open a web case during business hours\n"
    "- Bullets should be short and parallel (same grammar shape across items).\n"
    "- Keep total length tight: aim for under ~150 words unless the user asks "
    "for depth.\n"
    "- Do NOT use markdown bold (**...**), italics (*...*), headings (#, ##), "
    "blockquotes, tables, or code fences.\n"
    "- ABSOLUTELY NO page-number citations anywhere in the response. Never "
    "emit \"[p.3]\", \"[p. 3]\", \"[pp.3-4]\", \"(p.3)\", \"(page 3)\", "
    "\"on page 3\", or a trailing \"Sources:\" line. Not at the end, not "
    "mid-sentence, not in brackets, not in parentheses. The UI handles "
    "attribution separately. Example of FORBIDDEN: \"Open a TAC case when "
    "[p.3]: ...\". Example of CORRECT: \"Open a TAC case when: ...\".\n"
    "- Plain prose and simple bullets only.\n\n"
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I don't have that information in the Escalation Procedures KB for this "
    "section.\""
)


def answer_with_excerpts(
    *,
    section_label: str,
    question: str,
    excerpts: List[dict],
    history: Optional[List[dict]] = None,
    max_tokens: int = 700,
) -> str:
    """Call Bedrock Claude Haiku with the provided excerpts.

    ``excerpts`` items are ``{"page": int, "text": str}``.
    ``history`` items are ``{"role": "user"|"assistant", "text": str}``.
    """
    excerpt_block = "\n\n".join(
        f"[Section: {section_label}, Page {ex['page']}]\n{ex['text']}"
        for ex in excerpts
    ) or "(no excerpts found)"

    history_block = ""
    if history:
        rendered = []
        for turn in history[-6:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            text = (turn.get("text") or "").strip()
            if text:
                rendered.append(f"{role}: {text}")
        if rendered:
            history_block = "Prior turns (for context only):\n" + "\n".join(rendered) + "\n\n"

    prompt = (
        f"Section the user is asking about: {section_label}\n\n"
        f"{history_block}"
        f"Question: {question.strip()}\n\n"
        f"Excerpts from the Escalation Procedures KB:\n{excerpt_block}\n"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": _ANSWER_SYSTEM,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
    }
    resp = _get_client().invoke_model(
        modelId=settings.BEDROCK_HAIKU_MODEL,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))
    record_token_usage("escalation", settings.BEDROCK_HAIKU_MODEL, *extract_bedrock_usage(payload, resp))
    parts = payload.get("content", [])
    return "\n".join(
        item.get("text", "") for item in parts if item.get("type") == "text"
    ).strip() or "I don't have that information in the Escalation Procedures KB for this section."
