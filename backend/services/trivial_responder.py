"""
Trivial-response lookup table — zero-LLM canned replies for greetings,
farewells, thanks, and small-talk. The /ask endpoint calls
match_trivial_response() as its very first step so these inputs never
touch Bedrock, retrieval, or the answer cache.

A single word is canonical: short, normalized, whitespace+punctuation
tolerant. The 6-word cap on substring matches keeps genuine questions
like "how does hi impact the architecture" from being swallowed.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


TRIVIAL_RESPONSES: Dict[str, Dict[str, object]] = {
    "greeting": {
        "patterns": [
            "hi", "hello", "hey", "hiya", "yo",
            "good morning", "good afternoon", "good evening",
            "greetings", "howdy", "hola", "namaste", "sup",
            "hii", "helo", "heya",
        ],
        "response": "Hello! Ask me anything about your uploaded documents.",
    },
    "farewell": {
        "patterns": [
            "bye", "goodbye", "see you", "see ya", "later",
            "take care", "good night", "gn", "ttyl", "cya",
        ],
        "response": (
            "Goodbye! Feel free to come back whenever you need help with "
            "your documents."
        ),
    },
    "thanks": {
        "patterns": [
            "thanks", "thank you", "thx", "ty",
            "much appreciated", "appreciate it",
            "cheers", "thanks a lot", "thank you so much",
            "thank u", "tysm",
        ],
        "response": (
            "You're welcome! Let me know if you have more questions about "
            "your documents."
        ),
    },
    "how_are_you": {
        "patterns": [
            "how are you", "how r u", "how's it going", "hows it going",
            "how are you doing", "whats up", "what's up", "hows everything",
            "how is it going",
        ],
        "response": (
            "I'm doing well and ready to help with your documents. What "
            "would you like to know?"
        ),
    },
    "who_are_you": {
        "patterns": [
            "who are you", "what are you", "whats your name", "what's your name",
            "introduce yourself", "tell me about yourself",
        ],
        "response": (
            "I'm Acadia Log IQ — an AI assistant for your uploaded tickets, "
            "runbooks, and knowledge documents. I can look up specific "
            "incidents, aggregate across tickets, compare records, and walk "
            "through root causes and resolutions. What can I help with?"
        ),
    },
    "capabilities": {
        "patterns": [
            "what can you do", "what do you do", "help",
            "what are your capabilities", "how can you help",
            "what can i ask", "what do you know",
        ],
        "response": (
            "Here's what I'm built to help with, grounded in whatever "
            "documents you've uploaded:\n\n"
            "- Look up a specific record by its ID (ticket number, case "
            "number, KB article, Jira key, etc.)\n"
            "- Count, list, rank, or group records by customer, priority, "
            "SLA status, or any field in the source\n"
            "- Compare two or more records side by side\n"
            "- Walk through root causes, resolution steps, SOPs, and QA "
            "findings\n"
            "- Answer follow-up questions within the same session\n\n"
            "Ask me anything grounded in your uploads."
        ),
    },
    "affirmation": {
        "patterns": [
            "ok", "okay", "got it", "sure", "alright",
            "fine", "cool", "nice", "great", "awesome",
            "understood", "roger",
        ],
        "response": "Got it. Let me know your next question.",
    },
    "small_talk_refuse": {
        "patterns": [
            "tell me a joke", "sing a song", "write a poem",
            "whats the weather", "what's the weather",
            "who won the game", "tell me a story",
        ],
        "response": (
            "I'm focused on the documents you've uploaded — tickets, "
            "runbooks, KBs, and similar. Is there something from your "
            "documents I can help with?"
        ),
    },
}


# Normalization: strip common sentence punctuation, collapse whitespace,
# lowercase. Preserve hyphens and digits so 'INC-10015' survives (though
# it won't match any trivial pattern anyway — this is a safety belt).
_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\[\]{}]")
_WS_RE = re.compile(r"\s+")

# The 6-word cap is load-bearing: it prevents "hi" from matching inside
# "how does hi impact the architecture" (6+ words → not a greeting).
_MAX_WORDS_FOR_SUBSTRING_MATCH = 6


def _normalize(query: str) -> str:
    if not query:
        return ""
    q = query.strip().lower()
    q = _PUNCT_RE.sub(" ", q)
    q = _WS_RE.sub(" ", q).strip()
    return q


def match_trivial_response(query: str) -> Optional[Tuple[str, str]]:
    """Return (category, response_text) if the query is trivial/small-talk.

    Matching rules (evaluated in registry-declaration order):
      1. Normalize: lowercase + strip punctuation + collapse whitespace.
      2. Exact match: normalized query IS a pattern → hit.
      3. Short-query substring match: if the query is <= 6 words AND a
         pattern is a word-bounded substring of the normalized query → hit.
         The word-boundary guard prevents 'hi' from matching inside 'hiking'.

    Returns None for empty input, long queries, or anything containing a
    real-query shape (even trivially short ones).
    """
    if not query:
        return None
    normalized = _normalize(query)
    if not normalized:
        return None

    word_count = len(normalized.split())

    for category, entry in TRIVIAL_RESPONSES.items():
        patterns: List[str] = entry["patterns"]  # type: ignore[assignment]
        response: str = entry["response"]  # type: ignore[assignment]

        # 1. Exact match — fast path, size-independent.
        if normalized in patterns:
            return category, response

        # 2. Substring match — only for short queries.
        if word_count > _MAX_WORDS_FOR_SUBSTRING_MATCH:
            continue
        for pat in patterns:
            if " " in pat:
                # Multi-word pattern: look for exact substring.
                if pat in normalized:
                    return category, response
            else:
                # Single-word pattern: require word boundaries so 'hi' in
                # 'hiking' or 'hidden' doesn't count.
                if re.search(rf"\b{re.escape(pat)}\b", normalized):
                    return category, response

    return None


if __name__ == "__main__":
    cases = [
        ("hi", "greeting"),
        ("HELLO", "greeting"),
        ("hi!", "greeting"),
        ("hi there", "greeting"),
        ("thanks a lot", "thanks"),
        ("what can you do", "capabilities"),
        ("who are you?", "who_are_you"),
        ("tell me about INC-10015", None),
        ("how many Nebula-Corp tickets?", None),
        ("hi can you tell me about X", None),
    ]
    ok = 0
    for q, expected in cases:
        got = match_trivial_response(q)
        got_cat = got[0] if got else None
        status = "PASS" if got_cat == expected else "FAIL"
        ok += 1 if status == "PASS" else 0
        print(f"{status} | {q!r:45s} → {got_cat!r} (expected {expected!r})")
    print(f"\n{ok}/{len(cases)} passed")
