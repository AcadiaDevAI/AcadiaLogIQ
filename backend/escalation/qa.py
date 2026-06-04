"""Section-scoped retrieval + Haiku answer for the Escalation KB."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .bedrock_client import answer_with_excerpts, cosine, embed_text
from .sections import SECTION_LABELS
from .store import chunks_for_section


logger = logging.getLogger("acadia-log-iq")


def _top_k(
    query_vec: List[float], chunks: List[Dict], k: int
) -> List[Dict]:
    scored = []
    for c in chunks:
        emb = c.get("embedding") or []
        score = cosine(query_vec, emb)
        scored.append((score, c))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [c for _, c in scored[:k]]


def answer(
    *,
    section_id: str,
    question: str,
    history: Optional[List[Dict]] = None,
    top_k: int = 5,
) -> Dict:
    section_chunks = chunks_for_section(section_id)
    if not section_chunks:
        return {
            "answer": (
                "The Escalation Procedures KB hasn't been uploaded yet, or "
                "this section was not detected in the uploaded PDF."
            ),
            "sources": [],
        }

    query_vec = embed_text(question)
    top = _top_k(query_vec, section_chunks, top_k)
    excerpts = [{"page": int(c["page"]), "text": c["text"]} for c in top]

    text = answer_with_excerpts(
        section_label=SECTION_LABELS.get(section_id, section_id),
        question=question,
        excerpts=excerpts,
        history=history or [],
    )

    sources = [
        {"page": ex["page"], "snippet": ex["text"][:220]}
        for ex in excerpts
    ]
    return {"answer": text, "sources": sources}
