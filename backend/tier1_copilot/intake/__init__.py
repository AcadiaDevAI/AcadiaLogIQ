"""Sprint 9 — Universal Intake module.

Engineer pastes raw text from email / phone / portal / chat / note;
the extractor returns 1-4 validated structured candidates, the
diversifier rejects duplicate signatures, and the route handler
persists every extraction (picks + rejections) for future eval.

All modules are pure Python except `extractor.py` (one Bedrock Haiku
call via the shared `invoke_llm` helper). No new dependencies — fuzzy
matching uses stdlib `difflib`.
"""
