"""
Brief 4 / Opt 1 acceptance: 20 identical queries → 1 Bedrock call, 19 cache hits.
We stub _invoke_bedrock so we don't actually hit AWS. The rewrite_query
function should still drive through the cache path once and short-circuit
the remaining 19 invocations with a HIT log.
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services import query_rewriter as qr

_call_counter = {"n": 0}


def _fake_bedrock(prompt, max_tokens, temperature):
    _call_counter["n"] += 1
    # Return a minimal valid-shape rewrite JSON
    return '{"rewritten_query": "What caused INC-10015?", "was_rewritten": true, "reason": "pronoun_resolved", "confidence": 0.95}'


qr._invoke_bedrock = _fake_bedrock
qr.cache_clear()

history = [
    {"role": "user", "content": "Tell me about INC-10015"},
    {"role": "assistant", "content": "INC-10015 was a P1 incident at Enterprise-792 where 50 barcode scanners failed after firmware push."},
]

print("Running 20 identical rewrites...")
for i in range(20):
    result = qr.rewrite_query("what caused it?", recent_messages=history)
    print(f"  iter {i+1:2d}: was_rewritten={result.was_rewritten} reason={result.reason} conf={result.confidence:.2f}")

print()
print(f"Bedrock invocations: {_call_counter['n']}  (expected: 1)")
print(f"Cache hits (derived): {20 - _call_counter['n']}  (expected: 19)")
print(f"Cache stats: {qr.cache_stats()}")
