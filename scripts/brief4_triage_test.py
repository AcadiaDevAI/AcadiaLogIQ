"""
Brief 4 / Opt 3 smoke test: run classify_triage against the Brief 3
14-query regression matrix and print the verdicts.
"""
from __future__ import annotations

import io
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.triage_classifier import classify_triage

queries = [
    ("R1", "Tell me about INC-10001"),
    ("R2", "Tell me about INC-10005"),
    ("R3", "Tell me about INC-10008"),
    ("R4", "Tell me about INC-10041"),
    ("R5", "Tell me about INC-10002"),
    ("R6", "How many Nebula-Corp tickets?"),
    ("R7", "Which customer had the most incidents?"),
    ("R8", "How many tickets missed SLA?"),
    ("R9", "Which ticket had the highest resolution quality score?"),
    ("R10", "Compare INC-10005 and INC-10006"),
    ("R11", "What are common root causes for WiFi incidents?"),
    ("R12", "Tell me about Enterprise-338"),
    ("R13", "What resolutions were applied for Enterprise-338?"),
    ("R14", "What QA gaps were identified for Enterprise-338?"),
]

# Warmup pass
print("warmup...")
_ = classify_triage("warmup")

print("\n{:5s} {:55s} {:9s} {:15s} {:7s} {:6s} {:7s}".format(
    "tag", "query", "cx", "intent", "mode", "conf", "ms"
))
for tag, q in queries:
    t0 = time.perf_counter()
    r = classify_triage(q)
    ms = int((time.perf_counter() - t0) * 1000)
    if r.is_valid:
        print("{:5s} {:55.55s} {:9s} {:15s} {:7s} {:6.2f} {:5d}".format(
            tag, q, r.complexity, r.intent, r.mode_hint, r.confidence, ms,
        ))
    else:
        print("{:5s} {:55.55s} INVALID raw={!r:40.40} {:5d}".format(
            tag, q, r.raw_response, ms,
        ))
