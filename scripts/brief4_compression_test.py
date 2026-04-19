"""
Brief 4 / Opt 5 acceptance: compression reduces a section-targeted
identifier_exact chunk. We synthesize a representative 10k-char ticket
chunk (same section layout as contextual_ingestion_service emits) and
check that:

- "What was the resolution of INC-10015?" → compressed.
- "Tell me about INC-10015" → unchanged.
"""
from __future__ import annotations

import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval.context_builder import compress_chunk_for_query, SECTION_MAPPINGS


CHUNK = """TICKET INC-10015 | Enterprise-792 | P1
CUSTOMER: Enterprise-792
PRIORITY: P1
STATUS: Resolved
OPEN TIME: 2024-03-12T09:08:00Z
CLOSE TIME: 2024-03-12T17:41:00Z

SHORT DESCRIPTION:
50 barcode scanners failing after firmware push in Enterprise-792 warehouse.

LONG DESCRIPTION:
Users reported widespread scanner failures following the scheduled firmware
update pushed overnight. The update was intended to patch a known bluetooth
pairing regression but instead broke the symbology decoder on the entire
fleet. First reported by the warehouse shift-lead at 08:52; tickets began
aggregating in the queue within 20 minutes. Floor operations ground to a
halt for pick/pack flow — impact estimated ~180 minutes of halted shipping
activity before workaround began reaching the floor.

BUSINESS IMPACT:
- 180 minutes of floor downtime
- ~40k in lost shipping throughput for the shift
- Customer escalations from two retail partners due to SLA breaches

ROOT CAUSE:
A corrupted driver bundle baked into the signed firmware image at build
time. The vendor's QA process missed a regression in the symbology decoder
after a refactor to the bluetooth subsystem. The build system's smoke test
covered the bluetooth path but never exercised the barcode decode path
against the new codec library, so it shipped green.

ITIL 5-WHY ROOT CAUSE:
1. Scanners stopped decoding barcodes — because driver bundle corrupted.
2. Driver bundle corrupted — because vendor firmware signed a broken build.
3. Vendor shipped broken build — because QA regression gap on decode path.
4. Regression gap existed — because smoke test coverage refactor stale.
5. Coverage stale — because release-engineering team was down 2 FTEs.

RESOLUTION DETAIL:
The resolution consisted of rolling back the firmware to the last
known-good build (v3.14.2). Rollback script was pushed via the device
management console to all 50 scanners. Rollback completed within 35
minutes once the runbook was located and the rollback package identified.
Post-rollback, all scanners recovered barcode decoding on first beep and
operations resumed.

RESOLUTION STEPS:
- Identify scope (all 50 scanners confirmed affected).
- Contact Ribbon vendor support for rollback package.
- Stage rollback via MDM console.
- Push rollback firmware and force reboot.
- Verify decode on known-good test barcodes.
- Close ticket + schedule post-mortem.

RESOLUTION GROUPS:
- Tier-2 Endpoint Ops (primary)
- Vendor Support (Ribbon) (consultative)
- Warehouse Ops (coordination)

SOP EXECUTION STEPS:
1. Validated ticket scope vs known-outage pattern.
2. Opened vendor bridge with Ribbon.
3. Staged rollback firmware on MDM.
4. Executed rollback wave-by-wave.
5. Spot-checked 5 devices post-rollback.
6. Announced resolution and closed.

QA AUDITOR GAPS:
- Runbook did not reference the rollback package version explicitly.
- Vendor bridge invite took 22 minutes — should be under 10.
- Post-mortem template not completed at close time (followed up next day).

RESOLUTION QUALITY SCORE: 0.82

SLA TARGET MET: true

METADATA: {primary_id: INC-10015, customer: Enterprise-792, priority: P1, sla_target_met: true}
"""

# Pad the noisy descriptive sections so the chunk crosses the 5k threshold.
# We expand LONG DESCRIPTION and ROOT CAUSE so they look like real 10k
# production chunks. RESOLUTION sections stay intact.
_PAD = (
    "\nAdditional forensic detail captured during the incident: operators "
    "reported audio beep patterns that no longer matched the decode-success "
    "chime on the affected devices; device logs showed repeated codec-init "
    "failures between 08:52 and 09:14; the MDM console surfaced no alerts "
    "because the fleet was still reporting 'healthy' heartbeat status. Floor "
    "supervisors routed overflow picks to the backup laser-scanner kiosk "
    "(single-user throughput only) while Ops paged Endpoint Ops. Secondary "
    "symptom: the device self-diagnostic page timed out on every scanner, "
    "indicating the codec init was blocking the diagnostic callback too."
) * 6  # inflate
CHUNK = CHUNK.replace(
    "LONG DESCRIPTION:",
    "LONG DESCRIPTION:" + _PAD,
).replace(
    "ROOT CAUSE:",
    "ROOT CAUSE:" + _PAD,
)

print(f"Original chunk size: {len(CHUNK)} chars")
print()

queries = [
    "What was the resolution of INC-10015?",
    "Tell me about INC-10015",
    "What were the QA gaps for INC-10015?",
    "What was the root cause of INC-10015?",
    "What teams worked on INC-10015?",
    "What is the resolution quality score of INC-10015?",
]

for q in queries:
    compressed, was = compress_chunk_for_query(CHUNK, q)
    pct = int(100 * len(compressed) / len(CHUNK))
    tag = "COMPRESSED" if was else "UNCHANGED "
    print(f"{tag}  {len(CHUNK)} -> {len(compressed)} chars ({pct}%)  q={q!r}")

# Dump compressed content for the brief's required report
print("\n\n=== Compressed output for 'What was the resolution of INC-10015?' ===")
compressed, _ = compress_chunk_for_query(CHUNK, "What was the resolution of INC-10015?")
print(compressed)
