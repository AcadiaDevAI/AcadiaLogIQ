"""
Concurrency harness — demonstrates that the asyncio.to_thread fix keeps the
event loop free while a sync-blocking ingestion task runs.

Simulates:
  - A long sync-blocking function (stand-in for process_document)
  - A fast /ask-equivalent coroutine measured for latency

Runs the pattern BEFORE the fix (direct sync call inside async) and AFTER
(asyncio.to_thread wrap) and prints the observed interleaving.
"""
from __future__ import annotations

import asyncio
import time


# ── Sync-blocking work (stand-in for process_document + heavy phases) ──
def heavy_sync_work(label: str, seconds: float) -> str:
    t0 = time.perf_counter()
    end = t0 + seconds
    while time.perf_counter() < end:
        # Simulate CPU-bound work — real process_document does this with PDF
        # parsing + chunking + Haiku enrichment + DB inserts
        sum(range(10_000))
    return f"{label} done after {time.perf_counter() - t0:.2f}s"


# ── Fast /ask-equivalent: simulates a trivial response path ──
async def ask_like(tag: str) -> float:
    t0 = time.perf_counter()
    # Trivial match-like work — no I/O. Real /ask greeting path is ~50 ms.
    await asyncio.sleep(0)  # yield
    elapsed = time.perf_counter() - t0
    print(f"  [{ts()}] ask({tag}) returned in {elapsed*1000:.1f}ms")
    return elapsed


def ts() -> str:
    return f"{time.perf_counter():.3f}s"


# ── BEFORE: sync call directly inside async function (blocks event loop) ──
async def ingest_before_fix():
    print(f"  [{ts()}] ingest_before_fix start")
    heavy_sync_work("BEFORE-parse", 3.0)
    print(f"  [{ts()}] ingest_before_fix end")


# ── AFTER: sync call offloaded via asyncio.to_thread ──
async def ingest_after_fix():
    print(f"  [{ts()}] ingest_after_fix start")
    await asyncio.to_thread(heavy_sync_work, "AFTER-parse", 3.0)
    print(f"  [{ts()}] ingest_after_fix end")


async def scenario_before():
    print("\n=== BEFORE FIX (sync call inside async) ===")
    loop_start = time.perf_counter()
    ingestion_task = asyncio.create_task(ingest_before_fix())
    # Let ingestion grab the loop first
    await asyncio.sleep(0.05)

    latencies = []
    for i in range(5):
        lat = await ask_like(f"before-{i}")
        latencies.append(lat)

    await ingestion_task
    total = time.perf_counter() - loop_start
    max_ask = max(latencies) * 1000
    print(f"  BEFORE: total={total:.2f}s, max ask latency={max_ask:.1f}ms")
    return max_ask


async def scenario_after():
    print("\n=== AFTER FIX (asyncio.to_thread) ===")
    loop_start = time.perf_counter()
    ingestion_task = asyncio.create_task(ingest_after_fix())
    await asyncio.sleep(0.05)

    latencies = []
    for i in range(5):
        lat = await ask_like(f"after-{i}")
        latencies.append(lat)

    await ingestion_task
    total = time.perf_counter() - loop_start
    max_ask = max(latencies) * 1000
    print(f"  AFTER: total={total:.2f}s, max ask latency={max_ask:.1f}ms")
    return max_ask


async def main():
    print("Concurrency fix harness — simulates index_file_job vs /ask.\n")
    before_max = await scenario_before()
    after_max = await scenario_after()

    print("\n=== VERDICT ===")
    if after_max < before_max / 10:
        print(f"PASS — /ask latency dropped from {before_max:.1f}ms to "
              f"{after_max:.1f}ms (>=10x improvement)")
    else:
        print(f"UNEXPECTED — before={before_max:.1f}ms after={after_max:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
