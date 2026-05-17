"""Run pytest in chunks and write a summary file.

The CI sandbox here can't keep stdout open through the full pytest
unconfigure sequence, so we shell out per test directory and rely on
each subprocess's own stdio. Output is captured to results.txt.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

CHUNKS = [
    # path, label
    ("backend/tests/test_phase1_logic.py", "phase1_logic"),
    ("backend/tests/test_phase1_schema.py", "phase1_schema"),
    ("backend/tests/test_phase2_parser.py", "phase2_parser"),
    ("backend/tests/test_phase2_pipeline.py", "phase2_pipeline"),
    ("backend/tests/test_phase2_versioning.py", "phase2_versioning"),
    ("backend/tests/test_retrieval_Phase3.py", "phase3_retrieval"),
    ("backend/tests/test_agents.py", "agents"),
    ("backend/tests/test_validation.py", "validation"),
    ("backend/tests/test_sprint4_fingerprint.py", "sprint4_fingerprint"),
    ("backend/tests/test_sprint5_expert_template.py", "sprint5_expert_template"),
    ("backend/tests/services", "tests_services"),
    ("backend/tests/scripts", "tests_scripts"),
    ("backend/tests/tier1_copilot/journey", "journey_tests"),
    ("backend/tier1_copilot/tests", "tier1_unit_tests"),
    ("backend/tier1_copilot/intake/tests", "intake_tests"),
]

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "test_results.txt"


def run(target: str) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-p", "no:cacheprovider",
            "--tb=short",
            "--no-header",
            "-q",
            "--color=no",
            target,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out


def main() -> int:
    summary: list[str] = []
    overall = 0
    for target, label in CHUNKS:
        if not (ROOT / target).exists():
            line = f"SKIP   {label:30s} (path not found: {target})"
            print(line)
            summary.append(line)
            continue
        code, out = run(target)
        tail = "\n".join(out.splitlines()[-6:])
        if code == 0:
            line = f"PASS   {label:30s} :: {tail.splitlines()[-1] if tail else '(no output)'}"
        else:
            overall = code
            line = f"FAIL   {label:30s} :: {tail.splitlines()[-1] if tail else '(no output)'}"
        print(line)
        summary.append(line)
        # Append full chunk output to results.txt for inspection.
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(f"\n===== {label} ({target}) exit={code} =====\n")
            fh.write(out)
            fh.write("\n")
    print("\n----- SUMMARY -----")
    for line in summary:
        print(line)
    return overall


if __name__ == "__main__":
    RESULTS.write_text("", encoding="utf-8")
    sys.exit(main())
