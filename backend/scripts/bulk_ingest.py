#!/usr/bin/env python3
"""
Sprint 3-PREP-B — Bulk ingestion for SOPs, runbooks, KBs, contacts.

Usage:
  python -m backend.scripts.bulk_ingest \
    --path /data/sops/ \
    --doc-kind sop \
    --owner-id 11111111-2222-3333-4444-555555555555 \
    --batch-size 50 \
    --rate-limit-per-sec 10 \
    --resume-log /tmp/bulk_ingest.jsonl

  python -m backend.scripts.bulk_ingest \
    --s3 s3://my-bucket/kb-articles/ \
    --doc-kind kb \
    --owner-id <uuid>

Behavior:
- Walks every file under --path or --s3 prefix recursively.
- For each file, calls process_document(..., doc_kind=<flag>).
- Skips any file whose (filename, sha1) is in --resume-log.
- Token-bucket rate limit via --rate-limit-per-sec (default 10).
- Progress log every 50 files.
- Exit code 0 = all succeeded, 1 = partial failure, 2 = config error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

from backend.config import settings
from backend.services.contextual_ingestion_service import process_document

logger = logging.getLogger("bulk_ingest")


def iter_local(folder: Path) -> Iterator[Path]:
    for root, _, files in os.walk(folder):
        for fn in files:
            yield Path(root) / fn


def iter_s3(s3_uri: str) -> Iterator[Path]:
    # boto3 paginator over bucket/prefix; downloads each object to a
    # tempfile so the ingest pipeline gets a local Path. Caller is
    # responsible for cleaning up processed tempfiles after consuming.
    import tempfile
    import boto3

    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (expected s3://bucket/prefix): {s3_uri!r}")
    without_scheme = s3_uri[len("s3://"):]
    if "/" in without_scheme:
        bucket, prefix = without_scheme.split("/", 1)
    else:
        bucket, prefix = without_scheme, ""

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if key.endswith("/"):
                continue
            suffix = Path(key).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                s3.download_fileobj(bucket, key, tmp)
                tmp_path = Path(tmp.name)
            tmp_path = tmp_path.with_name(Path(key).name)
            Path(tmp.name).rename(tmp_path)
            yield tmp_path


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_resume(log_path: Optional[Path]) -> set:
    if not log_path or not log_path.exists():
        return set()
    done = set()
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok":
                fn = row.get("filename")
                sh = row.get("sha1")
                if fn and sh:
                    done.add(f"{fn}:{sh}")
    return done


def append_resume(log_path: Optional[Path], record: dict) -> None:
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bulk-ingest a folder or S3 prefix into the corpus.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", type=Path, help="Local folder to walk recursively")
    source.add_argument("--s3", type=str, help="S3 URI, e.g. s3://bucket/prefix/")
    parser.add_argument(
        "--doc-kind",
        required=True,
        choices=sorted(settings.VALID_DOC_KINDS),
        help="doc_kind tag applied to every file in this run.",
    )
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--rate-limit-per-sec", type=int, default=10)
    parser.add_argument("--resume-log", type=Path, default=None)
    args = parser.parse_args()

    if not getattr(settings, "LOGIQ_BULK_INGEST_BACKEND", False):
        logger.error("LOGIQ_BULK_INGEST_BACKEND=False — feature disabled.")
        return 2

    done = load_resume(args.resume_log)
    total = 0
    ok = 0
    failed = 0
    skipped = 0
    started = time.time()

    iter_source = iter_local(args.path) if args.path else iter_s3(args.s3)
    rate_ns = int(1e9 / max(args.rate_limit_per_sec, 1))
    last_call_ns = 0
    max_files = int(getattr(settings, "BULK_INGEST_MAX_FILES_PER_RUN", 10000))

    for path in iter_source:
        if total >= max_files:
            logger.warning(
                "Hit BULK_INGEST_MAX_FILES_PER_RUN=%d — stopping early. "
                "Re-run with --resume-log to continue.",
                max_files,
            )
            break

        total += 1
        digest = sha1_of(path)
        key = f"{path.name}:{digest}"
        if key in done:
            skipped += 1
            continue

        # Token-bucket rate limit
        now_ns = time.monotonic_ns()
        wait_ns = last_call_ns + rate_ns - now_ns
        if wait_ns > 0:
            time.sleep(wait_ns / 1e9)
        last_call_ns = time.monotonic_ns()

        try:
            result = process_document(
                local_path=path,
                filename=path.name,
                file_type=path.suffix.lower().lstrip(".") or "txt",
                owner_id=args.owner_id,
                fingerprint=digest,
                exact_duplicate_lookup=lambda _: None,
                version_candidate_lookup=lambda _: None,
                doc_kind=args.doc_kind,
            )
            if isinstance(result, dict) and result.get("status") == "rejected":
                failed += 1
                append_resume(args.resume_log, {
                    "filename": path.name, "sha1": digest,
                    "status": "rejected",
                    "reason": result.get("error_reason", "unknown"),
                })
            else:
                ok += 1
                append_resume(args.resume_log, {
                    "filename": path.name, "sha1": digest,
                    "status": "ok",
                })
        except Exception as exc:
            failed += 1
            logger.exception("Ingest failed for %s: %s", path, exc)
            append_resume(args.resume_log, {
                "filename": path.name, "sha1": digest,
                "status": "error", "reason": str(exc),
            })

        if total % 50 == 0:
            elapsed = time.time() - started
            rate = total / max(elapsed, 1e-6)
            remaining = None
            if ok + failed > 0:
                remaining = _format_eta((max_files - total) / max(rate, 1e-6))
            logger.info(
                "Processed %d files | ok=%d fail=%d skip=%d | %.1f files/sec | ETA(remaining cap) %s",
                total, ok, failed, skipped, rate, remaining or "—",
            )

    logger.info(
        "DONE | total=%d ok=%d fail=%d skip=%d in %.0fs",
        total, ok, failed, skipped, time.time() - started,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
