# Ingestion memory profile — sizing the ingest worker

The ingest-worker ECS task is currently sized at **1 vCPU / 2 GB**.
That budget was chosen for the worst-case file type (PDF via
PyMuPDF). This runbook documents the memory profile of every
supported parser so the next operator can adjust the budget if the
file mix changes.

## Supported file types

From `backend/config.py::Settings.ALLOWED_FILE_TYPES`:

| Extension | Parser | Library | Memory profile per 100 MB input | Notes |
| --- | --- | --- | --- | --- |
| `log`, `txt`, `md` | line streaming | stdlib | ~150 MB peak (3× file size for chunking + Titan batch) | Plain UTF-8; safe headroom for 4× the cap. |
| `json` | `json.loads` | stdlib | ~400 MB peak (full Python-object materialisation) | The 100 MB JSON-array case is the practical limit before we'd need a streaming parser. Most ingested JSON is <10 MB. |
| `pdf` | `process_document` → `fitz` (PyMuPDF) | `PyMuPDF>=1.24.0` | **~2 GB peak** on image-heavy / scanned PDFs at the 100 MB cap | Worst-case path. Scanned PDFs trigger pixmap materialisation; vector-only PDFs use ~3× file size. |
| `docx` | `python-docx` | `python-docx>=1.1.0` | ~500 MB peak (full XML tree in RAM) | OOXML is ~7× compression typical, so a 100 MB DOCX expands to ~700 MB in memory. |

## Why 2 GB is the right budget

The ingest worker is sized to the worst-case path (image-heavy PDF
at the 100 MB cap). Every other file type has comfortable headroom.
A 4 GB task would be more defensive but doubles the always-on cost
(~$36/mo → ~$72/mo) for headroom we can't measure ourselves needing
on the current corpus.

If you change `MAX_FILE_SIZE_MB` upward or accept new file types,
re-validate the memory ceiling:

```bash
# Quick check: ingest a worst-case file and measure container memory
docker stats acadialogiq-ingest-worker --no-stream
```

## What happens on OOM

ECS Fargate kills the container, which:

1. Postgres drops the in-flight DB transaction → the chunks/embeddings
   table state stays clean (no phantom partial inserts — see Phase 5
   migration design).
2. The job row stays in `running` until the stuck-job sweeper resets
   it back to `pending` with backoff. The sweeper runs every minute
   via EventBridge Scheduler (Terraform `scheduler_metrics.tf` pattern).
3. CloudWatch task-stopped event fires → Sentry alarm (when the alarm
   rule is wired in Phase 4 follow-up).
4. The next worker claim re-runs the job from scratch — idempotent
   because the chunk insert is atomic.

## Tuning levers

| Symptom | Lever |
| --- | --- |
| Frequent OOM on PDFs > 50 MB | Bump `ingest_worker_memory` to 4096 |
| Frequent OOM on DOCX > 80 MB | Same |
| Backlog builds during business hours | Bump `ingest_worker_desired_count` floor + raise `aws_appautoscaling_target.ingest_worker.max_capacity` |
| New file type added (e.g. `xlsx`) | Add to `ALLOWED_FILE_TYPES`, profile its parser, update this table |

## Not currently supported

* `xlsx` — `openpyxl` is in `requirements.txt` but `xlsx` is NOT in
  `ALLOWED_FILE_TYPES`. The XLSX path is reachable only via the
  legacy `etl_excel_contacts.py` ETL script. Adding `xlsx` to the
  uploads whitelist would require profiling `openpyxl` against
  a large workbook — single-sheet 100 MB workbooks can briefly
  use 2–4 GB. Don't enable without measurement.
* Streaming variants of PDF / DOCX (e.g. PyMuPDF's `tools.set_low_memory`
  flag) — not in scope today. Revisit if budget headroom becomes a
  recurring issue.
