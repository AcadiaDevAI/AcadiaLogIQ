# Acadia LogIQ — Production Readiness Summary

Plain-language checklist of every production-hardening step taken on this codebase. Use it to answer "how is this production grade?" / "what precautions are in place?" in stakeholder conversations.

Organised by concern area. Each bullet states **what** we did and **why it matters**.

---

## 1. Security & Authentication

- **Strict Clerk-only authentication.** Every protected route requires a valid Clerk JWT. The legacy "X-API-Key" fallback was removed — there is no anonymous mode.
- **Loud failure on misconfiguration.** If Clerk keys are missing, every protected route returns HTTP 503 instead of silently letting requests through. Operators see a critical log line at startup.
- **Narrow CORS allow-list.** Only the specific frontend origins can talk to the API. `Authorization` and `Content-Type` are the only allowed headers — no `X-API-Key` smuggle path.
- **PII scrubbing on error reports.** Before any error event leaves the server for Sentry, the request body, `Authorization`, `Cookie`, `X-API-Key`, and `X-Clerk-*` headers are stripped — customer ticket content never lands in a third-party SaaS dashboard.
- **Per-user rate limiting on expensive endpoints.** A single engineer (or a runaway script) cannot burn the Bedrock quota for the rest of the team. 5 Generate requests per minute per Clerk user_id.
- **IAM least-privilege.** API and worker each have their own task role. Bedrock IAM is scoped to `InvokeModel` only. Secrets Manager read is scoped to one exact ARN. No "AdministratorAccess" policies anywhere.
- **Secrets live in AWS Secrets Manager, never in git.** The backend bootstrap reads the secret blob at startup; the .env file remains only as a laptop-dev fallback.
- **Frontend bundle ships only public keys.** Clerk publishable key and Sentry DSN are both designed for browser embedding; no long-lived secret leaks through the JS bundle.
- **Job idempotency keys.** Duplicate POSTs from React StrictMode / rapid double-clicks never enqueue duplicate work — the `(kind, key)` UNIQUE constraint dedupes at the DB layer.

---

## 2. Observability — Knowing What's Happening

- **Structured JSON logs.** Every log line is a parseable JSON document with consistent fields (`ts`, `level`, `request_id`, `user_id`, `route`, `status`, `duration_ms`). Searching is one query, not a grep expedition.
- **Request-ID correlation.** Every HTTP request gets a unique ID injected into every downstream log line, including Bedrock calls and DB writes. One user click → one filter → every log line it produced.
- **User-ID correlation.** After Clerk validates the JWT, the user_id is auto-attached to all logs from that request. Filter by "show me everything user X did today."
- **CloudWatch ingestion via Docker awslogs driver.** No application-side log shipping code — Docker writes container stdout directly into a CloudWatch log group with 30-day retention.
- **Sentry backend integration.** Every unhandled exception fires a Sentry event with the full stack trace, request context, and user/release tags. Sentry deduplicates 1000 occurrences of the same error into 1 issue.
- **Sentry frontend integration.** Render-time React crashes show a friendly fallback UI and report the error to Sentry — no white-screen-of-death.
- **DSN-empty = no-op.** Sentry initialises only when a DSN is configured; absent DSN keeps the app working normally — useful during initial rollout.
- **Two health endpoints.** `/health/live` is a cheap 200-OK probe (used by ALB every 30 seconds × N containers); `/health` is the rich readiness check operators query manually.
- **Build identity in /health.** Returns `git_sha`, `build_timestamp`, `boot_time` so "is the new version actually deployed?" is one curl away.
- **Saved CloudWatch Logs Insights queries.** Pre-built queries for top-5 use cases (per-user activity, slow requests, error spikes, cache effectiveness) documented in `docs/observability/saved-queries.md`.
- **Custom CloudWatch metrics every minute.** `ReportJobsPending` + `IngestJobsPending` published by a scheduled cron — drives auto-scaling and alarms.

---

## 3. Scaling Foundation

- **Stateless API tier.** No in-process state. Any container can serve any request. Replacing the fleet is a rolling restart, not a coordination problem.
- **Gunicorn with uvicorn workers.** Production process manager with graceful reload, request-cap recycling (`--max-requests 1000`), and configurable worker count per container.
- **ECS Fargate, not raw EC2.** No AMI patching, no SSH, no EC2 maintenance. AWS manages the underlying compute.
- **Multi-AZ deployment.** API tasks run in at least two availability zones. One AZ outage doesn't take the product down.
- **Application Load Balancer with HTTPS.** TLS terminates at the ALB. HTTP requests are 301-redirected to HTTPS. Idle timeout sized for our request profile.
- **Auto-scaling on real signals.** API scales on ALB request count per task; workers scale on Postgres queue depth. CPU-based scaling is NOT used — the app is LLM-bound, CPU stays low while waiting on Bedrock.
- **Deployment circuit breaker.** ECS auto-rolls-back a failing deploy if the new tasks fail health checks. Bad code → automatic revert, no manual intervention.
- **Connection pool math fits RDS budget.** Pool sizing is env-driven; the production values (3+5 per worker × 4 workers × 2 containers = 64 connections) fit inside RDS `max_connections` with headroom.
- **One image, multiple roles via env var.** The same Docker image runs as `APP_ROLE=api`, `APP_ROLE=worker`, or any future role — no separate Dockerfiles, no separate CI builds.

---

## 4. Distributed Caching (No In-Process State)

- **Result cache for RCA + Gap Analysis** (`report_cache` table). Repeat clicks on the same incident return in milliseconds instead of re-running a 4-minute LLM call. Same cache is visible to every replica.
- **Exact-match answer cache moved to Postgres** (`answer_cache_exact`). Previously an in-memory Python dict — now multi-replica safe.
- **Semantic answer cache** (`semantic_answer_cache`). Catches paraphrased questions ("what caused X?" vs "root cause of X?") and serves the same answer.
- **Tier-1 answer cache** (`tier1_answer_cache`). Caches Tier-1 Copilot decisions cross-user.
- **All caches share the same failure-open pattern.** A DB hiccup on a cache read means "cache miss, run the real path." Caching never breaks the user request.
- **Cache invalidation on user feedback.** A 👎 on a generated report deletes the cached row so the next generation runs a fresh LLM. Stale results can't survive a single user complaint.
- **Glossary cache refreshes every 5 minutes.** New acronyms learned from uploaded documents propagate to all replicas within bounded time — no stale-on-ingest behaviour.

---

## 5. Rate Limiting

- **Per-user keying, not per-IP.** Engineers behind the same corporate NAT each get their own budget — one noisy engineer can't block the team.
- **5 Generate calls per minute per user** on RCA + Gap Analysis. Protects Bedrock quota and user experience.
- **100 standard requests per minute** on regular endpoints. Generous enough for real use, tight enough to stop abuse.
- **`SlowAPIMiddleware` properly registered.** Earlier setups silently failed because the ASGI middleware was missing — caught and fixed.
- **Limiter falls back to IP** if no user_id is bound (e.g. unauthenticated endpoints). No request escapes rate-limiting.

---

## 6. BM25 → Postgres FTS Migration

- **In-process BM25 index → Postgres-backed FTS.** The old BM25 lived in RAM in every replica — drift on ingest, memory waste, no real scaling. Replaced with Postgres full-text search.
- **Stored, indexed tsvector column.** `chunks.content_tsv` is a `GENERATED ALWAYS AS ... STORED` column with a GIN index — every query becomes a single index probe instead of a sequential scan.
- **`pg_trgm` extension** installed for trigram fallback on very short queries (≤2 tokens) — better recall on chat-style input.
- **Shadow-mode rollout.** Both BM25 and FTS run for 2 weeks side-by-side; every query's results are logged to a `retrieval_eval` table for comparison.
- **Golden-query eval harness** (`tests/retrieval/`). 100 real production queries baselined; pytest runs them on every PR touching retrieval code and fails the build if Recall@5 drops below 90 % of baseline.
- **Feature flag for cutover.** `RETRIEVAL_BM25_ENABLED=false` flips the keyword path to FTS-only without a redeploy. After 30 stable days, the BM25 module is deleted.

---

## 7. Upload, Ingestion & Workers

- **Browser uploads directly to S3.** A presigned-PUT URL is issued by the API; the file bytes never traverse the API tier. API bandwidth stays free for actual user requests.
- **Async ingestion via the worker fleet.** Previously, document parsing (PyMuPDF + Titan embedding batches) ran inside the API process via `BackgroundTasks`. A single 50 MB PDF could starve every concurrent user request. Now: API enqueues a job, returns immediately, worker container processes it.
- **Postgres-backed job queue.** Uses `SELECT FOR UPDATE SKIP LOCKED` — the same pattern Stripe, GitLab, and `pg_boss` use. Battle-tested at billions-of-jobs scale.
- **No Celery / Redis / SQS.** One transactional store (Postgres) for state and queue. Zero new infrastructure, zero new failure modes, zero new ops surface.
- **Two specialised worker fleets.** Report worker (1 GB RAM, RCA + Gap Analysis) and Ingest worker (2 GB RAM, document parsing). An OOM on a big PDF can't kill an in-flight RCA — different fleets, different blast radius.
- **`WORKER_KINDS` env var routes work.** Each worker fleet only claims the kinds it's authorised for. Adding a new job kind = update one env var.
- **Idempotent under retry.** Chunk inserts run in a single transaction; if a worker dies mid-batch, Postgres rolls back atomically. No phantom duplicates in the retrieval index.
- **Retry with exponential backoff.** 5 s → 30 s → 5 min → 30 min between attempts. Three attempts total, then permanent-fail and the row stays visible to ops.
- **Stuck-job sweeper.** A separate cron runs every minute, resets any row that's been "running" longer than its per-kind ceiling (10–45 min depending on job kind). A wedged worker can't silently block work.
- **Retention sweeper.** Terminal job rows older than 30 days are deleted daily. The actual cached output (`report_cache`) is kept indefinitely — it's the product.
- **Per-file-type memory profile documented.** PDF (PyMuPDF, ~2 GB peak), DOCX (python-docx, ~500 MB), JSON (~400 MB), log/txt/md (~150 MB). 2 GB ingest-worker budget covers all six supported types.
- **Shadow-mode rollback flag.** `INGESTION_VIA_WORKER=false` flips routes back to the legacy in-process path without a redeploy — used during staging cutover to revert in <2 minutes if needed.

---

## 8. Resilience & Recovery

- **Failure-open everywhere.** Cache errors fall back to a miss. Sentry init errors fall back to "monitoring disabled." Logging-config typos fall back to INFO. Nothing in the observability stack can take the app down.
- **DB connection ping before checkout.** `pool_pre_ping=True` catches stale connections before they hit the application.
- **Connection recycling every 30 minutes.** Prevents stale connections from accumulating on a long-running process.
- **Atomic transactions for ingestion.** Worker death mid-job = automatic rollback by Postgres; no half-written documents in the index.
- **Idempotency keys at every level.** Job queue, API routes, frontend polling — duplicates collapse to a single operation.
- **Frontend in-flight de-dupe.** React StrictMode dev-mode no longer double-fires expensive POSTs.
- **Frontend abort signal.** A user navigating away cancels the polling loop instead of holding a long XHR open.
- **RDS automated backups + Point-in-Time Recovery.** AWS-managed; configurable retention window.

---

## 9. Cost Guardrails

- **Caching is the single biggest cost lever.** Result cache for RCA + Gap Analysis means 1000 reads of the same incident = 1 Bedrock call. Steady-state hit rate ~70 % → cuts Bedrock spend by ~3×.
- **Per-user rate limit caps worst-case spend per user.** Even an automated misuse can't exceed 5 × N minutes of Bedrock cost per user.
- **Bedrock quota request runbook drafted.** Request increase + Provisioned Throughput plan documented; on-demand → provisioned crossover analysis included.
- **CloudWatch Logs retention set to 30 days.** Old logs auto-expire — no surprise storage bill.
- **Auto-scaling floors and ceilings.** API min/max 2/10 tasks; workers min/max 1/5 — bounded blast radius on a runaway scaling event.
- **EventBridge crons run in <30 s.** Per-minute cadence × Fargate task-start cost ≈ $8/mo total for both metrics and sweeper crons.

---

## 10. CI/CD & Deployment

- **Image tag = git SHA.** Every deploy is identifiable to a specific commit. Sentry release tags match. No "what's actually in prod?" confusion.
- **ECS rolling deploy with auto-rollback.** Bad deploy → deployment circuit breaker kicks in → service reverts to the previous task definition.
- **`services-stable` wait gate.** CI fails if the rollout doesn't reach steady state in 10 minutes.
- **Post-deploy smoke test.** A `curl /health/live` runs against the public ALB after every deploy.
- **Terraform IaC for everything.** VPC stays out-of-scope (avoid destructive plans), but ALB, ECS cluster, services, IAM roles, security groups, log groups, schedules — all version-controlled.
- **GitHub Actions workflow** triggers on push-to-main (dev), push-to-tag (prod), or manual dispatch (any env).
- **Shadow-mode flag for safe rollouts.** Schema changes ship first; behaviour changes ship second; observe; cut over; remove flag after 30 stable days.

---

## 11. Database Migrations Applied

7 production-grade migrations, all idempotent (`IF NOT EXISTS` everywhere), all non-destructive:

1. **`043_report_cache.sql`** — RCA + Gap Analysis result cache + feedback log.
2. **`044_answer_cache_exact.sql`** — Postgres-backed `/ask` exact-match cache.
3. **`045_chunks_fts_column.sql`** — `content_tsv` generated column + GIN index (plus `pg_trgm` extension).
4. **`046_retrieval_eval.sql`** — BM25-vs-FTS shadow log table.
5. **`047_report_jobs.sql`** — async LLM job queue (RCA + Gap Analysis workers).
6. **`048_ingestion_jobs_queue_columns.sql`** — promotes existing `ingestion_jobs` into a `SELECT FOR UPDATE SKIP LOCKED` queue.

---

## 12. AWS Services in Production Use

| Service | Role |
| --- | --- |
| **ECS Fargate** | Managed compute for api, worker, ingest_worker |
| **Application Load Balancer** | HTTPS termination, health checks, round-robin |
| **RDS Postgres** | Primary datastore (users, sessions, jobs, caches, chunks) |
| **Secrets Manager** | Backend config blob (DB URL, Bedrock keys, Clerk keys, Sentry DSN) |
| **CloudWatch Logs** | All container stdout (api + worker + ingest_worker + crons) |
| **CloudWatch Metrics** | Queue depth (`ReportJobsPending`, `IngestJobsPending`) |
| **CloudWatch Alarms** | Auto-scaling triggers + error-rate alarms |
| **EventBridge Scheduler** | Per-minute crons (metrics, stuck-job sweeper) + daily retention sweeper |
| **Bedrock** | Claude Haiku 4.5 (RCA, Gap Analysis, query rewrite) + Titan Embed V2 |
| **S3** | Document upload bucket (presigned PUT) |
| **ECR** | Backend Docker image registry |
| **IAM** | Per-service task roles, scoped Bedrock + Secrets permissions |
| **Sentry** (external) | Error monitoring + performance tracing |
| **Clerk** (external) | Authentication (JWT) |

---

## 13. Quick "Are You Production Grade?" Answers

> **"How do you handle one user blocking others?"**
> Per-user rate limit (5/min Generate). Different Fargate fleets for ingestion vs reports — an OOM on a big PDF doesn't kill an RCA in progress.

> **"What happens if a worker dies mid-job?"**
> Postgres rolls back the transaction atomically. The job row stays `running` until the stuck-job sweeper (runs every minute) resets it back to `pending`. A fresh worker claims it. No data corruption, no manual intervention.

> **"How do you know when something breaks?"**
> Sentry fires on every unhandled exception with stack trace + request_id + user_id. CloudWatch alarms trigger on 5xx rate or queue depth. Logs are structured JSON, queryable in seconds.

> **"Can you find one specific user's bad request?"**
> One Logs Insights query filtered by `user_id="user_…"` returns every log line that request produced, across api + workers.

> **"How does scaling work?"**
> Auto-scaling on real signals (request count, queue depth) — not CPU, which stays low for LLM-bound work. Floor is 2 API tasks across 2 AZs; ceiling is 10. Workers scale 1-5 each.

> **"What if AWS Bedrock throttles you?"**
> Worker catches the exception, soft-fails the job, re-queues with exponential backoff (5 s → 30 s → 5 min → 30 min). Three attempts before permanent-fail. The result cache means most repeat reads never hit Bedrock at all.

> **"How is data protected?"**
> Strict Clerk authentication (no anonymous mode); HTTPS-only public surface; secrets in AWS Secrets Manager; PII scrubbed from error reports; least-privilege IAM; multi-AZ DB with PITR backups.

> **"What's the rollback plan if a deploy goes bad?"**
> ECS deployment circuit breaker reverts automatically on health check failure. Manual rollback is `aws ecs update-service --task-definition <previous-rev>` — sub-2-minute revert. The `INGESTION_VIA_WORKER` flag adds a runtime fallback without redeploy.

> **"Cost containment?"**
> Bedrock cache cuts LLM spend ~3×. Per-user rate limits cap worst-case. Retention rules keep storage bounded. Auto-scaling floors/ceilings cap compute. CloudWatch logs auto-expire at 30 days.

---

**Bottom line:** the system is stateless API + dedicated worker fleets + Postgres-backed queues and caches + Fargate + ALB + IAM least-privilege + structured logs + Sentry + per-user rate limits + auto-rollback CI/CD + idempotent migrations — all standard production patterns, no exotic tech, no in-process state, no single points of failure that don't have a tested recovery path.
