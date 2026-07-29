# Scaling AcadiaLogIQ to 100 Concurrent Users — Target Numbers

Concrete capacity plan to serve **100 concurrent users** on the RAG/LLM stack
(FastAPI + Postgres/pgvector + AWS Bedrock). Numbers are derived from a stated
load model — re-run the math if your traffic pattern differs.

> These are **engineering targets**, not the current config. See
> "Current vs Target" for the delta. Validate with the load test in the last
> section before trusting them.

---

## 1. Load model (assumptions)

| Parameter | Value | Notes |
|---|---|---|
| Concurrent users | 100 | logged-in and active |
| Query cadence (steady) | 1 `/ask` per user / 30s | → **3.3 `/ask`/s** average |
| Query cadence (peak burst) | 1 per user / 10s | → **10 `/ask`/s** burst |
| Avg `/ask` latency | ~5 s | blended: cache hits ~1s, generations ~8–20s |
| Cache hit rate (semantic+answer) | ~40% | repeat/near-repeat queries served <1s |
| Avg tokens per generation | ~15k (in+out) | prompt is RAG-context-heavy |

**Concurrency math (Little's Law):** in-flight = arrival × latency.
- Steady: `3.3/s × 5s ≈ 17` concurrent heavy requests.
- Peak burst: `10/s × 5s ≈ 50` concurrent heavy requests.

**Design point: handle ~50 concurrent in-flight `/ask` with headroom** (plus
lighter traffic: `/status`, `/chat/sessions`, `/files`, presign, etc.).

---

## 2. Target architecture

```
            ┌────────────┐
  users ──▶ │    ALB     │ (HTTPS, sticky not required — stateless API)
            └─────┬──────┘
                  │
        ┌─────────┼─────────┬─────────┐
     ┌──▼──┐  ┌──▼──┐   ┌──▼──┐  ┌──▼──┐      4× API replicas
     │ api │  │ api │   │ api │  │ api │      (ECS Fargate or EC2 ASG)
     └──┬──┘  └──┬──┘   └──┬──┘  └──┬──┘
        └────────┴────┬────┴────────┘
                 ┌────▼─────┐
                 │ PgBouncer │  (transaction pooling)
                 └────┬─────┘
                 ┌────▼─────┐          ┌──────────┐
                 │   RDS     │          │ Bedrock  │◀── all api + workers
                 │ Postgres  │          │ Haiku/Titan
                 └───────────┘          └──────────┘

  workers (separate ECS services / containers, scale independently):
   • report-worker  ×2   (RCA / Gap)
   • ingest-worker  ×2   (PDF/JSON parse + embed)
```

Key change vs today: **stop running everything on one EC2 t3.medium**; run the
API as **N stateless replicas behind an ALB**, put **PgBouncer** in front of
RDS so replicas don't exhaust Postgres connections.

---

## 3. Current vs Target (the delta)

| Component | Current | Target (100 users) |
|---|---|---|
| **API compute** | 1× t3.medium, 1 container | **4× replicas**, 2 vCPU / 4 GB each (Fargate) or 2× c6i.xlarge in an ASG |
| **Gunicorn workers / replica** | 2 | **4** (`(2×vCPU)+1` ≈ 5 cap; 4 leaves memory headroom) |
| **Total API worker procs** | 2 | **16** (4 replicas × 4) |
| **Gunicorn timeout** | 120s | **120s** (keep — long generations) |
| **DB access** | direct to RDS | **PgBouncer** (transaction mode) in front |
| **DB pool / worker** | 3 + 5 overflow | **5 + 5** (→ to PgBouncer, not RDS) |
| **RDS instance** | small (tight budget) | **db.r6g.large** (2 vCPU/16 GB) — bump to xlarge if pgvector CPU-bound |
| **RDS max_connections** | tight | **≥ 300** (PgBouncer→PG capped at ~100; app→PgBouncer can be high) |
| **Report worker** | ×1 | **×2** |
| **Ingest worker** | ×1 | **×2** |
| **Bedrock Haiku** | account default | quota **≥ 800 RPM / ≥ 4M TPM** (see §5) |
| **Bedrock Titan embed** | account default | quota **≥ 1000 RPM** |
| **Caches** | on | on (keep warm) |

---

## 4. Connection budget (why PgBouncer)

Without pooling, connections multiply dangerously:

```
16 worker procs × (pool 5 + overflow 5) = 160 app connections
+ report workers   2 × (5+5)  = 20
+ ingest workers   2 × (5+5)  = 20
= ~200 direct Postgres connections   ← risky on a mid RDS class
```

**With PgBouncer (transaction pooling):**
- App side: 200 "connections" are cheap (multiplexed).
- PgBouncer → Postgres: cap `default_pool_size` so real PG backends stay
  **~80–100** total → comfortable on db.r6g.large.
- RDS `max_connections`: set **300** to leave admin/headroom.

PgBouncer config starting point:
```ini
pool_mode = transaction
max_client_conn = 500
default_pool_size = 20        # per (user,db) pair
reserve_pool_size = 5
```

---

## 5. Bedrock quotas (often the real ceiling)

Peak generation rate ≈ **10 `/ask`/s** minus cache hits ≈ **6 gens/s**.

| Model | Metric | Need (with headroom) | Action |
|---|---|---|---|
| Claude Haiku | Requests/min | **~800 RPM** (6/s ≈ 360, ×2 headroom) | Request quota increase |
| Claude Haiku | Tokens/min | **~4M TPM** (6/s × 15k × 60 ≈ 5.4M → round up) | Request quota increase |
| Titan Embed v2 | Requests/min | **~1000 RPM** (retrieval + ingest) | Request quota increase |

> Default on-demand Bedrock quotas are frequently **below** these. File the
> Service Quotas increase **early** (lead time), or use **provisioned
> throughput** if latency/SLA must be guaranteed. This is the single most
> common thing that silently caps concurrency.

---

## 6. Concrete env/config changes

**API replica (ECS task def / compose `environment`):**
```
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=120
GUNICORN_KEEPALIVE=75
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=5
# point the app at PgBouncer, not RDS directly:
DATABASE_URL=postgresql://<user>:<pw>@<pgbouncer-host>:6432/<db>
```

**Autoscaling (Fargate target tracking):**
- Scale API on **ALB RequestCountPerTarget** (e.g. target 30) or **CPU 60%**.
- Min 4 tasks, max 8. Scale-out cooldown 60s, scale-in 300s.

**Workers:** run 2 report + 2 ingest; scale ingest on `report_jobs` pending
count if bulk uploads spike.

---

## 7. Caching & rate limiting

- Keep **semantic cache + answer cache** enabled — at ~40% hit rate they cut
  generation load (and Bedrock spend) by roughly a third. Warm them after deploy.
- Review the **slowapi** limiter: `/escalation/ask` is `60/min`, `/upload`
  `10/min`. For 100 users these are per-client, so fine — but confirm no
  global limiter throttles legitimate burst traffic.

---

## 8. Scale-out gotchas

1. **In-process indexes:** BM25 + glossary are rebuilt from Postgres **per
   process at startup** and held in memory. With 16 worker procs each holds a
   copy → watch **per-replica memory** and **startup time** (each boot scans
   the corpus). At the current corpus (~6k chunks) this is small, but it grows
   with data. Consider a shared/warm index or lazy load if replicas balloon.
2. **Statelessness:** the API is otherwise stateless (sessions in Postgres,
   files in S3) — safe to run behind an ALB with no sticky sessions.
3. **Uploads:** confirm **S3** storage is configured (`STORAGE_TYPE=s3` +
   `S3_UPLOAD_BUCKET` in ASM) — local-disk uploads do **not** work across
   replicas.
4. **ASM read + Bedrock/S3 IAM:** every replica's task role needs
   `secretsmanager:GetSecretValue`, Bedrock invoke, and S3 rw on the bucket.

---

## 9. Validation (do this before trusting the numbers)

Load test (k6 / Locust) against a staging stack sized as above:
- **Ramp:** 0 → 100 VUs over 5 min, hold 15 min, each VU one `/ask` per 30s.
- **Burst test:** 100 VUs firing within 10s.
- **Pass criteria:** p95 `/ask` < 12s, error rate < 1%, no DB connection
  errors, Bedrock throttles = 0.
- **Watch:** ALB target response time, ECS CPU/mem, RDS connections + CPU,
  PgBouncer `SHOW POOLS`, Bedrock `ThrottlingException` count in CloudWatch.

Tune from the first bottleneck that appears (usually Bedrock quota, then DB, then workers).

---

## 10. Rollout order (lowest risk first)

1. Request **Bedrock quota** increases (longest lead time).
2. Stand up **RDS** (right class) + **PgBouncer**; point app at PgBouncer.
3. Containerize API as a **stateless service** behind an **ALB**; set replica
   count = 4, workers = 4.
4. Enable **autoscaling**.
5. Split **workers** to 2+2.
6. **Load test**, then right-size (replicas / RDS / quotas) from real numbers.

---

### TL;DR target
**4 API replicas × 4 workers behind an ALB → PgBouncer → db.r6g.large**, report+ingest workers ×2 each, **Bedrock Haiku ≥800 RPM / ≥4M TPM**, caches on. That comfortably serves 100 users at ~1 query/30s with burst headroom; the first thing to hit a wall will be Bedrock quota, so file that increase first.



I'll write a concrete scaling target doc to docs/SCALING_100_USERS.md.

● Write(docs\SCALING_100_USERS.md)
Wrote 204 lines to docs\SCALING_100_USERS.md
 Scaling AcadiaLogIQ to 100 Concurrent Users — Target Numbers

oncrete capacity plan to serve **100 concurrent users** on the RAG/LLM s
FastAPI + Postgres/pgvector + AWS Bedrock). Numbers are derived from a s
ated
oad model — re-run the math if your traffic pattern differs.

 These are **engineering targets**, not the current config. See
st
 section before trusting them.
… +220 lines

Created docs/SCALING_100_USERS.md with concrete target numbers. Highlights:

Load model — 100 users × 1 query/30s = ~3.3 /ask/s steady (10/s burst). By Little's Law that's ~17 concurrent generations steady, ~50 at peak → design point.

Target config (vs current):

┌──────────────────────┬──────────────┬───────────────────────────────────────────┐
│      Component       │   Current    │                  Target                   │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ API compute          │ 1× t3.medium │ 4 replicas (2 vCPU/4 GB, Fargate) behind  │
│                      │              │ an ALB                                    │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ Workers/replica      │ 2            │ 4 → 16 total                              │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ DB access            │ direct to    │ PgBouncer (transaction pooling)           │
│                      │ RDS          │                                           │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ DB pool/worker       │ 3+5          │ 5+5 (to PgBouncer)                        │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ RDS                  │ small        │ db.r6g.large, max_connections ≥ 300       │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ Report/ingest        │ ×1 each      │ ×2 each                                   │
│ workers              │              │                                           │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ Bedrock Haiku        │ default      │ ≥800 RPM / ≥4M TPM                        │
├──────────────────────┼──────────────┼───────────────────────────────────────────┤
│ Titan embed          │ default      │ ≥1000 RPM                                 │
└──────────────────────┴──────────────┴───────────────────────────────────────────┘

Key insights called out:
- Connection math — without pooling you'd hit ~200 direct Postgres conns; PgBouncer caps real PG backends to ~80–100.
- Bedrock quota is usually the real ceiling — file the increase first (longest lead time).
- Scale-out gotcha — BM25 + glossary are rebuilt in-memory per worker process; each of the 16 procs holds a copy (fine now, watch as corpus grows).
- Rollout order + a k6/Locust validation plan with pass criteria (p95 /ask < 12s, error < 1%, zero Bedrock throttles).

The doc is explicit that these are targets derived from the stated assumptions — the file includes the math so you can re-derive if your real traffic pattern differs. Want me to also generate the actual k6 load-test script or a PgBouncer/ECS task-definition snippet?

✻ Brewed for 1m 55s

❯ What would be the total cost per month and per user would be including llm, current services using aws? Could you analysis. What is it now?  What would be if we increase the users to 100? please answer

Here's a cost analysis. Important caveat up front: these are estimates in us-east-1 on-demand pricing, and the numbers hinge on query volume (which I'm assuming) — the LLM/Bedrock portion is the dominant variable cost and swings the total the most. The infra numbers are grounded in your actual docker-compose.ec2.yml + the scaling doc; the per-query LLM cost is grounded in your real [token_usage] log lines.

1. LLM cost per /ask (from your actual logs)

A single full /ask fires several Haiku calls. Summing one real cache-miss request from your logs (request_id=802c49ea9045):

┌──────────────────────────────────────────────────┬──────────┐
│                      Stage                       │   Cost   │
├──────────────────────────────────────────────────┼──────────┤
│ triage                                           │ $0.0008  │
├──────────────────────────────────────────────────┼──────────┤
│ query_rewrite                                    │ $0.0026  │
├──────────────────────────────────────────────────┼──────────┤
│ aggregation                                      │ $0.0039  │
├──────────────────────────────────────────────────┼──────────┤
│ router (main generation)                         │ $0.0181  │
├──────────────────────────────────────────────────┼──────────┤
│ relevancy                                        │ $0.0008  │
├──────────────────────────────────────────────────┼──────────┤
│ embeddings (Titan)                               │ ~$0.0000 │
├──────────────────────────────────────────────────┼──────────┤
│ Full cache-miss /ask                             │ ~$0.026  │
├──────────────────────────────────────────────────┼──────────┤
│ Cache-hit /ask (triage+rewrite+agg, then served) │ ~$0.007  │
└──────────────────────────────────────────────────┴──────────┘

Blended at ~40% cache hit rate: ≈ $0.018 per /ask.

2. Current (pilot) — estimated

Assumes: single t3.medium (api+workers+ui) + RDS, and ~15 active users × ~20 queries/day × 22 days ≈ 6,600 /ask/mo.

┌──────────────────────────────────────┬─────────────────────┐
│                 Item                 │        ~$/mo        │
├──────────────────────────────────────┼─────────────────────┤
│ EC2 t3.medium                        │ $30                 │
├──────────────────────────────────────┼─────────────────────┤
│ RDS Postgres (small, e.g. t3.medium) │ $50–60              │
├──────────────────────────────────────┼─────────────────────┤
│ S3 + CloudWatch + Secrets Mgr        │ $15–25              │
├──────────────────────────────────────┼─────────────────────┤
│ Data transfer                        │ $5–15               │
├──────────────────────────────────────┼─────────────────────┤
│ Clerk (auth, non-AWS)                │ $0–25               │
├──────────────────────────────────────┼─────────────────────┤
│ Fixed subtotal                       │ ~$110–150           │
├──────────────────────────────────────┼─────────────────────┤
│ Bedrock LLM (6,600 × $0.018)         │ ~$120               │
├──────────────────────────────────────┼─────────────────────┤
│ TOTAL                                │ ~$230–270/mo        │
├──────────────────────────────────────┼─────────────────────┤
│ Per user (÷15)                       │ ~$15–18 / user / mo │
└──────────────────────────────────────┴─────────────────────┘

3. At 100 users — estimated

Infra from docs/SCALING_100_USERS.md (4 API replicas + PgBouncer + db.r6g.large + 2+2 workers + ALB). Volume: 100 users × ~30 queries/day × 22 days ≈ 66,000 /ask/mo.

┌──────────────────────────────────────┬──────────────────┐
│                 Item                 │      ~$/mo       │
├──────────────────────────────────────┼──────────────────┤
│ API: 4× Fargate (2 vCPU/4 GB)        │ $290             │
├──────────────────────────────────────┼──────────────────┤
│ Workers: 2 report + 2 ingest         │ $120–160         │
├──────────────────────────────────────┼──────────────────┤
│ ALB                                  │ $25              │
├──────────────────────────────────────┼──────────────────┤
│ RDS db.r6g.large (single-AZ)         │ $205             │
├──────────────────────────────────────┼──────────────────┤
│ PgBouncer (small instance/task)      │ $15–30           │
├──────────────────────────────────────┼──────────────────┤
│ S3 + CloudWatch + Secrets + transfer │ $70–120          │
├──────────────────────────────────────┼──────────────────┤
│ Clerk                                │ ~$25             │
├──────────────────────────────────────┼──────────────────┤
│ Fixed subtotal                       │ ~$750–860        │
├──────────────────────────────────────┼──────────────────┤
│ Bedrock LLM (66,000 × $0.018)        │ ~$1,190          │
├──────────────────────────────────────┼──────────────────┤
│ TOTAL                                │ ~$1,950–2,050/mo │
├──────────────────────────────────────┼──────────────────┤
│ Per user (÷100)                      │ ~$20 / user / mo │
└──────────────────────────────────────┴──────────────────┘

(Multi-AZ RDS for HA would add ~$205; heavier usage at 50 queries/user/day pushes LLM to ~$2,000 and total to ~$2,800 → ~$28/user.)

4. Takeaways

- LLM dominates variable cost. At 100 users, Bedrock is ~60% of the bill and scales linearly with query volume; infra amortizes (economies of scale) but LLM does not.
- Per-user cost is roughly flat (~$15 → ~$20) — infra-per-user actually rises slightly (you over-provision for burst), while LLM-per-user depends purely on how much each user queries.
- Biggest cost levers (in order):
  a. Cache hit rate — every +10% cache hit ≈ −8% LLM cost. Already have semantic + answer caches; keep them warm.
  b. The main generation call ($0.018 of the $0.026) — trim RAG context size (14k input tokens is the driver), or route simple queries to cheaper/smaller handling.
  c. Right-size infra — single-AZ + EC2 (vs Fargate) can shave ~$150–200/mo if HA isn't required.
- Not modeled: Bedrock provisioned throughput (if you buy it for SLA, it's a fixed monthly commit that changes the math), and one-time ingest embedding costs (small unless bulk-loading large corpora).

Want me to save this as docs/COST_ANALYSIS.md, or re-run it with your actual numbers (real monthly query count, RDS class, expected queries/user, whether you need Multi-AZ)?
