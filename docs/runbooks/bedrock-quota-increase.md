# Runbook — Request Bedrock service-quota increase

Why: AWS Bedrock applies **account-level** RPM / TPM quotas per model.
Horizontal API scaling does **not** multiply this capacity — every
container in the cluster shares the same quota pool. The default
limits will throttle Gap Analysis / RCA generations once we exceed
~10 concurrent users, well before infrastructure scaling becomes the
bottleneck.

Action: file a service-quota increase **before** Phase 1 ships to
production. AWS turnaround is typically 1–2 weeks and on-demand
inference quota increases are reviewed by a human.

---

## 1. Where to file

AWS Console → **Service Quotas** → **AWS services** → **Amazon Bedrock**

Or the direct CLI form (faster for repeat increases):
```
aws service-quotas request-service-quota-increase \
  --service-code bedrock \
  --quota-code <code> \
  --desired-value <number> \
  --region us-east-1
```

## 2. Quotas to increase

For our primary model — Claude Haiku 4.5 on the on-demand inference
profile `us.anthropic.claude-haiku-4-5-20251001-v1:0`:

| Quota | Default | Request to | Why |
| --- | --- | --- | --- |
| Cross-region model invocations per minute for Claude Haiku 4.5 | 200 | **1000** | Worst-case fanout: 100 concurrent users × 2 panels each (Gap Analysis + Post-Mortem) = 200 calls/min sustained, with headroom for spikes |
| Cross-region model invocation tokens per minute for Claude Haiku 4.5 | 400 K | **2 M** | Each Gap Analysis prompt is ~14 K input + 16 K output tokens. 200 calls/min × 30 K = 6 M, so 2 M is intentionally tight — revisit after Phase 5 |
| Cross-region model invocations per minute for Titan Embed V2 (embeddings) | 2000 | **5000** | Embedding load scales linearly with ingestion + semantic-cache lookups |

> **Verify quota codes before submission** — AWS occasionally renames
> them. Browse the Service Quotas console for current code strings;
> the table above is a starting point, not a contract.

## 3. Justification text (paste into the ticket)

```
We are a SaaS platform (Acadia LogIQ) building incident-resolution
AI for IT operations teams. Current production traffic is light
(<10 concurrent users), but we are about to roll out an async
job pipeline (SQS + Fargate workers) that will let larger customers
queue up reports. Expected steady-state load over the next 90 days:

  * 50–200 concurrent users
  * 2 long-form report generations per user per session
  * Average input/output token sizes: 14K / 16K (Gap Analysis),
    9K / 5K (Post-Mortem), 4K / 3K (RCA panels)

The current quota of 200 RPM throttles us at ~5 concurrent users.
We have already implemented:
  * Per-user rate limiting (5 generations / min / user) via slowapi
  * Postgres-backed result cache — repeat reads of the same report
    do NOT hit Bedrock
  * Per-panel cache invalidation only on user-initiated regenerate

We expect ~70% cache hit rate at steady state, so the requested
1000 RPM gives ~3000 effective user-visible RPM with headroom for
ingestion-time embedding traffic.
```

## 4. After the increase lands

Once AWS confirms the new quotas:

1. Add CloudWatch alarms at 80% of each new ceiling (we don't want
   to discover we've outgrown the new limit by hitting it).
2. Update `_MAX_TOKENS_*` constants in
   `backend/tier1_copilot/gap_analysis/routes.py` only if we want
   to widen output budgets in tandem (currently we're tight for
   latency, not for quota).
3. Provisioned Throughput (Phase 5) becomes worth pricing when sustained
   load exceeds ~70% of the increased on-demand ceiling — at that point
   commit pricing typically beats pay-as-you-go.

## 5. Tracking

| Quota code | Filed on | AWS ticket # | Approved value | Date approved |
| --- | --- | --- | --- | --- |
| _Haiku 4.5 RPM_ | | | | |
| _Haiku 4.5 TPM_ | | | | |
| _Titan Embed V2 RPM_ | | | | |

Keep this table updated so the next operator can tell "have we already
asked for more?" at a glance.
