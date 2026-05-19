# Observability — Saved CloudWatch Logs Insights Queries

This is a cheat-sheet of the queries that pay off most often during
incident response and routine performance work. All queries assume
the backend log group `/acadialogiq/{env}/backend` is selected in the
CloudWatch Logs Insights console.

Every query parses each log line as JSON (already the case once
`LOG_FORMAT=json` is in effect). Fields referenced are emitted by
`backend/observability/log_formatters.py` and the FastAPI access-log
middleware.

---

## 1. Pull every log line for one user click

The single most useful query during a "this happened to me" call. Paste
the `X-Request-ID` value (the user can read it off the response
headers, or you find it in the access-log line).

```
fields @timestamp, level, msg, route, status, duration_ms
| filter request_id = "a8c7d2e14b3f"
| sort @timestamp asc
| limit 200
```

> Returns the access log + every application log line emitted while
> handling that request. Bedrock invocations, DB writes, journey
> events — all visible chronologically.

---

## 2. Last 50 errors grouped by route

Quick scan for "what's broken right now?"

```
fields @timestamp, route, status, msg, request_id
| filter level = "ERROR" or status >= 500
| sort @timestamp desc
| limit 50
```

Add `| stats count(*) by route` at the end to roll up.

---

## 3. Slow `/ask` calls (p95 latency by hour)

```
fields @timestamp, duration_ms
| filter route = "/ask" and status = 200
| stats pct(duration_ms, 95) as p95 by bin(1h)
| sort @timestamp asc
```

---

## 4. Per-user request volume (last 24 h)

Useful for spotting noisy clients and "is user X actually using the
product?" questions.

```
fields @timestamp, user_id, route
| filter ispresent(user_id)
| stats count(*) as requests by user_id
| sort requests desc
| limit 50
```

---

## 5. RCA / Gap Analysis cache effectiveness

The two reports log `cached=True/False` in their summary line. This
query tells you what fraction of generations are served from the
cache — a direct measure of LLM-cost savings.

```
fields @timestamp, msg
| filter logger = "acadia-log-iq" and msg like /\[gap_analysis\]/
| parse msg /gap_chars=(?<gc>\d+) cached=(?<gcache>True|False) pm_chars=(?<pmc>\d+) cached=(?<pmcache>True|False)/
| stats count(*) as n, sum(gcache="True") as gap_hits, sum(pmcache="True") as pm_hits by bin(1h)
| sort @timestamp asc
```

---

## 6. Auth misconfiguration alarm (heuristic)

If Clerk is misconfigured, every protected route 503s with a
specific CRITICAL log line. This query surfaces the count over the
last hour; alarm on `count > 0`.

```
filter level = "CRITICAL" and msg like /CLERK AUTH IS NOT CONFIGURED/
| stats count(*) by bin(5m)
```

---

## 7. Find the slow span inside a slow request

Once query #1 has surfaced a request_id, this gives you the
intra-request latency landscape — useful when "the whole request
took 70 seconds" needs to become "the Bedrock call took 65".

```
fields @timestamp, msg, module
| filter request_id = "a8c7d2e14b3f"
| sort @timestamp asc
| display @timestamp, msg
```

Combine with `parse` to extract `out_toks=` / `out_chars=` from the
Bedrock logger lines if you want timing per LLM hop.

---

## Field reference

| Field         | Source                                     | Example                  |
| ------------- | ------------------------------------------ | ------------------------ |
| `ts`          | formatter (ISO UTC)                        | `2026-05-18T20:15:12+00:00` |
| `level`       | logging level                              | `INFO`, `WARNING`, `ERROR` |
| `logger`      | `logging.getLogger(name)`                  | `acadia-log-iq`          |
| `msg`         | rendered log message                       | `POST /ask -> 200 (812.3ms)` |
| `request_id`  | ContextVar via `RequestContextMiddleware`  | 12-char hex slug or client-supplied |
| `user_id`     | ContextVar bound after Clerk JWT verify    | `user_3Cv79B9bJvfVrTAaanmcZU7x5fF` |
| `route`       | access-log `extra`                         | `/ask`                   |
| `method`      | access-log `extra`                         | `POST`                   |
| `status`      | access-log `extra`                         | `200`                    |
| `duration_ms` | access-log `extra` (rounded to 2 dp)       | `812.34`                 |
| `module`      | logging built-in                           | `api`                    |
| `host`        | formatter (`socket.gethostname()`)         | `ip-10-0-1-5`            |
| `extra.*`     | anything passed via `extra={...}` to logger | domain-specific          |

---

## Alarms worth setting up (CloudWatch Metric Filters)

| Filter pattern (Logs Insights flavour)              | Alarm on             | What it tells you |
| --------------------------------------------------- | -------------------- | ----------------- |
| `level = "ERROR" or status >= 500`                  | `count > 10` / 5 min | Spike in failures |
| `msg like /Bedrock.*ValidationException/`           | `count > 0`          | Model misconfigured |
| `msg like /CLERK AUTH IS NOT CONFIGURED/`           | `count > 0`          | Auth secrets missing |
| `msg like /report_cache.*failed/`                   | `count > 0`          | Cache DB unreachable |
