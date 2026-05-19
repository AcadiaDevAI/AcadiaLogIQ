# Runbook — `report_jobs` retention

The Phase 1 async-job pipeline writes a row to `report_jobs` for
every Generate request. The actual cached result (Markdown) lives
in `report_cache` and is the product — keep forever. The job row
is a workflow ledger and can be safely deleted once terminal.

## Default policy

* Sweep daily.
* Delete rows in `done` / `failed` / `cancelled` status older than
  **30 days**.
* Leave `pending` / `running` alone regardless of age — a stuck job
  is more valuable as a visible signal than as a silent gap.

## How to run

### Manual / dev

```
python -m backend.jobs.retention --days 30
# Dry-run first to see what would be deleted:
python -m backend.jobs.retention --days 30 --dry-run
```

### Production (ECS scheduled task)

Add to the Terraform stack (Phase 4 follow-up):

```hcl
resource "aws_scheduler_schedule" "job_retention" {
  name                = "${local.name_prefix}-job-retention"
  schedule_expression = "cron(15 3 * * ? *)"  # 03:15 UTC daily
  flexible_time_window { mode = "OFF" }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.worker.arn
      task_count          = 1
      launch_type         = "FARGATE"
      network_configuration {
        subnets         = var.private_subnet_ids
        security_groups = [aws_security_group.ecs_tasks.id]
      }
    }
    input = jsonencode({
      containerOverrides = [{
        name    = "worker"
        command = ["python", "-m", "backend.jobs.retention", "--days", "30"]
      }]
    })
  }
}
```

EventBridge Scheduler will call ECS `RunTask` once per day with the
overridden command. No new container image needed — the existing
worker task definition already has the code.

## Tuning the retention window

* **Shorter (7–14 days)** if storage is tight and you don't need
  the audit trail beyond a sprint.
* **Longer (90 days)** for compliance regimes that require a
  retention floor. The trade-off is index size on the
  `idx_report_jobs_by_user` and `idx_report_jobs_runnable` indexes —
  monitor `pg_relation_size('report_jobs')` and bump the RDS class
  if you cross ~80 % of disk.

## Monitoring

CloudWatch metric to watch:
```
filter logger="acadia-log-iq" and msg like /\[jobs.retention\]/
| stats max(deleted) by bin(1d)
```

Daily sweep volume should be smooth. A sudden spike or drop
typically indicates either an upstream traffic anomaly or a missed
sweep window — both worth investigating.
