# EventBridge Scheduler — stuck-job sweeper.
#
# Every minute, launches a one-shot ECS Fargate task that scans both
# queue tables (report_jobs, ingestion_jobs) for rows stuck in
# ``status='running'`` past their per-kind runtime ceiling and resets
# them via the existing mark_failed paths. See
# ``backend/jobs/sweeper.py`` for the runtime ceilings + soft-vs-permanent
# fail logic.
#
# Reuses
# ------
# * ``aws_iam_role.scheduler``                    (scheduler_metrics.tf)
# * ``aws_iam_role_policy.scheduler_ecs_runtask`` (scheduler_metrics.tf)
# * ``aws_ecs_task_definition.worker``            (ecs_worker.tf)
# * ``aws_ecs_cluster.main``                      (ecs_cluster.tf)
# * ``aws_security_group.ecs_tasks``              (security_groups.tf)
#
# Only the schedule + command override are new. Same image, same VPC
# wiring, same IAM — minimal blast radius for the addition.

resource "aws_scheduler_schedule" "stuck_job_sweeper" {
  name                = "${local.name_prefix}-stuck-job-sweeper"
  group_name          = "default"
  schedule_expression = "rate(1 minute)"

  # Strict timing — a stuck-job watchdog has to fire on a predictable
  # cadence. A flexible window would mean we could miss the moment a
  # truly-stuck row starts blocking auto-scaler signals.
  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.worker.arn
      task_count          = 1
      launch_type         = "FARGATE"
      platform_version    = "LATEST"

      network_configuration {
        subnets          = var.private_subnet_ids
        security_groups  = [aws_security_group.ecs_tasks.id]
        assign_public_ip = false
      }
    }

    # Container override — same image, different entrypoint. The
    # ``--dry-run`` flag is intentionally NOT set; in production we
    # want the sweeper to actually mark stuck rows failed.
    input = jsonencode({
      containerOverrides = [{
        name    = "worker"
        command = [
          "python", "-m", "backend.jobs.sweeper",
        ]
      }]
    })

    # If the task launch fails (cluster full, IAM blip), retry once.
    # Past 60 s the invocation drops — better to miss one tick than
    # build a queue of duplicate sweeps.
    retry_policy {
      maximum_event_age_in_seconds = 60
      maximum_retry_attempts       = 1
    }
  }
}
