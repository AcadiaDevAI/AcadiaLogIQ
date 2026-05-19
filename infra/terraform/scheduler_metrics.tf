# EventBridge Scheduler — emit queue-depth metrics every minute.
#
# Triggers a one-shot ECS Fargate task on the existing worker task
# definition, but overrides the container command to run
# ``python -m backend.jobs.metrics_publisher`` instead of the long-
# running worker loop. The same image, same code, same VPC + secrets
# wiring — only the command differs.
#
# Why EventBridge Scheduler (and not a sidecar cron inside the worker):
#   * The metric pipeline must NOT depend on the very service it
#     scales. If the worker fleet is wedged (queue full + tasks
#     stuck), a sidecar inside it would publish nothing — exactly
#     the time the auto-scaler needs the signal most.
#   * EventBridge Scheduler is AWS-managed, GA, and free at this
#     cadence (1440 invocations/day per schedule).
#   * Same primitive we'll use for the Phase-4 retention sweeper —
#     one shared pattern.
#
# Cost: ~$0 (free tier covers 14M invocations/month) + ~$3-5/mo for
# the per-minute Fargate task starts.


# ─────────────────────────────────────────────────────────────────
# IAM — EventBridge Scheduler assumes this role to call ECS RunTask
# and pass the existing task + execution roles to the launched task.
# ─────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${local.name_prefix}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
  tags               = local.common_tags
}

# Permissions the scheduler needs to launch an ECS task on our
# worker task definition. Scoped tightly to:
#   * ecs:RunTask on the worker family ARNs only (no wildcards
#     across the cluster — prevents a misconfigured schedule from
#     launching arbitrary task defs).
#   * iam:PassRole on exactly the two roles the worker uses.
resource "aws_iam_role_policy" "scheduler_ecs_runtask" {
  name = "${local.name_prefix}-scheduler-runtask"
  role = aws_iam_role.scheduler.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RunTaskOnWorkerFamilies"
        Effect = "Allow"
        Action = ["ecs:RunTask"]
        # Family wildcards cover the implicit revision suffix
        # (``-worker:*`` matches every revision under the family).
        Resource = [
          "${aws_ecs_task_definition.worker.arn}:*",
          aws_ecs_task_definition.worker.arn,
        ]
        Condition = {
          ArnEquals = {
            "ecs:cluster" = aws_ecs_cluster.main.arn
          }
        }
      },
      {
        Sid    = "PassWorkerRoles"
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.ecs_task_worker.arn,
        ]
      },
    ]
  })
}


# ─────────────────────────────────────────────────────────────────
# Schedule — invoke the metrics publisher every minute.
# ─────────────────────────────────────────────────────────────────

resource "aws_scheduler_schedule" "metrics_publisher" {
  name                = "${local.name_prefix}-metrics-publisher"
  group_name          = "default"
  # ``rate(1 minute)`` is the tightest cadence EventBridge Scheduler
  # accepts. Auto-scaling targets re-evaluate every minute, so a
  # tighter cadence would just burn Fargate task starts for no gain.
  schedule_expression = "rate(1 minute)"

  # No flexible window — we want metrics on time, not "sometime in
  # the next 15 minutes". The ``OFF`` mode means EventBridge fires
  # exactly when expected.
  flexible_time_window {
    mode = "OFF"
  }

  target {
    # Target the cluster ARN; the ecs_parameters block below tells
    # the scheduler which task definition to launch on it.
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.worker.arn
      task_count          = 1
      launch_type         = "FARGATE"
      # Latest platform version — needed for the awsvpc network
      # mode our task definitions use.
      platform_version = "LATEST"

      network_configuration {
        subnets          = var.private_subnet_ids
        security_groups  = [aws_security_group.ecs_tasks.id]
        assign_public_ip = false
      }
    }

    # The Input field on a Scheduler ECS target carries a JSON
    # blob with overrides for the launched task. Here we override
    # the container command to ``python -m backend.jobs.metrics_publisher``
    # — same image, same env vars, different entrypoint.
    input = jsonencode({
      containerOverrides = [{
        name    = "worker"
        command = [
          "python", "-m", "backend.jobs.metrics_publisher",
        ]
      }]
    })

    # If the task launch fails (cluster full, throttled), retry with
    # a tight cap so we don't fall behind. Past max_event_age the
    # scheduler drops the invocation — we'd rather miss one data
    # point than queue up a backlog.
    retry_policy {
      maximum_event_age_in_seconds = 60
      maximum_retry_attempts       = 1
    }
  }
}
