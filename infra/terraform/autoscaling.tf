# Auto-scaling policies.
#
# Why NOT CPU-based scaling:
#   The app is LLM-bound. Worker CPU stays low while it waits on
#   Bedrock; API CPU stays low while it polls the DB. CPU-based
#   target tracking would never trigger — by the time CPU climbed,
#   you'd be in trouble far upstream.
#
# What we scale on:
#   * api    → ALB ``ActiveConnectionCount`` per task. Proxy for
#              concurrent in-flight requests; goes up as soon as
#              more users hit Generate.
#   * worker → CloudWatch custom metric ``ReportJobsPending`` emitted
#              by the API every minute via a CW alarm + lambda or a
#              tiny scheduled task. Wire the publisher in Phase 4 —
#              for now the scaling target exists with a manual scale
#              policy.

# ─────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────

resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = 2  # always two tasks across two AZs for failover
  max_capacity       = 10
}

resource "aws_appautoscaling_policy" "api_active_connections" {
  name               = "${local.name_prefix}-api-active-conn"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    # ALB pre-defined metric: average active connections per ECS task.
    # 50 in-flight HTTP connections per task is a comfortable level
    # for the Phase-1 architecture where requests are sub-second.
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label = "${aws_lb.api.arn_suffix}/${aws_lb_target_group.api.arn_suffix}"
    }
    target_value       = 50.0
    scale_in_cooldown  = 120
    scale_out_cooldown = 60
  }
}

# ─────────────────────────────────────────────────────────────────
# Worker — placeholder scaling target. Phase 4 wires the CW metric
# publisher; until then, the worker fleet stays at desired_count.
# ─────────────────────────────────────────────────────────────────

resource "aws_appautoscaling_target" "worker" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = 1
  max_capacity       = 5
}

# Step-scaling on a custom CW metric. The metric publisher is added
# in Phase 4 (backend cron emits ``ReportJobsPending`` every minute).
# Until then, this policy is dormant — no metric, no triggers.
resource "aws_cloudwatch_metric_alarm" "worker_queue_high" {
  alarm_name          = "${local.name_prefix}-worker-queue-high"
  alarm_description   = "Triggers when pending job count exceeds 10 for 2 minutes"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  metric_name         = "ReportJobsPending"
  namespace           = "Acadia/LogIQ"
  period              = 60
  statistic           = "Average"
  threshold           = 10
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_appautoscaling_policy.worker_scale_up.arn]
  tags          = local.common_tags
}

resource "aws_appautoscaling_policy" "worker_scale_up" {
  name               = "${local.name_prefix}-worker-scale-up"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.worker.resource_id
  scalable_dimension = aws_appautoscaling_target.worker.scalable_dimension
  policy_type        = "StepScaling"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 60
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }
}

# ─────────────────────────────────────────────────────────────────
# Ingest worker — independent scaling target + policy.
#
# Uses a separate custom CW metric (``IngestJobsPending``) so a burst
# of file uploads scales out ingest workers without touching the
# report fleet. Like the report-worker policy, this is dormant until
# the Phase 4 metric publisher (the tiny cron that emits queue
# depths) is wired in production.
# ─────────────────────────────────────────────────────────────────

resource "aws_appautoscaling_target" "ingest_worker" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ingest_worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  # Ingestion is bursty — keep the floor at 1 (always one warm task)
  # and let the ceiling absorb upload spikes. 5 tasks × 1 vCPU is
  # plenty even with concurrent 100 MB PDFs.
  min_capacity = 1
  max_capacity = 5
}

resource "aws_cloudwatch_metric_alarm" "ingest_worker_queue_high" {
  alarm_name          = "${local.name_prefix}-ingest-worker-queue-high"
  alarm_description   = "Triggers when pending ingestion job count exceeds 5 for 2 minutes"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  metric_name         = "IngestJobsPending"
  namespace           = "Acadia/LogIQ"
  period              = 60
  statistic           = "Average"
  threshold           = 5
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_appautoscaling_policy.ingest_worker_scale_up.arn]
  tags          = local.common_tags
}

resource "aws_appautoscaling_policy" "ingest_worker_scale_up" {
  name               = "${local.name_prefix}-ingest-worker-scale-up"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.ingest_worker.resource_id
  scalable_dimension = aws_appautoscaling_target.ingest_worker.scalable_dimension
  policy_type        = "StepScaling"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 60
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }
}
