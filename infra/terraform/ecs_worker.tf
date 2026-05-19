# ECS worker service — async LLM job consumer.
#
# No ALB binding, no public port. Scales on the depth of the
# Postgres ``report_jobs`` queue (see autoscaling.tf — custom CW
# metric emitted by the API every minute).

resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name_prefix}-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_worker.arn

  container_definitions = jsonencode([{
    name      = "worker"
    image     = local.image
    essential = true

    environment = [
      { name = "APP_ROLE",             value = "worker" },
      # Pin the kinds set explicitly so a future code change that
      # adds a new ingestion-like kind can't silently drain into
      # this 1 GB report worker (which would OOM on a big PDF).
      # Adding a new report kind = update this list AND
      # ``local.report_worker_kinds``.
      { name = "WORKER_KINDS",         value = local.report_worker_kinds },
      { name = "APP_ENV",              value = var.env },
      { name = "AWS_REGION",           value = var.aws_region },
      # Workers hold a DB connection briefly per job — small pool
      # is fine and conserves the RDS budget for the API tier.
      { name = "DB_POOL_SIZE",         value = "2" },
      { name = "DB_MAX_OVERFLOW",      value = "2" },
      # Worker doesn't serve retrieval queries — no need for the
      # 5-minute glossary refresher loop.
      { name = "GLOSSARY_REFRESH_SECONDS", value = "0" },
      { name = "SENTRY_RELEASE",       value = var.image_tag },
      { name = "LOG_FORMAT",           value = "json" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "worker"
      }
    }
  }])

  tags = local.common_tags
}

resource "aws_ecs_service" "worker" {
  name            = "${local.name_prefix}-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_desired_count

  launch_type    = "FARGATE"
  propagate_tags = "TASK_DEFINITION"

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = local.common_tags
}
