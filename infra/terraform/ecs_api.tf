# ECS API service — public-facing FastAPI behind the ALB.
#
# Task definition + service. The service is bound to the ALB target
# group; new task revisions roll one container at a time with a
# circuit breaker that auto-rolls-back on health-check failure.

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task_api.arn

  container_definitions = jsonencode([{
    name      = "api"
    image     = local.image
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    # ``APP_ROLE`` is the switch the entrypoint script reads (see
    # backend/entrypoint.sh). Per-service so identical image, different
    # role per task definition.
    environment = [
      { name = "APP_ROLE",            value = "api" },
      { name = "APP_ENV",             value = var.env },
      { name = "AWS_REGION",          value = var.aws_region },
      # ── Worker routing flags ─────────────────────────────────
      # Defaults in code are FALSE (so local dev runs in-process
      # without a worker). Production explicitly opts in here —
      # without these env vars set, the API would silently fall
      # back to running LLMs + ingestion inside the request thread.
      { name = "REPORTS_VIA_WORKER",  value = "true" },
      { name = "INGESTION_VIA_WORKER", value = "true" },
      # Phase 0.4 — tight pool to fit RDS max_connections=80.
      # Bump when RDS class upgrades to db.t3.medium+ (max_conn ~150).
      { name = "DB_POOL_SIZE",        value = "3" },
      { name = "DB_MAX_OVERFLOW",     value = "5" },
      # Phase 2 — gunicorn worker count. (2 × vCPU) + 1 = 2 for the
      # 512-CPU default size. Adjust if you raise api_cpu.
      { name = "GUNICORN_WORKERS",    value = "2" },
      { name = "GUNICORN_TIMEOUT",    value = "120" },
      { name = "GUNICORN_KEEPALIVE",  value = "75" },
      # Observability — point Sentry's release tag at the image tag
      # so each deploy is identifiable in the Sentry UI.
      { name = "SENTRY_RELEASE",      value = var.image_tag },
      { name = "LOG_FORMAT",          value = "json" },
    ]

    # The secrets blob is loaded at boot by backend.core.secrets —
    # we only need the ARN here so ECS can grant pull permission via
    # the execution role. Individual keys are NOT mapped into the
    # container env one-by-one because the bootstrap reads the JSON
    # itself; matching one-by-one would just duplicate the contract.
    secrets = []

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }

    # Container-level health check (separate from the ALB target
    # group health check). Catches "container running but app
    # wedged" cases that the LB check might miss for ~30 s.
    healthCheck = {
      # Use /health/live (Phase 4) — the cheap probe. /health hits
      # the DB on every poll which is wasteful at 30 s × N containers.
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health/live || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }
  }])

  tags = local.common_tags
}

resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.api_desired_count

  launch_type    = "FARGATE"
  propagate_tags = "TASK_DEFINITION"

  # Auto-rollback if a deploy puts the service in a bad state
  # (failing health checks). Saves a 3-AM page.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.ecs_tasks.id]
    # Tasks have no public IP — they reach the internet through a
    # NAT in your VPC. (NAT gateway must already exist; we don't
    # manage it here.)
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  # Ignore desired_count drift from the auto-scaling target —
  # without this, every ``terraform apply`` would re-set the count
  # back to the variable's value, fighting the auto-scaler.
  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_lb_listener.https]
  tags       = local.common_tags
}
