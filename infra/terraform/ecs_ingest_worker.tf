# ECS ingest-worker service — document parsing + embedding generation.
#
# Why a separate service from the report worker:
#
#   * Memory profile is different. PyMuPDF can briefly use ~2 GB on a
#     50 MB PDF; Titan embedding batches add ~200 MB on top. A 1 GB
#     report worker would OOM on either spike.
#   * Failure isolation. If an ingest task crashes the container,
#     no report jobs in flight are lost — they live in their own
#     task. Single-service combined would mean an OOM on a PDF
#     ingest kills any RCA/Gap Analysis the container happened to
#     be running.
#   * Scaling signal is different. Ingestion is bursty (engineer
#     uploads 5 files); report jobs are steady-flow. Two services
#     means two auto-scaling policies tuned independently.
#
# Same image, same task role, same security group as the report
# worker — only WORKER_KINDS + the Fargate task size differ.

resource "aws_ecs_task_definition" "ingest_worker" {
  family                   = "${local.name_prefix}-ingest-worker"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ingest_worker_cpu
  memory                   = var.ingest_worker_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  # Reuses the report-worker task role: same Bedrock + Secrets-
  # Manager permissions. If you need ingest-specific scopes later
  # (e.g. wider S3 read), add a dedicated role here.
  task_role_arn = aws_iam_role.ecs_task_worker.arn

  container_definitions = jsonencode([{
    name      = "ingest_worker"
    image     = local.image
    essential = true

    environment = [
      { name = "APP_ROLE",                 value = "worker" },
      { name = "WORKER_KINDS",             value = local.ingest_worker_kinds },
      { name = "APP_ENV",                  value = var.env },
      { name = "AWS_REGION",               value = var.aws_region },
      # Same conservative DB pool — workers hold a connection only
      # briefly per job. The chunk-INSERT loop inside index_file_job
      # batches commits, so two connections cover the worst case.
      { name = "DB_POOL_SIZE",             value = "2" },
      { name = "DB_MAX_OVERFLOW",          value = "2" },
      # Disable the periodic glossary refresher — the ingest worker
      # never serves retrieval queries.
      { name = "GLOSSARY_REFRESH_SECONDS", value = "0" },
      { name = "SENTRY_RELEASE",           value = var.image_tag },
      { name = "LOG_FORMAT",               value = "json" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.ingest_worker.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "ingest-worker"
      }
    }
  }])

  tags = local.common_tags
}

resource "aws_ecs_service" "ingest_worker" {
  name            = "${local.name_prefix}-ingest-worker"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ingest_worker.arn
  desired_count   = var.ingest_worker_desired_count

  launch_type    = "FARGATE"
  propagate_tags = "TASK_DEFINITION"

  # Same auto-rollback story as the API + report worker: a failing
  # deploy reverts to the previous task def automatically.
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

  # Ignore desired_count drift from auto-scaling — without this,
  # ``terraform apply`` would tug the count back to the variable
  # default every run, fighting the auto-scaler.
  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = local.common_tags
}
