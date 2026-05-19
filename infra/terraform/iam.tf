# IAM — two role types per ECS service:
#
#   * Execution role  → ECS uses this to PULL the image and write
#                       container logs. Identical for api + worker
#                       (one role, attached to both task definitions).
#   * Task role       → the application's own runtime identity. We
#                       give it: read on the AWS Secrets Manager
#                       blob, invoke on Bedrock models, and the S3
#                       bucket used for uploads. Per-service in case
#                       we ever want to lock the worker down further.
#
# We deliberately do NOT attach ``AdministratorAccess`` or any of
# the broad managed policies — every grant is justified by the
# code reading or writing the underlying resource.

# ─────────────────────────────────────────────────────────────────
# Execution role — shared by both services.
# ─────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${local.name_prefix}-ecs-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.common_tags
}

# AWS-managed policy covering: ecr:GetAuthorizationToken,
# ecr:BatchGetImage, logs:CreateLogStream, logs:PutLogEvents.
# Everything an execution role legitimately needs.
resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow the execution role to read the secret at task startup
# (ECS injects the resolved values into the container env).
resource "aws_iam_policy" "ecs_execution_secrets_read" {
  name = "${local.name_prefix}-ecs-exec-secrets"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "AllowReadOfBackendSecret"
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [var.secrets_manager_arn]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_secrets_read" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = aws_iam_policy.ecs_execution_secrets_read.arn
}

# ─────────────────────────────────────────────────────────────────
# Task role — application runtime identity.
# ─────────────────────────────────────────────────────────────────

resource "aws_iam_role" "ecs_task_api" {
  name               = "${local.name_prefix}-task-api"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.common_tags
}

resource "aws_iam_role" "ecs_task_worker" {
  name               = "${local.name_prefix}-task-worker"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = local.common_tags
}

# Bedrock invocation — both services call ``bedrock-runtime:InvokeModel``.
# Scoped to the on-demand cross-region inference profiles we use; tighten
# to specific model IDs if you want a stricter audit story.
resource "aws_iam_policy" "bedrock_invoke" {
  name = "${local.name_prefix}-bedrock-invoke"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "AllowBedrockInvokeOnDemand"
      Effect = "Allow"
      Action = [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
      ]
      # Star is intentional inside the bedrock namespace — the model
      # inventory churns (new versions, regional profiles) too fast
      # to enumerate. The action set above is the actual blast radius.
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "api_bedrock" {
  role       = aws_iam_role.ecs_task_api.name
  policy_arn = aws_iam_policy.bedrock_invoke.arn
}

resource "aws_iam_role_policy_attachment" "worker_bedrock" {
  role       = aws_iam_role.ecs_task_worker.name
  policy_arn = aws_iam_policy.bedrock_invoke.arn
}

# Secrets-Manager read for the task role too (the backend's
# bootstrap reads the secret AT RUNTIME if the env var isn't set —
# see backend.core.secrets). Same ARN restriction as execution.
resource "aws_iam_role_policy_attachment" "api_secrets_read" {
  role       = aws_iam_role.ecs_task_api.name
  policy_arn = aws_iam_policy.ecs_execution_secrets_read.arn
}

resource "aws_iam_role_policy_attachment" "worker_secrets_read" {
  role       = aws_iam_role.ecs_task_worker.name
  policy_arn = aws_iam_policy.ecs_execution_secrets_read.arn
}
