# ECS cluster — Fargate-only.
#
# We use capacity providers (FARGATE + FARGATE_SPOT) so individual
# services can opt into spot for cost savings on the worker tier
# without affecting the API tier. Container Insights is on; the
# extra ~$2/mo per cluster is worth the dashboards.

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = local.common_tags
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
}
