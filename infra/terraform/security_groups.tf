# Security groups — the network ACL between the public internet,
# the ALB, and the ECS tasks.
#
# Layered model (most-locked-down at the right):
#
#   internet ---:443---> [alb_sg] ---:8000---> [ecs_tasks_sg] ---> RDS
#
# Nothing else gets through. RDS lives in its own SG with rules
# allowing ingress only from ``ecs_tasks_sg``; we don't manage that
# here because the RDS instance is a pre-existing resource (see
# README pre-requisites).

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb"
  description = "ALB ingress — public HTTPS + HTTP→HTTPS redirect"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTPS from anywhere"
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "HTTP (redirected to 443 at the listener level)"
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound — ALB needs to reach the ECS task SG on 8000"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-alb" })
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${local.name_prefix}-ecs-tasks"
  description = "ECS Fargate tasks — ingress from ALB only"
  vpc_id      = var.vpc_id

  ingress {
    description     = "App port from the ALB"
    protocol        = "tcp"
    from_port       = 8000
    to_port         = 8000
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All outbound — Bedrock, RDS, S3, Secrets Manager, ECR"
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-ecs-tasks" })
}
