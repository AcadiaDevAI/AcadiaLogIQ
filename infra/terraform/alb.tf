# Application Load Balancer + target group + listeners.
#
# HTTPS-only public surface. HTTP listener exists solely to issue
# a 301 redirect to HTTPS — leaving port 80 closed entirely is
# stricter but breaks any well-intentioned client that didn't
# upgrade-to-HTTPS before connecting.

resource "aws_lb" "api" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
  idle_timeout       = var.alb_idle_timeout_seconds

  # Drop "deletion protection" once you're confident in the rest of
  # the stack. Keeping it on means a fat-finger ``terraform destroy``
  # won't take the front door down with it.
  enable_deletion_protection = true
  tags                       = local.common_tags
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name_prefix}-api"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip" # Fargate awsvpc networking → IP targets

  # Liveness probe added in Phase 4 — returns a tight 200 + uptime
  # field without touching DB or BM25 state. /health (without /live)
  # remains the operator-facing readiness probe.
  health_check {
    path                = "/health/live"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  # ECS rolling deploys connect new tasks and drain old ones; this
  # gives in-flight requests time to finish on the old task.
  deregistration_delay = 30

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-api" })
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  # Default to the most recent generally-available policy. Bump as
  # AWS publishes newer ones — ``terraform plan`` will surface it.
  ssl_policy      = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}
