# CloudWatch log groups for both ECS services.
#
# Why explicit (rather than letting the awslogs driver auto-create):
# we need retention enforced from day one. Auto-created groups
# default to "Never expire" — a slow-burning bill nobody notices
# until the storage line surpasses everything else combined.

resource "aws_cloudwatch_log_group" "api" {
  name              = local.log_group_api
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = local.log_group_worker
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}

# Separate log group for the ingest worker keeps "report job ran"
# and "ingestion ran" log streams orthogonal — incident triage stays
# fast (you grep one group instead of one mixed stream).
resource "aws_cloudwatch_log_group" "ingest_worker" {
  name              = local.log_group_ingest_worker
  retention_in_days = var.log_retention_days
  tags              = local.common_tags
}
