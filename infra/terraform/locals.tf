# Computed identifiers + tags reused across resources.
#
# Centralised so a name change is one edit, not ten.

locals {
  # Standard 2-segment resource naming: ``acadialogiq-{env}-{role}``.
  # The env segment lets us run dev/staging/prod in the same account
  # without collisions; ECS service names must be unique per cluster.
  name_prefix = "acadialogiq-${var.env}"

  # ECR repository URL — assumes the repo lives in the same account
  # + region. Cross-account ECR pulls need a different ARN pattern;
  # add a variable when you hit that.
  ecr_url = "${var.aws_account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}"
  image   = "${local.ecr_url}:${var.image_tag}"

  # Single source of truth for log group names — referenced by the
  # task definitions AND the explicit aws_cloudwatch_log_group
  # resources so retention + KMS settings can be applied uniformly.
  log_group_api           = "/acadialogiq/${var.env}/backend"
  log_group_worker        = "/acadialogiq/${var.env}/worker"
  log_group_ingest_worker = "/acadialogiq/${var.env}/ingest-worker"

  # WORKER_KINDS values — kept centralised so a future re-org
  # (split RCA + Gap into separate workers, add re-ingest, etc.)
  # is a one-line edit instead of a hunt across task definitions.
  report_worker_kinds = "rca_customer,rca_internal,gap_analysis_gap,gap_analysis_pm"
  ingest_worker_kinds = "ingest_document"

  # Common tag set — provider-default tags add Project/Env/ManagedBy
  # to every resource automatically (see versions.tf). This local is
  # for resources that override or augment the defaults.
  common_tags = {
    Service = "acadialogiq"
    Env     = var.env
  }
}
