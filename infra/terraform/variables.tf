# Input variables — everything the operator MUST or MAY override.
#
# Defaults are chosen for a production dev/staging deployment. Prod
# usually wants ``api_desired_count = 2`` and a larger task size.
# Pass via ``-var`` on the CLI or a tfvars file checked into a
# separate, ops-only repo (NOT this repo — keep deployment-specific
# values out of git).

variable "env" {
  description = "Deployment environment. Used in resource names + tags."
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.env)
    error_message = "env must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region. RDS + Bedrock must be reachable in the same region."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS account ID — used to disambiguate ARNs in IAM policies."
  type        = string
}

variable "vpc_id" {
  description = "Existing VPC ID. We don't manage the VPC to avoid destructive plans."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnet IDs (one per AZ) for the ALB."
  type        = list(string)
  validation {
    condition     = length(var.public_subnet_ids) >= 2
    error_message = "ALB requires subnets in at least two AZs."
  }
}

variable "private_subnet_ids" {
  description = "Private subnet IDs (one per AZ) for the Fargate tasks."
  type        = list(string)
  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "Fargate services require subnets in at least two AZs."
  }
}

variable "acm_certificate_arn" {
  description = "ACM cert ARN for the ALB HTTPS listener. Must be in the same region."
  type        = string
}

# Image identity. Bump ``image_tag`` per deploy (typically a git SHA).
# The ECR repo URL is computed from the account_id + region in locals.tf
# so you don't repeat yourself.
variable "ecr_repository_name" {
  description = "ECR repository holding the backend image."
  type        = string
  default     = "acadialogiq-backend"
}

variable "image_tag" {
  description = "Image tag to deploy (typically a git SHA from CI)."
  type        = string
}

variable "secrets_manager_arn" {
  description = "Full ARN of the backend secrets blob. The task role is granted read on this exact ARN."
  type        = string
}

variable "api_desired_count" {
  description = "Initial API task count. Auto-scaling will adjust above/below this."
  type        = number
  default     = 2
}

variable "api_cpu" {
  description = "Fargate CPU units per API task. 512 = 0.5 vCPU."
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "Fargate memory per API task (MiB)."
  type        = number
  default     = 1024
}

variable "worker_desired_count" {
  description = "Initial worker task count."
  type        = number
  default     = 1
}

variable "worker_cpu" {
  description = "Fargate CPU units per worker task."
  type        = number
  default     = 512
}

variable "worker_memory" {
  description = "Fargate memory per worker task (MiB). Report jobs are Bedrock-bound and stay under 1 GB comfortably."
  type        = number
  default     = 1024
}

# Ingest worker — separate ECS service so PyMuPDF / Titan-embedding
# spikes can't OOM-kill an in-flight report job. Sized larger because
# parsing a >50 MB PDF can briefly need ~2 GB RAM.
variable "ingest_worker_desired_count" {
  description = "Initial ingest-worker task count."
  type        = number
  default     = 1
}

variable "ingest_worker_cpu" {
  description = "Fargate CPU units per ingest-worker task. 1024 = 1 vCPU — enough for PyMuPDF + parallel Titan batches."
  type        = number
  default     = 1024
}

variable "ingest_worker_memory" {
  description = "Fargate memory per ingest-worker task (MiB). 2048 covers worst-case PyMuPDF spike on a ~50 MB PDF."
  type        = number
  default     = 2048
}

variable "log_retention_days" {
  description = "CloudWatch log retention (days) for both services."
  type        = number
  default     = 30
}

variable "alb_idle_timeout_seconds" {
  description = "ALB idle timeout. Must exceed worst-case API request lifetime. With Phase 1 async jobs, API responses are sub-second — 60 s default is plenty."
  type        = number
  default     = 60
}
