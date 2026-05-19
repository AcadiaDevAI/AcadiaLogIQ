# Provider + Terraform version pins.
#
# Pinning to a specific Terraform minor + AWS provider major avoids
# the "provider upgrade silently changed a resource attribute"
# class of surprise — a real production failure mode. Bumping these
# is a deliberate, reviewable PR; floating constraints would let it
# happen at apply-time.

terraform {
  required_version = ">= 1.6.0, < 2.0.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
  # State lives in an S3 bucket you provide via -backend-config at
  # ``terraform init`` time. Encrypted by default; lock via DynamoDB
  # for multi-engineer safety.
  backend "s3" {}
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "acadialogiq"
      Env       = var.env
      ManagedBy = "terraform"
      Module    = "infra/terraform"
    }
  }
}
