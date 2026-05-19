# Example variable file — copy to ``prod.tfvars`` (NOT in git) and
# edit. Pass via ``terraform plan -var-file=prod.tfvars``.

env                  = "dev"
aws_region           = "us-east-1"
aws_account_id       = "123456789012"             # replace with your account
vpc_id               = "vpc-xxxxxxxxxxxxxxxxx"
public_subnet_ids    = ["subnet-aaa", "subnet-bbb"]
private_subnet_ids   = ["subnet-ccc", "subnet-ddd"]
acm_certificate_arn  = "arn:aws:acm:us-east-1:123456789012:certificate/xxxxxxxx"
secrets_manager_arn  = "arn:aws:secretsmanager:us-east-1:123456789012:secret:acadialogiq/dev/backend/secrets-XXXXXX"

ecr_repository_name  = "acadialogiq-backend"
image_tag            = "deadbeef"                 # set per deploy

api_desired_count           = 2
worker_desired_count        = 1
ingest_worker_desired_count = 1
# Bigger sizing for the ingest worker — PyMuPDF + Titan batches need ~2 GB
ingest_worker_cpu    = 1024
ingest_worker_memory = 2048
log_retention_days   = 30
