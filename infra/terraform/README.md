# Acadia LogIQ — Production Infrastructure (Terraform)

Defines the AWS resources behind the production deployment:

* **VPC + subnets** — assumed pre-existing (referenced by id). We deliberately
  don't manage the VPC here; you've already got one and a destructive plan
  ripping it out is a worse risk than a slightly looser blast radius.
* **ECS cluster** — Fargate-only, single cluster, two services.
* **ECS service `api`** — public-facing FastAPI behind the ALB.
* **ECS service `worker`** — the report-job consumer (RCA + Gap Analysis
  LLM generations). Pinned to `WORKER_KINDS=rca_customer,...` so it can't
  drain ingestion rows and OOM. 1 GB memory.
* **ECS service `ingest_worker`** — the document-ingestion consumer
  (PyMuPDF parse + Titan embeddings). Pinned to
  `WORKER_KINDS=ingest_document`. 1 vCPU / 2 GB memory — separate
  from the report worker so an OOM on a big PDF can't kill an
  in-flight report job.
* **Application Load Balancer** — HTTPS termination, listener rules,
  target group bound to `/health/live`, idle timeout 720 s.
* **CloudWatch log groups** — for both services, 30-day retention.
* **IAM** — task execution role + task role for each service.
* **Auto-scaling policies** — target tracking on in-flight requests
  (`api`) and pending job count (`worker`).

## Phase-3 ship list (what's here)

```
infra/terraform/
├── README.md               ← this file
├── versions.tf             ← provider pins
├── variables.tf            ← env, region, account_id, image_tag, …
├── locals.tf               ← computed names + WORKER_KINDS strings
├── alb.tf                  ← ALB + target group + listener
├── ecs_cluster.tf          ← cluster + capacity providers
├── ecs_api.tf              ← API task definition + service
├── ecs_worker.tf           ← report-worker task definition + service
├── ecs_ingest_worker.tf    ← ingest-worker task definition + service
├── iam.tf                  ← execution + task roles, log policies
├── security_groups.tf      ← ALB SG, ECS task SG, ingress rules
├── autoscaling.tf          ← target tracking + step scaling per service
├── logs.tf                 ← CloudWatch log groups (api, worker, ingest-worker)
└── outputs.tf              ← ALB DNS, service names, log group names
```

## Pre-requisites you must provide before `terraform apply`

These are deliberately NOT managed by this stack to avoid the
"oops Terraform deleted prod" failure mode. Pass them via tfvars
or environment variables (see `variables.tf` for the full list):

* **AWS account ID** (`var.aws_account_id`)
* **VPC ID** (`var.vpc_id`) and at least two **private subnet IDs**
  in different AZs (`var.private_subnet_ids`)
* At least one **public subnet ID** per AZ for the ALB
  (`var.public_subnet_ids`)
* **ACM certificate ARN** in the same region for the ALB HTTPS
  listener (`var.acm_certificate_arn`)
* **ECR repository** with the backend image you intend to deploy.
  Pass the image tag via `var.image_tag` (typically a git SHA).
* **AWS Secrets Manager** secret named
  `acadialogiq/${var.env}/backend/secrets` containing the same
  JSON the backend reads today (DATABASE_URL, BEDROCK_*, CLERK_*,
  SENTRY_DSN, etc.). The task role is granted `secretsmanager:GetSecretValue`
  on this exact ARN — nothing else.

## Bootstrap commands

```bash
cd infra/terraform

# One-time: configure the S3 backend for the state file. Pick a
# bucket name that already exists in your account.
terraform init \
  -backend-config="bucket=acadialogiq-tf-state-${ACCOUNT_ID}" \
  -backend-config="key=prod/terraform.tfstate" \
  -backend-config="region=us-east-1" \
  -backend-config="encrypt=true"

# Per-release: build + push image, then update the task definitions.
docker build -t "${ECR_URL}:${GIT_SHA}" -f backend/Dockerfile .
docker push "${ECR_URL}:${GIT_SHA}"

terraform plan -var="image_tag=${GIT_SHA}" -out=tfplan
terraform apply tfplan
```

`terraform apply` performs a blue/green ECS service update — the new
task definition is pushed, the service rolls instances one at a time
with a deployment circuit breaker. Zero-downtime.

## What happens on a quota-exhausted Bedrock call (operational note)

The worker handler raises a normal exception → `mark_failed` re-queues
with exponential backoff → after 3 attempts the row moves to `failed`
and a Sentry event fires (via the `LoggingIntegration` set up in Phase
3a). No infra changes needed for that path — it's handled at the
application layer.

## Pre-prod checklist

Before pointing real traffic at this stack:

- [ ] Bedrock quota increase ticket approved (see
  `docs/runbooks/bedrock-quota-increase.md`).
- [ ] RDS class upgraded to `db.t3.medium` (or larger) — the Phase 0.4
  pool math assumes `max_connections >= 150`. Default `db.t3.small`
  with 80 will throttle past 2 Fargate tasks.
- [ ] `var.acm_certificate_arn` validated and covers the domain you
  intend to attach (Route 53 record will be added in a follow-up
  module — not in this stack to avoid DNS coupling).
- [ ] Sentry DSN provisioned and stored in the backend secret blob.
- [ ] CloudWatch alarms configured (see
  `docs/observability/saved-queries.md` for the recommended set).
- [ ] First deploy uses `desired_count = 0` for both services, then
  scaled up after the ALB target group reports `healthy`. Avoids
  the "first container can't reach the DB" panic.

## Cost estimate (us-east-1 list price, 2026 rates)

| Resource | Quantity | $/month |
| --- | --- | --- |
| Fargate API tasks (2 × 0.5 vCPU, 1 GB) | 24×7 | $35 |
| Fargate report-worker tasks (1 × 0.5 vCPU, 1 GB) | 24×7 | $17 |
| Fargate ingest-worker tasks (1 × 1 vCPU, 2 GB) | 24×7 | $36 |
| ALB | 1 | $22 + LCU |
| CloudWatch Logs ingestion | ~7 GB/mo | $4 |
| **Approx total** | | **~$115/mo** |

Bedrock invocation cost is not included — it's per-token and scales
with usage, and the cache we built in Phase 13 dominates this number
in practice once steady-state traffic kicks in.
