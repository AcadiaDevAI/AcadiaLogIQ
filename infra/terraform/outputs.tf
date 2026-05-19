# Outputs — what the operator (or upstream Terraform) reads after
# ``terraform apply`` to wire DNS, CI/CD, monitoring, etc.

output "alb_dns_name" {
  description = "Public DNS name of the ALB. Point a Route 53 alias record at this to attach a custom domain."
  value       = aws_lb.api.dns_name
}

output "alb_arn" {
  description = "Full ARN of the ALB — useful for cross-stack references."
  value       = aws_lb.api.arn
}

output "ecs_cluster_name" {
  description = "ECS cluster name. CI/CD scripts use this with ``aws ecs update-service``."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_api_name" {
  description = "API service name in ECS — pass to ``aws ecs update-service``."
  value       = aws_ecs_service.api.name
}

output "ecs_service_worker_name" {
  description = "Report-worker service name in ECS — handles RCA + Gap Analysis LLM jobs."
  value       = aws_ecs_service.worker.name
}

output "ecs_service_ingest_worker_name" {
  description = "Ingest-worker service name in ECS — handles document ingestion (PyMuPDF + Titan embeddings)."
  value       = aws_ecs_service.ingest_worker.name
}

output "task_role_api_arn" {
  description = "Task role ARN for the API service. Grant your application identity-scoped resources to this role."
  value       = aws_iam_role.ecs_task_api.arn
}

output "task_role_worker_arn" {
  description = "Task role ARN for the worker service."
  value       = aws_iam_role.ecs_task_worker.arn
}

output "log_group_api" {
  description = "CloudWatch log group for the API service."
  value       = aws_cloudwatch_log_group.api.name
}

output "log_group_worker" {
  description = "CloudWatch log group for the report-worker service."
  value       = aws_cloudwatch_log_group.worker.name
}

output "log_group_ingest_worker" {
  description = "CloudWatch log group for the ingest-worker service."
  value       = aws_cloudwatch_log_group.ingest_worker.name
}
