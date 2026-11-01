output "docs_bucket_name" {
  value = aws_s3_bucket.docs.bucket
}

output "aws_region" {
  value = var.aws_region
}

output "knowledge_base_id" {
  value = aws_bedrockagent_knowledge_base.this.id
}

output "data_source_id" {
  value = aws_bedrockagent_data_source.docs.data_source_id
}

output "generation_model_arn" {
  value = local.generation_model_arn
}

output "rerank_model_arn" {
  value = local.rerank_model_arn
}

output "agent_ecr_repository_url" {
  value = aws_ecr_repository.agent.repository_url
}

output "agent_runtime_arn" {
  value = try(aws_bedrockagentcore_agent_runtime.agent[0].agent_runtime_arn, null)
}
