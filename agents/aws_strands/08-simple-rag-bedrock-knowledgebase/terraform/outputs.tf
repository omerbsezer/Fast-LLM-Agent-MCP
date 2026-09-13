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
