variable "aws_region" {
  description = "AWS region to deploy into. Must support Bedrock Knowledge Bases and OpenSearch Serverless."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Short name used as a prefix for resource names."
  type        = string
  default     = "simple-rag"
}

variable "docs_bucket_name" {
  description = "Globally-unique S3 bucket name for source documents. Leave null to auto-generate."
  type        = string
  default     = null
}

variable "embedding_model_id" {
  description = "Bedrock foundation model ID used for embeddings."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimension" {
  description = "Output vector dimension for the embedding model."
  type        = number
  default     = 1024
}

variable "generation_model_id" {
  description = "Bedrock inference profile ID used for answer generation via retrieve_and_generate. Must use a geo prefix (eu./us./apac.) matching aws_region's geography, since EU cross-region inference profiles only route within EU regions."
  type        = string
  default     = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "chunk_max_tokens" {
  description = "Max tokens per chunk for the Knowledge Base data source."
  type        = number
  default     = 300
}

variable "chunk_overlap_percentage" {
  description = "Overlap percentage between chunks."
  type        = number
  default     = 20
}

variable "admin_principal_arn" {
  description = "IAM principal ARN granted OpenSearch Serverless data-access for creating/managing the vector index via Terraform. Defaults to the caller's identity ARN; override with the underlying IAM role/user ARN if authenticating via an assumed role (STS assumed-role session ARNs are rejected by AOSS access policies)."
  type        = string
  default     = null
}
