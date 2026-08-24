variable "aws_region" {
  description = "AWS region for ECR, IAM, AgentCore Memory and AgentCore Runtime."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Base name used for the ECR repo, IAM role and AgentCore resources."
  type        = string
  default     = "strands-blog-agentcore"
}

variable "container_image_uri" {
  description = "Full ECR image URI:tag to deploy to the AgentCore Runtime, e.g. xx.dkr.ecr.eu-central-1.amazonaws.com/strands-blog-agentcore:latest"
  type        = string
  default     = ""
}

variable "memory_event_expiry_days" {
  description = "Days after which AgentCore Memory events expire (7-365). Kept short since memory here is per-run scratch, not long-term."
  type        = number
  default     = 30
}
