terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.18.0" # first version with aws_bedrockagentcore_* resources
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# --- ECR: holds the agent's ARM64 container image -------------------------
resource "aws_ecr_repository" "agent" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- IAM: execution role assumed by AgentCore Runtime ----------------------
# Pulls the container image, calls Bedrock models, writes logs/traces, and reads/writes AgentCore Memory events.
data "aws_iam_policy_document" "runtime_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:bedrock-agentcore:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
    }
  }
}

resource "aws_iam_role" "runtime_execution" {
  name               = "${var.project_name}-runtime-role"
  assume_role_policy = data.aws_iam_policy_document.runtime_trust.json
}

data "aws_iam_policy_document" "runtime_permissions" {
  statement {
    sid     = "BedrockModelInvoke"
    effect  = "Allow"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      # Wildcarded across regions, not just var.aws_region: "us.*" model IDs are cross-region inference profiles invoked from us-east-1 regardless of
      # where the Runtime itself is deployed (see agent.py's BEDROCK_REGION).
      "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }

  statement {
    sid       = "EcrPull"
    effect    = "Allow"
    actions   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.agent.arn]
  }

  statement {
    sid       = "EcrAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid       = "Logs"
    effect    = "Allow"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/bedrock-agentcore/*"]
  }

  statement {
    sid       = "Xray"
    effect    = "Allow"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
    resources = ["*"]
  }

  statement {
    sid    = "AgentCoreMemory"
    effect = "Allow"
    actions = [
      "bedrock-agentcore:CreateEvent",
      "bedrock-agentcore:ListEvents",
      "bedrock-agentcore:GetEvent",
      "bedrock-agentcore:GetMemory",
      "bedrock-agentcore:RetrieveMemoryRecords",
    ]
    resources = [aws_bedrockagentcore_memory.agent.arn]
  }
}

resource "aws_iam_role_policy" "runtime_permissions" {
  name   = "${var.project_name}-runtime-permissions"
  role   = aws_iam_role.runtime_execution.id
  policy = data.aws_iam_policy_document.runtime_permissions.json
}

# --- AgentCore Memory: short-term/event-only, no long-term strategies ------
resource "aws_bedrockagentcore_memory" "agent" {
  name                  = replace(var.project_name, "-", "_")
  description           = "Per-run scratch memory (Research/Sources/Critiques/Log) for the Strands blog pipeline."
  event_expiry_duration = var.memory_event_expiry_days
}

# --- AgentCore Runtime ------------------------------------------------------
resource "aws_bedrockagentcore_agent_runtime" "agent" {
  count = var.container_image_uri == "" ? 0 : 1

  agent_runtime_name = replace(var.project_name, "-", "_")
  role_arn           = aws_iam_role.runtime_execution.arn

  agent_runtime_artifact {
    container_configuration {
      container_uri = var.container_image_uri
    }
  }

  network_configuration {
    network_mode = "PUBLIC"
  }

  environment_variables = {
    AGENTCORE_MEMORY_ID = aws_bedrockagentcore_memory.agent.id
    AWS_REGION          = var.aws_region
  }
}
