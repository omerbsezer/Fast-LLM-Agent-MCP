# --- Agentic RAG agent: ECR + IAM + AgentCore Runtime ----------------------
# Deploys the Strands agent in ../agent to AWS Bedrock AgentCore Runtime.
# ../deploy.sh runs the whole flow: create ECR, build + push the arm64 image,
# record its URI in agent.auto.tfvars, then apply again to create the Runtime.
# No AgentCore Gateway is needed: the agent calls the Knowledge Base (which
# fronts OpenSearch, the embedding and rerank models) with plain AWS API calls
# signed by the execution role below.

resource "aws_ecr_repository" "agent" {
  name                 = "${var.project_name}-agent"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

data "aws_iam_policy_document" "agent_runtime_trust" {
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

resource "aws_iam_role" "agent_runtime_execution" {
  name               = "${var.project_name}-agent-runtime-role"
  assume_role_policy = data.aws_iam_policy_document.agent_runtime_trust.json
}

data "aws_iam_policy_document" "agent_runtime_permissions" {
  statement {
    sid     = "BedrockModelInvoke"
    effect  = "Allow"
    actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*",
    ]
  }

  statement {
    sid       = "KnowledgeBaseRetrieve"
    effect    = "Allow"
    actions   = ["bedrock:Retrieve"]
    resources = [aws_bedrockagent_knowledge_base.this.arn]
  }

  # Retrieve calls with a rerankingConfiguration also need Rerank on the caller.
  statement {
    sid       = "Rerank"
    effect    = "Allow"
    actions   = ["bedrock:Rerank"]
    resources = ["*"]
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
    sid    = "Logs"
    effect = "Allow"
    # PutResourcePolicy lets AgentCore deliver the agent's spans into its own log group.
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams", "logs:PutResourcePolicy"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/bedrock-agentcore/*"]
  }

  statement {
    sid       = "LogsDescribe"
    effect    = "Allow"
    actions   = ["logs:DescribeLogGroups"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:*"]
  }

  statement {
    sid       = "Xray"
    effect    = "Allow"
    actions   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords", "xray:GetSamplingRules", "xray:GetSamplingTargets"]
    resources = ["*"]
  }

  # AgentCore Observability metrics (sessions, latency, tokens) from ADOT.
  statement {
    sid       = "Metrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["bedrock-agentcore"]
    }
  }
}

resource "aws_iam_role_policy" "agent_runtime_permissions" {
  name   = "${var.project_name}-agent-runtime-permissions"
  role   = aws_iam_role.agent_runtime_execution.id
  policy = data.aws_iam_policy_document.agent_runtime_permissions.json
}

resource "aws_bedrockagentcore_agent_runtime" "agent" {
  count = var.agent_container_image_uri == "" ? 0 : 1

  agent_runtime_name = replace("${var.project_name}_agent", "-", "_")
  role_arn           = aws_iam_role.agent_runtime_execution.arn

  agent_runtime_artifact {
    container_configuration {
      container_uri = var.agent_container_image_uri
    }
  }

  network_configuration {
    network_mode = "PUBLIC"
  }

  environment_variables = {
    AWS_REGION          = var.aws_region
    KNOWLEDGE_BASE_ID   = aws_bedrockagent_knowledge_base.this.id
    GENERATION_MODEL_ID = var.generation_model_id
    RERANK_MODEL_ARN    = var.enable_reranking ? local.rerank_model_arn : ""
  }
}

# --- AgentCore Observability ----------------------------------------------
# Service-side tracing for the runtime (the "Tracing" toggle in the console).
# The agent's own spans (prompts, tool calls, model calls) come from ADOT
# inside the container; see ../agent/Dockerfile. Both need CloudWatch
# Transaction Search, which ../deploy.sh turns on once per account.

resource "aws_cloudwatch_log_delivery_source" "agent_traces" {
  count        = length(aws_bedrockagentcore_agent_runtime.agent)
  name         = "${var.project_name}-agent-traces"
  log_type     = "TRACES"
  resource_arn = aws_bedrockagentcore_agent_runtime.agent[0].agent_runtime_arn
}

resource "aws_cloudwatch_log_delivery_destination" "agent_traces" {
  count                     = length(aws_bedrockagentcore_agent_runtime.agent)
  name                      = "${var.project_name}-agent-traces"
  delivery_destination_type = "XRAY"
}

resource "aws_cloudwatch_log_delivery" "agent_traces" {
  count                    = length(aws_bedrockagentcore_agent_runtime.agent)
  delivery_source_name     = aws_cloudwatch_log_delivery_source.agent_traces[0].name
  delivery_destination_arn = aws_cloudwatch_log_delivery_destination.agent_traces[0].arn
}
