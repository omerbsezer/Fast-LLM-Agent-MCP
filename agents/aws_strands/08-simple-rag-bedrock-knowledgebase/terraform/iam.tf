data "aws_iam_policy_document" "bedrock_kb_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "bedrock_kb" {
  name               = "${var.project_name}-kb-role"
  assume_role_policy = data.aws_iam_policy_document.bedrock_kb_trust.json
}

data "aws_iam_policy_document" "bedrock_kb_permissions" {
  statement {
    sid       = "S3Read"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.docs.arn, "${aws_s3_bucket.docs.arn}/*"]
  }

  statement {
    sid       = "InvokeEmbeddingModel"
    actions   = ["bedrock:InvokeModel"]
    resources = ["arn:aws:bedrock:${var.aws_region}::foundation-model/${var.embedding_model_id}"]
  }

  statement {
    sid       = "OpenSearchServerlessAccess"
    actions   = ["aoss:APIAccessAll"]
    resources = [aws_opensearchserverless_collection.kb.arn]
  }
}

resource "aws_iam_role_policy" "bedrock_kb" {
  name   = "${var.project_name}-kb-policy"
  role   = aws_iam_role.bedrock_kb.id
  policy = data.aws_iam_policy_document.bedrock_kb_permissions.json
}
