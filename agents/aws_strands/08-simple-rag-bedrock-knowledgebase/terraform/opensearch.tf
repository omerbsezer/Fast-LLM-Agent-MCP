resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.project_name}-encrypt"
  type = "encryption"
  policy = jsonencode({
    Rules = [{
      ResourceType = "collection"
      Resource     = ["collection/${var.project_name}-kb"]
    }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.project_name}-network"
  type = "network"
  policy = jsonencode([{
    Rules = [{
      ResourceType = "collection"
      Resource     = ["collection/${var.project_name}-kb"]
    }]
    AllowFromPublic = true
  }])
}

resource "aws_opensearchserverless_collection" "kb" {
  name = "${var.project_name}-kb"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network,
  ]
}

resource "aws_opensearchserverless_access_policy" "kb" {
  name = "${var.project_name}-access"
  type = "data"
  policy = jsonencode([{
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${var.project_name}-kb"]
        Permission   = ["aoss:*"]
      },
      {
        ResourceType = "index"
        Resource     = ["index/${var.project_name}-kb/*"]
        Permission   = ["aoss:*"]
      }
    ]
    Principal = [
      aws_iam_role.bedrock_kb.arn,
      var.admin_principal_arn != null ? var.admin_principal_arn : data.aws_caller_identity.current.arn,
    ]
  }])
}

resource "time_sleep" "wait_for_access_policy" {
  depends_on      = [aws_opensearchserverless_access_policy.kb]
  create_duration = "60s"
}

provider "opensearch" {
  url               = aws_opensearchserverless_collection.kb.collection_endpoint
  healthcheck       = false
  aws_region        = var.aws_region
  sign_aws_requests = true
}

resource "opensearch_index" "kb_vector_index" {
  name      = "${var.project_name}-index"
  index_knn = true

  mappings = jsonencode({
    properties = {
      "bedrock-knowledge-base-default-vector" = {
        type      = "knn_vector"
        dimension = var.embedding_dimension
        method = {
          name       = "hnsw"
          engine     = "faiss"
          space_type = "l2"
        }
      }
      "AMAZON_BEDROCK_TEXT_CHUNK" = { type = "text" }
      "AMAZON_BEDROCK_METADATA"   = { type = "text", index = false }
    }
  })

  depends_on = [time_sleep.wait_for_access_policy]
}

resource "time_sleep" "wait_for_index" {
  depends_on      = [opensearch_index.kb_vector_index]
  create_duration = "30s"
}
