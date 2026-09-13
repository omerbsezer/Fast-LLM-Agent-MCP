resource "aws_s3_bucket" "docs" {
  bucket = var.docs_bucket_name != null ? var.docs_bucket_name : "${var.project_name}-docs-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "docs" {
  bucket = aws_s3_bucket.docs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket                  = aws_s3_bucket.docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
