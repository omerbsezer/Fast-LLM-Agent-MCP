## Simple RAG on AWS Bedrock KnowledgeBase, OpenSearch, S3 

A minimal Retrieval-Augmented Generation app: Terraform provisions the AWS
backend (S3 + Bedrock Knowledge Base + OpenSearch Serverless), and a local
Streamlit app handles document upload, Knowledge Base sync, and querying.

## Architecture

- **Terraform** (`terraform/`) — S3 bucket for source documents, an
  OpenSearch Serverless vector collection, and a Bedrock Knowledge Base
  wired to both.
- **Streamlit app** (`app/`) — runs locally, talks to AWS via boto3 using
  your local AWS credentials.

## Ingestion phase

- User uploads a document through the Streamlit UI.
- The app writes it straight to the S3 docs bucket.
- User clicks "Sync Knowledge Base" in the UI.
- This calls Bedrock's `StartIngestionJob` on the Knowledge Base's S3 data source.
- Bedrock reads new/changed objects from S3, chunks them (fixed-size, configurable via Terraform variables), embeds each chunk with **Titan Text Embeddings v2**, and writes the vectors into the OpenSearch Serverless collection.

## Querying phase

- User types a question into the Streamlit chat input.
- The app calls Bedrock's `retrieve_and_generate` against the Knowledge Base, using Claude Sonnet 4.5 as the generation model.
- Bedrock retrieves the most relevant chunks from OpenSearch Serverless and generates an answer with source citations.
- The answer and its sources are shown in the main chat pane and appended to the session's chat history in the left sidebar.

## Prerequisites

- **Bedrock model access enabled** in the target AWS account/region for `amazon.titan-embed-text-v2:0` and the Claude Sonnet 4.5 inference profile (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`). Enable this in the Bedrock console under "Model access" before applying — Terraform will provision the Knowledge Base successfully even without it, but ingestion and querying will fail with `AccessDeniedException`.
- **Local AWS credentials with permissions** to run Terraform, plus (since the app calls AWS directly with your local credentials, not through the Knowledge Base's IAM role): `s3:PutObject` on the docs bucket, `bedrock:StartIngestionJob`, `bedrock:GetIngestionJob`, `bedrock:RetrieveAndGenerate`, and `bedrock:InvokeModel` on both the inference profile ARN and the underlying foundation-model ARNs.
- Manually test on above on (GUI) AWS Console, Bedrock playground.

## Setup

1. **Provision AWS infrastructure**

   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

   This creates real, billable AWS resources (an OpenSearch Serverless collection has a minimum standing cost). Review the plan before confirming.

2. **Configure the app**

   ```bash
   cd ../app
   cp .env.example .env
   ```

   Fill in `.env` from the Terraform outputs:

   ```bash
   terraform -chdir=../terraform output
   ```

   Map `docs_bucket_name` → `DOCS_BUCKET_NAME`, `knowledge_base_id` → `KNOWLEDGE_BASE_ID`, `data_source_id` → `DATA_SOURCE_ID`, `generation_model_arn` → `GENERATION_MODEL_ARN`, `aws_region` → `AWS_REGION`.

3. **Install dependencies and run**

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Manual test plan

1. Upload a sample document via the UI.
2. Click "Sync Knowledge Base" and wait for the status to show `COMPLETE`.
3. Ask a question whose answer only appears in the uploaded document.
4. Verify the answer is correct and cites the uploaded document.

## Teardown

```bash
cd terraform
terraform destroy
```

## Note
- Be careful, be sure that OpenSearch Serverless is deleted after terraform destroy, it consumes cost in time. 