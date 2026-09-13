# Simple RAG on Bedrock Knowledge Bases — Design Spec

Date: 2026-09-13
Status: Approved for implementation

## Context

This directory (`08-agent-strands-agentic-RAG`) currently contains a leftover
blog-writing multi-agent pipeline (Planner/Generator/Evaluator agents) running
on AWS Bedrock AgentCore Runtime + Memory, with matching Terraform (ECR, IAM,
AgentCore Memory/Runtime resources). None of it is RAG-related — it appears to
have been copied from a sibling lab and never adapted. It is being replaced
entirely with a simple, from-scratch Retrieval-Augmented Generation project.

## Goal

Documents uploaded to S3 are ingested into a Bedrock Knowledge Base backed by
a vector database. A local Streamlit app provides two functions: uploading
documents (GUI → S3 → manual sync into the vector DB) and querying the
knowledge base (question → retrieved chunks → generated answer with
citations).

## Deletions

Remove entirely (superseded, unrelated to RAG):
- `agent.py`, `deploy.sh`, `invoke_client.py`, `Dockerfile`, `.dockerignore`
- `images/` (AgentCore screenshots)
- `requirements.txt` (rewritten for the new stack)
- `terraform/main.tf`, `terraform/variables.tf`, `terraform/outputs.tf` (rewritten from scratch)
- `output/` (blog pipeline output dir)
- `Readme.md` (rewritten)

Before deleting, the existing Terraform was reviewed (see chat history) to
confirm it only provisions AgentCore-specific resources (ECR repo, AgentCore
execution IAM role, AgentCore Memory, AgentCore Runtime) with no shared/
external dependencies — safe to remove wholesale.

## Vector database choice

AWS offers two vector store options usable with Bedrock Knowledge Bases:

| Option | Trade-off |
|---|---|
| **OpenSearch Serverless (chosen)** | Mature, fully documented Terraform support (`aws_opensearchserverless_*` resources have existed since aws provider ~5.x). Always-on minimum cost (~2 OCUs) since it's serverless-but-provisioned capacity, not the cheapest for a demo, but the safe/well-trodden path for Bedrock KB. |
| **S3 Vectors** | Newer (2025), pay-per-use with no minimum OCU floor — much cheaper for a low-traffic demo. Terraform coverage is newer/less battle-tested. Not chosen for this lab, but worth a "Reference" callout in the README as the cheaper alternative for readers who want to explore it themselves. |

## Region and models

Everything deploys in **eu-central-1** (matches the rest of this repo's labs).

- Embedding model: **Amazon Titan Text Embeddings V2** (`amazon.titan-embed-text-v2:0`), 1024 dimensions — the standard default for Bedrock KB.
- Generation model: **Amazon Nova Pro**, invoked via the **EU cross-region inference profile** `eu.amazon.nova-pro-v1:0` (ARN: `arn:aws:bedrock:eu-central-1:<account>:inference-profile/eu.amazon.nova-pro-v1:0`). Verified via AWS docs that Bedrock KB `RetrieveAndGenerate` accepts an inference-profile ARN as `modelArn`, and EU cross-region profiles keep routing within EU regions — no need to fall back to `us-east-1`. Model ID is a Terraform variable so it's a one-line swap if needed later.

## Architecture

```
Ingestion:
  Streamlit "Upload" page --(s3:PutObject)--> S3 bucket (raw docs)
  Streamlit "Upload" page --[Sync button]--> bedrock-agent StartIngestionJob
    -> Bedrock KB chunks + embeds (Titan Embed v2) -> OpenSearch Serverless vector index

Querying:
  Streamlit "Query" page --(bedrock-agent-runtime.RetrieveAndGenerate)-->
    retrieves chunks from OpenSearch Serverless, generates answer via
    eu.amazon.nova-pro-v1:0 inference profile, returns answer + source citations
```

## Components

| Component | Role |
|---|---|
| `terraform/` | S3 bucket (raw documents); OpenSearch Serverless collection + encryption/network/data-access security policies; IAM role assumed by Bedrock KB; `aws_bedrockagent_knowledge_base` + `aws_bedrockagent_data_source` (S3) resources; an `aws_iam_policy` resource documenting/holding the permissions the local Streamlit app needs (not auto-attached to any principal — output its ARN with instructions to attach it to the user's own IAM identity) |
| `scripts/create_vector_index.py` | One-time script that PUTs the KNN vector index mapping to the OpenSearch Serverless collection endpoint (using `opensearch-py` + SigV4 auth). Not a native Terraform resource, so it runs as a scripted step between two `terraform apply` phases — same pattern the old `deploy.sh` used for its Docker build step. |
| `deploy.sh` (rewritten) | 1) `terraform apply` (bucket + OSS collection + security policies + IAM role) → 2) `python3 scripts/create_vector_index.py` (creates the KNN index) → 3) `terraform apply` (Bedrock KB + Data Source, now that the index exists) |
| `app.py` (Streamlit) | Sidebar page switch: **Upload** (`st.file_uploader`, multi-file → S3 upload, lists current bucket contents, "Sync to Knowledge Base" button → `StartIngestionJob` + polls `GetIngestionJob`, surfaces `failureReasons` on error) and **Query** (question box → `RetrieveAndGenerate`, shows generated answer + expandable citations/source S3 URIs; empty `citations` is shown distinctly from a caught API exception) |
| `requirements.txt` (rewritten) | `streamlit`, `boto3`, `opensearch-py`, `requests-aws4auth`, `python-dotenv` |
| `Readme.md` (rewritten) | Explains the vector-DB choice (OpenSearch Serverless vs. S3 Vectors) with reasoning, then ingestion-phase bullets and querying-phase bullets, setup/model-access prerequisites, deploy/teardown instructions |

## Ingestion trigger

Manual "Sync" button in the Streamlit Upload page (calls `StartIngestionJob`)
— no S3-event/Lambda automation. Keeps the stack simple and cheap, consistent
with "simple RAG."

## Chunking

Default fixed-size chunking (e.g. 300 tokens, 20% overlap) set on the Bedrock
Data Source's `vector_ingestion_configuration` — no need to expose as a
variable beyond a sensible default.

## Error handling

- Ingestion failures surface `GetIngestionJob`'s `failureReasons` directly in the Streamlit UI.
- Query page distinguishes "no relevant sources found" (empty citations) from an API error (caught exception).

## Testing

No automated tests — consistent with the other labs in this repo, which are
thin UI + IaC demos with none. Verification is manual: `deploy.sh` succeeds
end-to-end, upload a sample PDF/TXT, click Sync, then ask a question in the
Query page and confirm a cited answer comes back.

## Out of scope

- No Lambda/event-driven auto-sync.
- No Strands agent framework on the query path (direct `RetrieveAndGenerate` API call instead, per user's explicit choice to keep this "simple RAG").
- No metadata filtering, multi-tenancy, or access control beyond IAM.
