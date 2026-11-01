from utils.aws_clients import get_bedrock_agent_client, get_bedrock_agent_runtime_client, get_config

# Vector search over-fetches candidates; the reranker keeps the best few.
# Must match agent/agent.py so Simple vs Agentic is a fair comparison.
NUM_CANDIDATES = 20
NUM_RESULTS = 5


def start_ingestion_job() -> str:
    cfg = get_config()
    resp = get_bedrock_agent_client().start_ingestion_job(
        knowledgeBaseId=cfg["knowledge_base_id"],
        dataSourceId=cfg["data_source_id"],
    )
    return resp["ingestionJob"]["ingestionJobId"]


def get_ingestion_job_status(job_id: str) -> dict:
    cfg = get_config()
    resp = get_bedrock_agent_client().get_ingestion_job(
        knowledgeBaseId=cfg["knowledge_base_id"],
        dataSourceId=cfg["data_source_id"],
        ingestionJobId=job_id,
    )
    job = resp["ingestionJob"]
    return {"status": job["status"], "failure_reasons": job.get("failureReasons", [])}


def _vector_search_configuration() -> dict:
    rerank_model_arn = get_config()["rerank_model_arn"]
    if not rerank_model_arn:
        return {"numberOfResults": NUM_RESULTS}
    return {
        "numberOfResults": NUM_CANDIDATES,
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": rerank_model_arn},
                "numberOfRerankedResults": NUM_RESULTS,
            },
        },
    }


def query(question: str) -> dict:
    cfg = get_config()
    resp = get_bedrock_agent_runtime_client().retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": cfg["knowledge_base_id"],
                "modelArn": cfg["generation_model_arn"],
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": _vector_search_configuration(),
                },
            },
        },
    )

    citations = []
    for citation in resp.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            uri = ref.get("location", {}).get("s3Location", {}).get("uri")
            if uri:
                citations.append(uri)

    return {"answer": resp["output"]["text"], "citations": citations}
