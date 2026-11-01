import json
import os
import re
import socket
import sys

import boto3
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

load_dotenv()

MAX_RETRIEVALS = 3
# Vector search over-fetches candidates; the reranker keeps the best few.
# Must match app/utils/bedrock_ops.py so Simple vs Agentic is a fair comparison.
NUM_CANDIDATES = 20
NUM_RESULTS_PER_RETRIEVAL = 5

DEFAULT_QUESTION = "What is retrieval augmented generation?"


def _config() -> dict:
    return {
        "region": os.environ["AWS_REGION"],
        "knowledge_base_id": os.environ["KNOWLEDGE_BASE_ID"],
        "model_id": os.environ["GENERATION_MODEL_ID"],
        # Optional: leave empty to disable reranking.
        "rerank_model_arn": os.environ.get("RERANK_MODEL_ARN", ""),
    }


def _kb_client():
    return boto3.client("bedrock-agent-runtime", region_name=_config()["region"])


def _vector_search_configuration() -> dict:
    rerank_model_arn = _config()["rerank_model_arn"]
    if not rerank_model_arn:
        return {"numberOfResults": NUM_RESULTS_PER_RETRIEVAL}
    return {
        "numberOfResults": NUM_CANDIDATES,
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": rerank_model_arn},
                "numberOfRerankedResults": NUM_RESULTS_PER_RETRIEVAL,
            },
        },
    }


def _retrieve(query: str) -> list[dict]:
    resp = _kb_client().retrieve(
        knowledgeBaseId=_config()["knowledge_base_id"],
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": _vector_search_configuration()},
    )
    results = []
    for r in resp.get("retrievalResults", []):
        uri = r.get("location", {}).get("s3Location", {}).get("uri")
        text = r.get("content", {}).get("text", "")
        results.append({"text": text, "uri": uri, "score": r.get("score")})
    return results


def _make_tools(state: dict) -> list:
    """Build the agent's tools bound to one request's state, so concurrent
    invocations in the same AgentCore container never share chunks/trace."""

    @tool
    def retrieve_kb(query: str) -> str:
        """Retrieve relevant chunks from the knowledge base for one specific query.

        Call this once per distinct sub-topic of a compound question, and again
        with a reformulated query if a previous retrieval looks thin or
        off-target. You may call this up to 3 times total per question."""
        num_calls = sum(1 for t in state["trace"] if t["type"] == "retrieve")
        if num_calls >= MAX_RETRIEVALS:
            return "Retrieval budget exhausted. Answer now using what you have already retrieved."

        results = _retrieve(query)
        seen = {c["text"] for c in state["chunks"]}
        new_results = [r for r in results if r["text"] not in seen]
        state["chunks"].extend(new_results)
        state["trace"].append(
            {
                "type": "retrieve",
                "query": query,
                "num_chunks": len(results),
                "num_new_chunks": len(new_results),
                "reranked": bool(_config()["rerank_model_arn"]),
                "sources": sorted({r["uri"] for r in results if r["uri"]}),
            }
        )
        if not results:
            return f"No chunks found for query: {query!r}"
        # Return full chunk text: the final answer is written from this tool
        # output, so truncating here would starve the answer of content.
        body = "\n\n".join(f"[Source: {r['uri']}]\n{r['text']}" for r in results)
        return f"Retrieved {len(results)} chunks for {query!r}:\n\n{body}"

    @tool
    def critique_context(question: str) -> str:
        """Assess whether everything retrieved so far is sufficient to fully
        answer the given question. Returns JSON: {"sufficient": bool,
        "missing_aspects": [str, ...]}. Call this after each retrieval, before
        deciding whether to retrieve again."""
        context = "\n\n".join(f"[{c['uri']}]\n{c['text']}" for c in state["chunks"])
        context = context or "(nothing retrieved yet)"

        critic = Agent(
            model=BedrockModel(
                model_id=_config()["model_id"], temperature=0.0, region_name=_config()["region"]
            ),
            system_prompt=(
                "You are a strict retrieval critic. Given a question and the context "
                "retrieved so far, decide if the context is sufficient to fully and "
                "accurately answer the question. Respond ONLY with JSON: "
                '{"sufficient": true|false, "missing_aspects": ["..."]}. No prose, no markdown fences.'
            ),
        )
        raw = str(critic(f"Question: {question}\n\nRetrieved context:\n{context}")).strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        try:
            verdict = json.loads(raw)
        except Exception:
            # Fail closed: an unreadable verdict is not evidence of sufficiency.
            # The retrieval budget in retrieve_kb still bounds the loop.
            verdict = {"sufficient": False, "missing_aspects": ["critic output could not be parsed"]}

        state["trace"].append({"type": "critique", **verdict})
        return json.dumps(verdict)

    return [retrieve_kb, critique_context]


def run(question: str) -> dict:
    state: dict = {"question": question, "trace": [], "chunks": []}

    agent = Agent(
        model=BedrockModel(
            model_id=_config()["model_id"], temperature=0.2, region_name=_config()["region"]
        ),
        tools=_make_tools(state),
        system_prompt=(
            "You are an agentic RAG assistant. Answer the user's question ONLY "
            "using information retrieved via retrieve_kb.\n\n"
            "Process:\n"
            "1. Call retrieve_kb with a focused query. For compound questions "
            "covering multiple topics, call it once per topic.\n"
            "2. Call critique_context(question) to check if you have enough to "
            "answer fully.\n"
            "3. If not sufficient and you still have retrieval budget, "
            "reformulate the query (or target a missing aspect) and retrieve "
            "again, then critique again.\n"
            "4. Once sufficient, or your retrieval budget is exhausted "
            f"(max {MAX_RETRIEVALS} retrieve_kb calls total), write the final "
            "answer using only retrieved content, citing sources by URI. If "
            "nothing relevant was found, say so plainly instead of guessing."
        ),
    )

    answer = str(agent(question)).strip()
    sources = sorted({c["uri"] for c in state["chunks"] if c["uri"]})
    return {"answer": answer, "citations": sources, "trace": state["trace"]}


if __name__ == "__main__":
    if "--serve" in sys.argv:
        # AWS Bedrock AgentCore Runtime entrypoint: serves POST /invocations on :8080.
        # Deployed via the Dockerfile + Terraform in ../terraform (see ../README.md).
        from bedrock_agentcore.runtime import BedrockAgentCoreApp

        app = BedrockAgentCoreApp()

        @app.entrypoint
        def _invoke(payload: dict, context=None) -> dict:
            question = payload.get("question") or payload.get("prompt") or DEFAULT_QUESTION
            result = run(question)
            # Echoed from inside the AgentCore container, so the app can show
            # that this answer really came from the deployed runtime.
            result["runtime"] = {
                "session_id": getattr(context, "session_id", None),
                "host": socket.gethostname(),
            }
            return result

        app.run()
    else:
        result = run(" ".join(sys.argv[1:]) or DEFAULT_QUESTION)
        print(json.dumps(result, indent=2))
