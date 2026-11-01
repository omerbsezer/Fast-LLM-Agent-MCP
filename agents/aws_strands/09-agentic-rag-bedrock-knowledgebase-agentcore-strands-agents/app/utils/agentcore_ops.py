import json
import time

from utils.aws_clients import get_bedrock_agentcore_client, get_config


def is_agentic_configured() -> bool:
    # A non-ARN value (e.g. "True") would be sent as an agent ID and rejected by AWS.
    return (get_config()["agent_runtime_arn"] or "").startswith("arn:aws:bedrock-agentcore:")


def query_agentic(question: str, session_id: str) -> dict:
    cfg = get_config()
    if not is_agentic_configured():
        raise RuntimeError(
            "AGENT_RUNTIME_ARN is not a Bedrock AgentCore runtime ARN. Deploy the agent "
            "with ./deploy.sh; it writes the ARN into app/.env."
        )

    # The same runtimeSessionId for every question in one browser session, so
    # AgentCore Observability groups them as one session.
    started = time.monotonic()
    resp = get_bedrock_agentcore_client().invoke_agent_runtime(
        agentRuntimeArn=cfg["agent_runtime_arn"],
        runtimeSessionId=session_id,
        qualifier="DEFAULT",
        payload=json.dumps({"question": question}).encode(),
    )
    result = json.loads(resp["response"].read())
    # Proof of execution on AgentCore: IDs returned by the AWS API, plus what
    # the agent reported from inside its container.
    result["agentcore"] = {
        "runtime_arn": cfg["agent_runtime_arn"],
        "session_id": resp.get("runtimeSessionId", session_id),
        "trace_id": resp.get("traceId"),
        "request_id": resp["ResponseMetadata"]["RequestId"],
        "status_code": resp.get("statusCode"),
        "latency_s": round(time.monotonic() - started, 1),
        "container": result.pop("runtime", None),
        "console_url": (
            f"https://{cfg['region']}.console.aws.amazon.com/bedrock-agentcore/observability"
            f"?region={cfg['region']}"
        ),
    }
    return result
