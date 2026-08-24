"""Invoke a deployed AgentCore Runtime via boto3.

Usage:
    export AGENT_RUNTIME_ARN=$(terraform -chdir=terraform output -raw agent_runtime_arn)
    
    python3 invoke_client.py "My blog topic" --runtime-arn arn:aws:bedrock-agentcore:...
    
    python3 invoke_client.py --read-memory <session_id>   # inspect a prior run's memory
    
"""
import argparse
import json
import os
import sys

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*", default=["AI Agents with Memory on AWS Bedrock AgentCore"])
    parser.add_argument("--read-memory", metavar="SESSION_ID", help="Read AgentCore Memory for a prior run's session_id instead of running the pipeline")
    parser.add_argument("--runtime-arn", default=os.getenv("AGENT_RUNTIME_ARN"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "eu-central-1"))
    parser.add_argument("--max-iter", type=int, default=3)
    args = parser.parse_args()

    if not args.runtime_arn:
        sys.exit("Set --runtime-arn or AGENT_RUNTIME_ARN (see: terraform output agent_runtime_arn)")

    if args.read_memory:
        payload = {"action": "read_memory", "session_id": args.read_memory}
    else:
        payload = {"topic": " ".join(args.topic), "max_iter": args.max_iter}

    client = boto3.client("bedrock-agentcore", region_name=args.region)
    response = client.invoke_agent_runtime(
        agentRuntimeArn=args.runtime_arn,
        qualifier="DEFAULT",
        payload=json.dumps(payload).encode(),
    )
    body = json.loads(response["response"].read())
    print(body["memory"] if args.read_memory and "memory" in body else json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
