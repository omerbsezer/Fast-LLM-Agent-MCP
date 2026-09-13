import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    return {
        "region": os.environ["AWS_REGION"],
        "docs_bucket": os.environ["DOCS_BUCKET_NAME"],
        "knowledge_base_id": os.environ["KNOWLEDGE_BASE_ID"],
        "data_source_id": os.environ["DATA_SOURCE_ID"],
        "generation_model_arn": os.environ["GENERATION_MODEL_ARN"],
    }


def get_s3_client():
    return boto3.client("s3", region_name=get_config()["region"])


def get_bedrock_agent_client():
    return boto3.client("bedrock-agent", region_name=get_config()["region"])


def get_bedrock_agent_runtime_client():
    return boto3.client("bedrock-agent-runtime", region_name=get_config()["region"])
