"""
Handles Titan embedding calls via Bedrock.
"""

import boto3
from backend.config import settings
from backend.services.token_usage import record_token_usage, extract_bedrock_usage


client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)


def embed(text):

    body = {
        "inputText": text
    }

    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=str(body)
    )
    record_token_usage("embeddings", settings.BEDROCK_EMBED_MODEL, *extract_bedrock_usage(None, response))

    return response