"""
Handles Titan embedding calls via Bedrock.
"""

import boto3
from backend.config import settings


client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)


def embed(text):

    body = {
        "inputText": text
    }

    response = client.invoke_model(
        modelId=settings.BEDROCK_EMBED_MODEL,
        body=str(body)
    )

    return response