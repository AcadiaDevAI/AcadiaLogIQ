from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


def _make_bedrock_runtime():
    boto_cfg = BotoConfig(
        retries={"max_attempts": 10, "mode": "adaptive"},
        read_timeout=120,
        connect_timeout=30,
        tcp_keepalive=True,
    )

    kwargs = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
        "config": boto_cfg,
    }

    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

    return boto3.client(**kwargs)


class BedrockHaikuClient:
    def __init__(self):
        self.client = _make_bedrock_runtime()

    def invoke_json(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        max_tokens = max_tokens or settings.HAIKU_MAX_TOKENS

        for attempt in range(1, settings.MAX_METADATA_RETRIES + 1):
            try:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": system,
                    "max_tokens": max_tokens,
                    "temperature": settings.HAIKU_TEMPERATURE,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}],
                        }
                    ],
                }

                response = self.client.invoke_model(
                    modelId=settings.BEDROCK_HAIKU_MODEL,
                    body=json.dumps(body).encode("utf-8"),
                    accept="application/json",
                    contentType="application/json",
                )

                payload = json.loads(response["body"].read().decode("utf-8"))
                content = payload.get("content", [])

                raw = "\n".join(
                    item.get("text", "")
                    for item in content
                    if item.get("type") == "text"
                ).strip()

                if not raw:
                    logger.warning("Haiku returned empty response text")
                    return None

                if raw.startswith("```"):
                    raw = raw.replace("```json", "").replace("```", "").strip()

                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    raw = raw[start:end + 1]

                try:
                    return json.loads(raw)
                except json.JSONDecodeError as json_exc:
                    logger.warning(
                        "Haiku returned invalid JSON: %s | raw=%s",
                        json_exc,
                        raw[:1200],
                    )
                    return None

            except Exception as exc:
                logger.warning(
                    "Haiku JSON invoke failed attempt %s/%s: %s",
                    attempt,
                    settings.MAX_METADATA_RETRIES,
                    exc,
                )
                if attempt >= settings.MAX_METADATA_RETRIES:
                    return None
                time.sleep(0.8 * attempt)


haiku_client = BedrockHaikuClient()