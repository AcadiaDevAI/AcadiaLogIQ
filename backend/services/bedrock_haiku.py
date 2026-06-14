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
    # Same tightening as the shared bedrock client in api.py — see the
    # comment there for the rationale. Per-attempt timeout and attempt
    # cap come from settings so they can be re-tuned per environment.
    boto_cfg = BotoConfig(
        retries={"max_attempts": settings.LLM_MAX_ATTEMPTS, "mode": "adaptive"},
        read_timeout=settings.LLM_READ_TIMEOUT_S,
        connect_timeout=settings.LLM_CONNECT_TIMEOUT_S,
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


def _extract_json_object(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""

    if value.startswith("```"):
        value = value.replace("```json", "").replace("```", "").strip()

    start = value.find("{")
    if start < 0:
        return value

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(value)):
        ch = value[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return value[start:i + 1]

    return value[start:]


class BedrockHaikuClient:
    def __init__(self):
        self.client = _make_bedrock_runtime()

    def _invoke_text(self, *, system: str, prompt: str, max_tokens: int) -> tuple[str, Optional[str]]:
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
        return raw, payload.get("stop_reason")

    def invoke_json(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        base_max_tokens = max_tokens or settings.HAIKU_MAX_TOKENS

        for attempt in range(1, settings.MAX_METADATA_RETRIES + 1):
            try:
                token_limit = base_max_tokens
                if attempt >= 2:
                    token_limit = max(base_max_tokens, 6000)

                raw, stop_reason = self._invoke_text(
                    system=system,
                    prompt=prompt,
                    max_tokens=token_limit,
                )

                if not raw:
                    logger.warning("Haiku returned empty response text | stop_reason=%s", stop_reason)
                    continue

                cleaned = _extract_json_object(raw)

                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError as json_exc:
                    logger.warning(
                        "Haiku returned invalid JSON on attempt %s/%s: %s | stop_reason=%s | raw_len=%d",
                        attempt,
                        settings.MAX_METADATA_RETRIES,
                        json_exc,
                        stop_reason,
                        len(cleaned),
                    )

                    if stop_reason == "max_tokens":
                        time.sleep(0.3 * attempt)
                        continue

                    repaired = cleaned.strip()

                    if repaired.count("{") > repaired.count("}"):
                        repaired += "}" * (repaired.count("{") - repaired.count("}"))
                    if repaired.count("[") > repaired.count("]"):
                        repaired += "]" * (repaired.count("[") - repaired.count("]"))

                    try:
                        return json.loads(repaired)
                    except Exception:
                        pass

            except Exception as exc:
                logger.warning(
                    "Haiku JSON invoke failed attempt %s/%s: %s",
                    attempt,
                    settings.MAX_METADATA_RETRIES,
                    exc,
                )

            if attempt < settings.MAX_METADATA_RETRIES:
                time.sleep(0.3 * attempt)

        return None

    def invoke_text(
        self,
        *,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Public free-form text invoke. Companion to invoke_json for callers
        that want raw text rather than parsed JSON.

        Used by the LLM timeout-fallback guard: when the primary chat model
        (Mistral) times out or throttles, this is the entry point for the
        Haiku failover. Returns "" on any failure — the caller decides
        whether to surface a decline message.
        """
        budget = max_tokens or settings.HAIKU_MAX_TOKENS
        try:
            raw, _stop = self._invoke_text(
                system=system,
                prompt=prompt,
                max_tokens=budget,
            )
            return (raw or "").strip()
        except Exception as exc:
            logger.warning("Haiku invoke_text failed: %s", exc)
            return ""


haiku_client = BedrockHaikuClient()