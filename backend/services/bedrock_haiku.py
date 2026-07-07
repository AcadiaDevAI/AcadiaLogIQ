from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig

from backend.config import settings
from backend.services.token_usage import (
    record_token_usage,
    get_usage_totals,      # re-exported for backward-compat callers
    reset_usage_totals,    # re-exported for backward-compat callers
)

logger = logging.getLogger("acadia-log-iq")


def _record_usage(usage: Dict[str, Any], context: str) -> None:
    """Adapter: every Bedrock invoke_model response carries a `usage` block
    with the exact input/output tokens billed. Hand it to the central
    per-org token-usage service, which logs + buffers + persists it. The
    `context` string doubles as the feature label (retry suffix stripped
    downstream). Never raises."""
    try:
        record_token_usage(
            feature=context or "haiku",
            model_id=settings.BEDROCK_HAIKU_MODEL,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
    except Exception as exc:  # pragma: no cover - accounting must never break calls
        logger.debug("[haiku.usage] failed to record usage: %s", exc)


def _make_bedrock_runtime():
    # Same tightening as the shared bedrock client in api.py — see the
    # comment there for the rationale. Per-attempt timeout and attempt
    # cap come from settings so they can be re-tuned per environment.
    boto_cfg = BotoConfig(
        retries={"max_attempts": settings.LLM_MAX_ATTEMPTS, "mode": "adaptive"},
        read_timeout=settings.LLM_READ_TIMEOUT_S,
        connect_timeout=settings.LLM_CONNECT_TIMEOUT_S,
        tcp_keepalive=True,
        # Pool sized to the concurrent Haiku/embedding fan-out + headroom so
        # workers reuse warm connections instead of churning them.
        max_pool_connections=max(
            settings.METADATA_CONCURRENCY, settings.EMBED_CONCURRENCY
        ) + 4,
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

    def _invoke_text(
        self, *, system: str, prompt: str, max_tokens: int, context: str = ""
    ) -> Tuple[str, Optional[str]]:
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
        # Capture the exact token counts Bedrock billed for this call.
        _record_usage(payload.get("usage") or {}, context)
        return raw, payload.get("stop_reason")

    def invoke_json(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        context: str = "invoke_json",
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
                    context=f"{context}#a{attempt}",
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
        context: str = "invoke_text",
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
                context=context,
            )
            return (raw or "").strip()
        except Exception as exc:
            logger.warning("Haiku invoke_text failed: %s", exc)
            return ""


haiku_client = BedrockHaikuClient()