"""
ServiceNow connector — optional integration for the Ticket Filter.

Self-contained module. Reads credentials from ``backend/env.bvk``
(lazily, on first call), calls the ServiceNow Table REST API with
HTTP Basic auth, requests the response as XML per the integration
brief, parses it into a JSON-safe dict, and returns it to the route
layer.

To migrate to AWS Secrets Manager later: add the three keys
``SERVICE_NOW_INSTANCE_URL`` / ``SERVICE_NOW_USERNAME`` /
``SERVICE_NOW_PASSWORD`` to the ASM secret JSON. The same
``os.environ`` lookup picks them up without any code change here.

Why stdlib only (urllib + xml.etree)
------------------------------------
``requests`` is NOT in the backend's pinned requirements. Sticking to
the stdlib avoids adding a new dependency for a single integration.

Failure modes — always surface, never swallow
---------------------------------------------
* Missing env vars            → ``ServiceNowConfigError`` (503 at the route)
* Network failure / timeout   → ``ServiceNowAPIError``    (502 at the route)
* HTTP 4xx/5xx from ServiceNow → ``ServiceNowAPIError``   (502 at the route)
* Unparseable XML             → ``ServiceNowAPIError``    (502 at the route)
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict
from xml.etree import ElementTree as ET

logger = logging.getLogger("acadia-log-iq")


# HTTP timeout in seconds. ServiceNow Table API typically responds
# in <2 s; capping aggressively means a hung integration can't pin
# an API worker thread.
_TIMEOUT_SECONDS = 10

# One-shot guard so env.bvk is read at most once per process.
_ENV_BVK_LOADED = False


def _load_env_bvk() -> None:
    """Idempotently load ``backend/env.bvk`` into ``os.environ``.

    Only fills gaps — never overrides a key that's already present.
    That way operator-exported vars and AWS Secrets Manager values
    keep winning over the local file (same precedence rule the core
    ``bootstrap_environment`` uses for ``.env``).
    """
    global _ENV_BVK_LOADED
    if _ENV_BVK_LOADED:
        return
    _ENV_BVK_LOADED = True

    # servicenow.py → ticket_filter → tier1_copilot → backend
    env_file = Path(__file__).resolve().parent.parent.parent / "env.bvk"
    if not env_file.exists():
        logger.debug(
            "[servicenow] env.bvk not found at %s — relying on os.environ only",
            env_file,
        )
        return

    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present (matches python-dotenv).
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        logger.info("[servicenow] loaded credentials from %s", env_file.name)
    except OSError as exc:
        logger.warning("[servicenow] could not read env.bvk: %s", exc)


class ServiceNowConfigError(Exception):
    """Required env vars are missing — operator misconfiguration."""


class ServiceNowAPIError(Exception):
    """ServiceNow was unreachable or returned an unexpected response."""


def _xml_element_to_dict(element: ET.Element) -> Any:
    """Recursive ElementTree → JSON-safe value.

    Rules:
      * Leaf with text  → string value
      * Leaf with empty → empty string
      * Parent          → dict of {child_tag: value}
      * Repeated tags   → list of values
      * Attributes      → "@attributes" sub-dict (rare for SN responses)
    """
    result: Dict[str, Any] = {}
    if element.attrib:
        result["@attributes"] = dict(element.attrib)

    children = list(element)
    if not children:
        text = (element.text or "").strip()
        if not result:
            return text
        result["#text"] = text
        return result

    for child in children:
        child_value = _xml_element_to_dict(child)
        if child.tag in result:
            existing = result[child.tag]
            if isinstance(existing, list):
                existing.append(child_value)
            else:
                result[child.tag] = [existing, child_value]
        else:
            result[child.tag] = child_value

    return result


def fetch_priority_1_incidents() -> Dict[str, Any]:
    """Hit ``GET /api/now/table/incident?priority=1`` and return parsed JSON.

    Returns a stable envelope so the frontend renders the same shape
    regardless of how many incidents ServiceNow ships back::

        {
            "source":       "servicenow",
            "instance_url": "https://dev320900.service-now.com",
            "query":        "priority=1",
            "count":        <int>,
            "incidents":    [ <dict>, ... ],   # always a list
            "raw":          <full nested dict>,
        }

    Raises
    ------
    ServiceNowConfigError
        SERVICE_NOW_* env vars not configured.
    ServiceNowAPIError
        Network failure, non-2xx response, or unparseable XML.
    """
    _load_env_bvk()

    instance_url = (os.environ.get("SERVICE_NOW_INSTANCE_URL") or "").strip().rstrip("/")
    username = (os.environ.get("SERVICE_NOW_USERNAME") or "").strip()
    password = os.environ.get("SERVICE_NOW_PASSWORD") or ""

    if not instance_url or not username or not password:
        raise ServiceNowConfigError(
            "ServiceNow credentials are not configured. Set "
            "SERVICE_NOW_INSTANCE_URL, SERVICE_NOW_USERNAME, and "
            "SERVICE_NOW_PASSWORD in backend/env.bvk or AWS Secrets Manager."
        )

    url = f"{instance_url}/api/now/table/incident?priority=1"

    # HTTP Basic auth per the integration brief. ServiceNow also
    # supports OAuth — switch when moving off the developer instance.
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    headers = {
        "Accept": "application/xml",
        "Authorization": f"Basic {token}",
        "User-Agent": "Acadia-LogIQ/1.0",
    }

    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read()
    except urllib.error.HTTPError as exc:
        # Read body for diagnostics but DO NOT log the Authorization header.
        snippet = ""
        try:
            snippet = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        logger.warning(
            "[servicenow] HTTP %s from %s: %s", exc.code, instance_url, snippet,
        )
        raise ServiceNowAPIError(
            f"ServiceNow returned HTTP {exc.code}. "
            "Check credentials and instance URL."
        ) from exc
    except (urllib.error.URLError, socket.timeout, ssl.SSLError) as exc:
        logger.warning(
            "[servicenow] network error contacting %s: %s", instance_url, exc,
        )
        raise ServiceNowAPIError(
            f"Could not reach ServiceNow at {instance_url}. {exc}"
        ) from exc

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning(
            "[servicenow] XML parse failed (status=%s len=%d): %s",
            status, len(body or b""), exc,
        )
        raise ServiceNowAPIError("ServiceNow response was not valid XML.") from exc

    parsed = _xml_element_to_dict(root)

    # Normalise the <result> wrapper into a flat list so the frontend
    # never has to branch on "single vs many" incidents.
    items: Any = []
    if isinstance(parsed, dict) and "result" in parsed:
        result_value = parsed["result"]
        if isinstance(result_value, list):
            items = result_value
        elif isinstance(result_value, dict) and result_value:
            items = [result_value]
        else:
            items = []

    logger.info(
        "[servicenow] fetched %d priority-1 incidents (http_status=%s)",
        len(items) if isinstance(items, list) else 0, status,
    )

    return {
        "source": "servicenow",
        "instance_url": instance_url,
        "query": "priority=1",
        "count": len(items) if isinstance(items, list) else 0,
        "incidents": items if isinstance(items, list) else [],
        "raw": parsed,
    }
