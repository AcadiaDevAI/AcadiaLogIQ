"""S3 lookup for the Escalation Procedures KB.

When ``STORAGE_TYPE=s3`` the consolidated PDF was uploaded via the
``/upload/presign`` browser-direct pipeline, which lands the file at
``{S3_KEY_PREFIX}/{tenant_slug}/{filename}``. We search there first,
then fall back to a bucket-prefix scan so the file is still found when
the bootstrap fires on behalf of a user other than the one who
originally uploaded it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from backend.config import settings
from backend.storage import get_s3_upload_provider
from backend.uploads.keys import derive_tenant_slug

from .sections import KB_FILENAME


logger = logging.getLogger("acadia-log-iq")


def _name_matches_kb(name: str) -> bool:
    target = KB_FILENAME.lower()
    candidate = (name or "").lower()
    if not candidate:
        return False
    if candidate == target or candidate.endswith("_" + target):
        return True
    if candidate.endswith(target):
        return True
    normalised_target = target.replace("_", " ").replace("-", " ")
    normalised_candidate = candidate.replace("_", " ").replace("-", " ")
    return normalised_candidate.endswith(normalised_target)


def _list_under_prefix(s3, prefix: str, *, max_keys: int = 1000) -> List[dict]:
    """Return ``[{Key, LastModified, Size}, ...]`` for every object below ``prefix``.

    Paginates so a deep tenant prefix doesn't truncate at 1000 objects.
    Empty/error returns an empty list so the caller can fall through.
    """
    objects: List[dict] = []
    continuation = None
    try:
        while True:
            kwargs = {
                "Bucket": s3.bucket_name,
                "Prefix": prefix,
                "MaxKeys": max_keys,
            }
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = s3.client.list_objects_v2(**kwargs)
            for item in resp.get("Contents", []):
                key = item.get("Key", "")
                if not key or key.endswith("/"):
                    continue
                objects.append({
                    "Key": key,
                    "LastModified": item.get("LastModified"),
                    "Size": int(item.get("Size", 0)),
                })
            if not resp.get("IsTruncated"):
                break
            continuation = resp.get("NextContinuationToken")
            if not continuation:
                break
    except Exception as exc:
        logger.warning(
            "[escalation.s3] list_objects_v2 failed prefix=%s err=%s",
            prefix, exc,
        )
    return objects


def find_in_s3(user_id: Optional[str] = None) -> Optional[Tuple[bytes, str]]:
    """Locate the consolidated KB PDF in S3 and return ``(bytes, filename)``.

    Search order:
      1. The current user's tenant prefix (``{S3_KEY_PREFIX}/{slug}/``)
         when ``user_id`` is supplied — covers the common case where the
         person opening the modal is also the uploader.
      2. The whole ``{S3_KEY_PREFIX}/`` prefix — covers the case where
         an admin uploaded under a different account.

    Returns ``None`` when S3 is not configured, the bucket scan fails,
    or no key in the bucket matches the KB filename pattern. The caller
    treats ``None`` the same as "no document on disk either" and reports
    "No document is uploaded." to the frontend.
    """
    s3 = get_s3_upload_provider()
    if s3 is None:
        return None

    root_prefix = (
        getattr(settings, "S3_KEY_PREFIX", "tenants") or "tenants"
    ).strip("/")

    search_prefixes: List[str] = []
    if user_id:
        try:
            slug = derive_tenant_slug(user_id)
        except Exception as exc:
            logger.warning("[escalation.s3] tenant_slug lookup failed: %s", exc)
            slug = None
        if slug:
            search_prefixes.append(f"{root_prefix}/{slug}/")
    # Bucket-wide fallback (still scoped to the tenants prefix so we
    # don't sweep every key in the bucket).
    search_prefixes.append(f"{root_prefix}/")

    seen_keys = set()
    matches: List[dict] = []
    for prefix in search_prefixes:
        for item in _list_under_prefix(s3, prefix):
            key = item["Key"]
            if key in seen_keys:
                continue
            seen_keys.add(key)
            filename = key.rsplit("/", 1)[-1]
            if _name_matches_kb(filename):
                matches.append({"key": key, "last_modified": item.get("LastModified")})
        # If a per-user prefix already produced a hit we don't need to
        # widen the scan; saves an extra ListObjects call against the
        # whole tenant root.
        if matches:
            break

    if not matches:
        logger.info(
            "[escalation.s3] no KB match in bucket=%s prefix=%s "
            "(searched %d prefixes, %d distinct keys)",
            s3.bucket_name, root_prefix,
            len(search_prefixes), len(seen_keys),
        )
        return None

    matches.sort(key=lambda m: m["last_modified"] or 0, reverse=True)
    winner = matches[0]
    try:
        body = s3.read_bytes(s3.make_storage_uri(winner["key"]))
    except Exception as exc:
        logger.warning(
            "[escalation.s3] download failed key=%s err=%s",
            winner["key"], exc,
        )
        return None

    filename = winner["key"].rsplit("/", 1)[-1]
    logger.info(
        "[escalation.s3] found KB in bucket=%s key=%s (%d bytes)",
        s3.bucket_name, winner["key"], len(body),
    )
    return body, filename
