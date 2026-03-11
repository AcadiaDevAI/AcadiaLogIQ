"""
S3 storage implementation.

This is ready for Phase 1 storage abstraction even if you keep local storage as
the active provider for now.
"""

from pathlib import Path
from typing import Optional

import boto3

from backend.config import settings
from backend.storage.base import StorageProvider


class S3StorageProvider(StorageProvider):
    def __init__(self, bucket_name: str, prefix: str = "uploads"):
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", region_name=settings.AWS_REGION)

    def save_bytes(self, relative_name: str, content: bytes) -> str:
        key = f"{self.prefix}/{relative_name}"
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=content)
        return f"s3://{self.bucket_name}/{key}"

    def delete(self, storage_uri: str) -> None:
        if not storage_uri.startswith("s3://"):
            return
        path = storage_uri.replace("s3://", "", 1)
        bucket, key = path.split("/", 1)
        self.client.delete_object(Bucket=bucket, Key=key)

    def resolve_local_path(self, storage_uri: str) -> Optional[Path]:
        # S3 objects are remote, so we return None.
        return None