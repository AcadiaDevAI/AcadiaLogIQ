"""
Local filesystem storage implementation.
"""

from pathlib import Path
from urllib.parse import quote, unquote

from backend.config import settings
from backend.storage.base import StorageProvider


class LocalStorageProvider(StorageProvider):
    def __init__(self, root: Path | None = None):
        self.root = root or settings.UPLOAD_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, relative_name: str, content: bytes) -> str:
        path = self.root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return f"local://{quote(str(path))}"

    def delete(self, storage_uri: str) -> None:
        path = self.resolve_local_path(storage_uri)
        if path and path.exists():
            path.unlink(missing_ok=True)

    def resolve_local_path(self, storage_uri: str):
        if not storage_uri or not storage_uri.startswith("local://"):
            return None
        raw = storage_uri.replace("local://", "", 1)
        return Path(unquote(raw))