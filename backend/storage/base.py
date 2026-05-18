"""
Storage abstraction for raw uploaded files.

This allows local filesystem now and S3 later without changing upload logic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class StorageProvider(ABC):
    @abstractmethod
    def save_bytes(self, relative_name: str, content: bytes) -> str:
        """
        Save file bytes and return a storage URI/path.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_uri: str) -> None:
        """
        Delete a previously stored file.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_local_path(self, storage_uri: str) -> Optional[Path]:
        """
        Return a readable local path when applicable.
        S3 provider returns None because the file is remote.
        """
        raise NotImplementedError

    def read_bytes(self, storage_uri: str) -> bytes:
        """
        Read raw bytes for a previously stored object.

        Default implementation resolves to a local Path; remote backends
        (S3) override this. Subclasses MUST override when the object is
        not addressable as a local file.
        """
        path = self.resolve_local_path(storage_uri)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"Cannot read bytes — storage URI not resolvable locally: {storage_uri}"
            )
        return path.read_bytes()
