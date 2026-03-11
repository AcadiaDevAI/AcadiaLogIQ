"""
File storage abstraction.
Local now, S3 later.
"""

from pathlib import Path
from backend.config import settings


class LocalStorage:

    def __init__(self):

        self.base = Path(settings.LOCAL_STORAGE_PATH)

        self.base.mkdir(exist_ok=True, parents=True)

    def save(self, file):

        path = self.base / file.filename

        with open(path, "wb") as f:
            f.write(file.file.read())

        return str(path)