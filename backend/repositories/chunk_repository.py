"""
Chunk database operations.
"""

from backend.domain.models.chunk import Chunk


class ChunkRepository:

    def __init__(self, db):
        self.db = db

    def insert_chunks(self, chunks):

        for chunk in chunks:
            self.db.add(chunk)

        self.db.commit()