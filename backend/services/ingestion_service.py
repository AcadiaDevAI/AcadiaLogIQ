"""
Handles document ingestion pipeline:
extract → chunk → embed → store.
"""

from backend.repositories.document_repository import DocumentRepository
from backend.repositories.chunk_repository import ChunkRepository
from backend.services.embedding_service import embed


class IngestionService:

    def __init__(self, db):

        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)

    def ingest(self, name, chunks):

        doc = self.doc_repo.create(name, None)

        chunk_objs = []

        for i, text in enumerate(chunks):

            embedding = embed(text)

            chunk_objs.append(
                Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=text,
                    embedding=embedding
                )
            )

        self.chunk_repo.insert_chunks(chunk_objs)

        return doc