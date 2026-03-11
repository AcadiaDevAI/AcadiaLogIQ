"""
Retrieves relevant chunks using pgvector similarity search.
"""

from sqlalchemy import text


class RetrievalService:

    def __init__(self, db):
        self.db = db

    def search(self, embedding, k=5):

        sql = text("""
        SELECT content
        FROM chunks
        ORDER BY embedding <-> :embedding
        LIMIT :k
        """)

        result = self.db.execute(sql, {"embedding": embedding, "k": k})

        return result.fetchall()