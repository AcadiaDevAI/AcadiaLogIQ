"""
Handles document database operations.
"""

from backend.domain.models.document import Document


class DocumentRepository:

    def __init__(self, db):
        self.db = db

    def create(self, name, fingerprint):

        doc = Document(name=name, fingerprint=fingerprint)

        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        return doc