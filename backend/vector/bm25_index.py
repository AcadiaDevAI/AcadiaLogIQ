"""
Keyword retrieval using BM25.
"""

from rank_bm25 import BM25Okapi


class BM25Index:

    def __init__(self, documents):

        tokenized = [d.split() for d in documents]

        self.index = BM25Okapi(tokenized)

    def search(self, query):

        return self.index.get_top_n(query.split(), documents, n=5)