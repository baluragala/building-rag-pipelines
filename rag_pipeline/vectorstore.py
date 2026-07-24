"""
vectorstore.py — a transparent, from-scratch vector store (Stage 3: RETRIEVAL).
==============================================================================

WHY
---
A "vector store" sounds like heavy infrastructure (FAISS, Chroma, Pinecone…).
Conceptually it is just: hold vectors + their documents, and given a query
vector return the nearest ones. We implement exactly that in ~40 lines of numpy
so learners see there is no magic — then point at where a real ANN index would
slot in for scale. Understanding the store from scratch is what lets you reason
about *why* a production store behaves the way it does.

WHAT / HOW
----------
`InMemoryVectorStore`:
  * add(docs, embeddings)          — store aligned docs + vectors
  * search(query_emb, k, filter)   — cosine top-k, with optional metadata filter
  * save/load                      — pickle, so a notebook can persist an index

Metadata filtering is first-class here because it's a retrieval technique on the
agenda ("apply metadata filtering") and a cheap, high-leverage quality lever:
constrain the search space *before* similarity ranking.
"""
from __future__ import annotations

import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .embeddings import cosine_similarity_matrix
from .loaders import Document


class InMemoryVectorStore:
    """A minimal cosine-similarity vector store backed by a numpy matrix."""

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None  # (n_docs, dim)

    # -- indexing -----------------------------------------------------------
    def add(self, docs: List[Document], embeddings: np.ndarray) -> "InMemoryVectorStore":
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must be the same length")
        self.documents.extend(docs)
        self.embeddings = (
            embeddings
            if self.embeddings is None
            else np.vstack([self.embeddings, embeddings])
        )
        return self

    def __len__(self) -> int:
        return len(self.documents)

    # -- search -------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        filter_fn: Optional[Callable[[Document], bool]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Return the top-`k` (Document, score) pairs by cosine similarity.

        Optional filtering restricts the candidate set BEFORE ranking:
          * metadata_filter: {"source": "...", "page": 3} — exact-match on keys
          * filter_fn:       a predicate for arbitrary logic (dates, ranges…)
        This is the "apply metadata filtering / optimise recall then precision"
        idea made concrete.
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        scores = cosine_similarity_matrix(query_embedding, self.embeddings)

        # Build the candidate index set (all, unless a filter narrows it).
        candidates = range(len(self.documents))
        if metadata_filter or filter_fn:
            candidates = [
                i
                for i in candidates
                if _matches(self.documents[i], metadata_filter, filter_fn)
            ]
            if not candidates:
                return []

        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], float(scores[i])) for i in ranked]

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: str) -> "InMemoryVectorStore":
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.documents = data["documents"]
        store.embeddings = data["embeddings"]
        return store


def _matches(doc: Document, metadata_filter, filter_fn) -> bool:
    if metadata_filter:
        for key, value in metadata_filter.items():
            if doc.metadata.get(key) != value:
                return False
    if filter_fn and not filter_fn(doc):
        return False
    return True
