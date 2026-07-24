"""
embeddings.py — helpers around vectorisation (part of Stage 3: RETRIEVAL).
=========================================================================

WHY
---
Retrieval by meaning needs a space where "distance = dissimilarity". Embeddings
give us that: a model maps text to a vector so that semantically similar texts
land near each other. Everything about dense retrieval — similarity search,
semantic chunking, reranking — rests on this one idea.

"What happens if this step is poorly designed?" Two silent traps: (1) comparing
vectors from DIFFERENT embedding models (their spaces are incompatible — nonsense
similarities); (2) forgetting to normalise, so vector *magnitude* leaks into your
similarity when you only meant to compare *direction*. Cosine similarity fixes
(2) by construction; this module makes both explicit.

WHAT / HOW
----------
Thin utilities on top of a `config.Embedder`: batch embedding, L2 normalisation,
and cosine similarity (single pair and matrix form). Kept separate from the
vector store so the maths is inspectable in the notebook.
"""
from __future__ import annotations

from typing import List

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot product == cosine similarity (float64, NaN-safe)."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = vectors / (norms + 1e-9)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity of one query vector against every row of `matrix`.
    Returns a 1-D array of scores aligned to matrix rows — the core operation
    of dense retrieval, written out so learners can see there is no magic.
    """
    q = normalize(np.asarray(query, dtype=np.float64).reshape(1, -1))
    m = normalize(np.asarray(matrix, dtype=np.float64))
    with np.errstate(all="ignore"):
        scores = (m @ q.T).ravel()
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def embed_documents(embedder, docs) -> np.ndarray:
    """Embed a list of Documents, returning an (n_docs, dim) array."""
    return embedder.embed([d.page_content for d in docs])


def embed_query(embedder, text: str) -> np.ndarray:
    """Embed a single query string, returning a 1-D vector."""
    return embedder.embed([text])[0]
