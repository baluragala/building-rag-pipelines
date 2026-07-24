"""
retrieval.py — Stage 3: RETRIEVAL (dense, sparse, hybrid, filtering, reranking).
================================================================================

WHY
---
Retrieval decides *what the model gets to see*. Generation cannot fix a retrieval
miss — if the relevant chunk never makes it into the context, no prompt on earth
recovers it. "What happens if this step is poorly designed?" You get the classic
failure trio: relevant chunk not retrieved (recall miss), irrelevant chunks
crowd it out (precision miss), or the right chunk is present but ranked too low
to survive the top-k cut.

The agenda's guidance is the mental model: **optimise recall first, then
precision.** Cast a wide net (hybrid retrieval → high recall), then tighten
(rerank → high precision). And the anti-pattern to avoid: **don't rely only on
top-k dense similarity** — dense retrieval misses exact terms (names, codes,
acronyms) that lexical search nails.

WHAT (techniques)
-----------------
* DenseRetriever      — embed query, cosine top-k over the vector store.
* BM25Retriever       — classic lexical/sparse ranking (exact-term matching).
* HybridRetriever     — fuse dense + BM25 with Reciprocal Rank Fusion (RRF).
* CrossEncoderReranker— re-score a candidate set with a cross-encoder (query and
                        doc read TOGETHER — far more accurate than bi-encoder
                        similarity, but too slow to run over the whole corpus).
* LLMReranker         — pointwise relevance scoring by an LLM (no extra model).

HOW
---
Every retriever exposes the same `.retrieve(query, k) -> List[(Document, score)]`
so they're swappable. Rerankers take a candidate list and return a re-ordered,
trimmed list. This uniformity is what lets the notebook A/B strategies cleanly.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

from .embeddings import embed_query
from .loaders import Document
from .vectorstore import InMemoryVectorStore

Scored = Tuple[Document, float]


# ===========================================================================
# Dense retrieval
# ===========================================================================
class DenseRetriever:
    """Embed the query and take cosine top-k from the vector store."""

    def __init__(self, store: InMemoryVectorStore, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 4, **search_kwargs) -> List[Scored]:
        q = embed_query(self.embedder, query)
        return self.store.search(q, k=k, **search_kwargs)


# ===========================================================================
# Sparse retrieval — BM25 from scratch (with rank_bm25 as an optional fast path)
# ===========================================================================
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    BM25 lexical ranking. We implement the scoring directly so the "why does
    lexical search catch exact terms that embeddings miss" point is inspectable:
    BM25 rewards documents that contain the query's *rare* terms, saturating on
    term frequency and normalising by document length.

    If `rank_bm25` is installed we can use it for speed, but the from-scratch
    version is the default so nothing is hidden.
    """

    def __init__(self, docs: List[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.corpus_tokens = [_tokenize(d.page_content) for d in docs]
        self.doc_len = [len(t) for t in self.corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freqs = [Counter(t) for t in self.corpus_tokens]
        # Document frequency per term, for IDF.
        df: Counter = Counter()
        for tokens in self.corpus_tokens:
            for term in set(tokens):
                df[term] += 1
        self.df = df
        self.N = len(docs)

    def _idf(self, term: str) -> float:
        n_qi = self.df.get(term, 0)
        # BM25 idf with +0.5 smoothing; clamp at 0 to avoid negatives.
        return max(0.0, math.log((self.N - n_qi + 0.5) / (n_qi + 0.5) + 1.0))

    def _score(self, query_terms: List[str], i: int) -> float:
        score, tf, dl = 0.0, self.term_freqs[i], self.doc_len[i]
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += self._idf(term) * (freq * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def retrieve(self, query: str, k: int = 4) -> List[Scored]:
        query_terms = _tokenize(query)
        scores = [(i, self._score(query_terms, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.docs[i], float(s)) for i, s in scores[:k] if s > 0]


# ===========================================================================
# Hybrid retrieval — Reciprocal Rank Fusion of dense + sparse
# ===========================================================================
class HybridRetriever:
    """
    Combine a dense and a sparse retriever. We use **Reciprocal Rank Fusion**:
    each retriever contributes 1/(rank + c) for every doc it ranks; the fused
    score is the sum. RRF is robust because it fuses *ranks*, not raw scores —
    so we never have to reconcile cosine similarities (0–1) with BM25 scores
    (unbounded). This directly serves "optimise recall first": the union of two
    different retrievers surfaces more of the truly relevant chunks.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        rrf_c: int = 60,
        pool_k: int = 10,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_c = rrf_c
        self.pool_k = pool_k  # how many to pull from each retriever before fusing

    def retrieve(self, query: str, k: int = 4) -> List[Scored]:
        dense_hits = self.dense.retrieve(query, k=self.pool_k)
        sparse_hits = self.sparse.retrieve(query, k=self.pool_k)

        fused: dict[int, float] = {}
        registry: dict[int, Document] = {}

        def add(hits: List[Scored]):
            for rank, (doc, _) in enumerate(hits):
                key = id(doc)  # identity is fine within one retrieval call
                registry[key] = doc
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_c + rank + 1)

        add(dense_hits)
        add(sparse_hits)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(registry[key], score) for key, score in ranked]


# ===========================================================================
# Reranking — recall net first, precision scalpel second
# ===========================================================================
class CrossEncoderReranker:
    """
    A cross-encoder reads the query and a candidate TOGETHER and outputs one
    relevance score. Because it attends across both, it is far more accurate
    than bi-encoder cosine similarity — but it must run once per candidate, so
    you only ever apply it to a small shortlist the first-stage retriever
    produced. This is the textbook "retrieve wide, rerank narrow" pattern.

    Needs sentence-transformers; if unavailable, `available` is False and the
    notebook falls back to the LLMReranker or shows the concept only.
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.available = False
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model)
            self.available = True
        except Exception as e:  # pragma: no cover
            print(f"[reranker] cross-encoder unavailable ({e}); use LLMReranker instead.")

    def rerank(self, query: str, candidates: List[Scored], top_n: int = 4) -> List[Scored]:
        if not self.available:
            return candidates[:top_n]
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(s)) for (doc, _), s in reranked[:top_n]]


class LLMReranker:
    """
    Pointwise LLM reranking: ask the LLM to score each candidate's relevance to
    the query on a 0–10 scale, then reorder. No extra model to install — reuses
    the generator you already have. Slower/pricier per query than a cross-encoder
    but requires zero new infrastructure, which is why it's popular in practice.
    """

    def __init__(self, llm):
        self.llm = llm

    def _score_one(self, query: str, text: str) -> float:
        prompt = (
            "Rate how relevant the PASSAGE is to answering the QUESTION, on a "
            "scale of 0 (irrelevant) to 10 (directly answers it). Reply with ONLY "
            f"the number.\n\nQUESTION: {query}\n\nPASSAGE: {text[:800]}\n\nScore:"
        )
        try:
            out = self.llm.generate(prompt)
            m = re.search(r"\d+(\.\d+)?", out)
            return float(m.group()) if m else 0.0
        except Exception:
            return 0.0

    def rerank(self, query: str, candidates: List[Scored], top_n: int = 4) -> List[Scored]:
        scored = [(doc, self._score_one(query, doc.page_content)) for doc, _ in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# ===========================================================================
# LangChain PARALLEL MAPPING (thin, illustrative).
# ===========================================================================
def build_langchain_retriever(chunks: List[Document], k: int = 4):
    """
    The same first-stage dense retrieval via LangChain + an in-memory FAISS-like
    store. Shown so learners see the framework wraps the exact steps we built by
    hand: embed chunks → store → similarity search. (Requires langchain-openai +
    a FAISS/Chroma backend; kept optional.)
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document as LCDoc

    lc_docs = [LCDoc(page_content=c.page_content, metadata=c.metadata) for c in chunks]
    store = FAISS.from_documents(lc_docs, OpenAIEmbeddings(model="text-embedding-3-small"))
    return store.as_retriever(search_kwargs={"k": k})
