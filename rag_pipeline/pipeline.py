"""
pipeline.py — the end-to-end modular RAG pipeline (the whole session in one class).
==================================================================================

WHY
---
The session's thesis: **RAG performance is the cumulative result of design
decisions across ALL stages, not just the LLM.** This class makes that literal —
every stage is a swappable knob, and a `RAGTrace` records what each stage did so
you can walk the pipeline diagram at every step and, in the Conclusion, take a
"bad RAG output" and diagnose *which stage* caused it.

WHAT
----
`RAGPipeline` wires the six stages together:
    Loading → Chunking → (Embedding) → Vector store → Retrieval
    → (Rerank) → Augmentation → Generation
and returns an answer plus a full trace (retrieved chunks, scores, context,
timings). Change chunker, retriever, injection strategy, reranker, or prompt
independently and re-run.

HOW
---
`ingest(docs)` builds the index. `query(question)` runs the read path and returns
a dict {answer, sources, trace}. The trace is the debugging surface used in
notebook 08.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from . import augmentation as aug
from .chunking import recursive_chunk
from .config import get_embedder, get_llm
from .embeddings import embed_documents
from .generation import generate_answer
from .loaders import Document
from .retrieval import DenseRetriever
from .vectorstore import InMemoryVectorStore


@dataclass
class RAGTrace:
    """A record of one query's journey through the pipeline — for debugging."""

    question: str
    retrieved: List[dict] = field(default_factory=list)  # {label, score, preview}
    context: str = ""
    answer: str = ""
    timings_ms: dict = field(default_factory=dict)

    def show(self) -> str:
        lines = [f"QUESTION: {self.question}", "", "RETRIEVED:"]
        for r in self.retrieved:
            lines.append(f"  [{r['id']}] score={r['score']:.4f}  {r['label']}")
            lines.append(f"       {r['preview']}")
        lines += ["", f"TIMINGS(ms): {self.timings_ms}", "", f"ANSWER:\n{self.answer}"]
        return "\n".join(lines)


class RAGPipeline:
    """
    An explicitly modular RAG pipeline. Defaults use the config providers and the
    from-scratch components, but every stage accepts an override so the notebooks
    can compare strategies without rewriting the wiring.
    """

    def __init__(
        self,
        embedder=None,
        llm=None,
        chunker: Callable[[List[Document]], List[Document]] = None,
        retriever_factory: Optional[Callable] = None,
        injection: str = "stuff",  # "stuff" | "map_reduce" | "refine"
        reranker=None,
        k: int = 4,
        max_context_tokens: int = 2000,
        grounded: bool = True,
    ):
        self.embedder = embedder or get_embedder()
        self.llm = llm or get_llm()
        self.chunker = chunker or (lambda docs: recursive_chunk(docs, 500, 50))
        self.retriever_factory = retriever_factory  # (store, embedder) -> retriever
        self.injection = injection
        self.reranker = reranker
        self.k = k
        self.max_context_tokens = max_context_tokens
        self.grounded = grounded

        self.store = InMemoryVectorStore()
        self.chunks: List[Document] = []
        self.retriever = None

    # -- write path ---------------------------------------------------------
    def ingest(self, docs: List[Document]) -> "RAGPipeline":
        """Chunk → embed → store. Builds the retriever."""
        self.chunks = self.chunker(docs)
        embeddings = embed_documents(self.embedder, self.chunks)
        self.store = InMemoryVectorStore().add(self.chunks, embeddings)
        if self.retriever_factory:
            self.retriever = self.retriever_factory(self.store, self.embedder)
        else:
            self.retriever = DenseRetriever(self.store, self.embedder)
        return self

    # -- read path ----------------------------------------------------------
    def query(self, question: str) -> dict:
        """Retrieve → (rerank) → augment → generate, returning answer + trace."""
        if self.retriever is None:
            raise RuntimeError("Call ingest(docs) before query().")

        trace = RAGTrace(question=question)

        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k=max(self.k, 4))
        trace.timings_ms["retrieve"] = _ms(t0)

        if self.reranker is not None:
            t0 = time.perf_counter()
            hits = self.reranker.rerank(question, hits, top_n=self.k)
            trace.timings_ms["rerank"] = _ms(t0)
        else:
            hits = hits[: self.k]

        trace.retrieved = [
            {
                "id": i,
                "label": aug._source_label(doc),
                "score": float(score),
                "preview": doc.page_content[:100].replace("\n", " "),
            }
            for i, (doc, score) in enumerate(hits, start=1)
        ]

        # Augmentation + generation, dispatched by injection strategy.
        t0 = time.perf_counter()
        if self.injection == "map_reduce":
            answer = aug.map_reduce(self.llm, question, hits)
            context = aug.format_context(hits, max_tokens=self.max_context_tokens)
        elif self.injection == "refine":
            answer = aug.refine(self.llm, question, hits)
            context = aug.format_context(hits, max_tokens=self.max_context_tokens)
        else:  # stuff
            context = aug.stuff(hits, max_tokens=self.max_context_tokens)
            answer = generate_answer(self.llm, question, context, grounded=self.grounded)
        trace.timings_ms["augment+generate"] = _ms(t0)

        trace.context = context
        trace.answer = answer

        return {
            "question": question,
            "answer": answer,
            "sources": aug.collect_sources(hits),
            "trace": trace,
        }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
