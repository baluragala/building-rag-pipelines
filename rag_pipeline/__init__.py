"""
rag_pipeline — a modular, teachable Retrieval-Augmented Generation pipeline.
===========================================================================

Built for the "Building RAG Pipelines" teaching session. Each
module maps to one stage of the pipeline, in the order data flows:

    loaders      → Stage 1: Loading   (ingestion, cleaning, metadata)
    chunking     → Stage 2: Chunking  (fixed / recursive / semantic + overlap)
    embeddings   → Stage 3: Retrieval (vectorisation utilities)
    vectorstore  → Stage 3: Retrieval (from-scratch cosine vector store)
    retrieval    → Stage 3: Retrieval (dense / BM25 / hybrid / rerank / filter)
    augmentation → Stage 4: Augmentation (stuff / map-reduce / refine + sources)
    generation   → Stage 5: Generation (grounded prompting, citations)
    evaluation   → Stage 6: Evaluation (metrics, heuristics, RAGAS, datasets)
    pipeline     → the whole thing, wired together with a debuggable trace
    config       → provider-agnostic LLM + embedder factory (OpenAI default,
                   Anthropic + local + offline-mock fallbacks)

Design intent: every stage is a swappable knob, so learners can *feel* how each
design decision changes the final answer. Frameworks (LangChain) appear as thin
"parallel mappings" — never the primary path.
"""

from .config import get_embedder, get_llm, current_config, RagConfig
from .loaders import Document
from .pipeline import RAGPipeline, RAGTrace

__all__ = [
    "get_embedder",
    "get_llm",
    "current_config",
    "RagConfig",
    "Document",
    "RAGPipeline",
    "RAGTrace",
]

__version__ = "1.0.0"
