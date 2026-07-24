"""
chunking.py — Stage 2 of the pipeline: CHUNKING.
================================================

WHY
---
Two hard limits force chunking: (1) embeddings compress a whole passage into ONE
vector — the longer the passage, the more meaning is averaged away; (2) the LLM
context window and cost budget are finite, so you can't stuff whole documents.
Chunking is how you slice documents into retrievable, embeddable units.

"What happens if this step is poorly designed?" This is the session's central
cause→effect: **bad chunking → poor retrieval → bad answer.** Chunks too LARGE
dilute the embedding (the relevant sentence is drowned out by surrounding text,
so similarity drops and you miss it). Chunks too SMALL fragment ideas across
boundaries (the answer needs two chunks that never co-occur, so the model gets
half the story). Splitting mid-sentence or mid-table destroys meaning outright.

WHAT (strategies & trade-offs)
------------------------------
* fixed-size     — split every N characters/tokens. Simple, fast, structure-blind.
* recursive      — try to split on the biggest natural boundary that fits
                   (paragraph → line → sentence → word). Structure-aware; the
                   sensible default.
* semantic       — split where the *topic* shifts, detected by a drop in
                   embedding similarity between adjacent sentences. Best
                   boundaries, highest cost.
* overlap        — repeat a little text between neighbours so an idea that
                   straddles a boundary still appears whole in at least one chunk.

DO:   preserve semantic boundaries; add a modest overlap (10–20%).
DON'T: make chunks so large the signal is diluted, or so small ideas fragment.

HOW
---
All chunkers take and return `Document`s, carrying source metadata forward and
adding a `chunk` index. We measure size in characters by default (portable) and
offer a token-based option via tiktoken. LangChain TextSplitters are shown as
the parallel mapping.
"""
from __future__ import annotations

import re
from typing import Callable, List

import numpy as np

from .loaders import Document


# ---------------------------------------------------------------------------
# Size measurement — characters by default, tokens if tiktoken is available.
# ---------------------------------------------------------------------------
def char_len(text: str) -> int:
    return len(text)


def token_len(text: str, model: str = "gpt-4o-mini") -> int:
    """Token count via tiktoken (falls back to a ~4 chars/token estimate)."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 1) Fixed-size chunking with overlap.
# ---------------------------------------------------------------------------
def fixed_size_chunk(
    docs: List[Document],
    chunk_size: int = 500,
    overlap: int = 50,
    length_fn: Callable[[str], int] = char_len,
) -> List[Document]:
    """
    Slide a window of `chunk_size` (measured by `length_fn`) with `overlap`
    carried between windows. Structure-blind: it will happily cut mid-sentence.
    That flaw is the teaching point — compare its output to the recursive
    splitter's and the fragmentation is visible.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    out: List[Document] = []
    for doc in docs:
        text = doc.page_content
        # For char length we can index directly; for token length we walk words.
        if length_fn is char_len:
            step = chunk_size - overlap
            for i, start in enumerate(range(0, max(1, len(text)), step)):
                piece = text[start : start + chunk_size]
                if piece.strip():
                    out.append(_child(doc, piece, i))
        else:
            out.extend(_windowed_by_length(doc, text, chunk_size, overlap, length_fn))
    return out


def _windowed_by_length(doc, text, chunk_size, overlap, length_fn) -> List[Document]:
    words = text.split()
    chunks, cur, idx = [], [], 0
    for w in words:
        cur.append(w)
        if length_fn(" ".join(cur)) >= chunk_size:
            chunks.append(_child(doc, " ".join(cur), idx))
            idx += 1
            # carry an overlap tail of words
            tail, acc = [], 0
            for tw in reversed(cur):
                acc += length_fn(tw) + 1
                tail.insert(0, tw)
                if acc >= overlap:
                    break
            cur = tail
    if cur:
        chunks.append(_child(doc, " ".join(cur), idx))
    return chunks


# ---------------------------------------------------------------------------
# 2) Recursive chunking — the structure-aware default.
# ---------------------------------------------------------------------------
def recursive_chunk(
    docs: List[Document],
    chunk_size: int = 500,
    overlap: int = 50,
    separators: List[str] | None = None,
    length_fn: Callable[[str], int] = char_len,
) -> List[Document]:
    """
    Split on the LARGEST natural boundary that keeps chunks under `chunk_size`,
    recursing to finer separators only when a piece is still too big. Default
    separator ladder: blank line (paragraph) → newline → sentence → space.
    This preserves semantic boundaries far better than fixed-size — the do.
    """
    separators = separators or ["\n\n", "\n", ". ", " ", ""]
    out: List[Document] = []
    for doc in docs:
        pieces = _recursive_split(doc.page_content, separators, chunk_size, length_fn)
        pieces = _merge_with_overlap(pieces, chunk_size, overlap, length_fn)
        for i, piece in enumerate(pieces):
            if piece.strip():
                out.append(_child(doc, piece.strip(), i))
    return out


def _recursive_split(text, separators, chunk_size, length_fn) -> List[str]:
    if length_fn(text) <= chunk_size or not separators:
        return [text]
    sep, *rest = separators
    parts = text.split(sep) if sep else list(text)
    result: List[str] = []
    for part in parts:
        piece = part + (sep if sep and sep != "" else "")
        if length_fn(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_recursive_split(part, rest, chunk_size, length_fn))
    return result


def _merge_with_overlap(pieces, chunk_size, overlap, length_fn) -> List[str]:
    """Greedily pack small pieces up to chunk_size, adding overlap between chunks."""
    merged: List[str] = []
    cur = ""
    for p in pieces:
        if length_fn(cur + p) <= chunk_size:
            cur += p
        else:
            if cur:
                merged.append(cur)
            cur = (_tail(cur, overlap) + p) if overlap and cur else p
    if cur:
        merged.append(cur)
    return merged


def _tail(text: str, overlap: int) -> str:
    return text[-overlap:] if overlap < len(text) else text


# ---------------------------------------------------------------------------
# 3) Semantic chunking — split where the topic shifts.
# ---------------------------------------------------------------------------
def semantic_chunk(
    docs: List[Document],
    embedder,
    threshold_percentile: int = 90,
    max_chars: int = 1200,
) -> List[Document]:
    """
    Sentence-level semantic chunking:
      1. split into sentences,
      2. embed each sentence,
      3. compute the cosine DISTANCE between consecutive sentences,
      4. cut where the distance exceeds the `threshold_percentile` — i.e. where
         the topic shifts the most.
    Produces the most coherent chunks but costs one embedding per sentence.

    `embedder` is any object from config.get_embedder(). With the mock embedder
    the boundaries are approximate but the MECHANISM is fully demonstrable.
    """
    out: List[Document] = []
    for doc in docs:
        sentences = _split_sentences(doc.page_content)
        if len(sentences) <= 1:
            out.append(_child(doc, doc.page_content, 0))
            continue
        embs = embedder.embed(sentences)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        dists = [1.0 - float(embs[i] @ embs[i + 1]) for i in range(len(sentences) - 1)]
        cut = np.percentile(dists, threshold_percentile) if dists else 1.0

        chunks, cur, idx = [], [sentences[0]], 0
        for i, d in enumerate(dists):
            over_len = len(" ".join(cur)) > max_chars
            if d >= cut or over_len:  # topic shift OR chunk too big
                chunks.append(_child(doc, " ".join(cur), idx))
                idx += 1
                cur = [sentences[i + 1]]
            else:
                cur.append(sentences[i + 1])
        if cur:
            chunks.append(_child(doc, " ".join(cur), idx))
        out.extend(chunks)
    return out


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter; good enough for teaching (no nltk dep).
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# LangChain PARALLEL MAPPING.
# ---------------------------------------------------------------------------
def recursive_chunk_langchain(docs: List[Document], chunk_size=500, overlap=50):
    """
    The framework equivalent of `recursive_chunk`. Same idea, same separator
    ladder — reinforcing that the framework abstracts the mechanics but you
    still own the design decision (size, overlap, what counts as a boundary).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    lc_docs = splitter.create_documents(texts, metadatas=metas)
    # Map back to our Document type so downstream stages are provider-neutral.
    return [Document(page_content=d.page_content, metadata=d.metadata) for d in lc_docs]


# ---------------------------------------------------------------------------
# Analysis helper — used in the "compare across chunk sizes" demo.
# ---------------------------------------------------------------------------
def chunk_stats(chunks: List[Document], length_fn: Callable[[str], int] = char_len) -> dict:
    sizes = [length_fn(c.page_content) for c in chunks]
    if not sizes:
        return {"n_chunks": 0}
    return {
        "n_chunks": len(chunks),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "mean": round(float(np.mean(sizes)), 1),
        "median": float(np.median(sizes)),
    }


def _child(parent: Document, text: str, idx: int) -> Document:
    meta = dict(parent.metadata)
    meta["chunk"] = idx
    return Document(page_content=text, metadata=meta)
