"""
augmentation.py — Stage 4: AUGMENTATION (context injection strategies).
=======================================================================

WHY
---
Retrieval found the chunks; augmentation decides HOW they enter the prompt. This
is the stage learners most often conflate with retrieval, so name the distinction
sharply: **retrieval = which chunks; augmentation = how those chunks are packed
into the context the LLM reads.** "What happens if this step is poorly designed?"
You blow the context budget (cost + latency), or you bury the answer in the
middle of a long context where LLMs attend to it least (the "lost in the middle"
effect), or you hand the model unlabelled text it can't cite.

WHAT (strategies & trade-offs)
------------------------------
* Naive stuffing  — concatenate all chunks into one prompt. Simple; fails when
                    chunks exceed the window or dilute attention.
* Map-reduce      — answer from each chunk independently (map), then combine the
                    partial answers (reduce). Scales to many chunks; more calls.
* Refine          — start with one chunk's answer, then iteratively revise it as
                    each further chunk is read. Good for cumulative reasoning;
                    sequential, so slower.
* Return sources  — always keep chunk provenance so answers can be cited and
                    audited. Non-negotiable for trustworthy RAG.

DO:   control context size; structure the prompt; label each chunk with its source.
DON'T: dump everything and hope; drop provenance.

HOW
---
Pure functions that take retrieved (Document, score) pairs and produce either a
formatted context string (for stuffing) or a synthesised answer (map-reduce /
refine, which need the LLM). Token budgeting uses the chunking.token_len helper.
"""
from __future__ import annotations

from typing import List, Tuple

from .chunking import token_len
from .loaders import Document

Scored = Tuple[Document, float]


# ---------------------------------------------------------------------------
# Formatting context with visible provenance (enables citations downstream).
# ---------------------------------------------------------------------------
def format_context(
    hits: List[Scored],
    max_tokens: int | None = 2000,
    numbered: bool = True,
) -> str:
    """
    Turn retrieved chunks into a labelled context block, respecting a token
    budget. Each chunk is tagged with a source id like [1] (title/page) so the
    generation prompt can ask the model to cite [1], [2], … — grounding the
    answer and making it auditable.

    `max_tokens=None` disables budgeting (useful to *demonstrate* what happens
    when you don't control context size).
    """
    lines, used = [], 0
    for i, (doc, _score) in enumerate(hits, start=1):
        src = _source_label(doc)
        header = f"[{i}] ({src})" if numbered else f"({src})"
        block = f"{header}\n{doc.page_content.strip()}"
        block_tokens = token_len(block)
        if max_tokens is not None and used + block_tokens > max_tokens:
            # Budget exhausted — stop rather than silently overflowing the window.
            break
        lines.append(block)
        used += block_tokens
    return "\n\n".join(lines)


def _source_label(doc: Document) -> str:
    meta = doc.metadata
    title = meta.get("title", meta.get("source", "unknown"))
    page = meta.get("page")
    chunk = meta.get("chunk")
    label = str(title)
    if page is not None:
        label += f", p.{page}"
    if chunk is not None:
        label += f", chunk {chunk}"
    return label


# ---------------------------------------------------------------------------
# 1) Naive stuffing.
# ---------------------------------------------------------------------------
def stuff(hits: List[Scored], max_tokens: int | None = 2000) -> str:
    """Return a single stuffed context string. The simplest injection strategy."""
    return format_context(hits, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# 2) Map-reduce.
# ---------------------------------------------------------------------------
def map_reduce(llm, question: str, hits: List[Scored]) -> str:
    """
    MAP: ask the LLM to extract an answer (or "NO ANSWER") from each chunk on its
    own. REDUCE: combine the non-empty partials into a final answer. This keeps
    each call's context small and scales past the window, at the cost of N+1
    calls. Great when relevant evidence is spread across many chunks.
    """
    partials = []
    for i, (doc, _s) in enumerate(hits, start=1):
        prompt = (
            f"Answer the QUESTION using ONLY the passage. If the passage does not "
            f"help, reply exactly 'NO ANSWER'.\n\nQUESTION: {question}\n\n"
            f"CONTEXT:\n[{i}] ({_source_label(doc)})\n{doc.page_content}\n\nAnswer:"
        )
        out = llm.generate(prompt).strip()
        if "NO ANSWER" not in out.upper():
            partials.append(f"[{i}] {out}")

    if not partials:
        return "The retrieved context does not contain the answer."

    reduce_prompt = (
        "Combine the PARTIAL ANSWERS below into one coherent, non-repetitive "
        f"answer to the QUESTION. Keep the source tags.\n\nQUESTION: {question}\n\n"
        "PARTIAL ANSWERS:\n" + "\n".join(partials) + "\n\nFinal answer:"
    )
    return llm.generate(reduce_prompt).strip()


# ---------------------------------------------------------------------------
# 3) Refine.
# ---------------------------------------------------------------------------
def refine(llm, question: str, hits: List[Scored]) -> str:
    """
    Build the answer incrementally: seed it from the first chunk, then for each
    subsequent chunk ask the LLM to REFINE the running answer with any new,
    relevant detail. Good when later evidence should adjust an earlier claim;
    strictly sequential, so it is the slowest of the three.
    """
    if not hits:
        return "No context was retrieved."

    first_doc = hits[0][0]
    answer = llm.generate(
        f"Answer the QUESTION using ONLY the context.\n\nQUESTION: {question}\n\n"
        f"CONTEXT:\n[1] ({_source_label(first_doc)})\n{first_doc.page_content}\n\nAnswer:"
    ).strip()

    for i, (doc, _s) in enumerate(hits[1:], start=2):
        answer = llm.generate(
            "Refine the EXISTING ANSWER using the NEW CONTEXT. Add or correct "
            "detail only if the new context is relevant; otherwise return the "
            f"existing answer unchanged.\n\nQUESTION: {question}\n\n"
            f"EXISTING ANSWER: {answer}\n\n"
            f"NEW CONTEXT:\n[{i}] ({_source_label(doc)})\n{doc.page_content}\n\n"
            "Refined answer:"
        ).strip()
    return answer


# ---------------------------------------------------------------------------
# Sources — the return-sources-with-responses requirement.
# ---------------------------------------------------------------------------
def collect_sources(hits: List[Scored]) -> List[dict]:
    """Compact source list to return alongside an answer for auditability."""
    return [
        {
            "id": i,
            "label": _source_label(doc),
            "source": doc.metadata.get("source"),
            "score": round(float(score), 4),
        }
        for i, (doc, score) in enumerate(hits, start=1)
    ]
