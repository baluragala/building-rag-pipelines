"""
evaluation.py — Stage 6: EVALUATION (heuristics, metrics, RAGAS, datasets).
===========================================================================

WHY
---
"How do you know your RAG system is any good?" Without measurement you are just
guessing, and every stage decision (chunk size, k, reranker, prompt) becomes
opinion. Evaluation turns RAG from art into engineering. It also localises
failure: RAG can break in the RETRIEVAL stage (wrong chunks) or the GENERATION
stage (right chunks, wrong answer) — and the fix is completely different. Metrics
that separate the two are what let you debug instead of flail.

"What happens if this step is poorly designed?" You optimise the wrong thing:
tune the prompt for weeks when the real problem was recall, or ship a system that
scores well on fluency while quietly hallucinating.

WHAT (the two families)
-----------------------
RETRIEVAL metrics (is the right context found?):
  * precision@k, recall@k, hit_rate@k, MRR — computed against ground-truth
    relevant chunk ids.
GENERATION metrics (is the answer faithful and correct?):
  * faithfulness heuristic — is the answer supported by the retrieved context?
  * answer relevance       — does the answer address the question?
  * RAGAS                   — the framework (faithfulness, answer correctness,
                             context precision/recall) wrapped optionally.

Also: how to BUILD an eval dataset — questions, ground truths, and crucially
NEGATIVE tests (questions the corpus can't answer, where the correct output is
"I don't know"). Metrics vs rubrics: numbers for what's countable, human rubrics
for what isn't.

HOW
---
Pure functions over a small `EvalExample` schema, plus an optional RAGAS bridge
and an LLM-as-judge faithfulness scorer for when you have a key but not RAGAS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .embeddings import cosine_similarity, embed_query


# ---------------------------------------------------------------------------
# Eval dataset schema.
# ---------------------------------------------------------------------------
@dataclass
class EvalExample:
    """
    One evaluation item. `relevant_ids` are the ids of chunks/documents that
    truly answer the question (ground truth for retrieval). `ground_truth` is the
    reference answer (for generation). `is_negative=True` marks a question the
    corpus CANNOT answer — the correct behaviour is an "I don't know" refusal.
    """

    question: str
    ground_truth: str = ""
    relevant_ids: List[str] = field(default_factory=list)
    is_negative: bool = False
    metadata: dict = field(default_factory=dict)


def load_eval_dataset(path: str) -> List[EvalExample]:
    """Load a JSONL eval file into EvalExample objects."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(
                EvalExample(
                    question=d["question"],
                    ground_truth=d.get("ground_truth", ""),
                    relevant_ids=d.get("relevant_ids", []),
                    is_negative=d.get("is_negative", False),
                    metadata=d.get("metadata", {}),
                )
            )
    return examples


# ---------------------------------------------------------------------------
# Retrieval metrics — measured against ground-truth relevant ids.
# ---------------------------------------------------------------------------
def _id_of(doc) -> str:
    """Stable id for a chunk: source#chunk (matches the eval dataset convention)."""
    src = doc.metadata.get("source", "?")
    chunk = doc.metadata.get("chunk")
    page = doc.metadata.get("page")
    if chunk is not None:
        return f"{src}#chunk{chunk}"
    if page is not None:
        return f"{src}#page{page}"
    return str(src)


def source_id_of(doc) -> str:
    """
    SOURCE-level id: just the file's basename (e.g. 'acme_pricing.md'). We use
    this for evaluation by default because it is robust to chunk size — the
    ground-truth in the eval dataset refers to which *document* answers a
    question, which doesn't change when you re-tune the chunker. (Chunk-level
    relevance is more precise but brittle across configs; discuss both in class.)
    """
    src = str(doc.metadata.get("source", "?"))
    return src.replace("\\", "/").split("/")[-1]


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Of the top-k retrieved, what fraction are relevant? (precision)"""
    if k == 0:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i in set(relevant_ids))
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Of all relevant chunks, what fraction appear in the top-k? (recall)"""
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    hits = sum(1 for i in relevant_ids if i in top)
    return hits / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Did AT LEAST ONE relevant chunk make the top-k? (1.0/0.0)"""
    top = set(retrieved_ids[:k])
    return 1.0 if any(i in top for i in relevant_ids) else 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant hit — rewards ranking it high."""
    rel = set(relevant_ids)
    for rank, i in enumerate(retrieved_ids, start=1):
        if i in rel:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever, examples: List[EvalExample], k: int = 4, id_fn: Callable = source_id_of
) -> dict:
    """
    Run a retriever over the eval set and average the retrieval metrics. Skips
    negative examples (they have no relevant ids by design). This is the number
    to watch FIRST when debugging — if recall is low, no prompt fix will help.

    `id_fn` maps a retrieved doc to the id space of the eval dataset. Defaults to
    SOURCE-level (`source_id_of`) so ground truth is robust to chunk size; pass
    `id_fn=_id_of` for chunk-level evaluation.
    """
    p, r, hit, mrrs = [], [], [], []
    for ex in examples:
        if ex.is_negative or not ex.relevant_ids:
            continue
        hits = retriever.retrieve(ex.question, k=k)
        ids = [id_fn(doc) for doc, _ in hits]
        p.append(precision_at_k(ids, ex.relevant_ids, k))
        r.append(recall_at_k(ids, ex.relevant_ids, k))
        hit.append(hit_rate_at_k(ids, ex.relevant_ids, k))
        mrrs.append(mrr(ids, ex.relevant_ids))
    n = max(1, len(p))
    return {
        f"precision@{k}": round(sum(p) / n, 3),
        f"recall@{k}": round(sum(r) / n, 3),
        f"hit_rate@{k}": round(sum(hit) / n, 3),
        "mrr": round(sum(mrrs) / n, 3),
        "n_evaluated": len(p),
    }


# ---------------------------------------------------------------------------
# Generation heuristics — cheap, no framework needed.
# ---------------------------------------------------------------------------
def faithfulness_heuristic(answer: str, context: str, embedder) -> float:
    """
    Cheap faithfulness proxy: embed the answer and the context and return their
    cosine similarity. High similarity ⇒ the answer's content overlaps the
    context (a weak but useful "is it grounded?" signal). NOT a substitute for
    RAGAS/LLM-judge — it's the "heuristic" half of the agenda's "heuristics AND
    frameworks".
    """
    if not answer.strip() or not context.strip():
        return 0.0
    return max(0.0, cosine_similarity(embed_query(embedder, answer), embed_query(embedder, context)))


def answer_relevance_heuristic(answer: str, question: str, embedder) -> float:
    """Does the answer address the question? Embedding cosine as a proxy."""
    if not answer.strip():
        return 0.0
    return max(0.0, cosine_similarity(embed_query(embedder, answer), embed_query(embedder, question)))


def refusal_correct(answer: str) -> bool:
    """True if the answer looks like an 'I don't know' refusal (for negatives)."""
    a = answer.lower()
    return any(p in a for p in ["i don't know", "i do not know", "cannot answer", "no answer", "not contain"])


def llm_judge_faithfulness(llm, answer: str, context: str) -> float:
    """
    LLM-as-judge faithfulness: ask the model whether every claim in the answer is
    supported by the context. Returns a 0–1 score. Use when you have a key but
    don't want the full RAGAS dependency.
    """
    prompt = (
        "You are a strict grader. Is EVERY claim in the ANSWER supported by the "
        "CONTEXT? Reply with a single number from 0 (fully unsupported / "
        "hallucinated) to 1 (fully supported).\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore (0-1):"
    )
    import re

    out = llm.generate(prompt)
    m = re.search(r"[01](\.\d+)?", out)
    return float(m.group()) if m else 0.0


# ---------------------------------------------------------------------------
# RAGAS bridge (optional heavy dependency).
# ---------------------------------------------------------------------------
def evaluate_with_ragas(records: List[dict]) -> Optional[dict]:
    """
    Evaluate with RAGAS if installed. `records` is a list of dicts with keys:
      question, answer, contexts (list[str]), ground_truth.
    Returns the aggregated RAGAS scores (faithfulness, answer_relevancy,
    context_precision, context_recall) or None if RAGAS isn't available.

    RAGAS itself calls an LLM+embeddings under the hood, so it needs a key. The
    notebook shows the workflow even when this returns None.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as e:  # pragma: no cover
        print(f"[ragas] not available ({e}); showing conceptual workflow only.")
        return None

    ds = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r.get("ground_truth", ""),
            }
            for r in records
        ]
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)


# ---------------------------------------------------------------------------
# RAGAS-STYLE metrics computed directly with the LLM (no heavy dependency).
# ---------------------------------------------------------------------------
# RAGAS is powerful but its dependency web is notoriously brittle to install in a
# shared environment (it pins specific langchain/pydantic/numpy versions). Under
# the hood, though, RAGAS's core metrics are just LLM-as-judge prompts. So we
# reproduce the SAME four metrics with direct LLM/embedding calls. This ALWAYS
# runs with the OpenAI stack, needs no extra install, and — bonus — demystifies
# what RAGAS actually does. Use `evaluate_with_ragas` (below) for the real library.
def _judge_number(llm, prompt: str) -> float:
    import re

    try:
        out = llm.generate(prompt)
    except Exception:
        return 0.0
    m = re.search(r"[01](?:\.\d+)?", out)
    try:
        return max(0.0, min(1.0, float(m.group()))) if m else 0.0
    except Exception:
        return 0.0


def context_precision_llm(llm, question: str, contexts: List[str]) -> float:
    """Fraction of the retrieved contexts that are actually relevant (LLM judge)."""
    if not contexts:
        return 0.0
    hits = 0
    for ctx in contexts:
        p = (
            "Is the PASSAGE relevant to answering the QUESTION? Reply 1 for yes, "
            f"0 for no, nothing else.\n\nQUESTION: {question}\n\n"
            f"PASSAGE: {ctx[:800]}\n\nAnswer:"
        )
        hits += round(_judge_number(llm, p))
    return hits / len(contexts)


def context_recall_llm(llm, ground_truth: str, contexts: List[str]) -> float:
    """Fraction of the reference answer's content supported by the contexts (0-1)."""
    if not ground_truth.strip() or not contexts:
        return 0.0
    joined = "\n".join(c[:600] for c in contexts)
    p = (
        "Estimate the fraction (a number from 0 to 1) of the REFERENCE ANSWER's "
        "factual content that is supported by the CONTEXT.\n\n"
        f"REFERENCE ANSWER: {ground_truth}\n\nCONTEXT:\n{joined}\n\nFraction (0-1):"
    )
    return _judge_number(llm, p)


def evaluate_ragas_style(records: List[dict], llm, embedder) -> dict:
    """
    Compute RAGAS's four metrics using only the LLM + embedder (no `ragas` install).
    Each record needs: question, answer, contexts (list[str]), ground_truth.

      * faithfulness       — is every claim in the answer supported by the context?
      * answer_relevancy   — does the answer address the question? (embedding cosine)
      * context_precision  — how many retrieved contexts are relevant?
      * context_recall     — how much of the ground truth do the contexts cover?
    """
    f, ar, cp, cr = [], [], [], []
    for r in records:
        ctx = "\n\n".join(r.get("contexts", []))
        f.append(llm_judge_faithfulness(llm, r["answer"], ctx))
        ar.append(answer_relevance_heuristic(r["answer"], r["question"], embedder))
        cp.append(context_precision_llm(llm, r["question"], r.get("contexts", [])))
        cr.append(context_recall_llm(llm, r.get("ground_truth", ""), r.get("contexts", [])))
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else 0.0
    return {
        "faithfulness": mean(f),
        "answer_relevancy": mean(ar),
        "context_precision": mean(cp),
        "context_recall": mean(cr),
        "n_evaluated": len(records),
    }


# ---------------------------------------------------------------------------
# Building an eval dataset from a corpus (auto-draft questions with an LLM).
# ---------------------------------------------------------------------------
def draft_questions_from_chunks(llm, chunks, n_per_chunk: int = 1) -> List[EvalExample]:
    """
    Bootstrap an eval set: for a sample of chunks, ask the LLM to write a
    question the chunk answers, and record the chunk id as ground-truth relevant.
    A starting point a human should then curate — never ship auto-generated
    ground truth unreviewed. (See the handout: metrics vs rubrics.)
    """
    examples: List[EvalExample] = []
    for doc in chunks:
        prompt = (
            "Write ONE clear question that the passage below directly answers. "
            "Reply with only the question.\n\nPASSAGE:\n" + doc.page_content[:800]
        )
        q = llm.generate(prompt).strip()
        examples.append(EvalExample(question=q, relevant_ids=[_id_of(doc)]))
    return examples
