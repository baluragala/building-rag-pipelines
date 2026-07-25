"""
generation.py — Stage 5: GENERATION (prompting with retrieved context).
=======================================================================

WHY
---
The LLM is the last stage — and the one most people wrongly treat as the *whole*
system. Its job in RAG is narrow but critical: synthesise a grounded answer from
the context it was handed, cite it, and refuse when the context doesn't support
an answer. "What happens if this step is poorly designed?" The model ignores the
context and answers from parametric memory (hallucination), or pads with fluent
filler, or fails to say "I don't know" when it should. Most of these are
*prompt* failures, not model failures — which is exactly why prompt design is a
RAG design decision, not an afterthought.

WHAT (safe RAG prompting)
-------------------------
* Instruction prompts — tell the model to use ONLY the context.
* Grounding           — instruct it to answer "I don't know" if the context is
                        insufficient (this single line kills a lot of hallucination).
* Citations           — require [n] tags tied to the numbered context blocks.
* Structure           — separate SYSTEM (role/rules) from the CONTEXT + QUESTION.

HOW
---
A canonical RAG prompt template plus two entry points:
  * generate_answer(...)      — answer from a pre-formatted context string.
  * answer_with_sources(...)  — retrieve→augment→generate in one call, returning
                                the answer AND its sources.
We also expose an UNGROUNDED template so the notebook can contrast grounded vs
ungrounded prompting ("compare outputs with/without structured prompts").
"""
from __future__ import annotations

from typing import List, Tuple

from .augmentation import collect_sources, format_context
from .loaders import Document

Scored = Tuple[Document, float]


# ---------------------------------------------------------------------------
# Prompt templates — the design surface of the generation stage.
# ---------------------------------------------------------------------------
RAG_SYSTEM = (
    "You are a precise assistant that answers strictly from the provided context. "
    "Follow these rules without exception:\n"
    "1. Use ONLY the information in CONTEXT. Do not use outside knowledge.\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided context.\"\n"
    "3. Cite the context blocks you used with their bracketed numbers, e.g. [1], [2].\n"
    "4. Be concise and do not speculate."
)

RAG_PROMPT = """Answer the QUESTION using only the CONTEXT below. Cite sources as [n].

CONTEXT:
{context}

QUESTION: {question}

Grounded answer (with citations):"""

# Deliberately weak template used ONLY to demonstrate the failure mode of
# ungrounded prompting in the notebook comparison.
UNGROUNDED_PROMPT = """Here is some context, but answer however you like.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------
def generate_answer(
    llm,
    question: str,
    context: str,
    grounded: bool = True,
    system: str | None = None,
) -> str:
    """
    Generate an answer from an already-formatted `context` string. `grounded`
    toggles between the safe RAG template and the weak one — this is the switch
    the notebook flips to show how much the prompt (not the model) determines
    faithfulness.
    """
    template = RAG_PROMPT if grounded else UNGROUNDED_PROMPT
    prompt = template.format(context=context, question=question)
    sys = system if system is not None else (RAG_SYSTEM if grounded else None)
    return llm.generate(prompt, system=sys).strip()


def answer_with_sources(
    llm,
    retriever,
    question: str,
    k: int = 4,
    max_context_tokens: int = 2000,
    reranker=None,
    grounded: bool = True,
) -> dict:
    """
    One-shot RAG: retrieve → (optional rerank) → format context → generate.
    Returns a dict with the answer, the sources (for auditability), and the raw
    context — everything you need to show and debug the result.
    """
    hits: List[Scored] = retriever.retrieve(question, k=max(k, 4))
    if reranker is not None:
        hits = reranker.rerank(question, hits, top_n=k)
    else:
        hits = hits[:k]

    context = format_context(hits, max_tokens=max_context_tokens)
    answer = generate_answer(llm, question, context, grounded=grounded)
    return {
        "question": question,
        "answer": answer,
        "sources": collect_sources(hits),          # compact labels (for display)
        "context": context,                         # the single formatted context string
        "contexts": [doc.page_content for doc, _ in hits],  # the raw chunk TEXTS (RAGAS needs these)
    }


# ---------------------------------------------------------------------------
# LangChain PARALLEL MAPPING — the RAG chain in framework form.
# ---------------------------------------------------------------------------
def build_rag_chain_langchain(lc_retriever, model: str = "gpt-4o-mini"):
    """
    The same retrieve→prompt→generate flow expressed as a LangChain LCEL chain.
    Note it encodes the SAME design decisions we made by hand (grounding rule,
    citation instruction) — the framework does not make them for you.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [("system", RAG_SYSTEM), ("human", RAG_PROMPT)]
    )

    def _format(docs):
        return "\n\n".join(
            f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)
        )

    return (
        {"context": lc_retriever | _format, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model=model, temperature=0)
        | StrOutputParser()
    )
