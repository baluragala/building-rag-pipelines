# Building RAG Pipelines — Teaching Package Design

**Date:** 2026-07-24
**Source agenda:** `Agenda_C8_GenAI_W4S1_Building_RAG_Pipelines_SME.pdf` (upGrad, 180-min live session)

## Goal
A complete, extensive hands-on teaching package for a 180-minute live session that teaches RAG as a **modular pipeline** (Loading → Chunking → Retrieval → Augmentation → Generation → Evaluation), following a **WHY → WHAT → HOW** pedagogy. Learners build and analyse a modular RAG pipeline, understand how each stage's design decisions cumulatively determine output quality, and can diagnose failure modes.

## Non-negotiable constraints (from agenda + user)
- **Pedagogy:** every stage = WHY (motivation + "what happens if this step is poorly designed?") → WHAT (concept, trade-offs, do's/don'ts, failure modes) → HOW (from-scratch code first, then LangChain as a *parallel mapping*).
- Predict-before-run cells; compare outputs across chunk sizes / retrieval strategies / injection methods; a pipeline diagram revisited at every stage; LangChain kept as parallel mapping, not the primary focus.
- **Stack:** OpenAI default (`gpt-4o-mini` LLM, `text-embedding-3-small` embeddings). Provider-agnostic config; Anthropic Claude LLM adapter (Anthropic has no embeddings API); offline **mock** fallback so notebooks render with zero keys.
- **Notebooks must be Google Colab compatible** (self-bootstrap: pip install, fetch package, keys via getpass / colab userdata).

## Components
1. `rag_pipeline/` — reusable from-scratch package, single source of truth:
   `config.py, loaders.py, chunking.py, embeddings.py, vectorstore.py, retrieval.py, augmentation.py, generation.py, evaluation.py, pipeline.py`.
2. `notebooks/` — 8 Colab-compatible guided notebooks, one per agenda stage.
3. `data/corpus/` — small multi-format corpus; `data/eval/eval_dataset.jsonl` — questions, ground truths, contexts, negatives.
4. `slides/rag_pipelines.html` — self-contained reveal.js deck, SVG pipeline diagram per stage, speaker notes.
5. `teaching/` — instructor_guide.md (minute-by-minute), learner_handout.md, exercises.md, solutions/.

## Notebook ↔ agenda ↔ timing
| # | Notebook | Stage | Min |
|---|----------|-------|-----|
|01|rag_foundations|architecture, standalone-LLM limits, failure points|15|
|02|loading|PDF/HTML/MD/API, clean+metadata, manual vs LangChain loaders|15|
|03|chunking|fixed/recursive/semantic, size vs overlap, manual vs TextSplitters|20|
|04|retrieval|embeddings, dense, BM25+hybrid, filtering, reranking|30|
|05|augmentation_generation|stuffing/map-reduce/refine, sources, prompts, grounding, citations|40|
|06|evaluation|failure modes, latency/cost, metrics, RAGAS, building eval datasets|35|
|07|advanced_rag|Naïve/Advanced/Modular, agentic, multi-hop, Graph RAG, adaptive, long-context|15|
|08|conclusion_end_to_end|full pipeline, design→output, debugging a "bad RAG output"|10|

## Success criteria
- Every notebook runs top-to-bottom in Colab with only an OpenAI key (or the mock fallback with none).
- Each stage contains explicit WHY/WHAT/HOW sections, a predict-before-run cell, and at least one output comparison.
- The package is imported (not duplicated) by notebooks; from-scratch and LangChain paths shown side by side.
- Instructor guide maps to the 180-minute clock; slides revisit the pipeline diagram at each stage.
