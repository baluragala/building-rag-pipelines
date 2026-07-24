# Building RAG Pipelines

A complete, hands-on teaching package for the 180-minute upGrad live session
**"Building RAG Pipelines"**. It teaches Retrieval-Augmented Generation (RAG) as a
**modular pipeline** — Loading → Chunking → Retrieval → Augmentation → Generation →
Evaluation — following a **WHY → WHAT → HOW** pedagogy, with the recurring question
*"What happens if this step is poorly designed?"*

> **The one takeaway:** *RAG performance is the cumulative result of design
> decisions across all stages — not just the choice of LLM.*

---

## What's in the box

```
building-rag-pipelines/
├── rag_pipeline/            # Reusable, from-scratch Python package (the "not just frameworks" story)
│   ├── config.py            #   provider-agnostic LLM + embedder factory (OpenAI default / Anthropic / offline mock)
│   ├── loaders.py           #   Stage 1 — ingestion, cleaning, metadata (PDF/HTML/MD/API)
│   ├── chunking.py          #   Stage 2 — fixed / recursive / semantic chunking + overlap
│   ├── embeddings.py        #   Stage 3 — vectorisation & cosine similarity utilities
│   ├── vectorstore.py       #   Stage 3 — from-scratch numpy cosine store + metadata filtering
│   ├── retrieval.py         #   Stage 3 — dense / BM25 / hybrid (RRF) / cross-encoder & LLM rerank
│   ├── augmentation.py      #   Stage 4 — stuff / map-reduce / refine + source provenance
│   ├── generation.py        #   Stage 5 — grounded prompting, citations, grounded-vs-ungrounded
│   ├── evaluation.py        #   Stage 6 — precision@k/recall@k/MRR, faithfulness, RAGAS, eval datasets
│   └── pipeline.py          #   end-to-end RAGPipeline + debuggable RAGTrace
├── notebooks/               # 8 Colab-compatible guided notebooks, one per agenda stage
│   ├── 01_rag_foundations.ipynb          (15 min · conceptual)
│   ├── 02_loading.ipynb                  (15 min · concept + demo)
│   ├── 03_chunking.ipynb                 (20 min · guided coding)
│   ├── 04_retrieval.ipynb                (30 min · guided coding)
│   ├── 05_augmentation_generation.ipynb  (40 min · guided coding)
│   ├── 06_evaluation.ipynb               (35 min · guided analysis)
│   ├── 07_advanced_rag.ipynb             (15 min · conceptual)
│   └── 08_conclusion_end_to_end.ipynb    (10 min · end-to-end + debugging)
├── data/
│   ├── corpus/              # Fictional "Acme Cloud" docs (pricing, security, SLA, API, onboarding, FAQ)
│   └── eval/eval_dataset.jsonl   # 21 questions w/ ground truths + 3 NEGATIVE tests (unanswerable → refuse)
├── slides/rag_pipelines.html   # Self-contained reveal.js deck (61 slides), SVG pipeline diagram + speaker notes
├── teaching/
│   ├── instructor_guide.md      # Minute-by-minute run sheet mapped to the 180-min agenda
│   ├── learner_handout.md       # Take-home study notes, cheat sheets, glossary
│   ├── exercises.md             # Graded practice exercises per stage + capstone
│   └── solutions/solutions.md   # Worked solutions
├── requirements.txt  ·  .env.example  ·  .gitignore
└── docs/superpowers/specs/      # design spec for this package
```

---

## Quick start (local)

```bash
pip install -r requirements.txt          # or just: pip install numpy openai tiktoken rank-bm25 beautifulsoup4
cp .env.example .env                      # add your OPENAI_API_KEY (optional)
jupyter lab                               # open notebooks/01_rag_foundations.ipynb
```

**No API key? It still runs.** Every module falls back to a deterministic **offline
mock** provider, so the whole session — chunking comparisons, hybrid retrieval,
evaluation, even grounded-vs-ungrounded refusal — works with zero keys. Set a real
key for fluent, high-quality generation.

## Quick start (Google Colab)

The notebooks are Colab-compatible. Each notebook's **first cell is a bootstrap**
that installs dependencies and makes `rag_pipeline` importable. Before class:

1. Push this folder to a Git repo.
2. In each notebook's bootstrap cell, set `REPO_URL` to that repo (one line, marked
   `# INSTRUCTOR: set this`).
3. Learners open a notebook in Colab and run the bootstrap cell — it clones the repo
   and installs deps. (Alternatively, upload the `rag_pipeline/` folder and `data/`
   via the Colab file browser; the bootstrap detects and uses them.)
4. For real generation, set `OPENAI_API_KEY` via `getpass` or Colab's secrets
   (`userdata`) — shown in the "Choose your providers" cell. With no key, it runs on
   the mock automatically.

---

## Provider configuration (agnostic by design)

The instructor notes require the session to stay *agnostic of embedding/LLM choice*.
You swap providers with **environment variables only** — no code changes:

| Variable | Default | Options |
|----------|---------|---------|
| `RAG_LLM_PROVIDER` | `openai` | `openai` · `anthropic` · `mock` |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | any model of the chosen provider |
| `RAG_EMBED_PROVIDER` | `openai` | `openai` · `sentence-transformers` · `mock` |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` | e.g. `all-MiniLM-L6-v2` for local |

> Note: **Anthropic has no embeddings API** — so even when you pick Claude for
> generation, embeddings come from OpenAI or a local model. That separation of
> retrieval and generation is itself a teachable architecture point.

---

## The 30-second demo

```python
from rag_pipeline.loaders import load_directory
from rag_pipeline.pipeline import RAGPipeline

docs = load_directory("data/corpus")
pipe = RAGPipeline(k=3).ingest(docs)                 # chunk → embed → store
out  = pipe.query("How much does the Growth plan cost per month?")

print(out["answer"])                                 # grounded answer
print([s["label"] for s in out["sources"]])          # with citations
print(out["trace"].show())                            # glass-box: what each stage did
```

---

## How to teach with this (pedagogy)

Every stage is taught **WHY** (motivation + "what if poorly designed?") → **WHAT**
(concepts, trade-offs, do's/don'ts, failure modes) → **HOW** (from-scratch code first,
LangChain as a *parallel mapping*, never the primary focus). Threaded throughout:
**predict-before-run**, **compare outputs** (chunk sizes / retrieval strategies /
injection methods), **cause→effect** (bad chunking → poor retrieval → bad answer),
and the closing **debug-a-bad-output** workflow (localise the failing stage, then fix
*that* stage). Start with the pipeline diagram and revisit it at every stage.

- **Instructors:** start with `teaching/instructor_guide.md` (minute-by-minute) and
  project `slides/rag_pipelines.html`.
- **Learners:** work the `notebooks/` in order; keep `teaching/learner_handout.md`
  open; practise with `teaching/exercises.md`.

---

## A result you'll reproduce

On the eval set, **hybrid retrieval beats dense** — `recall@3 ≈ 0.96` vs `≈ 0.85` —
because hybrid catches both exact-term queries (`HTTP 429`, via BM25) and paraphrases
(via dense embeddings). That single evidence-based comparison is how you justify a
design decision to a stakeholder: not vibes, measurement.

---

## Additional reading
OpenAI Retrieval & Vector embeddings docs · `sentence-transformers` · LangChain
Retrieval docs · RAGAS.
