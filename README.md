# Building RAG Pipelines

A complete, hands-on teaching package for the 180-minute session
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

## Local setup with a virtual environment (recommended)

Always work inside a **virtual environment** so this project's packages stay isolated
from your system Python. You need **Python 3.9+** installed first
([python.org/downloads](https://www.python.org/downloads/) — on Windows tick
*"Add Python to PATH"* during install).

### Option A — one-shot setup script

From the project root:

| OS | Command |
|----|---------|
| **macOS / Linux** | `bash scripts/setup.sh` |
| **Windows (PowerShell)** | `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |

The script creates `.venv/`, installs everything in `requirements.txt` plus
JupyterLab, and registers a **"Python (RAG Pipelines)"** Jupyter kernel. When it
finishes, activate the environment (see the activate command for your OS below) and
run `jupyter lab`.

### Option B — manual steps (same result)

**macOS / Linux (bash / zsh)**
```bash
python3 -m venv .venv                 # 1. create the virtual environment
source .venv/bin/activate             # 2. activate it  (prompt now shows (.venv))
python -m pip install --upgrade pip   # 3. upgrade pip
pip install -r requirements.txt       # 4. install dependencies
pip install jupyterlab ipykernel      # 5. (for local notebooks) install Jupyter
cp .env.example .env                  # 6. optional: add OPENAI_API_KEY to .env
jupyter lab                           # 7. open notebooks/01_rag_foundations.ipynb
```

**Windows — PowerShell**
```powershell
py -3 -m venv .venv                   # 1. create the virtual environment
.\.venv\Scripts\Activate.ps1          # 2. activate it  (prompt now shows (.venv))
python -m pip install --upgrade pip   # 3. upgrade pip
pip install -r requirements.txt       # 4. install dependencies
pip install jupyterlab ipykernel      # 5. (for local notebooks) install Jupyter
copy .env.example .env                # 6. optional: add OPENAI_API_KEY to .env
jupyter lab                           # 7. open notebooks/01_rag_foundations.ipynb
```
> If PowerShell blocks activation with a script-execution error, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and retry, or use Option A.

**Windows — Command Prompt (cmd.exe)**
```bat
py -3 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab ipykernel
copy .env.example .env
jupyter lab
```

### Everyday use

| Action | macOS / Linux | Windows (PowerShell) |
|--------|---------------|----------------------|
| **Activate** the env | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| **Deactivate** | `deactivate` | `deactivate` |
| Run a notebook | `jupyter lab` → pick the **Python (RAG Pipelines)** kernel | same |
| Run the smoke test | `python -c "import rag_pipeline; print('ok')"` | same |

> **No API key? It still runs.** Every module falls back to a deterministic **offline
> mock** provider, so the whole session — chunking comparisons, hybrid retrieval,
> evaluation, even grounded-vs-ungrounded refusal — works with zero keys inside the
> venv. Set a real `OPENAI_API_KEY` (in `.env` or your shell) for fluent generation.

## Quick start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/building-rag-pipelines/blob/main/notebooks/01_rag_foundations.ipynb)
&nbsp; ← start here (Notebook 01). Every notebook has its own **Open in Colab** badge at the top.

The notebooks are Colab-compatible and **already point at this repo**
(`REPO_URL = https://github.com/baluragala/building-rag-pipelines.git` in each
notebook's first cell), so there's nothing to configure:

1. Open a notebook in Colab (e.g. via *File → Open notebook → GitHub*).
2. Run the **first cell (bootstrap)** — it clones this repo, installs dependencies,
   and makes `rag_pipeline` importable. *(Fork the repo and change that one
   `REPO_URL` line if you host your own copy. Alternatively, upload the
   `rag_pipeline/` folder and `data/` via the Colab file browser — the bootstrap
   detects and uses them.)*
3. **Set your OpenAI key once in Colab Secrets** — left sidebar → 🔑 key icon → add
   a secret named `OPENAI_API_KEY`, and toggle **Notebook access** on. The "Choose
   your providers" cell reads it automatically and **defaults to the OpenAI stack**
   (`gpt-4o-mini` + `text-embedding-3-small`). With no key it falls back to the
   offline mock so the notebook still runs.

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
