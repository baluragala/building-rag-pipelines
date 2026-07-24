# Instructor Guide — Building RAG Pipelines (C8-W4-S1)

**Format:** 180-minute upGrad live session (graduate level)
**Modality:** Live coding + discussion. 8 Google-Colab-compatible notebooks driven from a shared `rag_pipeline` package.
**Pedagogy:** Every stage runs **WHY → WHAT → HOW**. WHY always includes the recurring question **"What happens if this step is poorly designed?"** HOW is **from-scratch first, LangChain as a parallel mapping (secondary)**.
**Central thesis (say it three times):** *RAG performance is the cumulative result of design decisions across ALL stages — not just the choice of LLM.*

> **How to use this guide.** The minute-by-minute run sheet (Section 3) is the spine. Sections 1–2 are your pre-flight. Sections 4–6 are the facilitation playbook, Q&A bank, and timing contingencies you dip into live. Every stage block in Section 3 gives you: the WHY hook, the "what if poorly designed?" question, WHAT talking points, the HOW (notebook + cells), predict-before-run checkpoints, the output comparison to run, and discussion questions with model answers.

---

## 1. Session overview, objectives, and pre-req check

### 1.1 One-paragraph overview
Learners already know *that* RAG is a retriever + a generator. This session makes them fluent in RAG as a **modular pipeline of six stages** — Loading → Chunking → Retrieval → Augmentation → Generation → Evaluation — bracketed by Foundations and Advanced/Conclusion. The through-line is causal: a decision made in an early stage propagates. **Bad chunking → poor retrieval → bad answer.** We build each stage from scratch (so there is no magic), then show the LangChain equivalent as a thin mapping to reinforce that *frameworks abstract mechanics but do not eliminate design decisions.*

### 1.2 Learning objectives
By the end, a learner can:
1. **Diagram** the RAG pipeline and name what each stage decides and what it can and cannot fix.
2. **Explain and predict** the effect of chunk size/overlap, retrieval strategy (dense/sparse/hybrid), reranking, and context-injection method (stuff/map-reduce/refine) on the final answer.
3. **Ground** a generation prompt so the system cites sources and refuses when context is insufficient.
4. **Evaluate** a RAG system with retrieval metrics (precision@k, recall@k, hit-rate, MRR) and generation metrics (faithfulness, answer relevance, RAGAS/LLM-judge), and **build an eval dataset** including negative tests.
5. **Diagnose a "bad RAG output" stage-by-stage** and name the cheapest fix.

### 1.3 Pre-req check (2-min opener — do this live)
Ask for a thumbs-up on each; you are calibrating, not gatekeeping:
- "You've called an LLM API and constructed a prompt." (Y/N)
- "You know RAG = a retriever feeding a generator." (Y/N)
- "You've seen LangChain components (loaders, splitters, retrievers)." (Y/N)
- "You're comfortable with Python + basic text handling." (Y/N)

**If several are shaky on LangChain:** reassure them — LangChain is deliberately *secondary* here; every concept is built from scratch first. **If several lack keys:** point to the mock fallback (Section 2) — the class runs end-to-end with zero keys.

---

## 2. Materials checklist & setup

### 2.1 What you (instructor) need open
- [ ] All 8 notebooks pre-loaded in Colab tabs: `notebooks/01_rag_foundations.ipynb` … `08_conclusion_end_to_end.ipynb`.
- [ ] The pipeline diagram (Section 3.0) on a slide you can return to at every stage.
- [ ] `data/corpus/` (Acme Cloud docs) and `data/eval/eval_dataset.jsonl` visible.
- [ ] This guide + the learner handout.
- [ ] **A prepared "bad RAG output"** for the Evaluation diagnosis (Section 4.2). Have it captured ahead of time in case live generation behaves.

### 2.2 How the notebooks bootstrap in Colab
Each notebook's first cell self-installs and imports the package:
```python
# Colab bootstrap (top of every notebook)
!pip -q install numpy tiktoken
# (optional extras) !pip -q install openai langchain-openai langchain-community sentence-transformers pypdf ragas
# make the rag_pipeline package importable (clone or add to path)
import sys; sys.path.append("/content/building-rag-pipelines")
from rag_pipeline import get_embedder, get_llm, current_config, Document
print(current_config())
```

### 2.3 The mock fallback — the class ALWAYS runs
`config.get_embedder()` and `config.get_llm()` resolve **explicit args → env vars → OpenAI default → deterministic mock**. If a key/library is missing they print a one-line notice and return `MockEmbedder` / `MockLLM`.
- `MockEmbedder` — hash-based bag-of-words vectors; **real vector space** (cosine, chunking comparisons, store mechanics all behave), only semantic quality is missing.
- `MockLLM` — honest **extractive** stand-in: it stitches the retrieved context, never invents facts. This makes the "generation quality gap" visible the moment a real LLM is plugged in.
- **Teaching win:** the mock lets you demo pipeline *mechanics* to a keyless room, then flip one env var to show the quality jump.

### 2.4 Setting a key (for learners who have one)
```python
import os, getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI key: ")   # or Colab: from google.colab import userdata
# Provider switches (all optional):
# os.environ["RAG_LLM_PROVIDER"]="openai"|"anthropic"|"mock"
# os.environ["RAG_LLM_MODEL"]="gpt-4o-mini"
# os.environ["RAG_EMBED_PROVIDER"]="openai"|"st"|"mock"
# os.environ["RAG_EMBED_MODEL"]="text-embedding-3-small"
```
Default stack: **OpenAI `gpt-4o-mini` + `text-embedding-3-small`**. Anthropic Claude is available for generation only (no Anthropic embeddings — a teachable point: **retrieval and generation are genuinely separable and can use different providers**). **Keep the session agnostic of the specific model** — never let a debate about "which embedding model" derail the *design-decisions* message.

---

## 3. Minute-by-minute run sheet (180 minutes)

### 3.0 The pipeline diagram — draw it once, revisit it EVERY stage
Put this on screen at 0:00 and physically point at the current stage each time you move on.

```
                         ┌──────────── INGEST (write path) ────────────┐
   raw sources  ──▶  [1 LOADING]  ──▶  [2 CHUNKING]  ──▶  (EMBED)  ──▶  [VECTOR STORE]
   pdf/md/html/api      clean +          fixed/recursive/    text→vec       numpy cosine
                        metadata         semantic + overlap                 + filtering
                                                                                  │
   user question  ─────────────────────────────────────────────────────────▶  [3 RETRIEVAL]
                                                              dense / BM25 / hybrid(RRF) → (RERANK)
                                                                                  │
                                                                          [4 AUGMENTATION]
                                                                       stuff / map-reduce / refine
                                                                        + labelled context [n]
                                                                                  │
                                                                          [5 GENERATION]
                                                                     grounded prompt, cite [n],
                                                                     refuse if insufficient
                                                                                  │
                                                                             answer + sources
                                                                                  │
                                                                          [6 EVALUATION]
                                                          retrieval metrics + faithfulness/RAGAS
                                                          ──▶ diagnose which stage failed ──▶ loop
```
**Recurring line:** "Generation is the LAST stage, not the WHOLE system. Every arrow before it constrains what it can do."

### 3.1 Master timing table

| Time | Min | Stage | Mode | Notebook |
|------|-----|-------|------|----------|
| 0:00–0:15 | 15 | RAG Foundations | Concept | 01 |
| 0:15–0:30 | 15 | Loading | Concept + demo | 02 |
| 0:30–0:50 | 20 | Chunking | Concept + guided coding | 03 |
| 0:50–1:20 | 30 | Retrieval | Concept + guided coding | 04 |
| 1:20–2:00 | 40 | Augmentation & Generation | Concept + guided coding | 05 |
| 2:00–2:35 | 35 | Evaluation | Concept + guided analysis | 06 |
| 2:35–2:50 | 15 | Explore Advanced RAG | Concept | 07 |
| 2:50–3:00 | 10 | Conclusion / end-to-end | Concept + demo | 08 |

Each block below expands its internal minutes.

---

### STAGE 1 — RAG Foundations (0:00–0:15, 15 min) — Notebook 01

**Internal breakdown:** 2 opener/pre-req · 8 WHY+WHAT · 5 diagram + notebook 01 skim.

**WHY hook (say this):** "You already know an LLM can hallucinate and can't know your private or fresh data. RAG fixes that by *retrieving* evidence and making the model answer from it. But the retriever is only half the story — and the half most people never tune."

**"What happens if this step (the whole pipeline) is poorly designed?"** — Standalone LLMs fail in three ways RAG must address: (a) **knowledge cutoff / no private data**, (b) **hallucination** — confident fabrication, (c) **no provenance** — you can't audit the answer. Preview that each downstream stage can *reintroduce* these failures if designed poorly.

**WHAT talking points:**
- RAG = **retrieve relevant context, then generate grounded on it.** Two sub-systems, six pipeline stages.
- **The modular view (this session's spine):** Loading, Chunking, Retrieval, Augmentation, Generation, Evaluation. Each is a *swappable knob*.
- **Failure localisation preview:** RAG breaks in RETRIEVAL (wrong chunks) or GENERATION (right chunks, wrong answer) — different bugs, different fixes. Evaluation is how you tell them apart.
- Set the thesis: **cumulative design decisions**, not "just use a bigger LLM."

**HOW (notebook 01):** Run the bootstrap cell; `print(current_config())`. Show `RAGPipeline` exists as the end-to-end object you'll build up to. Do NOT deep-dive code yet — 01 is conceptual. Point at the diagram.

**Discussion Q (1):** "Name one thing a bigger LLM CANNOT fix in a RAG system." → *Model answer:* If the relevant chunk was never retrieved, no LLM can answer from it — retrieval recall is a hard ceiling generation cannot raise. Also: provenance/citations come from the pipeline, not model size.

**Discussion Q (2):** "Where could hallucination sneak back in, even with retrieval?" → *Model answer:* In generation, if the prompt doesn't force grounding the model answers from parametric memory; in augmentation, if the answer is buried mid-context ("lost in the middle"); in loading/chunking, if the right text was never ingested cleanly.

---

### STAGE 2 — Loading (0:15–0:30, 15 min) — Notebook 02

**Internal breakdown:** 6 WHY+WHAT · 7 demo (load + clean + metadata) · 2 LangChain mapping + discuss.

**WHY hook:** "Retrieval can only ever surface what ingestion let in. Loading is the quiet stage everyone skips — and where a shocking share of RAG failures are *born*."

**"What happens if this step is poorly designed?"** — A PDF loaded as one giant blob with page furniture (headers, footers, page numbers) glued into sentences will **chunk badly → embed noisily → retrieve wrong spans.** *Garbage in → garbage retrieved → confident garbage out.* Also: **drop the metadata and you lose citations and metadata filtering forever downstream.**

**WHAT talking points (do's & don'ts):**
- **DO:** clean text (but *preserve* paragraph structure — it's semantic signal for the chunker), preserve structure (Markdown headings, PDF pages), **attach metadata** (`source`, `title`, `page`).
- **DON'T:** dump raw bytes, ignore encoding/noise, strip newlines entirely, or throw away provenance.
- The `Document` atom = `page_content` + `metadata` (same shape as LangChain → mapping is a one-liner).

**HOW (notebook 02, from-scratch first):**
```python
from rag_pipeline.loaders import load_directory, load_markdown, clean_text
docs = load_directory("data/corpus")          # dispatches by extension
print(len(docs), docs[0].metadata)            # note title/source/loader metadata
print(clean_text("line\r\n\r\n\r\n   12   \r\nnext"))   # show noise removal
```
- Live-load `acme_pricing.md`; show the `title` came from the H1, `source` is the path.
- Show `load_pdf(per_page=True)` attaches a `page` number → page-level citations later.
- **LangChain parallel mapping (secondary, ~90s):** `load_directory_langchain(folder)` — "the framework abstracts the *loading*, not the design decision of *what to clean and what metadata to keep*."

**Predict-before-run checkpoint:** Before running `clean_text` on a blob with a stray page number and triple blank lines: "Predict what survives." → It collapses 3+ newlines to a paragraph break, removes the isolated page number, keeps single paragraph breaks.

**Output comparison to run:** Load the FAQ **as HTML with vs without** bs4 tag-stripping (show boilerplate `<script>/<nav>` removal), OR clean vs `clean=False` on one doc — point at the noise you'd otherwise embed.

**Discussion Q:** "Why keep `page` in metadata if the user only sees an answer?" → *Model answer:* Citations ("[2] Pricing, p.1"), metadata-filtered retrieval (search only one doc/section), and auditability. Provenance is a *loading* decision you can't recover later.

---

### STAGE 3 — Chunking (0:30–0:50, 20 min) — Notebook 03

**Internal breakdown:** 6 WHY+WHAT · 3 **predict** · 8 guided coding · 3 compare/discuss.

**WHY hook:** "Two hard limits force chunking: embeddings compress a whole passage into ONE vector (longer = more meaning averaged away), and the context window + cost are finite. Chunking is how you slice documents into retrievable, embeddable units."

**"What happens if this step is poorly designed?"** — This is the session's central cause→effect, **say it explicitly: bad chunking → poor retrieval → bad answer.**
- Chunks too **LARGE** → the relevant sentence is *diluted* by surrounding text, similarity drops, you miss it.
- Chunks too **SMALL** → ideas *fragment* across boundaries; the answer needs two chunks that never co-occur.
- Splitting **mid-sentence / mid-table** destroys meaning outright.

**WHAT talking points (strategies & trade-offs):**
- **fixed-size** — split every N chars/tokens. Simple, fast, **structure-blind** (cuts mid-sentence).
- **recursive** — split on the largest natural boundary that fits (paragraph → line → sentence → word). **Structure-aware; the sensible default.**
- **semantic** — split where the *topic* shifts (drop in adjacent-sentence embedding similarity). Best boundaries, highest cost (one embedding per sentence).
- **overlap** — repeat a little text between neighbours so a straddling idea appears whole in at least one chunk. **DO: 10–20% overlap.**

**HOW (notebook 03, from-scratch first):**
```python
from rag_pipeline.chunking import (fixed_size_chunk, recursive_chunk,
    semantic_chunk, chunk_stats, token_len, recursive_chunk_langchain)
fx  = fixed_size_chunk(docs, chunk_size=500, overlap=50)
rc  = recursive_chunk(docs,  chunk_size=500, overlap=50)
print(chunk_stats(fx)); print(chunk_stats(rc))
print(fx[0].page_content[-60:])   # look: fixed cut mid-word/sentence
print(rc[0].page_content[-60:])   # recursive ended on a boundary
```
- Show `semantic_chunk(docs, embedder)` conceptually (works even with mock embedder — mechanism is demonstrable).
- Show `token_len` vs char length: "size in *tokens* is what the model and your bill actually see."
- **LangChain parallel mapping (secondary):** `recursive_chunk_langchain(docs, 500, 50)` — same separator ladder. "The framework abstracts the mechanics; **you still own size, overlap, and what counts as a boundary.**"

**Predict-before-run checkpoints (do BOTH):**
1. "I'll run `fixed_size_chunk` at size 200 then 800 on the pricing doc. **Predict** which produces more chunks and which will fragment the '$199 Growth plan' sentence." → 200 → more, smaller chunks, higher fragmentation risk; 800 → fewer, but the exact fact may sit inside a diluted chunk.
2. "**Predict** what the tail of a fixed-size chunk looks like vs a recursive one." → fixed ends mid-token; recursive ends on `\n\n`/`. `.

**Output comparison to run (required):** Same doc through `fixed_size_chunk` vs `recursive_chunk` at the **same** size/overlap → compare `chunk_stats` (n_chunks, min/max/mean) AND eyeball two boundaries. Then vary size ∈ {200, 500, 1000} and read the cheat-sheet trade-off aloud (small→precise but fragmented; large→context-rich but diluted).

**Discussion Q (1):** "Overlap costs storage and duplicates text. Why pay it?" → *Model answer:* An idea that straddles a boundary would otherwise be split across two chunks and appear whole in neither; 10–20% overlap makes it recoverable in at least one, protecting recall. Too much overlap wastes tokens and inflates the index.

**Discussion Q (2):** "You retrieve the right *document* but the answer sentence is cut in half. Which knob?" → *Model answer:* Chunk size too small / boundary not preserved → use recursive splitting and/or add overlap; this is a chunking bug masquerading as a retrieval miss.

---

### STAGE 4 — Retrieval (0:50–1:20, 30 min) — Notebook 04

**Internal breakdown:** 8 WHY+WHAT · 4 **predict** · 12 guided coding · 6 compare/discuss.

**WHY hook:** "Retrieval decides *what the model gets to see*. **Generation cannot fix a retrieval miss** — if the relevant chunk never enters the context, no prompt on earth recovers it."

**"What happens if this step is poorly designed?"** — the classic failure trio:
1. **Recall miss** — relevant chunk not retrieved at all.
2. **Precision miss** — irrelevant chunks crowd out the good one.
3. **Ranking miss** — the right chunk is present but ranked below the top-k cut.

**WHAT talking points:**
- Embeddings give a space where **distance = dissimilarity**; cosine compares *direction*, not magnitude (normalise!). Two silent traps: comparing vectors from **different embedding models** (incompatible spaces), and **forgetting to normalise**.
- **Vector store** is not magic — hold vectors + docs, return nearest by cosine. `InMemoryVectorStore` does exactly this in numpy, plus **metadata filtering** (constrain the search space *before* ranking) and save/load.
- **Strategies:**
  - **Dense** (`DenseRetriever`) — semantic, but misses exact terms (names, codes, acronyms like `AES-256`, `HTTP 429`).
  - **Sparse / BM25** (`BM25Retriever`, from scratch) — lexical exact-term matching; rewards rare query terms.
  - **Hybrid** (`HybridRetriever`) — **Reciprocal Rank Fusion** of dense + sparse. Fuses *ranks* not raw scores (so no reconciling 0–1 cosine with unbounded BM25). The union surfaces more truly-relevant chunks.
- **Mental model (repeat it):** **optimise RECALL first, then PRECISION. Retrieve wide, rerank narrow.**
  - **Rerankers:** `CrossEncoderReranker` (reads query+doc together — far more accurate than bi-encoder cosine, too slow for the whole corpus → shortlist only); `LLMReranker` (pointwise 0–10 scoring, zero new infra, slower/pricier).
- **Anti-pattern to name:** *don't rely only on top-k dense similarity.*

**HOW (notebook 04, from-scratch first):**
```python
from rag_pipeline.embeddings import embed_documents, cosine_similarity
from rag_pipeline.vectorstore import InMemoryVectorStore
from rag_pipeline.retrieval import (DenseRetriever, BM25Retriever,
    HybridRetriever, CrossEncoderReranker, LLMReranker)

chunks = recursive_chunk(docs, 500, 50)
store  = InMemoryVectorStore().add(chunks, embed_documents(embedder, chunks))
dense  = DenseRetriever(store, embedder)
bm25   = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, bm25)

q = "How are API calls above the plan limit charged?"
for name, r in [("dense",dense),("bm25",bm25),("hybrid",hybrid)]:
    print(name, [ (round(s,3), d.metadata['title']) for d,s in r.retrieve(q, k=3) ])
```
- Show a **metadata filter**: `store.search(embed_query(...), k=3, metadata_filter={"title": "Acme Pricing"})`.
- Show **retrieve-wide-rerank-narrow**: pull k=10 from hybrid, then `LLMReranker(llm).rerank(q, hits, top_n=3)` (or CrossEncoder if `sentence-transformers` installed; `.available` guards it).
- **LangChain parallel mapping (secondary):** `build_langchain_retriever(chunks, k)` — FAISS + OpenAIEmbeddings. "Same steps we built by hand: embed → store → similarity search."

**Predict-before-run checkpoints (do all four are too many — pick 2):**
1. Query **"What encryption does Acme use at rest?"** (ground truth `AES-256`). "**Predict:** dense or BM25 wins on the acronym?" → BM25/lexical nails the exact term; dense may drift. Hybrid should get both.
2. "**Predict** hybrid's recall vs dense's over the eval set." → Hybrid ≥ dense (union of two nets). *(Payoff: the real observed number below.)*

**Output comparison to run (required):** dense vs sparse vs hybrid on 2 queries — a **factual/acronym** query (`AES-256`, `HTTP 429`, `$199`) and a **paraphrase** query. Then the money slide:
> **Observed result to quote:** on this corpus **hybrid recall@3 ≈ 0.96 beats dense ≈ 0.85.** "Same chunks, same LLM — a *retrieval design decision* moved recall 11 points. This is the thesis in one number."

**Discussion Q (1):** "Why does BM25 beat dense on `AES-256` but lose on 'how is my data protected when stored'?" → *Model answer:* BM25 matches exact tokens (great for codes/acronyms, blind to paraphrase); dense matches meaning (great for paraphrase, can miss rare exact tokens). Hybrid + RRF captures both, which is why it wins overall.

**Discussion Q (2):** "You have recall@10 = 0.95 but precision@3 = 0.4. What do you change?" → *Model answer:* Recall is fine (the good chunk is in the pool), precision/ranking is the problem → add a reranker (retrieve wide at k=10, rerank narrow to 3). Don't touch chunking or the prompt yet.

---

### STAGE 5 — Augmentation & Generation (1:20–2:00, 40 min) — Notebook 05

**Internal breakdown:** 4 augmentation WHY+WHAT · 3 predict · 10 augmentation coding/compare · 4 generation WHY+WHAT · 12 generation coding/compare · 7 grounding/citations + discuss.

#### 5A. Augmentation (context injection)

**WHY hook:** "Retrieval found the chunks; augmentation decides **HOW** they enter the prompt. This is the stage learners most often conflate with retrieval."
**Name the distinction sharply (write it up):** **retrieval = WHICH chunks; augmentation = HOW those chunks are packed into the context the LLM reads.**

**"What happens if this step is poorly designed?"** — you blow the context budget (cost + latency), bury the answer **in the middle** of a long context where LLMs attend least (**"lost in the middle"**), or hand the model **unlabelled** text it can't cite.

**WHAT (strategies & trade-offs):**
- **stuff** — concatenate all chunks into one prompt. Simple; fails when chunks exceed the window or dilute attention. *(1 LLM call.)*
- **map-reduce** — answer from each chunk independently (map), combine partials (reduce). Scales past the window; **N+1 calls.** Good when evidence is spread across many chunks.
- **refine** — seed from chunk 1, iteratively revise as each further chunk is read. Good for cumulative reasoning; **sequential → slowest.**
- **`format_context`** labels each chunk `[n] (source)` and **respects a token budget** (`max_tokens`; `None` disables it to *demonstrate* overflow). **`collect_sources`** returns provenance. **DO:** control size, structure the prompt, label with source. **DON'T:** dump everything and hope; drop provenance.

**HOW (notebook 05, from-scratch first):**
```python
from rag_pipeline.augmentation import stuff, map_reduce, refine, format_context, collect_sources
hits = hybrid.retrieve(q, k=4)
print(format_context(hits, max_tokens=800))     # note the [1] (Acme Pricing…) labels
print(stuff(hits))                              # 1 call path
print(map_reduce(llm, q, hits))                 # N+1 calls
print(refine(llm, q, hits))                     # sequential
```

**Predict-before-run checkpoint:** "For a question whose answer lives in **one** chunk vs one spread across **four** chunks — **predict** which injection method wins each." → single-chunk: stuff is fine and cheapest; spread: map-reduce/refine shine.

**Output comparison to run (required):** same `hits`, same question through **stuff vs map-reduce vs refine** — compare answer completeness AND rough call count/latency. Then run `format_context(hits, max_tokens=None)` vs a tight budget to show truncation behaviour.

#### 5B. Generation (grounded prompting)

**WHY hook:** "The LLM is the LAST stage — and the one most people wrongly treat as the *whole* system. Its job is narrow but critical: synthesise a grounded answer from the context it was handed, **cite it, and refuse when the context doesn't support an answer.**"

**"What happens if this step is poorly designed?"** — the model ignores context and answers from parametric memory (**hallucination**), pads with fluent filler, or fails to say "I don't know" when it should. **Most of these are *prompt* failures, not model failures** — which is why prompt design is a RAG *design decision*.

**WHAT (safe RAG prompting):**
- **Instruction:** use ONLY the context.
- **Grounding rule** (the single highest-leverage line): *answer "I don't know based on the provided context." when the context is insufficient.* Kills a lot of hallucination.
- **Citations:** require `[n]` tags tied to the numbered blocks.
- **Structure:** separate `RAG_SYSTEM` (role/rules) from `RAG_PROMPT` (context + question). `UNGROUNDED_PROMPT` exists ONLY to demo the failure mode.

**HOW (notebook 05):**
```python
from rag_pipeline.generation import generate_answer, answer_with_sources, RAG_SYSTEM, RAG_PROMPT
ctx = stuff(hits)
print(generate_answer(llm, q, ctx, grounded=True))    # cites [n], refuses if thin
print(generate_answer(llm, q, ctx, grounded=False))   # ungrounded — watch it drift
res = answer_with_sources(llm, hybrid, q, k=4)         # retrieve→augment→generate in one call
print(res["answer"]); print(res["sources"])
```
- **The negative test live:** ask **"What is the CEO of Acme Cloud's name?"** (not in corpus). Grounded → *"I don't know based on the provided context."* Ungrounded → likely a fabricated name. **This is the whole grounding lesson in one contrast.**
- **LangChain parallel mapping (secondary):** `build_rag_chain_langchain(lc_retriever)` — an LCEL chain that encodes the **same** `RAG_SYSTEM`/grounding/citation decisions. "The framework does not make these decisions for you."

**Output comparison to run (required):** grounded vs ungrounded prompt on (a) an answerable factual query and (b) an unanswerable one → grounding changes both faithfulness and refusal behaviour.

**Discussion Q (1):** "The answer is fluent, confident, and wrong. The right chunk WAS retrieved. Which stage?" → *Model answer:* Generation (or augmentation "lost in the middle") — not retrieval. Fix the prompt (grounding + citations) before touching the retriever or the model.

**Discussion Q (2):** "Why require `[n]` citations even if users won't read them?" → *Model answer:* Citations force the model to tie each claim to a retrieved block (reducing fabrication) and make faithfulness auditable — for you and for evaluation.

---

### STAGE 6 — Evaluation (2:00–2:35, 35 min) — Notebook 06

**Internal breakdown:** 8 WHY+WHAT · 10 metrics coding · 10 **"bad RAG output" diagnosis** · 4 eval-dataset guide · 3 discuss.

**WHY hook:** "How do you know your RAG system is any good? Without measurement, every stage decision — chunk size, k, reranker, prompt — is just opinion. Evaluation turns RAG from art into engineering, and it **localises failure**: RAG breaks in RETRIEVAL (wrong chunks) or GENERATION (right chunks, wrong answer) — and the fix is completely different."

**"What happens if this step is poorly designed?"** — you **optimise the wrong thing**: tune the prompt for weeks when the real problem was recall, or ship a system that scores well on fluency while quietly hallucinating.

**WHAT (two metric families):**
- **RETRIEVAL metrics** (is the right context found?), vs ground-truth relevant ids:
  - `precision_at_k` — of top-k, fraction relevant.
  - `recall_at_k` — of all relevant, fraction in top-k. **Watch this FIRST when debugging.**
  - `hit_rate_at_k` — did ≥1 relevant chunk make top-k (1/0).
  - `mrr` — reciprocal rank of first hit (rewards ranking it high).
  - `evaluate_retriever(retriever, examples, k, id_fn=source_id_of)` — averages them, **skips negatives**. Default **source-level** ids (robust to chunk size); pass `id_fn=_id_of` for chunk-level.
- **GENERATION metrics** (is the answer faithful & correct?):
  - `faithfulness_heuristic` — embedding cosine of answer vs context (cheap proxy, NOT a substitute for RAGAS).
  - `answer_relevance_heuristic` — answer vs question cosine.
  - `refusal_correct` — did a negative get an "I don't know"?
  - `llm_judge_faithfulness` — LLM grades 0–1 "is every claim supported?"
  - `evaluate_with_ragas` — faithfulness, answer_relevancy, context_precision/recall (needs a key; shows workflow even when it returns None).
- **Metrics vs rubrics:** numbers for what's countable; human rubrics for what isn't. **Negative tests are non-negotiable** (the eval set has 3: mobile app, CEO name, Starter phone hotline — all correct answer = refusal).

**HOW (notebook 06):**
```python
from rag_pipeline.evaluation import (load_eval_dataset, evaluate_retriever,
    faithfulness_heuristic, refusal_correct, llm_judge_faithfulness, evaluate_with_ragas)
ex = load_eval_dataset("data/eval/eval_dataset.jsonl")   # 21 examples, 3 negatives
print("dense ", evaluate_retriever(dense,  ex, k=3))
print("hybrid", evaluate_retriever(hybrid, ex, k=3))     # recall ↑ vs dense
```

**THE SET-PIECE — lead the "bad RAG output" diagnosis (10 min, see Section 4.2 for the script):** Present a prepared bad answer and walk the pipeline diagram **backwards**, stage by stage, using the trace, until you localise the cause. This is the highest-value 10 minutes of the session.

**Predict/compare:** predict hybrid vs dense recall@3 before running (callback to the 0.96 vs 0.85 result); run both live.

**Eval-dataset mini-guide (say aloud, 4 min):** an `EvalExample` = `question` + `ground_truth` + `relevant_ids` (+ `is_negative`). Steps: (1) pick questions real users ask; (2) label the relevant source(s) — source-level is robust to chunk size; (3) **add negatives** (unanswerable → refusal); (4) `draft_questions_from_chunks(llm, chunks)` bootstraps candidates **but a human must curate — never ship auto-generated ground truth unreviewed.**

**Discussion Q (1):** "recall@3 is high but faithfulness is low. Retrieval or generation bug?" → *Model answer:* Generation — the right context is being retrieved but the answer isn't grounded in it. Fix the prompt (grounding/citations), not the retriever.

**Discussion Q (2):** "Why evaluate at source-level by default instead of chunk-level?" → *Model answer:* Ground truth 'which document answers this' doesn't change when you re-tune the chunker; chunk-level ids are more precise but brittle across configs. Use chunk-level when you're specifically tuning chunking.

---

### STAGE 7 — Explore Advanced RAG (2:35–2:50, 15 min) — Notebook 07

**Internal breakdown:** 12 concept tour · 3 discuss. Conceptual only — no live coding.

**WHY hook:** "Everything so far is *Naïve/Advanced RAG* on a single hop. Real systems hit questions one retrieval can't answer. Here's the map of where the field goes — and every one of these is still *the same six stages*, rearranged."

**"What happens if poorly designed?"** — advanced patterns add moving parts (more calls, more latency, more failure surface). Reach for them only when evaluation shows a concrete gap; complexity you can't measure is complexity you can't defend.

**WHAT (tour — keep each to ~90s):**
- **Naïve → Advanced → Modular RAG** — from a straight line to pre/post-retrieval optimisation (query rewriting, reranking) to fully swappable modules (what our `RAGPipeline` already is).
- **Agentic RAG** — an agent decides *whether* and *what* to retrieve, can call tools, can loop.
- **Multi-hop RAG** — chain retrievals: answer sub-question 1, use it to retrieve for sub-question 2.
- **Graph RAG** — retrieve over a knowledge graph, not just chunks; good for entity/relationship questions.
- **Adaptive / self-RAG** — the system decides when it has enough evidence or should retrieve more.
- **Long-context** — huge windows reduce *some* chunking pressure but don't remove retrieval (cost, "lost in the middle", and you still shouldn't stuff everything).

**Reinforce:** frameworks and fancy patterns **abstract but do not eliminate design decisions.** Each pattern still needs good loading, chunking, grounding, and evaluation.

**Discussion Q:** "When would you choose multi-hop over just increasing k?" → *Model answer:* When the question decomposes into dependent sub-questions where the second retrieval *depends on the first answer* (e.g., 'what's the SLA credit for the tier that includes SSO?'). Bigger k widens one hop; it can't chain reasoning.

---

### STAGE 8 — Conclusion / end-to-end (2:50–3:00, 10 min) — Notebook 08

**Internal breakdown:** 6 end-to-end demo · 4 recap + closing takeaway.

**WHY hook:** "Let's assemble the whole pipeline in one object and prove the thesis by *changing one knob at a time*."

**HOW (notebook 08):**
```python
from rag_pipeline import RAGPipeline
from rag_pipeline.retrieval import DenseRetriever, BM25Retriever, HybridRetriever

pipe = RAGPipeline(injection="stuff", k=4).ingest(docs)   # defaults: recursive chunker, dense
out  = pipe.query("How are overage API calls charged?")
print(out["answer"]); print(out["sources"])
print(out["trace"].show())     # retrieved chunks + scores + timings — the debugging surface
```
- **Live A/B:** swap `retriever_factory=lambda store,emb: HybridRetriever(DenseRetriever(store,emb), BM25Retriever(pipe.chunks))` or change `injection`/`k`/`reranker`/chunker and re-`query`. Show the trace change.
- Close with a 60-second walk of the pipeline diagram, pointing at each stage's one key decision.

**Closing takeaway (say verbatim):** **"RAG performance is the cumulative result of design decisions across ALL stages — not just the choice of LLM. Loading, chunking, retrieval, augmentation, generation, and evaluation each set a ceiling the next stage cannot raise. Frameworks abstract the mechanics; you still own the design."**

---

## 4. Facilitation notes

### 4.1 Running predict-then-run (chunking & retrieval especially)
1. **State the exact experiment** before executing ("size 200 vs 800 on the pricing doc").
2. **Force a commitment** — poll, chat, or hands: "more chunks at 200 or 800?" Silence is not prediction; make them pick.
3. **Run it.**
4. **Reconcile** — name *why* the surprise happened in causal terms ("more chunks because step = size − overlap shrank"). The learning is in the gap between prediction and result, so never skip the reconcile.
5. Keep predictions binary/directional so everyone can commit fast.

### 4.2 Leading the "bad RAG output" diagnosis (the Evaluation set-piece)
Present a concrete bad answer to an Acme question — e.g. Q: *"What's the service credit if Growth misses its SLA?"* → A (bad): a confident, wrong percentage with no citation. Then walk the diagram **backwards**, asking the room at each stage:

1. **Generation first (cheapest to check):** Is it grounded/cited? Run `generate_answer(..., grounded=True)` vs the bad ungrounded one. If grounding fixes it → **generation bug (prompt).** Stop.
2. **Augmentation:** Was the answer buried mid-context or was the budget too tight (truncated the SLA chunk)? Inspect `format_context(hits)`. If the credit table got truncated → **augmentation bug.**
3. **Retrieval:** `pipe.query(q)["trace"].show()` — is the SLA chunk even in the retrieved set with a decent score? `evaluate_retriever(retriever, [that_example], k=3)`. If recall = 0 → **retrieval bug** (try hybrid / bump k / rerank).
4. **Chunking:** If the SLA credit sentence was split across two chunks → **chunking bug** (recursive + overlap).
5. **Loading:** If the SLA doc never ingested cleanly (table mangled) → **loading bug.**

**The lesson to land:** *diagnose in order of cost — check the last stage first, walk backwards, and let the metrics/trace tell you where the ceiling was set.* Cause→effect: a low number in an early stage caps everything after it.

### 4.3 Keeping LangChain as parallel mapping (not the main event)
- Always build **from scratch first**, then show the LangChain one-liner as "the same thing, wrapped."
- Cap each LangChain detour at ~60–90 seconds. Use the sentence: **"The framework abstracts the mechanics; it does not make the design decision for you"** every time.
- If a learner wants to go deep on LangChain internals, park it: "Great question for the forum — today we're learning the *decisions*, and those are framework-independent."
- The mapping functions to point at: `load_directory_langchain`, `recursive_chunk_langchain`, `build_langchain_retriever`, `build_rag_chain_langchain`. Note each encodes the *same* choices we made by hand.

### 4.4 Common misconceptions & how to correct them
| Misconception | Correction |
|---|---|
| **"Retrieval and augmentation are the same thing."** | Retrieval = WHICH chunks. Augmentation = HOW they're packed into the prompt (stuff/map-reduce/refine, labelling, budget). Write both on the board and keep them separate all session. |
| **"Just use a bigger/better LLM."** | A bigger LLM cannot answer from a chunk that was never retrieved (recall ceiling), cannot cite what it wasn't given, and won't refuse unless the prompt tells it to. Quote hybrid 0.96 vs dense 0.85: *same LLM*, better retrieval. |
| **"Smaller chunks are always more precise / bigger chunks are always better."** | Both extremes fail: too large → embedding dilution → recall miss; too small → idea fragmentation → half the answer. It's a trade-off tuned per corpus, not a monotonic dial. |
| **"Dense embeddings retrieval is state-of-the-art, BM25 is old."** | Dense misses exact terms (codes, acronyms, IDs). BM25 nails them. Hybrid (RRF) beats both here. Old ≠ useless. |
| **"More overlap = better."** | Overlap protects straddling ideas but wastes tokens, inflates the index, and duplicates hits. 10–20% is the sweet spot. |
| **"Evaluation is optional / eyeballing is enough."** | Without metrics you can't tell a retrieval bug from a generation bug, so you optimise the wrong stage. Show `evaluate_retriever` localising the fault. |
| **"RAGAS/LLM-judge numbers are ground truth."** | They're LLM-generated proxies (they call a model under the hood). Combine with human rubrics; never ship auto-generated ground truth unreviewed. |
| **"Long context windows kill RAG / make chunking obsolete."** | Bigger windows ease *some* chunking pressure but retrieval still matters for cost, latency, and 'lost in the middle'. You still shouldn't stuff everything. |

---

## 5. Anticipated Q&A

1. **"Do I need an API key to follow along?"** — No. The mock embedder + mock LLM run every notebook offline (real vector space, honest extractive answers). Plug a key in later to see the quality jump.
2. **"OpenAI vs Anthropic vs local embeddings — which is best?"** — Deliberately out of scope; the config is provider-agnostic. Note Anthropic has *no embeddings API*, so even a Claude generator uses OpenAI/local embeddings — retrieval and generation are separable.
3. **"What chunk size should I use?"** — There's no universal answer; it depends on your docs and questions. Start ~300–800 tokens, 10–20% overlap, recursive splitter, then *measure* recall and tune. (Point at the cheat sheet.)
4. **"How do I pick k?"** — Retrieve wide for recall (k=8–10 into a reranker), present narrow for precision (top 3–4 to the LLM). Let `precision@k`/`recall@k` guide it.
5. **"When is hybrid worth the extra complexity?"** — When queries mix exact terms (codes, names, prices) with paraphrase. Our eval shows +11 pts recall for near-zero code cost. Measure on your own eval set.
6. **"RRF — why fuse ranks instead of scores?"** — Because dense cosine (0–1) and BM25 (unbounded) aren't comparable. Ranks are. RRF sums 1/(c+rank), robustly favouring docs both retrievers rank high.
7. **"Cross-encoder vs LLM reranker?"** — Cross-encoder: more accurate, needs `sentence-transformers`, cheap per call. LLM reranker: zero new infra, reuses your generator, slower/pricier per query. Both run on a *shortlist* only.
8. **"stuff vs map-reduce vs refine — default?"** — stuff for few chunks that fit the window (1 call, cheapest). map-reduce when evidence is spread across many chunks. refine when later evidence should revise earlier claims. Start with stuff; escalate when evaluation shows a gap.
9. **"How do I stop hallucination?"** — Three levers: retrieve the right context (recall), ground the prompt ("answer 'I don't know' if insufficient" + require `[n]` citations), and evaluate faithfulness. Most hallucination is a *prompt* failure, not a model failure.
10. **"How big should my eval set be?"** — Start small (20–30 curated examples, like our 21) covering factual, list, paraphrase, and **negative** questions. Quality and coverage beat size; grow it as real failures surface.
11. **"Why 3 negative tests? Aren't they just failures?"** — Negatives test *refusal* — the correct answer is "I don't know." A system that never refuses is a system that hallucinates on out-of-scope questions. Use `refusal_correct`.
12. **"Does the trace add latency in production?"** — `RAGTrace` is lightweight (previews + timings). Keep it in staging/debug; it's the surface you use to localise a bad output to a stage.
13. **"Where does a real vector DB (FAISS/Chroma/Pinecone) fit?"** — Exactly where `InMemoryVectorStore` sits — same interface (add + search-by-cosine), swapped for an ANN index at scale. We built it in numpy so there's no magic to fear.
14. **"Can I use different providers for retrieval and generation?"** — Yes, and often you should. `config` makes each independent (`RAG_EMBED_PROVIDER` vs `RAG_LLM_PROVIDER`).

---

## 6. Timing contingencies

**If running LONG (need to cut ~10–15 min):**
- **Cut first:** the LangChain parallel-mapping detours (they're secondary by design) — mention they exist, skip running them.
- Cut `semantic_chunk` live run; describe it and show `chunk_stats` from fixed vs recursive only.
- In Retrieval, cut the cross-encoder path (guard on `.available`), keep `LLMReranker` conceptual.
- In Augmentation, demo stuff + map-reduce; describe refine without running.
- Advanced RAG (Stage 7) is the biggest compressible block — trim to Naïve→Advanced→Modular + one pattern (agentic or multi-hop) and the "still the same six stages" line.
- **Never cut:** the chunking predict-then-run, the dense-vs-hybrid comparison (0.96 vs 0.85), the grounded-vs-ungrounded contrast, and the "bad RAG output" diagnosis. Those four carry the thesis.

**If running AHEAD (need to fill ~10–15 min):**
- Expand the **"bad RAG output"** diagnosis — have the room propose fixes at each stage and run them live via `trace.show()`.
- Add a second predict-then-run in Retrieval (metadata filtering: predict how filtering to one doc changes precision).
- Run `evaluate_with_ragas` if keys are present, or `llm_judge_faithfulness` on a grounded vs ungrounded answer, and compare to the cheap `faithfulness_heuristic`.
- Let learners tune `RAGPipeline` knobs (chunk size, k, injection, reranker) in Stage 8 and race for the best recall/faithfulness on the eval set.
- Deepen a chosen Advanced pattern (multi-hop or Graph RAG) with a whiteboard walk-through mapping it back to the six stages.

---

*End of instructor guide. Companion: `learner_handout.md`.*
