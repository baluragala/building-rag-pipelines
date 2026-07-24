# Learner Handout — Building RAG Pipelines (C8-W4-S1)

**Take-home study notes for the 180-minute session.** Keep this next to the 8 notebooks and the `rag_pipeline` package.

**The one idea to remember:** *RAG performance is the cumulative result of design decisions across ALL stages — Loading, Chunking, Retrieval, Augmentation, Generation, Evaluation — not just the choice of LLM. Each stage sets a ceiling the next one cannot raise.*

**How to read each stage:** **WHY** it exists → **WHAT** the concept and trade-offs are → **HOW** to try it (from scratch first; LangChain is a thin parallel mapping). Watch for the recurring question **"What happens if this step is poorly designed?"** — the answer is the failure mode you must guard against.

---

## The pipeline (keep this in view)

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

**Cause → effect chain:** bad **loading** → noisy text → bad **chunking** → poor **retrieval** → weak **augmentation** → bad **generation**. A miss upstream cannot be repaired downstream. This is why we evaluate to find *which* stage failed.

**Setup reminder:** notebooks run with **zero API keys** via the deterministic mock (`get_embedder()` / `get_llm()` fall back to `MockEmbedder`/`MockLLM`). To use real models: set `OPENAI_API_KEY` (via `getpass`), optionally `RAG_LLM_PROVIDER`/`RAG_EMBED_PROVIDER`. Default stack = OpenAI `gpt-4o-mini` + `text-embedding-3-small`. The session is **agnostic to the specific model** — the lessons are about *design decisions*, not model choice.

---

## Stage 0 — RAG Foundations *(Notebook 01)*

**WHY.** A standalone LLM has three limits RAG addresses: (1) **knowledge cutoff / no private data**, (2) **hallucination** (confident fabrication), (3) **no provenance** (can't audit the answer). RAG retrieves evidence and makes the model answer *from it*.

**WHAT.** RAG = *retrieve relevant context, then generate grounded on it.* The powerful mental model for this session is the **modular pipeline** of six swappable stages. RAG breaks in one of two places — **RETRIEVAL** (wrong chunks) or **GENERATION** (right chunks, wrong answer) — and the two need completely different fixes.

**What if poorly designed?** Any downstream stage can *reintroduce* the very failures RAG was meant to fix.

**HOW (API to try).** `from rag_pipeline import get_embedder, get_llm, current_config, RAGPipeline, Document` — `print(current_config())` to see the active providers.

---

## Stage 1 — Loading *(Notebook 02)*

**WHY.** Retrieval can only surface what ingestion let in. A large share of RAG failures are *born* here.

**WHAT.** Turn raw sources (text/Markdown/HTML/PDF/API) into `Document`s (`page_content` + `metadata`). Clean the noise but **preserve structure** (paragraph breaks, headings, page numbers are semantic signal). **Attach metadata** — you need it for citations and metadata-filtered retrieval later.

**What if poorly designed?** A PDF loaded as one blob with page furniture glued into sentences → chunks badly → embeds noisily → retrieves wrong spans. *Garbage in → garbage retrieved → confident garbage out.* Drop metadata → no citations, no filtering, ever.

**Do's & don'ts.**
- DO: `clean_text` (normalise line endings, collapse blank-line runs, strip stray page numbers) **without** killing paragraph breaks; preserve Markdown/PDF structure; attach `source`, `title`, `page`.
- DON'T: dump raw bytes; ignore encoding/noise; embed nav/boilerplate; discard provenance.

**Failure modes.** Boilerplate embedded as content; encoding garbage; lost page/section provenance; a table flattened into one line.

**Key trade-off.** Aggressive cleaning removes noise but can destroy structure — clean *conservatively*.

**HOW (API to try).**
```python
from rag_pipeline.loaders import load_directory, load_markdown, load_pdf, load_html, load_api, clean_text
docs = load_directory("data/corpus")     # dispatch by extension; every doc carries metadata
# parallel mapping (secondary): load_directory_langchain("data/corpus")
```

---

## Stage 2 — Chunking *(Notebook 03)*

**WHY.** Embeddings compress a passage into ONE vector (longer = more meaning averaged away), and context windows + cost are finite. Chunking slices documents into retrievable, embeddable units.

**WHAT.** Three strategies + overlap:
- **fixed-size** — every N chars/tokens. Simple, fast, **structure-blind** (cuts mid-sentence).
- **recursive** — split on the largest natural boundary that fits (paragraph → line → sentence → word). **Structure-aware; the default.**
- **semantic** — split where the topic shifts (drop in adjacent-sentence similarity). Best boundaries, highest cost.
- **overlap** — repeat a little text between neighbours so a straddling idea appears whole in ≥1 chunk.

**What if poorly designed? (the session's central cause→effect):** **bad chunking → poor retrieval → bad answer.** Too **large** → the key sentence is *diluted*, similarity drops, recall miss. Too **small** → ideas *fragment* across chunks, the model gets half the story. Mid-sentence cuts destroy meaning.

**Do's & don'ts.** DO preserve semantic boundaries; add 10–20% overlap. DON'T dilute (too big) or fragment (too small).

**Failure modes.** Fact split across two chunks; answer diluted inside a huge chunk; table/list cut apart.

### Chunk size-vs-overlap cheat sheet
| Chunk size | Effect | Good for | Risk |
|---|---|---|---|
| **Small (≈100–300 tok)** | precise, focused embeddings | fact lookup, short QA | idea **fragmentation**; more chunks, bigger index |
| **Medium (≈300–800 tok)** | balanced — the usual default | most RAG | mild dilution; tune with overlap |
| **Large (≈800–1500 tok)** | rich context per chunk | narrative, reasoning | **dilution** → similarity drops → recall miss; cost |
| **Overlap 0%** | no duplication, smallest index | clean-boundary docs | ideas split at boundaries lost |
| **Overlap 10–20%** | straddling ideas recoverable | **recommended default** | small token/index overhead |
| **Overlap >30%** | maximum safety | rare | wasted tokens, duplicate hits, bloated index |

**HOW (API to try).**
```python
from rag_pipeline.chunking import (fixed_size_chunk, recursive_chunk, semantic_chunk,
    chunk_stats, token_len, recursive_chunk_langchain)
rc = recursive_chunk(docs, chunk_size=500, overlap=50)
print(chunk_stats(rc))                 # n_chunks, min/max/mean/median
# Predict-then-run: fixed vs recursive at the SAME size — eyeball the boundaries.
```

---

## Stage 3 — Retrieval *(Notebook 04)*

**WHY.** Retrieval decides *what the model gets to see.* **Generation cannot fix a retrieval miss** — if the chunk never enters the context, no prompt recovers it.

**WHAT.** Embeddings put text in a space where **distance = dissimilarity**; cosine compares direction, not magnitude (**normalise!**). A vector store just holds vectors + docs and returns the nearest. Strategies:
- **Dense** — semantic similarity; misses exact terms (codes, acronyms like `AES-256`, `HTTP 429`).
- **Sparse / BM25** — lexical exact-term matching; rewards rare query terms; blind to paraphrase.
- **Hybrid** — **Reciprocal Rank Fusion (RRF)** of dense + sparse; fuses *ranks* (robust — no reconciling 0–1 cosine with unbounded BM25). Usually best.
- **Rerankers** — **cross-encoder** (reads query+doc together, very accurate, shortlist only) or **LLM reranker** (0–10 scoring, zero new infra, slower). Apply to a shortlist, not the whole corpus.
- **Metadata filtering** — constrain the candidate set *before* ranking (cheap precision lever).

**What if poorly designed?** The failure trio: **recall miss** (relevant chunk not retrieved), **precision miss** (junk crowds it out), **ranking miss** (right chunk below the top-k cut).

**The mental model.** **Optimise RECALL first, then PRECISION. Retrieve wide, rerank narrow.** Anti-pattern: relying only on top-k dense similarity.

**Silent traps.** Comparing vectors from *different* embedding models (incompatible spaces); forgetting to normalise (magnitude leaks into similarity).

### Dense vs sparse vs hybrid
| | **Dense** | **Sparse (BM25)** | **Hybrid (RRF)** |
|---|---|---|---|
| Matches on | meaning / semantics | exact tokens / terms | both |
| Great at | paraphrase, synonyms | codes, acronyms, names, IDs | mixed queries |
| Weak at | rare exact terms | paraphrase, synonyms | (little — extra compute) |
| Needs | an embedding model | just the corpus tokens | dense + sparse |
| **Observed here** | **recall@3 ≈ 0.85** | strong on exact-term Qs | **recall@3 ≈ 0.96** ✅ |

> **Remember this result:** on the Acme corpus, **hybrid recall@3 ≈ 0.96 beat dense ≈ 0.85** — same chunks, same LLM. An 11-point gain from a *retrieval design decision*. That's the thesis in one number.

**HOW (API to try).**
```python
from rag_pipeline.embeddings import embed_documents, embed_query, cosine_similarity
from rag_pipeline.vectorstore import InMemoryVectorStore
from rag_pipeline.retrieval import (DenseRetriever, BM25Retriever, HybridRetriever,
    CrossEncoderReranker, LLMReranker)
store  = InMemoryVectorStore().add(chunks, embed_documents(embedder, chunks))
dense  = DenseRetriever(store, embedder)
hybrid = HybridRetriever(dense, BM25Retriever(chunks))
hits   = hybrid.retrieve("How are overage API calls charged?", k=10)   # retrieve wide
hits   = LLMReranker(llm).rerank("...", hits, top_n=3)                  # rerank narrow
# metadata filter: store.search(embed_query(embedder, q), k=3, metadata_filter={"title":"Acme Pricing"})
# parallel mapping (secondary): build_langchain_retriever(chunks, k=4)  # FAISS
```

---

## Stage 4 — Augmentation *(Notebook 05)*

**WHY.** Retrieval found the chunks; augmentation decides **HOW** they enter the prompt.

**WHAT — the distinction to burn in:** **retrieval = WHICH chunks; augmentation = HOW they're packed into the context the LLM reads.** Strategies:
- **stuff** — concatenate all chunks into one prompt. Simplest; **1 call**. Fails past the window or dilutes attention.
- **map-reduce** — answer per chunk (map), then combine (reduce). Scales past the window; **N+1 calls**.
- **refine** — seed from chunk 1, revise as each further chunk is read. Cumulative reasoning; **sequential → slowest**.
- Always **label chunks `[n] (source)`** (`format_context`) and **keep a token budget**; return **sources** (`collect_sources`) for auditability.

**What if poorly designed?** Blow the context budget (cost + latency); bury the answer **mid-context** where LLMs attend least (**"lost in the middle"**); hand the model unlabelled text it can't cite.

**Do's & don'ts.** DO control context size, structure the prompt, label each chunk with its source. DON'T dump everything and hope; don't drop provenance.

### Context-injection comparison
| | **stuff** | **map-reduce** | **refine** |
|---|---|---|---|
| LLM calls | 1 | N + 1 | N (sequential) |
| Scales past window? | ✗ | ✓ | ✓ |
| Best when | few chunks fit the window | evidence spread across many chunks | later evidence should revise earlier |
| Cost / latency | lowest | higher (parallelisable) | highest (sequential) |
| Risk | dilution, "lost in the middle" | reduce step drops nuance | error propagation across steps |

**HOW (API to try).**
```python
from rag_pipeline.augmentation import stuff, map_reduce, refine, format_context, collect_sources
print(format_context(hits, max_tokens=800))   # labelled [1] (source) blocks, budgeted
ctx = stuff(hits)                              # or map_reduce(llm, q, hits) / refine(llm, q, hits)
```

---

## Stage 5 — Generation *(Notebook 05)*

**WHY.** The LLM is the **last** stage, not the whole system. Its job: synthesise a **grounded** answer from the given context, **cite it**, and **refuse** when context is insufficient.

**WHAT — safe RAG prompting.**
- **Instruction:** use ONLY the context.
- **Grounding rule (highest leverage):** answer *"I don't know based on the provided context."* when context is insufficient.
- **Citations:** require `[n]` tags tied to the numbered blocks.
- **Structure:** separate `RAG_SYSTEM` (role/rules) from `RAG_PROMPT` (context + question). `UNGROUNDED_PROMPT` exists only to demonstrate the failure.

**What if poorly designed?** The model answers from parametric memory (**hallucination**), pads with fluent filler, or won't say "I don't know." **Most of these are *prompt* failures, not model failures.**

**Do's & don'ts.** DO ground, cite, and structure the prompt; DON'T let the model use outside knowledge or skip refusal on out-of-scope questions.

**Failure modes.** Confident wrong answer despite correct context; no citations; answering an unanswerable question instead of refusing.

**HOW (API to try).**
```python
from rag_pipeline.generation import generate_answer, answer_with_sources, RAG_SYSTEM, RAG_PROMPT
print(generate_answer(llm, q, ctx, grounded=True))     # cites [n]; refuses if thin
print(generate_answer(llm, q, ctx, grounded=False))    # ungrounded — watch it drift
res = answer_with_sources(llm, hybrid, q, k=4)          # retrieve→augment→generate; returns sources
# Try the negative: "What is the CEO of Acme Cloud's name?" → grounded should refuse.
# parallel mapping (secondary): build_rag_chain_langchain(lc_retriever)  # same rules, LCEL form
```

---

## Stage 6 — Evaluation *(Notebook 06)*

**WHY.** Without measurement, every design choice is opinion. Evaluation turns RAG into engineering and **localises failure** — RETRIEVAL (wrong chunks) vs GENERATION (right chunks, wrong answer) need different fixes.

**WHAT — two metric families.**
- **RETRIEVAL** (vs ground-truth relevant ids): `precision_at_k`, `recall_at_k`, `hit_rate_at_k`, `mrr`. `evaluate_retriever` averages them and **skips negatives**; source-level ids by default (robust to chunk size).
- **GENERATION**: `faithfulness_heuristic` (cheap cosine proxy), `answer_relevance_heuristic`, `refusal_correct` (for negatives), `llm_judge_faithfulness` (0–1), and **RAGAS** (`evaluate_with_ragas`: faithfulness, answer_relevancy, context_precision/recall).

**What if poorly designed?** You optimise the wrong thing — tune the prompt for weeks when the real problem was recall, or ship a fluent hallucinator.

**Debugging order.** Watch **recall FIRST** — if the right context isn't retrieved, no prompt fix helps.

### Metrics reference (plain English)
| Metric | Plain-English meaning | Family |
|---|---|---|
| **precision@k** | Of the top-k retrieved, what fraction are actually relevant? | retrieval |
| **recall@k** | Of all truly relevant chunks, what fraction made the top-k? | retrieval |
| **hit_rate@k** | Did *at least one* relevant chunk make the top-k? (1/0) | retrieval |
| **MRR** | How high was the *first* relevant hit ranked? (1/rank) | retrieval |
| **faithfulness** | Is every claim in the answer supported by the retrieved context? | generation |
| **answer relevance** | Does the answer actually address the question? | generation |
| **refusal correct** | On an unanswerable question, did it correctly say "I don't know"? | generation |
| **RAGAS** | Framework bundle: faithfulness, answer relevancy, context precision/recall | both |

### How to build a RAG eval dataset (mini-guide)
1. **Collect real questions** users actually ask (factual, list, paraphrase).
2. **Write ground truths** — the reference answer.
3. **Label relevant sources** — which document(s) answer it. **Source-level** ids are robust to chunk-size changes; chunk-level is more precise but brittle.
4. **Add NEGATIVE tests** — questions the corpus *cannot* answer, where the correct output is a **refusal**. (Our set has 3 of 21: mobile app, CEO name, Starter phone hotline.) A system that never refuses hallucinates on out-of-scope questions.
5. **Metrics vs rubrics** — numbers for what's countable (retrieval metrics), human rubrics for what isn't (tone, completeness).
6. **Bootstrap, then curate** — `draft_questions_from_chunks(llm, chunks)` proposes candidates, but **never ship auto-generated ground truth unreviewed.**

**HOW (API to try).**
```python
from rag_pipeline.evaluation import (load_eval_dataset, evaluate_retriever,
    faithfulness_heuristic, refusal_correct, llm_judge_faithfulness, evaluate_with_ragas, EvalExample)
ex = load_eval_dataset("data/eval/eval_dataset.jsonl")   # 21 examples, 3 negatives
print("dense ", evaluate_retriever(dense,  ex, k=3))
print("hybrid", evaluate_retriever(hybrid, ex, k=3))     # recall ↑ vs dense
```

**Diagnosing a "bad RAG output" (walk the pipeline backwards — cheapest first):**
1. **Generation** — grounded/cited? If `grounded=True` fixes it → prompt bug.
2. **Augmentation** — answer buried mid-context or budget truncated the key chunk?
3. **Retrieval** — is the right chunk even retrieved? Check `trace.show()` / `evaluate_retriever` on that example. recall=0 → retrieval bug.
4. **Chunking** — was the answer sentence split across chunks?
5. **Loading** — did the source ingest cleanly at all?

---

## Stage 7 — Explore Advanced RAG *(Notebook 07)*

**WHY.** Single-hop RAG can't answer everything. These patterns are where the field goes — and each is still *the same six stages*, rearranged.

**WHAT (the map).**
- **Naïve → Advanced → Modular RAG** — straight line → pre/post-retrieval optimisation (query rewriting, reranking) → fully swappable modules (what `RAGPipeline` already is).
- **Agentic RAG** — an agent decides whether/what to retrieve, can use tools and loop.
- **Multi-hop RAG** — chain retrievals; the second retrieval depends on the first answer.
- **Graph RAG** — retrieve over a knowledge graph; strong for entity/relationship questions.
- **Adaptive / self-RAG** — the system decides when it has enough evidence.
- **Long-context** — bigger windows ease *some* chunking pressure but don't remove retrieval (cost, "lost in the middle").

**What if poorly designed?** More moving parts = more latency and failure surface. Reach for advanced patterns only when evaluation shows a concrete gap.

**Key takeaway.** Frameworks and fancy patterns **abstract but do not eliminate design decisions.** Each still needs good loading, chunking, grounding, and evaluation.

---

## Stage 8 — Conclusion / end-to-end *(Notebook 08)*

**WHY.** Assemble everything in one object and prove the thesis by changing one knob at a time.

**WHAT / HOW.**
```python
from rag_pipeline import RAGPipeline
pipe = RAGPipeline(injection="stuff", k=4).ingest(docs)   # defaults: recursive chunker, dense retriever
out  = pipe.query("How are overage API calls charged?")
print(out["answer"]); print(out["sources"])
print(out["trace"].show())     # retrieved chunks + scores + timings — the debugging surface
```
Swap `chunker`, `retriever_factory`, `injection` (`stuff`/`map_reduce`/`refine`), `reranker`, or `k` and re-`query` to *feel* each design decision move the answer.

**Closing takeaway.** **RAG performance is the cumulative result of design decisions across ALL stages — not just the choice of LLM. Frameworks abstract the mechanics; you still own the design.**

---

## Glossary

- **Embedding** — a text mapped to a vector so that semantically similar texts land near each other.
- **Vector store** — holds vectors + their documents; returns the nearest to a query vector (here, numpy cosine).
- **Cosine similarity** — similarity by the *angle* between vectors (direction, not magnitude); needs normalisation.
- **Dense retrieval** — retrieval by embedding similarity (semantic).
- **BM25** — classic sparse/lexical ranking; rewards documents containing the query's *rare* exact terms; saturates on term frequency, normalises by length.
- **Sparse retrieval** — token/term-based retrieval (BM25 is the canonical example).
- **Hybrid retrieval** — combine dense + sparse for both meaning and exact terms.
- **RRF (Reciprocal Rank Fusion)** — fuse rankings by summing 1/(c + rank); robust because it uses ranks, not incomparable raw scores.
- **Reranker** — a second, more accurate scorer applied to a shortlist (cross-encoder or LLM).
- **Cross-encoder** — reads query and document *together* to score relevance; very accurate, too slow for the whole corpus.
- **Metadata filtering** — restrict the candidate set by metadata (source, page) *before* similarity ranking.
- **Chunk** — a retrievable, embeddable slice of a document.
- **Chunk overlap** — text repeated between adjacent chunks so straddling ideas survive whole in ≥1 chunk.
- **Context window** — the maximum tokens an LLM can read at once; finite → you can't stuff everything.
- **Augmentation / context injection** — HOW retrieved chunks are packed into the prompt (stuff/map-reduce/refine).
- **"Lost in the middle"** — LLMs attend least to content in the middle of a long context.
- **Grounding** — instructing the model to answer only from provided context and refuse when it's insufficient.
- **Hallucination** — confident, fluent output not supported by the context (or by fact).
- **Provenance / citation** — tracking which source each claim came from (`[n]` tags).
- **Faithfulness** — whether every claim in the answer is supported by the retrieved context.
- **Negative test** — a question the corpus can't answer, where the correct output is a refusal.
- **RAGAS** — a framework for RAG metrics (faithfulness, answer relevancy, context precision/recall).
- **Token** — the unit the model (and your bill) actually count; roughly ¾ of a word in English.

---

## Additional readings

- **OpenAI — Retrieval / RAG guide:** https://platform.openai.com/docs/guides/retrieval
- **OpenAI — Embeddings guide:** https://platform.openai.com/docs/guides/embeddings
- **sentence-transformers (bi-encoders & cross-encoders):** https://www.sbert.net/
- **LangChain — Retrieval / RAG docs:** https://python.langchain.com/docs/concepts/rag/
- **LangChain — Text splitters:** https://python.langchain.com/docs/concepts/text_splitters/
- **RAGAS — RAG evaluation:** https://docs.ragas.io/
- **BM25 (Robertson & Zaragoza, *The Probabilistic Relevance Framework*)** — the classic reference on BM25/IDF.
- **"Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)** — why mid-context evidence is under-attended.
- **Reciprocal Rank Fusion (Cormack et al., 2009)** — the fusion method behind `HybridRetriever`.

---

*End of learner handout. Companion: `instructor_guide.md`.*
