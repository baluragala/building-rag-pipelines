# Building RAG Pipelines — Practice Exercises

**Session:** C8-W4-S1 · **Duration:** 180 minutes · **Package:** `rag_pipeline` · **Corpus:** Acme Cloud docs

---

## The one idea to hold in your head

> **RAG quality is the cumulative result of design decisions across *all* pipeline stages — not just the LLM.**

Every exercise below is designed to make you *feel* that sentence. A great model on top of a bad loader,
a careless chunker, or a shallow retriever still produces a bad answer. The failures compound:

```
Loading → Chunking → Embedding → Retrieval → (Rerank) → Augmentation → Generation → Evaluation
   │          │           │           │           │            │             │            │
 noise    fragment/    wrong       recall     precision     lost-in-      hallucinate   can't
 in        dilute      space       miss       miss          the-middle    / no refusal  measure
```

Each stage is a **knob**. These exercises turn the knobs one at a time, ask you to **predict the effect
first**, then measure it — so cause and effect become visible instead of vibes.

---

## How to use this document

- Exercises are graded: **⭐ warm-up · ⭐⭐ core · ⭐⭐⭐ challenge**. Do the ⭐ and ⭐⭐ in order; reach for ⭐⭐⭐ if you have time.
- Every exercise has a **Task**, a **Predict first** prompt (where it matters), a **Hint**, and a **What you should observe** line. Do not skip *Predict first* — being wrong on a prediction is where the learning happens.
- Worked solutions with runnable code live in `solutions/solutions.md`. Try before you peek.

### Shared setup (run once at the top of your notebook)

```python
from rag_pipeline.config import get_embedder, get_llm, current_config
from rag_pipeline.loaders import load_directory, load_markdown, load_html, clean_text, Document
from rag_pipeline.chunking import (
    fixed_size_chunk, recursive_chunk, semantic_chunk, chunk_stats, token_len,
)
from rag_pipeline.embeddings import embed_documents, embed_query, cosine_similarity
from rag_pipeline.vectorstore import InMemoryVectorStore
from rag_pipeline.retrieval import (
    DenseRetriever, BM25Retriever, HybridRetriever, CrossEncoderReranker, LLMReranker,
)
from rag_pipeline.augmentation import stuff, map_reduce, refine, format_context, collect_sources
from rag_pipeline.generation import generate_answer, answer_with_sources
from rag_pipeline.evaluation import (
    load_eval_dataset, evaluate_retriever, faithfulness_heuristic,
    answer_relevance_heuristic, refusal_correct, EvalExample,
)
from rag_pipeline.pipeline import RAGPipeline

print(current_config())          # which providers are active?
embedder = get_embedder()        # OpenAI default; falls back to offline mock with no key
llm = get_llm()                  # OpenAI default; falls back to offline mock with no key

CORPUS = "data/corpus"
EVAL = "data/eval/eval_dataset.jsonl"
docs = load_directory(CORPUS)    # 6 Acme Cloud documents
examples = load_eval_dataset(EVAL)   # 21 items (18 answerable, 3 negatives)
```

> **Note on providers.** With an API key you get real semantic embeddings and a fluent LLM, and the
> qualitative results described in the solutions hold. With **no key** the package uses a deterministic
> **mock** embedder (bag-of-words hashing) and an honest **mock** LLM (extractive, never invents facts).
> The *mechanics* of every exercise still run offline; only the semantic *quality* gap narrows. Where a
> result depends on real embeddings, the exercise says so.

---

# Stage 1 — Loading

*Retrieval can only ever surface what ingestion let in. Garbage in → garbage retrieved → confident garbage out.*

### 1.1 ⭐ Load the corpus and inspect what came in

**Task.** Load `data/corpus/` and print, for each `Document`, its `metadata` keys and the length of
`page_content`. Confirm all six Acme docs loaded and that each carries a `source`, `title`, and `loader`.

**Hint.** `load_directory(CORPUS)` dispatches by extension. Iterate `docs` and read `doc.metadata`.

**What you should observe.** Six documents, each with provenance metadata already attached. Metadata is
not decoration — it is what makes citations and metadata-filtered retrieval possible three stages later.

---

### 1.2 ⭐⭐ Measure how `clean_text` changes downstream chunk counts

**Task.** Load `acme_faq.html` **twice** — once through the normal HTML loader (which cleans), and once as
*raw* text with cleaning disabled (read the file yourself into a `Document`). Chunk both with
`recursive_chunk(..., chunk_size=300, overlap=0)` and compare `chunk_stats(...)`. Then repeat the idea for
a Markdown file by calling `clean_text` vs. not.

**Predict first.** Will the *un-cleaned* HTML produce **more** or **fewer** chunks than the cleaned one? Why?
Which one embeds the words "Home | Docs | Pricing | Contact" (nav boilerplate) into a chunk?

**Hint.** The clean HTML loader drops `<script>`, `<style>`, `<nav>`, `<footer>` and collapses whitespace.
Build the dirty comparison with `Document(page_content=open(path).read(), metadata={"source": path})`.

**What you should observe.** Cleaning changes both the chunk **count** and the chunk **content**: uncleaned
text carries navigation/footer noise that will later get embedded and can be retrieved instead of the
answer. A loading decision silently sets the ceiling on retrieval quality.

---

### 1.3 ⭐⭐⭐ Attach custom metadata, then filter on it at retrieval time

**Task.** Re-load the corpus but tag each document with a `category` in metadata (e.g. `pricing`, `security`,
`sla`, `api`, `onboarding`, `faq`) derived from its filename. Chunk, embed, and index. Then run the query
*"What uptime does Enterprise guarantee?"* **twice**: once unconstrained, and once restricted to
`category == "sla"` using the vector store's `metadata_filter`. Show the retrieved source labels each time.

**Predict first.** Before running: which retrieved chunks will the filter *remove*, and will the top answer
chunk change rank once irrelevant categories are excluded?

**Hint.** Loaders accept `**extra_meta`, so `load_markdown(path, category="sla")` stamps it on. At search
time, `DenseRetriever.retrieve(query, k=4, metadata_filter={"category": "sla"})` forwards the filter to
`InMemoryVectorStore.search`, which narrows candidates *before* ranking.

**What you should observe.** The filter constrains the search space *before* similarity ranking — a cheap,
high-leverage precision lever. This is a *loading* decision (you must attach the metadata up front) paying
off at the *retrieval* stage: the cumulative-design principle in miniature.

---

# Stage 2 — Chunking

*The central cause→effect of the session: **bad chunking → poor retrieval → bad answer.***

### 2.1 ⭐ See fixed-size vs. recursive fragmentation with your own eyes

**Task.** Chunk the corpus with `fixed_size_chunk(docs, 200, 0)` and with `recursive_chunk(docs, 200, 0)`.
Print the first three chunks of each. Find a place where the fixed-size splitter cuts mid-sentence or
mid-table and the recursive splitter does not.

**Hint.** `fixed_size_chunk` is structure-blind (it slices every N characters); `recursive_chunk` splits on
the largest natural boundary that fits (paragraph → line → sentence → word).

**What you should observe.** Fixed-size chunking severs ideas at arbitrary character offsets; recursive
chunking respects boundaries. Same size budget, very different chunk *meaning* — and meaning is what gets
embedded.

---

### 2.2 ⭐⭐ The chunk-size / overlap sweep — predict, then measure

**Task.** For every combination of `chunk_size ∈ {200, 500, 1000}` and `overlap ∈ {0, 50, 150}`
(9 configs), build an index with `recursive_chunk`, wrap a `DenseRetriever`, and run
`evaluate_retriever(retriever, examples, k=4)`. Tabulate `recall@4`, `precision@4`, and `mrr` per config.
Also record `chunk_stats` (chunk count and mean size) for each.

**Predict first.** Write down, *before running*, which single config you think maximises `recall@4` and why.
Do you expect tiny chunks (200) or huge chunks (1000) to win? What does overlap trade off (recall vs. index
size and redundancy)?

**Hint.** Reuse a helper: `build_dense(chunker) -> retriever` that chunks, embeds with
`embed_documents`, indexes with `InMemoryVectorStore().add(...)`, and returns `DenseRetriever(store, embedder)`.
Loop the 9 configs. Keep `k=4` fixed so only chunking varies.

**What you should observe.** There is a sweet spot, not a monotonic trend. Too-small chunks fragment ideas
(recall of multi-fact answers drops); too-large chunks dilute the embedding (the relevant sentence is averaged
away). A modest overlap recovers ideas that straddle a boundary. **You cannot pick a chunk size without
measuring — it is a design decision, not a default.**

---

### 2.3 ⭐⭐⭐ Construct a fragmentation failure and a dilution failure

**Task.** Find (or craft) one question that is answered **correctly at chunk_size=500** but **fails with
chunk_size=200** (fragmentation), and one that is answered correctly at **500** but degrades at **1000**
(dilution). Use a multi-fact question for fragmentation (e.g. the Enterprise multi-hop question, which needs
SSO + 99.99% SLA evidence) and an exact-fact-buried-in-a-long-doc question for dilution. Show the retrieved
chunks and answer for each size.

**Predict first.** For your fragmentation question, which *two* facts must co-occur in one chunk (or be
retrieved together) for the answer to be complete? Predict which size splits them apart.

**Hint.** Run the same question through `answer_with_sources(llm, retriever, q, k=4)` at each size and read
the `context` field. For dilution, use the mock or real embedder and watch the target sentence's chunk fall
out of the top-4 as neighbouring text swells the chunk.

**What you should observe.** The *same query* flips from right to wrong purely because of a chunking knob.
This is the session thesis made undeniable: the LLM never changed — the retrieval it was fed did.

---

# Stage 3 — Retrieval

*Optimise **recall first** (cast a wide net), then **precision** (tighten). Never rely only on top-k dense similarity.*

### 3.1 ⭐ Dense vs. BM25 vs. Hybrid on the eval set

**Task.** Build all three retrievers over one fixed index (recursive chunks, size 500/overlap 50) and run
`evaluate_retriever(retriever, examples, k=4)` on each. Report `precision@4`, `recall@4`, `hit_rate@4`, `mrr`
side by side.

**Predict first.** Which retriever will have the highest `recall@4`? Rank dense, BM25, hybrid *before* running.

**Hint.** `dense = DenseRetriever(store, embedder)`; `sparse = BM25Retriever(chunks)`;
`hybrid = HybridRetriever(dense, sparse, rrf_c=60, pool_k=10)`. All three share the same `.retrieve(query, k)`
signature, so you can loop over them.

**What you should observe.** Hybrid usually meets or beats both parents because Reciprocal Rank Fusion takes
the *union* of what each retriever finds. Fusing ranks (not raw scores) is what makes recall go up without a
score-normalisation headache.

---

### 3.2 ⭐⭐ Find the query where BM25 wins and the query where dense wins

**Task.** Using single queries (not the whole set), demonstrate:
- a query where **BM25 beats dense** — an exact token like *"What HTTP status code is returned when I hit
  the rate limit?"* (the token **HTTP 429** / **429**), or *"AES-256"*.
- a query where **dense beats BM25** — a **paraphrase** with no shared keywords, like *"Can I use a Gmail
  address to sign up for an organisation account?"* (the doc says "personal email domains such as gmail.com
  are not accepted").

For each, print the top-3 `(source, score)` from BM25 and from dense and note which puts the correct
document at rank 1.

**Predict first.** For the "HTTP 429" query, why might a pure embedding miss it? For the Gmail paraphrase,
why might BM25 miss it?

**Hint.** BM25 rewards documents containing the query's *rare exact terms* (429, AES-256); it has no notion
of synonyms. Dense retrieval matches *meaning*, so it bridges paraphrases but can smear over exact codes.
(With the offline **mock** embedder the dense win on paraphrase is weaker — note this if you have no key.)

**What you should observe.** Neither retriever dominates. BM25 nails exact terms; dense bridges vocabulary
gaps. This is *why* hybrid exists — and why "just use vector search" is the anti-pattern the session warns
against.

---

### 3.3 ⭐⭐ Add a metadata filter to sharpen a retrieval

**Task.** Take a query that pulls chunks from the wrong document (e.g. a pricing question that also surfaces
SLA chunks because both mention tiers). Add a `metadata_filter` or `filter_fn` at search time to constrain
retrieval to the right document/category, and re-measure precision on that single query.

**Predict first.** Will filtering raise **precision**, **recall**, or both? Which one can filtering *hurt* if
you over-constrain (e.g. filter to the wrong document for a multi-hop question)?

**Hint.** `dense.retrieve(q, k=4, metadata_filter={"category": "pricing"})` or
`filter_fn=lambda d: "pricing" in d.metadata["source"]`. The store narrows candidates before ranking.

**What you should observe.** Filtering trades recall for precision. Used well it removes distractors; used
carelessly (over-constraining a multi-hop question) it *removes the evidence you needed*. A precision tool,
not a free lunch.

---

### 3.4 ⭐⭐⭐ Add a reranker and measure the precision lift

**Task.** Start from the hybrid retriever with a wide net (`pool_k=10`). Retrieve a candidate pool, then apply
a reranker (`CrossEncoderReranker()` if `sentence-transformers` is installed, else `LLMReranker(llm)`) to cut
to the top `k=4`. Compare `precision@4` / `mrr` on the eval set **with vs. without** the reranker, using the
"retrieve wide, rerank narrow" pattern.

**Predict first.** Will reranking improve **recall@4** or **precision@4** more? (Think about what a reranker
can and cannot do: can it retrieve a chunk the first stage never surfaced?)

**Hint.** Wrap the reranker into a tiny retriever-like object, or evaluate manually: for each example, pull a
pool of 10, `reranker.rerank(q, pool, top_n=4)`, then score the reranked ids. `CrossEncoderReranker` exposes
`.available`; if `False`, fall back to `LLMReranker`.

**What you should observe.** Reranking lifts **precision and MRR** (better ordering of what you already
retrieved) but **cannot raise recall beyond the pool** — if the first stage missed a chunk, the reranker
never sees it. Hence: recall net *first*, precision scalpel *second*.

---

# Stage 4 — Augmentation & Generation

*Retrieval decides **which** chunks; augmentation decides **how** they enter the prompt; the prompt decides whether the model **grounds or hallucinates**.*

### 4.1 ⭐⭐ Stuff vs. Map-Reduce vs. Refine on a multi-hop question

**Task.** Take the multi-hop question *"Which plan gives SSO and a 99.99% SLA?"* (evidence lives across
`acme_pricing.md`, `acme_security.md`, and `acme_sla.md`). Retrieve `k=6` with the hybrid retriever, then
produce three answers: `stuff` → `generate_answer`, `map_reduce(llm, q, hits)`, and `refine(llm, q, hits)`.
Compare the answers for completeness (does it name **Enterprise** *and* cite the SSO and SLA evidence?).

**Predict first.** Which strategy is most likely to assemble facts spread across three documents into one
complete answer? Which makes the most LLM calls?

**Hint.** All three take the same `hits` list. `stuff` builds one context string (`stuff(hits, max_tokens=2000)`)
then `generate_answer(llm, q, context)`. `map_reduce` and `refine` call the LLM per chunk internally.

**What you should observe.** For scattered evidence, map-reduce and refine tend to be more complete because
they read each chunk deliberately, at the cost of extra calls/latency. Stuffing is cheapest but can bury the
key sentence in a long context ("lost in the middle"). Same chunks, different assembly, different answer.

---

### 4.2 ⭐⭐ Grounded vs. ungrounded prompt on a NEGATIVE question (watch it hallucinate)

**Task.** Ask a negative question the corpus **cannot** answer — *"Does Acme Cloud offer a native mobile app
for iOS and Android?"* Retrieve context, then generate an answer twice: `grounded=True` and `grounded=False`.
Print both.

**Predict first.** What will the *ungrounded* prompt do when the context has no answer? What single instruction
in the grounded template prevents that?

**Hint.** `generate_answer(llm, q, context, grounded=True)` vs. `grounded=False`. The grounded template forces
*"I don't know based on the provided context."* when the context is insufficient; the ungrounded one says
"answer however you like." (Requires a real LLM to see a fabricated answer — the mock LLM is extractive and
won't invent, but you'll still see the template difference.)

**What you should observe.** The ungrounded prompt invents a plausible mobile-app answer from parametric
memory; the grounded prompt **refuses**. The hallucination was a *prompt* failure, not a *model* failure —
which is exactly why prompt design is a RAG design decision.

---

### 4.3 ⭐⭐ Verify that citations point to real sources

**Task.** Run `answer_with_sources(llm, retriever, "How often are encryption keys rotated?", k=4)`. Confirm
the answer contains `[n]` citation tags, and that each `n` maps to a source in the returned `sources` list
whose document actually contains the cited fact (90 days → `acme_security.md`).

**Hint.** The return dict has `answer`, `sources` (with `label` and `source`), and `context`. Cross-check the
`[n]` in the answer against `sources[n-1]`, and grep the source file for the fact.

**What you should observe.** Grounding + numbered context + a citation instruction make answers auditable. A
citation that points to a document *not* containing the fact is a red flag you can catch automatically —
trust in RAG comes from provenance, not fluency.

---

### 4.4 ⭐⭐⭐ Blow the context budget on purpose

**Task.** Retrieve `k=8` and format context with `format_context(hits, max_tokens=None)` (budgeting off) vs.
`max_tokens=300` (tight). Measure `token_len(context)` for each and observe which chunks get dropped when the
budget is tight. Then answer the same question from each context and compare.

**Predict first.** With budgeting **off**, what happens to cost/latency and to the model's attention on the
key sentence? With a *tight* budget, which chunk might get cut — and could it be the one with the answer?

**Hint.** `format_context` stops adding blocks once `max_tokens` is exceeded (it truncates by dropping later
chunks, not by mangling text). Compare the two contexts and the resulting answers.

**What you should observe.** No budget → bloated context, higher cost, and the answer risks being "lost in the
middle." Too tight → you may cut the answer-bearing chunk. Augmentation is a real knob between cost and
correctness — retrieval order suddenly matters a lot.

---

# Stage 5 — Evaluation

*Without measurement, every stage decision is opinion. Metrics also **localise failure**: retrieval vs. generation are different bugs with different fixes.*

### 5.1 ⭐ Compute the baseline scorecard

**Task.** For a default pipeline (recursive 500/50, dense retriever, `k=4`), print the retrieval scorecard
from `evaluate_retriever(retriever, examples, 4)`. Then, for three answerable questions, compute
`faithfulness_heuristic(answer, context, embedder)` and `answer_relevance_heuristic(answer, question, embedder)`.

**Hint.** `evaluate_retriever` automatically **skips the 3 negative examples** (they have no relevant ids).
The heuristics are cheap cosine proxies — high faithfulness means the answer overlaps its context.

**What you should observe.** You now have a single number to defend every later change against. The retrieval
metrics and the generation heuristics measure *different* things — hold onto that; the next exercise depends on it.

---

### 5.2 ⭐⭐ Extend the eval dataset (with a negative) and re-measure

**Task.** Add **three new questions** to the eval set — two answerable (with correct `relevant_ids`, e.g. a
webhook-signing question → `acme_api.md`, and a data-retention question → `acme_faq.html`) and **one negative**
(a question the corpus can't answer, e.g. *"Does Acme offer a Slack integration?"*, `is_negative=True`,
`relevant_ids=[]`). Re-run `evaluate_retriever` on the extended answerable set, and check that your pipeline
**refuses** the new negative with `refusal_correct(answer)`.

**Predict first.** Will adding a negative change the retrieval metrics? (Recall which examples
`evaluate_retriever` skips.) What *should* the correct behaviour on the negative be?

**Hint.** Build `EvalExample(question=..., ground_truth=..., relevant_ids=["acme_api.md"], is_negative=False)`
and append to `examples`. For the negative, `is_negative=True, relevant_ids=[]`. Then answer it with a
**grounded** pipeline and assert `refusal_correct(answer) is True`.

**What you should observe.** Negatives don't move retrieval precision/recall (they're skipped) — they test a
*different* capability: **knowing when to say "I don't know."** A RAG system that never refuses is not
safe; a good eval set forces the question.

---

### 5.3 ⭐⭐⭐ The "bad RAG output" diagnosis — retrieval or generation?

**Task.** You are given a wrong answer for some question. Using the pipeline's `trace` and the two metric
families, decide **which stage failed**: did retrieval fail to surface the right chunk (a *retrieval* bug),
or did the right chunk get retrieved but the model answered wrongly (a *generation* bug)? Write a short
decision procedure and apply it to at least two cases (induce one retrieval failure with tiny chunks or an
over-tight metadata filter; induce one generation failure with `grounded=False`).

**Predict first.** What signal in the `trace.retrieved` list tells you retrieval succeeded? What signal in
faithfulness/refusal tells you generation failed *despite* good retrieval?

**Hint.** `result = pipeline.query(q); print(result["trace"].show())`. Check whether the ground-truth source
appears in `trace.retrieved`. If **yes but the answer is wrong** → generation (fix prompt/injection/model).
If **no** → retrieval (fix chunking/retriever/k/filter). `faithfulness_heuristic` on `trace.context` vs.
`trace.answer` separates the two.

**What you should observe.** The fix for a retrieval miss (re-chunk, add hybrid, raise k) is completely
different from the fix for a generation miss (ground the prompt, change injection). **Metrics that separate
the two are what let you debug instead of flail** — the whole reason evaluation exists.

---

# Capstone — Assemble the best pipeline you can

### C ⭐⭐⭐ Maximise the eval-set score by tuning the whole pipeline

**Task.** Configure a single `RAGPipeline` that maximises quality on the eval set by choosing, and
**justifying**, each of these knobs:

1. **Chunker** — size, overlap, and strategy (`fixed_size_chunk` / `recursive_chunk` / `semantic_chunk`).
2. **Retriever** — dense / BM25 / **hybrid** (via `retriever_factory`).
3. **Reranker** — none / cross-encoder / LLM.
4. **Injection** — `stuff` / `map_reduce` / `refine`.
5. **k** — how many chunks reach the prompt.

Then evaluate on **both** axes:
- **Retrieval:** `evaluate_retriever(pipeline.retriever, examples, k)` → precision/recall/MRR.
- **Generation:** for the answerable questions, `faithfulness_heuristic`; for the **3 negatives**, the
  refusal rate via `refusal_correct`. A good pipeline scores high on faithful answers **and** refuses all
  negatives.

**Predict first.** Before tuning, write your hypothesis for the single best configuration and *why*, phrased
in terms of the cumulative-design principle (e.g. "hybrid for recall, rerank for precision, recursive 500/50
so multi-fact chunks stay whole, grounded stuff with k=5 so the multi-hop evidence co-occurs").

**Hint.** `RAGPipeline(chunker=lambda d: recursive_chunk(d, 500, 50),
retriever_factory=lambda store, emb: HybridRetriever(DenseRetriever(store, emb), BM25Retriever(store.documents)),
injection="stuff", reranker=CrossEncoderReranker(), k=5)`. Call `.ingest(docs)` then evaluate. Change **one
knob at a time** and keep a table — that discipline is the point.

**What you should observe / the takeaway.** Your best score comes from **no single hero knob** — it is the
*stack* of good decisions: clean loading, whole-idea chunks, a wide-recall retriever, a precision reranker, a
budgeted grounded prompt, and negatives in your eval set to keep you honest. Swap any one back to a bad
choice and the score drops. That is the session's entire thesis, proven on your own numbers:
**RAG quality is cumulative across all stages, not just the LLM.**

---

## Self-check: can you now answer these?

1. Name one failure that is *born* in loading but only *visible* at retrieval.
2. Why is there a chunk-size sweet spot rather than "bigger is always better"?
3. Give a query BM25 wins and one dense wins, and say why in one sentence each.
4. What can a reranker **not** fix, no matter how good it is?
5. Which single prompt instruction prevents most hallucinations on negative questions?
6. Given a wrong answer, what is the *first* number you look at to decide retrieval-vs-generation?
