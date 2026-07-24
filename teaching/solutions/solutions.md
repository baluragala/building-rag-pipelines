# Building RAG Pipelines — Worked Solutions

**Session:** C8-W4-S1 · Companion to `../exercises.md`

Every solution below assumes the shared setup from `exercises.md` has run:

```python
from rag_pipeline.config import get_embedder, get_llm, current_config
from rag_pipeline.loaders import load_directory, load_markdown, load_html, clean_text, Document
from rag_pipeline.chunking import fixed_size_chunk, recursive_chunk, semantic_chunk, chunk_stats, token_len
from rag_pipeline.embeddings import embed_documents, embed_query, cosine_similarity
from rag_pipeline.vectorstore import InMemoryVectorStore
from rag_pipeline.retrieval import DenseRetriever, BM25Retriever, HybridRetriever, CrossEncoderReranker, LLMReranker
from rag_pipeline.augmentation import stuff, map_reduce, refine, format_context, collect_sources
from rag_pipeline.generation import generate_answer, answer_with_sources
from rag_pipeline.evaluation import load_eval_dataset, evaluate_retriever, faithfulness_heuristic, answer_relevance_heuristic, refusal_correct, EvalExample
from rag_pipeline.pipeline import RAGPipeline

embedder = get_embedder()
llm = get_llm()
docs = load_directory("data/corpus")
examples = load_eval_dataset("data/eval/eval_dataset.jsonl")
```

A reusable helper used throughout (chunk → embed → index → dense retriever):

```python
def build_dense(chunker):
    chunks = chunker(docs)
    embs = embed_documents(embedder, chunks)
    store = InMemoryVectorStore().add(chunks, embs)
    return DenseRetriever(store, embedder), chunks, store
```

> **Provider note.** Expected *numbers* below assume the default OpenAI stack (real embeddings + `gpt-4o-mini`).
> With the offline **mock** embedder, absolute scores differ and semantic (dense-wins-on-paraphrase) effects
> shrink, but every ordering and mechanism reproduces. Numbers are illustrative, not exact — the *direction*
> is the lesson.

---

# Stage 1 — Loading

## 1.1 ⭐ Load and inspect

```python
docs = load_directory("data/corpus")
print(len(docs), "documents")
for d in docs:
    print(f"{len(d.page_content):5d} chars | keys={list(d.metadata)} | title={d.metadata['title']!r}")
```

**Expected result.** Six documents. Each already carries `source`, `title`, `loader` (and `page` for PDFs).
Example: `acme_pricing.md → title='Acme Cloud Pricing', loader='load_markdown'`.

**Why (cause→effect).** The loader attaches provenance *at ingestion*, which is the only moment that
information exists cleanly. Skip it here and you can never cite or metadata-filter later — a loading omission
becomes an un-fixable gap three stages downstream.

---

## 1.2 ⭐⭐ How `clean_text` changes chunk counts

```python
path = "data/corpus/acme_faq.html"

clean_docs = load_html(path)                                   # cleaned: nav/footer/script dropped
raw_docs   = [Document(open(path, encoding="utf-8").read(), {"source": path})]  # NOT cleaned

for name, dset in [("clean", clean_docs), ("raw", raw_docs)]:
    chunks = recursive_chunk(dset, chunk_size=300, overlap=0)
    print(name, chunk_stats(chunks))
    print("   first chunk:", chunks[0].page_content[:120].replace("\n", " "))
```

**Expected result.** The **raw** version yields **more chunks** and its first chunks contain HTML tags plus
the nav bar text `Home | Docs | Pricing | Contact` and the `© Acme Cloud` footer. The **clean** version is
shorter, tag-free, and starts at the actual FAQ content.

```
clean {'n_chunks': ~5, 'mean': ~250 ...}
raw   {'n_chunks': ~8, 'mean': ~230 ...}   # tags + boilerplate inflate the count
```

**Why.** Uncleaned markup is *text that gets embedded*. Nav/footer boilerplate becomes retrievable noise that
can outrank the answer. `clean_text` also collapses whitespace and strips page furniture — fewer, cleaner
chunks. The quality ceiling of every later stage was just set by a loader flag.

---

## 1.3 ⭐⭐⭐ Attach metadata, then filter at retrieval

```python
import os
tagged = []
for path in sorted(__import__("glob").glob("data/corpus/*")):
    cat = os.path.splitext(os.path.basename(path))[0].replace("acme_", "")  # pricing, security, ...
    if path.endswith(".html"):
        tagged += load_html(path, category=cat)
    elif path.endswith(".md"):
        tagged += load_markdown(path, category=cat)

chunks = recursive_chunk(tagged, 500, 50)
store  = InMemoryVectorStore().add(chunks, embed_documents(embedder, chunks))
dense  = DenseRetriever(store, embedder)

q = "What uptime does Enterprise guarantee?"
print("UNFILTERED:", [(d.metadata['category'], round(s,3)) for d, s in dense.retrieve(q, k=4)])
print("SLA ONLY :", [(d.metadata['category'], round(s,3))
                     for d, s in dense.retrieve(q, k=4, metadata_filter={"category": "sla"})])
```

**Expected result.** Unfiltered, the top-4 mixes `sla` with `pricing`/`security` chunks (all mention tiers).
With `metadata_filter={"category": "sla"}`, only SLA chunks are candidates and the 99.99% chunk ranks #1
cleanly.

**Why.** `InMemoryVectorStore.search` applies the filter to build the candidate set *before* cosine ranking.
Metadata you chose to attach in **Stage 1** becomes a precision lever in **Stage 3** — the cumulative-design
principle in a single query.

---

# Stage 2 — Chunking

## 2.1 ⭐ Fixed-size vs. recursive fragmentation

```python
fx = fixed_size_chunk(docs, 200, 0)
rc = recursive_chunk(docs, 200, 0)
for name, cs in [("fixed", fx), ("recursive", rc)]:
    print(f"\n== {name} ==")
    for c in cs[:3]:
        print(repr(c.page_content[:90]))
```

**Expected result.** `fixed_size_chunk` cuts at exact 200-char offsets — a chunk may end `"...costs **$199 per mo"`
and the next begins `"nth** and includes 15 seats..."`, severing the price from the plan. `recursive_chunk`
breaks on paragraph/sentence boundaries, keeping `"The Growth plan costs $199 per month..."` intact.

**Why.** Fixed-size splitting is structure-blind; the mid-sentence cut destroys the very fact you want to
embed. Recursive splitting preserves the semantic unit. Same 200-char budget, very different meaning per
chunk — and meaning is what the embedder sees.

---

## 2.2 ⭐⭐ The chunk-size / overlap sweep

```python
import itertools, pandas as pd
rows = []
for size, ov in itertools.product([200, 500, 1000], [0, 50, 150]):
    if ov >= size:
        continue
    retr, chunks, _ = build_dense(lambda d, s=size, o=ov: recursive_chunk(d, s, o))
    m = evaluate_retriever(retr, examples, k=4)
    rows.append({"size": size, "overlap": ov, "n_chunks": len(chunks),
                 **{k: m[k] for k in ("recall@4", "precision@4", "mrr")}})
print(pd.DataFrame(rows).to_string(index=False))
```

**Expected result (illustrative, real embeddings).**

```
 size  overlap  n_chunks  recall@4  precision@4   mrr
  200        0       ~55      0.78         0.30  0.71
  200       50      ~70       0.83         0.31  0.74
  500        0      ~22       0.92         0.47  0.88
  500       50      ~26       0.95         0.48  0.90   <-- sweet spot
  500      150      ~34       0.94         0.46  0.89
 1000        0      ~12       0.88         0.40  0.83
 1000       50      ~14       0.86         0.39  0.81
```

**Prediction check.** If you guessed 1000 would win "because more context per chunk," the numbers correct you:
recall *peaks around 500* and falls at 1000. **500/50** is typically best.

**Why.** Too small (200) → ideas fragment across chunks, so multi-fact answers lose recall. Too large (1000)
→ the relevant sentence is *averaged away* in one vector (dilution), so similarity drops and it falls out of
top-4. A modest overlap (50) recovers boundary-straddling ideas without exploding the index. **There is no
default chunk size — you measure.**

---

## 2.3 ⭐⭐⭐ A fragmentation failure and a dilution failure

```python
def answer_at(size, overlap, q, k=4):
    retr, _, _ = build_dense(lambda d: recursive_chunk(d, size, overlap))
    out = answer_with_sources(llm, retr, q, k=k)
    srcs = [s["label"] for s in out["sources"]]
    return out["answer"], srcs

# Fragmentation: multi-fact answer needs SSO + 99.99% SLA evidence together
q_frag = "Which plan gives SSO and a 99.99% SLA?"
print("size=500:", answer_at(500, 50, q_frag))   # complete: names Enterprise, cites both facts
print("size=200:", answer_at(200, 0,  q_frag))   # partial: SSO and SLA land in separate tiny chunks

# Dilution: an exact fact buried in a long chunk
q_dil = "How often are encryption keys rotated?"
print("size=500 :", answer_at(500, 50,  q_dil))  # "every 90 days", security chunk at rank 1
print("size=1000:", answer_at(1000, 50, q_dil))  # 90-day sentence diluted; chunk may drop from top-4
```

**Expected result.** At **200/0** the multi-hop question returns an incomplete answer — SSO evidence
(`acme_security.md`) and the 99.99% SLA evidence (`acme_sla.md`/`acme_pricing.md`) end up in different tiny
chunks and not all survive the top-4. At **500/50** all needed chunks co-occur and the answer names
**Enterprise** with citations. Conversely, the key-rotation fact is retrieved crisply at **500** but at
**1000** the "rotated every 90 days" sentence shares a bloated chunk with BYOK/KMS text, its similarity
drops, and it can fall out of the top-4.

**Why.** The LLM never changed between runs — only the chunking knob did. Fragmentation starves a multi-fact
answer; dilution hides a single fact inside a vector dominated by neighbouring text. **bad chunking → poor
retrieval → bad answer**, demonstrated on the same query.

---

# Stage 3 — Retrieval

## 3.1 ⭐ Dense vs. BM25 vs. Hybrid

```python
retr, chunks, store = build_dense(lambda d: recursive_chunk(d, 500, 50))
dense  = retr
sparse = BM25Retriever(chunks)
hybrid = HybridRetriever(dense, sparse, rrf_c=60, pool_k=10)

for name, r in [("dense", dense), ("bm25", sparse), ("hybrid", hybrid)]:
    print(name, evaluate_retriever(r, examples, k=4))
```

**Expected result (illustrative).**

```
dense  {'precision@4': 0.47, 'recall@4': 0.85, 'hit_rate@4': 0.94, 'mrr': 0.88, ...}
bm25   {'precision@4': 0.46, 'recall@4': 0.88, 'hit_rate@4': 0.94, 'mrr': 0.86, ...}
hybrid {'precision@4': 0.49, 'recall@4': 0.96, 'hit_rate@4': 1.00, 'mrr': 0.91, ...}
```

**Prediction check.** Hybrid should top your ranking: **hybrid recall@4 ≈ 0.96 > bm25 ≈ 0.88 ≈ dense ≈ 0.85**.

**Why.** Reciprocal Rank Fusion sums `1/(rank + c)` across both retrievers, so hybrid retrieves the *union* of
what each finds. Because RRF fuses *ranks* not raw scores, it sidesteps reconciling cosine (0–1) with
unbounded BM25 scores. Optimise recall first → cast the widest net.

---

## 3.2 ⭐⭐ Where BM25 wins and where dense wins

```python
def top3(r, q):
    return [(d.metadata['source'].split('/')[-1], round(s, 3)) for d, s in r.retrieve(q, k=3)]

q_exact = "What HTTP status code is returned when I hit the rate limit?"   # token: 429
q_para  = "Can I use a Gmail address to sign up for an organisation account?"  # paraphrase

print("EXACT  bm25 :", top3(sparse, q_exact))   # acme_api.md rank 1 (matches '429' / 'http')
print("EXACT  dense:", top3(dense,  q_exact))
print("PARA   bm25 :", top3(sparse, q_para))    # weak: few shared tokens with 'personal email domains'
print("PARA   dense:", top3(dense,  q_para))    # acme_onboarding.md rank 1 (meaning match)
```

**Expected result.** BM25 puts `acme_api.md` at rank 1 for the **HTTP 429** query — it matches the exact,
rare tokens `429`/`http`, which is precisely what lexical scoring rewards. Dense may rank it lower because
"429" is not semantically distinctive. On the **Gmail paraphrase**, dense puts `acme_onboarding.md` first
(the doc says "personal email domains such as gmail.com are not accepted" — no shared keyword with the
question), while BM25 struggles because the query and passage share almost no tokens.

**Why.** BM25 wins on exact terms (codes, acronyms, IDs) because it matches tokens directly; dense wins on
paraphrase because it matches *meaning*, bridging the vocabulary gap. Neither dominates — which is the whole
argument for hybrid and against "just use vector search." *(With the mock embedder the dense paraphrase win
is muted, since it is bag-of-words-ish; note this if running offline.)*

---

## 3.3 ⭐⭐ Metadata filter to sharpen retrieval

```python
# reuse the category-tagged index from 1.3 (dense over `tagged`)
q = "How are API calls above the plan limit charged?"   # answer in pricing; api/sla also mention 'calls'
before = dense.retrieve(q, k=4)
after  = dense.retrieve(q, k=4, filter_fn=lambda d: "pricing" in d.metadata["source"])
print("before:", [d.metadata['source'].split('/')[-1] for d, _ in before])
print("after :", [d.metadata['source'].split('/')[-1] for d, _ in after])
```

**Expected result.** Before filtering the top-4 mixes `acme_pricing.md` with `acme_api.md`
(rate-limit/"calls" chunks). After constraining to pricing, all four candidates are from `acme_pricing.md`
and precision@4 on this query rises (the `$0.50 per 1,000 calls` overage chunk dominates).

**Why.** Filtering narrows candidates *before* ranking, raising **precision** by removing distractors. But it
can **hurt recall** if over-applied: filter this same index to `pricing` for the *multi-hop* SSO+SLA question
and you delete the security/SLA evidence — the answer becomes impossible. A precision scalpel, used with care.

---

## 3.4 ⭐⭐⭐ Add a reranker and measure the lift

```python
class RerankRetriever:
    """Wrap a base retriever + reranker so evaluate_retriever can score it."""
    def __init__(self, base, reranker, pool=10):
        self.base, self.reranker, self.pool = base, reranker, pool
    def retrieve(self, query, k=4):
        pool = self.base.retrieve(query, k=self.pool)
        return self.reranker.rerank(query, pool, top_n=k)

reranker = CrossEncoderReranker()
if not getattr(reranker, "available", False):
    reranker = LLMReranker(llm)          # fallback when sentence-transformers is absent

hybrid_reranked = RerankRetriever(hybrid, reranker, pool=10)
print("hybrid          :", evaluate_retriever(hybrid, examples, k=4))
print("hybrid + rerank :", evaluate_retriever(hybrid_reranked, examples, k=4))
```

**Expected result (illustrative).**

```
hybrid          precision@4 0.49  recall@4 0.96  mrr 0.91
hybrid + rerank precision@4 0.61  recall@4 0.96  mrr 0.97   <-- precision & MRR up, recall unchanged
```

**Prediction check.** The reranker lifts **precision@4** and **MRR**, not recall.

**Why.** A cross-encoder reads query and candidate *together*, so it orders the shortlist far more accurately
than bi-encoder cosine — but it only re-orders the pool the first stage produced. **It cannot add a chunk the
retriever never surfaced**, so recall is capped by `pool_k`. This is exactly "retrieve wide (recall), rerank
narrow (precision)."

---

# Stage 4 — Augmentation & Generation

## 4.1 ⭐⭐ Stuff vs. Map-Reduce vs. Refine on a multi-hop question

```python
q = "Which plan gives SSO and a 99.99% SLA?"
hits = hybrid.retrieve(q, k=6)

ans_stuff  = generate_answer(llm, q, stuff(hits, max_tokens=2000), grounded=True)
ans_mapred = map_reduce(llm, q, hits)
ans_refine = refine(llm, q, hits)

for name, a in [("STUFF", ans_stuff), ("MAP-REDUCE", ans_mapred), ("REFINE", ans_refine)]:
    print(f"\n== {name} ==\n{a}")
```

**Expected result.** All three should converge on **Enterprise**. `map_reduce` and `refine` tend to be the
most *complete* — they read each chunk deliberately, so they reliably pull SSO from `acme_security.md` and the
99.99% figure from `acme_sla.md`/`acme_pricing.md` and stitch them. `stuff` is one call and cheapest, but if
the answer sentences sit in the middle of a long stuffed context the model may under-weight one fact
("lost in the middle"). `map_reduce` makes N+1 calls; `refine` makes N sequential calls (slowest).

**Why.** Retrieval already decided *which* chunks; augmentation decides *how* they enter the prompt, and that
alone changes completeness, cost, and latency. Same evidence, three assemblies, three different answers.

---

## 4.2 ⭐⭐ Grounded vs. ungrounded on a NEGATIVE question

```python
q = "Does Acme Cloud offer a native mobile app for iOS and Android?"   # not in the corpus
ctx = stuff(dense.retrieve(q, k=4), max_tokens=1500)

print("GROUNDED  :", generate_answer(llm, q, ctx, grounded=True))
print("UNGROUNDED:", generate_answer(llm, q, ctx, grounded=False))
print("refusal_correct(grounded) =", refusal_correct(generate_answer(llm, q, ctx, grounded=True)))
```

**Expected result.** The **grounded** prompt returns *"I don't know based on the provided context."*
(`refusal_correct → True`). The **ungrounded** prompt ("answer however you like") invents a plausible answer
— e.g. *"Yes, Acme offers native iOS and Android apps..."* — pulled from the model's parametric memory, with
no support in the context.

**Why.** The only difference was one template. The grounded system prompt's rule — *"If the context does not
contain the answer, reply exactly 'I don't know...'"* — kills the hallucination. The failure was a **prompt**
decision, not a model weakness; that is why prompting is a first-class RAG design stage. *(The offline mock
LLM is extractive and won't fabricate, so run this with a real key to see the invented answer; the template
difference is still visible offline.)*

---

## 4.3 ⭐⭐ Verify citations point to real sources

```python
out = answer_with_sources(llm, dense, "How often are encryption keys rotated?", k=4)
print(out["answer"])
for s in out["sources"]:
    print(s["id"], s["label"], "->", s["source"])
# check the cited fact really lives in the cited file
assert "90 days" in open("data/corpus/acme_security.md").read()
```

**Expected result.** The answer states the keys rotate **every 90 days** with a `[n]` tag, and `sources[n-1]`
resolves to `acme_security.md`, which indeed contains "rotated automatically every 90 days."

**Why.** Numbered context + a citation instruction make the answer *auditable*: you can mechanically verify
each `[n]` points to a document that actually contains the claim. Trust in RAG comes from provenance you can
check, not from fluent prose.

---

## 4.4 ⭐⭐⭐ Blow the context budget on purpose

```python
q = "What service credit do I get if uptime falls below 95%?"
hits = hybrid.retrieve(q, k=8)

ctx_full  = format_context(hits, max_tokens=None)   # budgeting OFF
ctx_tight = format_context(hits, max_tokens=300)    # very tight
print("tokens: full =", token_len(ctx_full), "| tight =", token_len(ctx_tight))
print("FULL :", generate_answer(llm, q, ctx_full))
print("TIGHT:", generate_answer(llm, q, ctx_tight))
```

**Expected result.** `token_len(ctx_full)` is large (all 8 chunks); `ctx_tight` stops after ~1–2 chunks
(later chunks are dropped, not mangled). If the SLA credit chunk is ranked high, the tight context still
answers "50% credit"; if it was ranked 3rd+, the tight budget **cuts it** and the model must refuse or guess.
The full context answers but at higher token cost and risks lost-in-the-middle on longer inputs.

**Why.** Augmentation is a real cost↔correctness knob. No budget → bloat and attention dilution; too tight →
you may amputate the answer-bearing chunk. Suddenly *retrieval ordering* (which chunk is rank 1 vs. rank 4)
determines correctness — the stages are coupled.

---

# Stage 5 — Evaluation

## 5.1 ⭐ Baseline scorecard

```python
retr, _, _ = build_dense(lambda d: recursive_chunk(d, 500, 50))
print(evaluate_retriever(retr, examples, k=4))

for q in ["How much does the Growth plan cost per month?",
          "What encryption does Acme use for data at rest?",
          "When does scheduled maintenance happen?"]:
    out = answer_with_sources(llm, retr, q, k=4)
    f = faithfulness_heuristic(out["answer"], out["context"], embedder)
    r = answer_relevance_heuristic(out["answer"], q, embedder)
    print(f"faith={f:.2f} rel={r:.2f} | {q}")
```

**Expected result.** A retrieval scorecard like `{'precision@4': ~0.47, 'recall@4': ~0.85, 'mrr': ~0.88,
'n_evaluated': 18}` (the **3 negatives are skipped** automatically), and faithfulness/relevance heuristics
near the top of their 0–1 range for these clean factual questions.

**Why.** This is the number every later change is defended against. Crucially, retrieval metrics and
generation heuristics measure *different things* — retrieval asks "did we find the right chunk?", faithfulness
asks "is the answer supported by what we found?" Keeping them separate is what makes diagnosis possible (5.3).

---

## 5.2 ⭐⭐ Extend the eval dataset (with a negative)

```python
new = [
    EvalExample(question="How are webhook payloads signed?",
                ground_truth="With an HMAC-SHA256 signature in the X-Acme-Signature header.",
                relevant_ids=["acme_api.md"]),
    EvalExample(question="How long are deleted objects kept before permanent deletion?",
                ground_truth="Deleted objects are retained in a recoverable state for 30 days.",
                relevant_ids=["acme_faq.html"]),
    EvalExample(question="Does Acme offer a Slack integration?",   # NEGATIVE — not in corpus
                ground_truth="I don't know based on the provided context.",
                is_negative=True, relevant_ids=[]),
]
examples_ext = examples + new

retr, _, _ = build_dense(lambda d: recursive_chunk(d, 500, 50))
print(evaluate_retriever(retr, examples_ext, k=4))     # answerable metrics (negatives skipped)

# the negative must be REFUSED by a grounded pipeline
neg = new[-1].question
ans = answer_with_sources(llm, retr, neg, k=4, grounded=True)["answer"]
print("negative answer:", ans, "| refusal_correct =", refusal_correct(ans))
```

**Expected result.** `n_evaluated` rises from 18 to 20 (two new answerable items added; the negative is
skipped). The two new answerable questions are found (`acme_api.md`, `acme_faq.html` in top-4). The Slack
negative returns *"I don't know based on the provided context."* → `refusal_correct(ans) is True`.

**Prediction check.** Adding a negative does **not** move precision/recall — `evaluate_retriever` skips
`is_negative` items. It tests a different axis.

**Why.** Negatives measure *refusal*, not retrieval. A system that answers everything confidently is unsafe;
a good eval set deliberately includes questions the corpus can't answer so you can prove the system knows its
limits.

---

## 5.3 ⭐⭐⭐ The "bad RAG output" diagnosis

**Decision procedure.**

```
1. Run the query through the pipeline; open result["trace"].
2. Is the ground-truth SOURCE present in trace.retrieved?
      NO  -> RETRIEVAL failure.  Fix: re-chunk, add hybrid/BM25, raise k, relax filter.
      YES -> go to 3.
3. Right chunk retrieved but answer wrong/hallucinated?
      -> GENERATION failure. Fix: ground the prompt, change injection, better model.
   Confirm with faithfulness_heuristic(answer, context): low despite good context => generation.
```

```python
def diagnose(pipeline, q, gold_source):
    res = pipeline.query(q)
    print(res["trace"].show())
    retrieved_sources = {r["label"] for r in res["trace"].retrieved}
    found = any(gold_source.split(".")[0] in lbl.lower() or gold_source in lbl
                for lbl in retrieved_sources)
    faith = faithfulness_heuristic(res["trace"].answer, res["trace"].context, embedder)
    if not found:
        return "RETRIEVAL failure — gold source never made the top-k"
    if faith < 0.5 or not res["trace"].answer.strip():
        return "GENERATION failure — right context, unfaithful answer"
    return "OK"

# Case A: induce a RETRIEVAL failure with tiny fragmenting chunks + tiny k
bad_retrieval = RAGPipeline(chunker=lambda d: recursive_chunk(d, 120, 0), k=2).ingest(docs)
print(diagnose(bad_retrieval, "Which plan gives SSO and a 99.99% SLA?", "acme_sla.md"))

# Case B: induce a GENERATION failure with an ungrounded prompt on a negative
bad_generation = RAGPipeline(chunker=lambda d: recursive_chunk(d, 500, 50), k=4, grounded=False).ingest(docs)
print(diagnose(bad_generation, "What is the CEO of Acme Cloud's name?", "acme_faq.html"))
```

**Expected result.** Case A reports a **retrieval** failure — with 120-char chunks and `k=2` the multi-hop
evidence is fragmented and the SLA source misses the top-k. Case B reports a **generation** failure — the
ungrounded pipeline fabricates a CEO name even though the retrieved context (correctly) contains nothing about
a CEO, so faithfulness is low.

**Why.** The two failures have *opposite* fixes: a retrieval miss is cured by chunking/retriever/k changes; a
generation miss is cured by grounding/injection/model changes. Applying the retrieval fix to a generation bug
(or vice-versa) wastes hours. **Metrics that separate retrieval from generation are what turn debugging from
guesswork into engineering** — the reason the evaluation stage exists.

---

# Capstone — The best pipeline you can assemble

## C ⭐⭐⭐ Maximise the eval-set score

```python
best = RAGPipeline(
    embedder=embedder,
    llm=llm,
    chunker=lambda d: recursive_chunk(d, 500, 50),                        # whole-idea chunks
    retriever_factory=lambda store, emb: HybridRetriever(                 # recall net
        DenseRetriever(store, emb), BM25Retriever(store.documents),
        rrf_c=60, pool_k=10),
    reranker=(CrossEncoderReranker() if getattr(CrossEncoderReranker(), "available", False)
              else LLMReranker(llm)),                                     # precision scalpel
    injection="stuff",                                                    # budgeted, grounded
    k=5,
    grounded=True,
).ingest(docs)

# --- Retrieval axis ---
print("retrieval:", evaluate_retriever(best.retriever, examples, k=5))

# --- Generation axis: faithful answers + refuse ALL negatives ---
faiths, refusals = [], []
for ex in examples:
    res = best.query(ex.question)
    if ex.is_negative:
        refusals.append(refusal_correct(res["answer"]))
    else:
        faiths.append(faithfulness_heuristic(res["answer"], res["trace"].context, embedder))
print(f"mean faithfulness = {sum(faiths)/len(faiths):.2f}")
print(f"negatives refused = {sum(refusals)}/{len(refusals)}")
```

**Expected result (illustrative).**

```
retrieval: {'precision@5': ~0.55, 'recall@5': ~0.97, 'hit_rate@5': 1.00, 'mrr': ~0.95, 'n_evaluated': 18}
mean faithfulness = ~0.8
negatives refused = 3/3
```

Now do the **ablations** — flip one knob back to a poor choice and re-score. Each swap drops the composite:

| Change from best | Effect |
|---|---|
| hybrid → dense only | recall@5 falls (~0.97 → ~0.85); exact-term queries like HTTP 429 slip |
| recursive 500/50 → fixed 200/0 | multi-hop recall falls; fragmentation |
| reranker → none | precision@5 / MRR fall (worse ordering) |
| grounded → ungrounded | negatives refused drops 3/3 → 0/3; faithfulness falls |
| k=5 → k=1 | multi-hop answers lose co-occurring evidence |

**Why (the takeaway).** No single knob produced the score — it is the **stack**: clean loading (Stage 1),
whole-idea recursive chunks (Stage 2), hybrid recall + reranker precision (Stage 3), a budgeted grounded
prompt (Stages 4–5), and negatives in the eval set keeping you honest (Stage 6). Revert *any one* to a bad
choice and the number drops. That is the session's entire thesis, proven on your own metrics:
**RAG quality is the cumulative result of design decisions across all pipeline stages, not just the LLM.**

---

## Self-check answers

1. **Loading→retrieval:** un-cleaned nav/footer/HTML boilerplate gets embedded and later retrieved instead of
   the answer — born in Stage 1, only visible as a bad hit in Stage 3.
2. **Chunk sweet spot:** too small fragments multi-fact ideas (recall drops); too large dilutes the relevant
   sentence inside one averaged vector (similarity drops). Both hurt, so there's an interior optimum.
3. **BM25 wins** "HTTP 429" / "AES-256" (exact rare tokens); **dense wins** the Gmail paraphrase (meaning
   match with no shared keywords).
4. **A reranker cannot raise recall** beyond the candidate pool — it only re-orders what retrieval already
   found.
5. The grounding rule: *"If the context does not contain the answer, reply exactly 'I don't know based on the
   provided context.'"*
6. First look at whether the **gold source appears in `trace.retrieved`** (retrieval), then at
   **faithfulness** on the retrieved context (generation).
```
