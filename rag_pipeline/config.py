"""
config.py — provider-agnostic factory for the LLM and the embedding model.
============================================================================

WHY this file exists
--------------------
The instructor notes for this session say: *"keep the session agnostic of
embedding model/LLM choice."* But hands-on code has to actually run. Those two
goals only coexist if the *choice of provider* lives behind a single, small
interface. That is what this module is: one place that answers two questions —

    1. "Give me something that turns text into vectors."   -> get_embedder()
    2. "Give me something that turns a prompt into text."  -> get_llm()

Everything else in `rag_pipeline` depends on the tiny `Embedder` / `LLM`
protocols below and never imports `openai` or `anthropic` directly. Swap a
provider by changing an environment variable; no pipeline code changes.

WHAT you get
------------
* Default stack = OpenAI (`text-embedding-3-small` + `gpt-4o-mini`).
* Optional Anthropic Claude LLM adapter (Anthropic has no embeddings API, so
  embeddings still come from OpenAI or a local model).
* Optional local `sentence-transformers` embeddings (free, no key).
* A deterministic **mock** fallback for BOTH so the notebooks render end-to-end
  with zero API keys — invaluable in a classroom where not everyone has a key.

The mock is not a toy afterthought: a hash-based embedding is still a real
vector space, so cosine similarity, chunking comparisons, and the vector-store
mechanics all behave sensibly. Only the *semantic quality* is missing.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

import numpy as np

# Load a .env file if python-dotenv is available (Colab users usually set env
# vars via getpass instead, which also works — see the notebook bootstrap).
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# The two interfaces the whole pipeline is written against.
# ---------------------------------------------------------------------------
@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn a list of strings into a 2-D float array."""

    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:  # (n_texts, dim)
        ...


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a prompt (+ optional system) into a string."""

    name: str

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        ...


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------
class MockEmbedder:
    """
    Deterministic, offline embeddings. Each text is hashed into a fixed-length
    pseudo-random vector, then L2-normalised. Same text -> same vector, and
    texts sharing tokens land nearer each other than random pairs, so the
    *mechanics* of retrieval are demonstrable without any API call.

    Do NOT use this for real quality claims — it has no semantic understanding.
    It exists so a learner with no key can still run every notebook.
    """

    def __init__(self, dim: int = 384):
        self.name = "mock-embedder"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        # Build a vector by hashing each lowercased token into the dimension
        # space — a poor-man's bag-of-words hashing embedding.
        v = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        # A tiny hash of the whole string breaks ties so identical bags differ.
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        v += np.random.default_rng(seed).normal(0, 0.01, self.dim).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])


class OpenAIEmbedder:
    """OpenAI embeddings — the default. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI  # imported lazily so the mock path needs nothing

        self.client = OpenAI()
        self.name = f"openai:{model}"
        self.model = model
        # Dimensions for the small/large models; queried lazily on first call.
        self.dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(
            model, 1536
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        # The API accepts a batch; we send everything in one call for the small
        # corpora used in this session. For big corpora you would batch here.
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class SentenceTransformerEmbedder:
    """Local, free embeddings via sentence-transformers. No API key needed."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=False), dtype=np.float32
        )


# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
class MockLLM:
    """
    Offline stand-in for a generator. It does NOT invent facts: it produces a
    grounded, extractive answer by stitching together the retrieved context it
    is given. This is deliberately honest — it lets us demonstrate the *shape*
    of a RAG answer (grounded, with sources) even with no key, and it makes the
    "generation quality" gap obvious when a learner later plugs in a real LLM.
    """

    def __init__(self):
        self.name = "mock-llm"

    # A tiny stopword set so the grounding heuristic keys off content words only.
    _STOP = {
        "what", "which", "does", "have", "with", "your", "that", "this", "from",
        "when", "where", "will", "would", "should", "could", "there", "their",
        "about", "into", "much", "many", "long", "acme", "cloud", "offer", "provide",
    }

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        import re

        marker = "CONTEXT:"
        if marker not in prompt:
            # Non-RAG prompt (e.g. query rewriting / reranking) — echo an honest note.
            return (
                "[mock-llm] (offline stand-in) A real LLM would produce a fluent answer "
                "here. Set RAG_LLM_PROVIDER=openai with a key for real output."
            )

        # Split the context block from any trailing QUESTION so overlap is clean.
        after = prompt.split(marker, 1)[1]
        context_only = after.split("QUESTION:", 1)[0] if "QUESTION:" in after else after

        # Grounding heuristic: if the prompt is GROUNDED (its system prompt carries the
        # "I don't know" rule) and the question shares NO content word with the context,
        # refuse — mirroring what a real grounded LLM does on an unanswerable question.
        grounded = bool(system) and "i don't know" in system.lower()
        if grounded and "QUESTION:" in prompt:
            question = prompt.split("QUESTION:", 1)[1].split("\n")[0]
            qwords = {
                w for w in re.findall(r"[a-z0-9]{4,}", question.lower())
                if w not in self._STOP
            }
            ctx_low = context_only.lower()
            if qwords and not any(w in ctx_low for w in qwords):
                return "I don't know based on the provided context."

        snippet = " ".join(context_only.split())[:600]
        return (
            "[mock-llm] Based only on the retrieved context, here is a grounded "
            f"extract:\n{snippet}\n\n(Plug in a real LLM via RAG_LLM_PROVIDER for "
            "a fluent, synthesised answer.)"
        )


class OpenAIChat:
    """OpenAI chat completion — the default generator. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI()
        self.name = f"openai:{model}"
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.choices[0].message.content or ""


class AnthropicChat:
    """
    Anthropic Claude generator (optional). Requires ANTHROPIC_API_KEY.

    NOTE: Anthropic has no embeddings endpoint, so even when you pick Claude for
    generation your embeddings still come from OpenAI or a local model. That is
    a real, teachable architecture point — retrieval and generation are
    genuinely separable and can use different providers.
    """

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        self.name = f"anthropic:{model}"
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "You are a helpful, grounded assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate any text blocks in the response.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------------------
# Factories — the only functions the rest of the pipeline calls.
# ---------------------------------------------------------------------------
@dataclass
class RagConfig:
    """A snapshot of the active provider choices, handy for logging in demos."""

    llm_provider: str
    llm_model: str
    embed_provider: str
    embed_model: str

    def __str__(self) -> str:
        return (
            f"LLM={self.llm_provider}:{self.llm_model}  |  "
            f"Embeddings={self.embed_provider}:{self.embed_model}"
        )


def current_config() -> RagConfig:
    return RagConfig(
        llm_provider=os.getenv("RAG_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
        embed_provider=os.getenv("RAG_EMBED_PROVIDER", "openai"),
        embed_model=os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small"),
    )


def get_embedder(provider: str | None = None, model: str | None = None) -> Embedder:
    """
    Return an Embedder. Resolution order:
      explicit args -> env (RAG_EMBED_PROVIDER / RAG_EMBED_MODEL) -> openai.
    If the chosen provider can't be constructed (missing key/lib), we fall back
    to the mock and print a clear one-line notice so the classroom keeps moving.
    """
    provider = (provider or os.getenv("RAG_EMBED_PROVIDER", "openai")).lower()
    model = model or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
    try:
        if provider == "mock":
            return MockEmbedder()
        if provider in ("sentence-transformers", "st", "local"):
            return SentenceTransformerEmbedder(model)
        if provider == "openai":
            return OpenAIEmbedder(model)
        raise ValueError(f"Unknown embed provider: {provider}")
    except Exception as e:  # missing key, missing lib, offline, etc.
        print(f"[config] embedder '{provider}' unavailable ({e}); using MockEmbedder.")
        return MockEmbedder()


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """
    Return an LLM. Resolution order:
      explicit args -> env (RAG_LLM_PROVIDER / RAG_LLM_MODEL) -> openai.
    Falls back to the honest MockLLM if the provider can't be constructed.
    """
    provider = (provider or os.getenv("RAG_LLM_PROVIDER", "openai")).lower()
    model = model or os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    try:
        if provider == "mock":
            return MockLLM()
        if provider == "openai":
            return OpenAIChat(model)
        if provider == "anthropic":
            return AnthropicChat(model)
        raise ValueError(f"Unknown LLM provider: {provider}")
    except Exception as e:
        print(f"[config] LLM '{provider}' unavailable ({e}); using MockLLM.")
        return MockLLM()
