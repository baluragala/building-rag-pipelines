"""
loaders.py — Stage 1 of the pipeline: LOADING (data ingestion).
===============================================================

WHY
---
Retrieval can only ever surface what ingestion let in. Loading is the quiet
stage everyone skips — and it is where a shocking share of RAG failures are
*born*. "What happens if this step is poorly designed?" A PDF loaded as one
giant blob with page furniture (headers, footers, page numbers) glued into the
sentences will chunk badly, embed noisily, and retrieve the wrong spans. The
model never had a chance. Garbage in -> garbage retrieved -> confident garbage
out.

WHAT (do's & don'ts, from the agenda)
-------------------------------------
DO:   clean text, preserve structure, attach metadata (source, title, page).
DON'T: dump raw bytes, ignore encoding/noise, or throw away where a span came
       from (you need that for citations and metadata filtering later).

HOW
---
We define a tiny `Document` (page_content + metadata) — the same shape LangChain
uses, so the "parallel mapping" to LangChain DocumentLoaders is a one-liner.
Then we implement manual loaders for text, Markdown, HTML, PDF and a simple API,
plus a `clean_text` helper. Every loader ATTACHES METADATA — that discipline is
the whole point of this stage.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Document:
    """
    The atom of the pipeline. `page_content` is the text; `metadata` carries
    everything you'll want later: the source path, a title, a page number, and
    (after chunking) chunk indices. Keeping metadata on the document is what
    makes citations and metadata-filtered retrieval possible downstream.
    """

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.page_content[:70].replace("\n", " ")
        return f"Document({preview!r}…, metadata={self.metadata})"


# ---------------------------------------------------------------------------
# Cleaning — the "clean text, don't ignore noise" do.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Conservative cleanup that preserves paragraph structure:
      * normalise Windows/Mac line endings,
      * collapse runs of blank lines to a single blank (keeps paragraph breaks),
      * collapse intra-line whitespace,
      * strip a few common PDF artefacts (form-feeds, isolated page numbers).

    We intentionally do NOT strip newlines entirely — paragraph boundaries are
    semantic signal the recursive/semantic chunkers will use later.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")  # form feed (PDF page break)
    # Remove lines that are just a page number, e.g. "  12  ".
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Collapse 3+ newlines to a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse spaces/tabs but keep newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Manual loaders (the "from scratch" story).
# ---------------------------------------------------------------------------
def load_text(path: str, clean: bool = True, **extra_meta) -> List[Document]:
    """Load a plain-text / .md file as a single Document with metadata."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    content = clean_text(raw) if clean else raw
    meta = {
        "source": path,
        "title": _title_from_path(path),
        "loader": "load_text",
        **extra_meta,
    }
    return [Document(page_content=content, metadata=meta)]


def load_markdown(path: str, clean: bool = True, **extra_meta) -> List[Document]:
    """
    Markdown loader that captures the first H1 as the title. We keep the
    Markdown syntax (headings, lists) because it is *structure* — the recursive
    splitter can later break on headings, which preserves semantic boundaries.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    title = _first_markdown_h1(raw) or _title_from_path(path)
    content = clean_text(raw) if clean else raw
    meta = {"source": path, "title": title, "loader": "load_markdown", **extra_meta}
    return [Document(page_content=content, metadata=meta)]


def load_html(path_or_html: str, is_html_string: bool = False, **extra_meta) -> List[Document]:
    """
    HTML loader. Prefers BeautifulSoup (proper parsing, drops <script>/<style>);
    falls back to a crude tag-stripping regex if bs4 is unavailable so the demo
    still works offline.
    """
    if is_html_string:
        html, source = path_or_html, "<string>"
    else:
        with open(path_or_html, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        source = path_or_html

    title = None
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()  # DON'T embed navigation/boilerplate noise
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = soup.get_text(separator="\n")
    except Exception:
        # Offline fallback: strip tags with a regex (imperfect but functional).
        text = re.sub(r"<[^>]+>", " ", html)

    content = clean_text(text)
    meta = {
        "source": source,
        "title": title or _title_from_path(source),
        "loader": "load_html",
        **extra_meta,
    }
    return [Document(page_content=content, metadata=meta)]


def load_pdf(path: str, per_page: bool = True, **extra_meta) -> List[Document]:
    """
    PDF loader using pypdf. `per_page=True` returns ONE Document per page with a
    `page` number in metadata — this is the do: preserve structure and keep the
    provenance you need for page-level citations. If pypdf is missing we raise a
    friendly error rather than silently returning nothing.
    """
    try:
        from pypdf import PdfReader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "load_pdf needs `pypdf` (pip install pypdf). "
            "For the offline demo use the Markdown/HTML corpus instead."
        ) from e

    reader = PdfReader(path)
    title = (reader.metadata.title if reader.metadata else None) or _title_from_path(path)
    docs: List[Document] = []
    if per_page:
        for i, page in enumerate(reader.pages):
            content = clean_text(page.extract_text() or "")
            if not content:
                continue  # skip blank/scanned pages that yielded no text
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path,
                        "title": title,
                        "page": i + 1,
                        "loader": "load_pdf",
                        **extra_meta,
                    },
                )
            )
    else:
        content = clean_text(
            "\n\n".join(p.extract_text() or "" for p in reader.pages)
        )
        docs.append(
            Document(
                page_content=content,
                metadata={"source": path, "title": title, "loader": "load_pdf", **extra_meta},
            )
        )
    return docs


def load_api(url: str, json_path: str | None = None, timeout: int = 20, **extra_meta) -> List[Document]:
    """
    Ingest from an HTTP API/URL. `json_path` (dot-notation, e.g. "data.body")
    optionally extracts a field from a JSON response. Demonstrates that
    ingestion is not just files — many real corpora arrive over the wire.
    """
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        data = resp.json()
        text = _dig(data, json_path) if json_path else str(data)
        text = text if isinstance(text, str) else str(text)
    elif "html" in ctype:
        return load_html(resp.text, is_html_string=True, source=url, **extra_meta)
    else:
        text = resp.text
    meta = {"source": url, "title": url, "loader": "load_api", **extra_meta}
    return [Document(page_content=clean_text(text), metadata=meta)]


def load_directory(
    folder: str, patterns: tuple[str, ...] = ("*.md", "*.txt", "*.html", "*.pdf")
) -> List[Document]:
    """Walk a folder and dispatch each file to the right loader by extension."""
    docs: List[Document] = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True)):
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".pdf":
                    docs.extend(load_pdf(path))
                elif ext in (".html", ".htm"):
                    docs.extend(load_html(path))
                elif ext == ".md":
                    docs.extend(load_markdown(path))
                else:
                    docs.extend(load_text(path))
            except Exception as e:  # keep going if one file is broken
                print(f"[loaders] skipped {path}: {e}")
    return docs


# ---------------------------------------------------------------------------
# LangChain PARALLEL MAPPING (kept deliberately thin — not the primary focus).
# ---------------------------------------------------------------------------
def load_directory_langchain(folder: str):
    """
    The same job via LangChain DocumentLoaders. Shown so learners can map the
    from-scratch idea to the framework — and see that the framework abstracts
    the *loading*, not the *design decisions* (you still choose what to clean
    and what metadata to keep).
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader)
    return loader.load()  # returns LangChain Documents (same page_content/metadata shape)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _title_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0].replace("_", " ").replace("-", " ").title()


def _first_markdown_h1(text: str) -> str | None:
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _dig(data: Any, dotted: str) -> Any:
    cur = data
    for key in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key, "")
        else:
            return ""
    return cur
