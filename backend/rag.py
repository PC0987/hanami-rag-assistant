"""Core RAG logic: chunking, embedding + retrieval with Chroma, and answer generation with Claude."""
import glob
import os
import re
import time

import chromadb
from anthropic import Anthropic

MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

# Two prompt versions so they can be compared in the eval (v2 is stricter about unknown answers).
PROMPTS = {
    "v1": (
        "You are a support assistant. Answer the question using the context below. "
        "Cite sources like [1]. Reply in the same language as the question."
    ),
    "v2": (
        "You are a support assistant. Answer ONLY from the numbered context below. "
        "If the context does not contain the answer, say you could not find it in the documentation "
        "and do not guess. Keep the answer short and cite sources like [1]. "
        "Reply in the same language as the question."
    ),
}

_llm = None


def llm() -> Anthropic:
    global _llm
    if _llm is None:
        _llm = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _llm


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Pack paragraphs into chunks of roughly `size` characters, with a small overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > size:
            chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if overlap else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def _embedding_kwargs() -> dict:
    """Default: Chroma's built-in English MiniLM. Set EMBEDDING_MODEL for a multilingual model."""
    name = os.getenv("EMBEDDING_MODEL")
    if not name:
        return {}
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return {"embedding_function": SentenceTransformerEmbeddingFunction(model_name=name)}


class RagIndex:
    def __init__(self, chunk_size: int = 500, top_k: int = 3, prompt_version: str = "v2"):
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.prompt_version = prompt_version
        tag = "ml" if os.getenv("EMBEDDING_MODEL") else "default"
        self.client = chromadb.Client()  # in-memory; the docs are small so we rebuild on startup
        self.col = self.client.get_or_create_collection(f"docs_{chunk_size}_{tag}", **_embedding_kwargs())
        if self.col.count() == 0:
            self._build()

    def _build(self):
        ids, docs, metas = [], [], []
        for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.md"))):
            name = os.path.basename(path)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for i, chunk in enumerate(chunk_text(text, self.chunk_size, overlap=self.chunk_size // 10)):
                ids.append(f"{name}-{i}")
                docs.append(chunk)
                metas.append({"source": name})
        self.col.add(ids=ids, documents=docs, metadatas=metas)

    def retrieve(self, question: str) -> list[dict]:
        res = self.col.query(query_texts=[question], n_results=self.top_k)
        return [
            {"text": t, "source": m["source"]}
            for t, m in zip(res["documents"][0], res["metadatas"][0])
        ]

    def answer(self, question: str) -> dict:
        start = time.time()
        passages = self.retrieve(question)
        context = "\n\n".join(f"[{i + 1}] ({p['source']}) {p['text']}" for i, p in enumerate(passages))
        resp = llm().messages.create(
            model=MODEL,
            max_tokens=500,
            system=PROMPTS[self.prompt_version],
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return {
            "answer": resp.content[0].text,
            "passages": passages,
            "context": context,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "latency_ms": int((time.time() - start) * 1000),
        }
