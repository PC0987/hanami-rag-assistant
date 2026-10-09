"""Core RAG logic: chunking, embedding + retrieval with Chroma, and answer generation with Claude or Gemini."""
import glob
import os
import re
import time

import chromadb


def _provider() -> str:
    """LLM_PROVIDER=anthropic|google. If unset, use Google when only a Google key is present."""
    name = os.getenv("LLM_PROVIDER", "").lower()
    if name in ("anthropic", "google"):
        return name
    has_google = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return "google" if has_google and not os.getenv("ANTHROPIC_API_KEY") else "anthropic"


PROVIDER = _provider()
DEFAULT_MODELS = {"anthropic": "claude-haiku-4-5-20251001", "google": "gemini-2.5-flash"}
MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS[PROVIDER])
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

_clients: dict = {}


def llm():
    """The provider's SDK client, created on first use."""
    if PROVIDER not in _clients:
        if PROVIDER == "google":
            from google import genai

            _clients[PROVIDER] = genai.Client()  # reads GEMINI_API_KEY or GOOGLE_API_KEY
        else:
            from anthropic import Anthropic

            _clients[PROVIDER] = Anthropic()  # reads ANTHROPIC_API_KEY
    return _clients[PROVIDER]


def generate(prompt: str, system: str | None = None, max_tokens: int = 500, model: str | None = None) -> dict:
    """Call the configured LLM. Returns {"text", "input_tokens", "output_tokens"}."""
    model = model or MODEL
    if PROVIDER == "google":
        from google.genai import types

        resp = llm().models.generate_content(
            model=model,
            contents=prompt,
            # Gemini 2.5 models count hidden "thinking" tokens against the limit, so leave headroom
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens + 1024),
        )
        usage = resp.usage_metadata
        return {
            "text": resp.text or "",
            "input_tokens": (usage.prompt_token_count or 0) if usage else 0,
            "output_tokens": (usage.candidates_token_count or 0) if usage else 0,
        }
    kwargs = {"system": system} if system else {}
    resp = llm().messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kwargs
    )
    return {
        "text": resp.content[0].text,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


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
        out = generate(
            f"Context:\n{context}\n\nQuestion: {question}",
            system=PROMPTS[self.prompt_version],
            max_tokens=500,
        )
        return {
            "answer": out["text"],
            "passages": passages,
            "context": context,
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "latency_ms": int((time.time() - start) * 1000),
        }
