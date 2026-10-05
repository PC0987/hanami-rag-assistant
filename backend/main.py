"""FastAPI app: /ask (RAG answer), /feedback (thumbs up/down), /stats (dashboard data)."""
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
from rag import RagIndex

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    state["index"] = RagIndex(
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        top_k=int(os.getenv("TOP_K", "3")),
        prompt_version=os.getenv("PROMPT_VERSION", "v2"),
    )
    yield


app = FastAPI(title="Hanami Cloud support assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class FeedbackRequest(BaseModel):
    id: int
    value: int = Field(description="1 for thumbs up, -1 for thumbs down")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        result = state["index"].answer(req.question)
    except Exception as e:  # e.g. missing API key or LLM outage
        log.error(json.dumps({"event": "ask_error", "error": str(e)}))
        raise HTTPException(status_code=502, detail="The answer service failed. Check the server logs.")
    sources = sorted({p["source"] for p in result["passages"]})
    qid = db.log_query(
        req.question, result["answer"], sources,
        result["latency_ms"], result["input_tokens"], result["output_tokens"],
    )
    log.info(json.dumps({
        "event": "ask", "id": qid, "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
        "sources": sources,
    }))
    return {
        "id": qid,
        "answer": result["answer"],
        "sources": result["passages"],
        "latency_ms": result["latency_ms"],
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.value not in (1, -1):
        raise HTTPException(status_code=400, detail="value must be 1 or -1")
    if not db.set_feedback(req.id, req.value):
        raise HTTPException(status_code=404, detail="query not found")
    log.info(json.dumps({"event": "feedback", "id": req.id, "value": req.value}))
    return {"ok": True}


@app.get("/stats")
def stats():
    return db.stats()
