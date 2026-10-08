"""FastAPI server for the Lord of the Rings RAG chatbot.

Replaces the Streamlit app: serves a designed single page app at /, exposes
GET /api/health and a token-streaming POST /api/chat (Server-Sent Events).

Local:
    uvicorn server:app --reload --port 8000
Dokku / Heroku (see Procfile):
    uvicorn server:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_runtime

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="LOTR RAG Chat",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str | None = None
    prompt: str | None = None

    def text(self) -> str:
        return (self.message or self.prompt or "").strip()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    """Report what the app can do right now, without forcing a build."""
    return {
        "status": "ok",
        "app": "LOTR RAG Chat",
        "model": rag_runtime.model_name(),
        "retrieval_k": rag_runtime.RETRIEVAL_K,
        # ready: the model path works (routing + general questions)
        "ready": rag_runtime.models_ready(),
        # knowledge_base_ready: PGVector is reachable (Tolkien questions)
        "knowledge_base_ready": rag_runtime.store_ready(),
        "missing_env": rag_runtime.missing_env(),
        "runtime_error": rag_runtime.models_error() or rag_runtime.store_error(),
        "models_error": rag_runtime.models_error(),
        "store_error": rag_runtime.store_error(),
    }


@app.post("/api/agent/reset")
def reset_runtime() -> dict:
    """Clear the cached vectorstore/model so the next request rebuilds it."""
    rag_runtime.reset()
    return {"status": "ok", "message": "Runtime cache cleared."}


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    message = request.text()

    def event_stream():
        for frame in rag_runtime.stream_answer(message):
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "close"}\n\n'

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # stop nginx/Dokku from buffering the stream so tokens arrive live
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
