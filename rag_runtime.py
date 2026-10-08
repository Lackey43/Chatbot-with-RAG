"""Lazy runtime for the Lord of the Rings RAG chatbot.

Every heavy import (langchain, PGVector, Google Gemini) and every credential
check is deferred to first use, so the web server boots and serves the UI even
when the API keys or the vector database are not reachable yet.

The chat model and the vectorstore are built independently:

* the model path (routing + general answers) needs only the API keys, so
  general questions keep working while PostgreSQL is down;
* the vectorstore is built on demand, the first time a Tolkien question needs
  retrieval, and a connection failure is reported for that one request instead
  of taking the whole app offline.

GET /api/health reports the state of both.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterator

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional; the platform may inject env vars
    pass

# The model name is defined in models/models.py; reported here for /api/health.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
RETRIEVAL_K = 8

REQUIRED_ENV = ("GOOGLE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL")

_MODELS: dict[str, Any] | None = None
_MODELS_ERROR: str | None = None
_STORE: Any = None
_STORE_ERROR: str | None = None
_LOCK = threading.Lock()


def missing_env() -> list[str]:
    """Names of required environment variables that are not set."""
    return [name for name in REQUIRED_ENV if not os.getenv(name)]


def model_name() -> str:
    if _MODELS is not None:
        return getattr(_MODELS["google_model"], "model", DEFAULT_MODEL)
    return DEFAULT_MODEL


def models_ready() -> bool:
    return _MODELS is not None


def store_ready() -> bool:
    return _STORE is not None


def models_error() -> str | None:
    return _MODELS_ERROR


def store_error() -> str | None:
    return _STORE_ERROR


def reset() -> None:
    """Drop everything cached so the next request rebuilds it."""
    global _MODELS, _MODELS_ERROR, _STORE, _STORE_ERROR
    with _LOCK:
        _MODELS = None
        _MODELS_ERROR = None
        _STORE = None
        _STORE_ERROR = None


def _summarise(exc: BaseException) -> str:
    """One-line, length-capped description of an exception for the UI.

    Tracebacks from the database driver are enormous (every pooler host is
    listed), so keep the first line and cap the rest.
    """
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    message = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    return message if len(message) <= 300 else message[:297] + "..."


def _build_models() -> dict[str, Any]:
    """Embeddings, chat model, router and prompts. No network calls."""
    from models.models import embeddings_model, google_model, query_checker
    from prompt.prompts import default_prompt, simple_prompt

    return {
        "embeddings_model": embeddings_model,
        "google_model": google_model,
        "query_checker": query_checker,
        "simple_prompt": simple_prompt,
        "default_prompt": default_prompt,
    }


def get_models() -> dict[str, Any]:
    """Return the model bundle, building it once. Raises RuntimeError."""
    global _MODELS, _MODELS_ERROR
    if _MODELS is not None:
        return _MODELS

    with _LOCK:
        if _MODELS is not None:
            return _MODELS

        missing = missing_env()
        if missing:
            _MODELS_ERROR = "Missing environment variables: " + ", ".join(missing)
            raise RuntimeError(_MODELS_ERROR)

        try:
            _MODELS = _build_models()
            _MODELS_ERROR = None
        except Exception as exc:  # noqa: BLE001 - message is shown to the user
            _MODELS_ERROR = _summarise(exc)
            raise RuntimeError(_MODELS_ERROR) from exc

    return _MODELS


def get_retriever() -> Any:
    """Return the PGVector retriever, building it once. Raises RuntimeError.

    Building connects to PostgreSQL and creates the vector extension if it is
    missing, so an unreachable DATABASE_URL surfaces here.
    """
    global _STORE, _STORE_ERROR
    if _STORE is not None:
        return _STORE

    with _LOCK:
        if _STORE is not None:
            return _STORE

        models = get_models()
        try:
            from langchain_postgres import PGVector

            store = PGVector(
                embeddings=models["embeddings_model"],
                connection=os.getenv("DATABASE_URL"),
            )
            _STORE = store.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            _STORE_ERROR = None
        except Exception as exc:  # noqa: BLE001 - message is shown to the user
            _STORE_ERROR = _summarise(exc)
            raise RuntimeError(_STORE_ERROR) from exc

    return _STORE


def _is_lotr(query: str, query_checker: Any) -> bool:
    """Route the question: Tolkien-related, or general?"""
    response = query_checker.invoke(query)
    return bool(getattr(response, "is_about_lotr", False))


def _stream_text(chain: Any, inputs: dict[str, Any]) -> Iterator[str]:
    """Yield text chunks from an LCEL chain, tolerating chunk formats."""
    for chunk in chain.stream(inputs):
        content = getattr(chunk, "content", None)
        if content is None:
            yield str(chunk)
        elif isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    if text:
                        yield text
                elif isinstance(part, str):
                    yield part
        else:
            yield str(content)


def _clean_metadata(metadata: Any) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def stream_answer(query: str) -> Iterator[dict[str, Any]]:
    """Yield JSON-serialisable frames for one question.

    Frame sequence: route -> retrieval (only when routed to RAG) -> token* ->
    done. Any failure yields a single error frame instead.
    """
    query = (query or "").strip()
    if not query:
        yield {"type": "error", "message": "Please enter a question."}
        return

    try:
        models = get_models()
    except RuntimeError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    try:
        is_lotr = _is_lotr(query, models["query_checker"])
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"Routing failed: {type(exc).__name__}: {exc}"}
        return

    if is_lotr:
        path = "RAG - Lord of the Rings knowledge base"
        yield {"type": "route", "is_lotr": True, "path": path}

        try:
            documents = get_retriever().invoke(query)
        except RuntimeError as exc:
            yield {"type": "error", "message": f"Knowledge base unavailable: {exc}"}
            return
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"Retrieval failed: {type(exc).__name__}: {exc}"}
            return

        chunks = [
            {"content": doc.page_content, "metadata": _clean_metadata(getattr(doc, "metadata", {}))}
            for doc in documents
        ]
        yield {"type": "retrieval", "count": len(chunks), "chunks": chunks}

        context = "\n\n".join(doc.page_content.replace("\n\n", " ") for doc in documents)
        chain = models["simple_prompt"] | models["google_model"]
        inputs: dict[str, Any] = {"context": context, "question": query}
    else:
        path = "Direct model - general knowledge"
        yield {"type": "route", "is_lotr": False, "path": path}
        chain = models["default_prompt"] | models["google_model"]
        inputs = {"question": query}

    answer = ""
    try:
        for token in _stream_text(chain, inputs):
            answer += token
            yield {"type": "token", "text": token}
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"Generation failed: {type(exc).__name__}: {exc}"}
        return

    yield {"type": "done", "answer": answer, "path": path, "is_lotr": is_lotr}
