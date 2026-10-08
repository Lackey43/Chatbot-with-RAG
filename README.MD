# LOTR RAG Chat

A retrieval-augmented chatbot over J.R.R. Tolkien's The Lord of the Rings.

Questions about the books are routed to a PostgreSQL + PGVector knowledge base,
which retrieves the most relevant passages before Google Gemini answers.
Everything else goes straight to the model, with no retrieval.

Built by Claude Daigan.

## What changed

This used to be a Streamlit app. It is now a FastAPI service that serves a
designed single page front end and streams answers token by token over
Server-Sent Events. Streamlit is no longer a dependency.

## Run locally

    pip install -r requirements.txt
    uvicorn server:app --reload --port 8000

Then open http://127.0.0.1:8000

A `.env` file in the project root supplies the credentials:

    GOOGLE_API_KEY=...
    OPENROUTER_API_KEY=...
    DATABASE_URL=postgresql+psycopg://user:password@host:5432/dbname

The app boots without them. The status line in the left rail and
`GET /api/health` both report exactly what is missing.

## Deploy (Dokku / Heroku)

The Procfile runs uvicorn on the platform-assigned port:

    web: uvicorn server:app --host 0.0.0.0 --port $PORT

Set the three variables on the platform, for example with Dokku:

    dokku config:set chatbot-with-rag GOOGLE_API_KEY=... OPENROUTER_API_KEY=... DATABASE_URL=...

Answers stream over Server-Sent Events, and nginx buffers upstream responses by
default, which would deliver the whole answer at once instead of token by token.
Turn that off for this app:

    dokku nginx:set chatbot-with-rag proxy-buffering off

## Endpoints

    GET  /                the single page app
    GET  /api/health      readiness, model, and any missing configuration
    POST /api/chat        {"message": "..."} -> Server-Sent Events token stream
    POST /api/agent/reset rebuild the cached vectorstore and model
    GET  /api/docs        OpenAPI docs

The `/api/chat` stream emits JSON frames: `route`, then `retrieval` (RAG path
only), then `token` for each chunk, then `done`. Failures arrive as a single
`error` frame.

## Loading documents

    python -c "from Actions.upload_documentsvector import upload; upload()"

Reads every file in Documents/, splits it, and writes the embeddings into the
PGVector collection. Requires DATABASE_URL.

## Layout

    server.py          FastAPI app: routes and the SSE stream
    rag_runtime.py     lazy runtime: PGVector + Gemini, routing, streaming
    static/            front end (index.html, styles.css, app.js)
    models/models.py   embeddings, chat model, structured-output router
    prompt/prompts.py  system prompts for the RAG and direct paths
    Actions/           document ingestion
    main.py            original command line version
