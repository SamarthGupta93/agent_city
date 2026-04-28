# Simple RAG

A conversational Retrieval-Augmented Generation (RAG) API built with LangGraph, PGVector, and Google Gemini. Supports multi-turn conversations with session persistence backed by PostgreSQL.

## Architecture

- **Retriever** — Google `gemini-embedding-001` embeddings stored in PGVector
- **Generator** — Google `gemini-2.5-flash-lite` LLM with conversation history
- **Session memory** — LangGraph `PostgresSaver` checkpointer (same Postgres instance)
- **API** — FastAPI served via Uvicorn inside Docker

## Prerequisites

- [Postgres.app](https://postgresapp.com/) (native macOS PostgreSQL with pgvector)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A Google AI API key ([aistudio.google.com](https://aistudio.google.com/))

## Setup

### 1. Enable pgvector in Postgres.app

Open a psql session and run:

```sql
CREATE DATABASE rag;
\c rag
CREATE EXTENSION IF NOT EXISTS vector;
```

> The `pgvector/pgvector:pg16` Docker image used by the app already has the extension available — the `CREATE EXTENSION` is handled automatically by LangChain on first connection. You only need the native Postgres.app database if you want to run the indexing pipeline outside Docker.

### 2. Add documents

Place PDF files you want to index in the `docs/raw/` directory:

```
docs/
  raw/
    your_document.pdf
    another_document.pdf
```

Only `.pdf` files are supported. The indexing pipeline will skip any other formats.

### 3. Configure environment

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
VECTOR_DB_URI="postgresql+psycopg://postgres:admin@localhost:5433/rag"
POSTGRES_SESSION_URI="postgresql://postgres:admin@localhost:5433/rag"
COLLECTION_NAME="documents"
GOOGLE_API_KEY="your-google-api-key"
GOOGLE_MODEL="gemini-2.5-flash-lite"
GOOGLE_EMBEDDING_MODEL="gemini-embedding-001"
```

> **Port note:** Docker maps the container's Postgres to host port `5433` to avoid conflicts with Postgres.app running on `5432`. Use `localhost:5433` in `.env` for local scripts; the Docker app service uses the internal hostname `postgres:5432` automatically via `docker-compose.yml`.

### 4. Start the application

```bash
docker compose up --build
```

This starts two services:
- `postgres` — PGVector-enabled PostgreSQL on host port `5433`
- `app` — FastAPI server on port `8000`

Wait for the health check to pass before indexing (you'll see `healthy` in the Docker logs).

### 5. Index documents

With Docker running, execute the indexing pipeline from your local environment:

```bash
uv run python main.py
```

This runs three sequential steps:
1. **Load** — extracts text and metadata from each PDF in `docs/raw/`
2. **Chunk** — splits text into overlapping chunks (1000 chars, 200 overlap) saved to `docs/chunks/`
3. **Index** — embeds each chunk and upserts into PGVector

Re-indexing is safe — chunks use deterministic IDs and upsert on conflict.

> If you add new documents later, drop the existing chunk files first to avoid stale data:
> ```bash
> rm docs/chunks/*.txt docs/chunks/*.json
> ```
> Then re-run `uv run python main.py`.

### 6. Interact with the API

Open the interactive API docs in your browser:

```
http://localhost:8000/docs
```

#### Chat endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-123",
    "message": "What topics does the document cover?"
  }'
```

Response:

```json
{
  "response": "The document covers ...",
  "doc_ids": ["abc123", "def456"]
}
```

Use the same `session_id` across requests to maintain conversation context.

#### Streaming endpoint

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session-123", "message": "Summarise section 3"}'
```

Returns Server-Sent Events with `doc_ids` followed by streamed `token` chunks.

## Project structure

```
simple-rag/
├── agents/simple_rag/   # ConversationalRAG (LangGraph + PostgresSaver)
├── rag/                 # SimpleRAG (single-turn, no session)
├── retriever/
│   ├── doc_processor.py # PDF loading and chunking pipelines
│   ├── indexer.py       # PGVector upsert pipeline
│   └── models.py        # Document dataclass
├── evals/               # Evaluation dataset
├── docs/
│   ├── raw/             # Drop PDFs here
│   ├── text/            # Intermediate extracted text (auto-generated)
│   └── chunks/          # Chunked documents (auto-generated)
├── api.py               # FastAPI app
├── main.py              # CLI entry point (indexing + local chat)
├── docker-compose.yml
└── .env.example
```

## Resetting the database

To wipe all vectors and session data and start fresh:

```bash
docker compose down -v   # removes the pgdata volume
docker compose up --build
```

Then re-run the indexing pipeline.
